import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [3, 4, 5]

def _make_schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W

def _verify(U, V, W, n, m, p):
    r = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _try_rank(r, n, m, p, deadline, max_attempts=200):
    """Try to find a rank-r decomposition using structured random matrices."""
    small_coeffs = [-1, 0, 1]
    
    for attempt in range(max_attempts):
        if time.time() > deadline:
            return None
        
        # Generate U and V with small integer entries
        # Use a mix: some columns are basis-like, others are random small ints
        U = np.zeros((n * m, r))
        V = np.zeros((m * p, r))
        
        # Strategy: build U and V column by column
        # Each column of U has entries from small_coeffs
        # Each column of V has entries from small_coeffs
        # But we need the product structure to allow half-integer W solutions
        
        for c in range(r):
            # Randomly choose a pattern
            pattern = random.random()
            if pattern < 0.3:
                # Sparse: few non-zero entries
                U[:, c] = 0
                V[:, c] = 0
                nz_u = random.randint(1, min(3, n * m))
                nz_v = random.randint(1, min(3, m * p))
                u_idx = random.sample(range(n * m), nz_u)
                v_idx = random.sample(range(m * p), nz_v)
                for idx in u_idx:
                    U[idx, c] = random.choice([-1, 1])
                for idx in v_idx:
                    V[idx, c] = random.choice([-1, 1])
            elif pattern < 0.6:
                # Medium density
                U[:, c] = [random.choice(small_coeffs) for _ in range(n * m)]
                V[:, c] = [random.choice(small_coeffs) for _ in range(m * p)]
            else:
                # Dense with small values
                U[:, c] = [random.choice([-1, 0, 1]) for _ in range(n * m)]
                V[:, c] = [random.choice([-1, 0, 1]) for _ in range(m * p)]
        
        # Now try to solve for W
        W = np.zeros((p * n, r))
        valid = True
        
        for i in range(n):
            for k in range(p):
                # Build M: m x r where M[j, c] = U[i*m+j, c] * V[j*p+k, c]
                M = np.zeros((m, r))
                for j in range(m):
                    M[j, :] = U[i * m + j, :] * V[j * p + k, :]
                
                # Solve M @ w = ones(m)
                # We want w with entries in {0, 0.5, 1} (or more generally half-integers)
                # First try least squares
                try:
                    w_ls, residuals, rank_m, sv = np.linalg.lstsq(M, np.ones(m), rcond=1e-10)
                    
                    # Try rounding to half-integers
                    w_rounded = np.round(w_ls * 2) / 2
                    if np.allclose(M @ w_rounded, np.ones(m), atol=1e-8):
                        # Check if all entries are half-integers (they should be by construction)
                        # Also check they're in a reasonable range
                        if np.all(np.abs(w_rounded) <= 10):
                            W[k * n + i, :] = w_rounded
                        else:
                            valid = False
                            break
                    else:
                        # Try integer rounding
                        w_int = np.round(w_ls)
                        if np.allclose(M @ w_int, np.ones(m), atol=1e-8):
                            W[k * n + i, :] = w_int
                        else:
                            # Try to find a solution via null space
                            # If M has rank < r, there are multiple solutions
                            # Find particular solution + null space
                            if rank_m < r:
                                # Compute null space
                                _, _, vh = np.linalg.svd(M)
                                null_space = vh[rank_m:].T  # (r x (r-rank))
                                # Particular solution
                                w_p = w_ls
                                # Try small combinations in null space
                                found = False
                                for t1 in range(-2, 3):
                                    if found:
                                        break
                                    for t2 in range(-2, 3):
                                        if found:
                                            break
                                        w_candidate = w_p.copy()
                                        if null_space.shape[1] > 0:
                                            w_candidate += t1 * null_space[:, 0]
                                        if null_space.shape[1] > 1:
                                            w_candidate += t2 * null_space[:, 1]
                                        w_cand_rounded = np.round(w_candidate * 2) / 2
                                        if np.allclose(M @ w_cand_rounded, np.ones(m), atol=1e-8):
                                            if np.all(np.abs(w_cand_rounded) <= 10):
                                                W[k * n + i, :] = w_cand_rounded
                                                found = True
                                                break
                                if not found:
                                    valid = False
                                    break
                            else:
                                valid = False
                                break
                except:
                    valid = False
                    break
            
            if not valid:
                break
        
        if valid:
            # Full verification
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            if _verify(U_list, V_list, W_list, n, m, p):
                return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Schoolbook decomposition as fallback
    U_schoolbook, V_schoolbook, W_schoolbook = _make_schoolbook()
    
    # Lower bound on rank: max(n*m, m*p, p*n) = max(12, 20, 15) = 20
    # Upper bound: schoolbook = 60
    # Try ranks from 25 to 60, starting from lower values
    # But be strategic: try a few key values
    
    # Known results: for 3x4x5, the rank is likely in the 40s-50s range
    # Try ranks in increasing order of likelihood of success
    rank_candidates = list(range(25, 61))
    # Shuffle but keep some structure: try lower ranks first with fewer attempts
    random.shuffle(rank_candidates)
    # Ensure we try some low ranks
    rank_candidates = sorted(rank_candidates, key=lambda r: abs(r - 45))
    
    for r in rank_candidates:
        if time.time() > deadline - 5.0:
            break
        
        # Allocate more attempts to ranks near the expected range
        if 35 <= r <= 55:
            max_attempts = 50
        elif 25 <= r < 35:
            max_attempts = 20
        else:
            max_attempts = 10
        
        result = _try_rank(r, n, m, p, deadline, max_attempts)
        if result is not None:
            return result
    
    # If no better rank found, return schoolbook
    return {"U": U_schoolbook, "V": V_schoolbook, "W": W_schoolbook}
