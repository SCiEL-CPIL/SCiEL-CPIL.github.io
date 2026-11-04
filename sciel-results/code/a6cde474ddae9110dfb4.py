"""Matrix multiplication 4x4 by 4x9 tensor decomposition."""
import numpy as np
import time
from itertools import product

DIMENSIONS = [4, 4, 9]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    schoolbook_rank = n * m * p  # 144

    # Precompute the tensor target
    # u = i*m + j (i in [0,n), j in [0,m))
    # v = j'*p + k (j' in [0,m), k in [0,p))
    # w = k'*n + i' (k' in [0,p), i' in [0,n))
    # T[u,v,w] = 1 iff j == j' AND i == i' AND k == k'
    # For given (u,v): i = u//m, j = u%m, j' = v//p, k = v%p
    # If j == j': target w = k*n + i, else no valid w

    target_w = {}
    for u in range(nm):
        i = u // m
        j = u % m
        for v in range(mp):
            jp = v // p
            k = v % p
            if j == jp:
                target_w[(u, v)] = k * n + i
            else:
                target_w[(u, v)] = -1

    # Schoolbook solution
    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank), dtype=int)
        V = np.zeros((mp, rank), dtype=int)
        W = np.zeros((pn, rank), dtype=int)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = i * m + j
                    v = j * p + k
                    w = k * n + i
                    U[u, col] = 1
                    V[v, col] = 1
                    W[w, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit

    best_solution = schoolbook()
    best_rank = n * m * p

    def is_valid_solution(U, V, W):
        """Check if U, V, W form a valid decomposition with integer/half-integer entries."""
        # Check entries are integer or half-integer
        for mat in [U, V, W]:
            scaled = mat * 2
            if not np.allclose(scaled, np.round(scaled), atol=1e-9):
                return False

        # Check the tensor equation
        for u in range(nm):
            for v in range(mp):
                tw = target_w[(u, v)]
                for w in range(pn):
                    expected = 1.0 if w == tw else 0.0
                    val = np.dot(U[u, :] * V[v, :], W[w, :])
                    if abs(val - expected) > 1e-9:
                        return False
        return True

    def try_rank_structured(r, num_trials=20):
        """
        Try to find a valid decomposition of rank r using structured random search.
        Uses the observation that for matrix multiplication, we can exploit the
        structure by fixing some entries and solving for others.
        """
        for trial in range(num_trials):
            if time.time() > deadline:
                return None

            # Strategy: Use a structured approach where we fix U and V to have
            # specific patterns and solve for W, then verify.
            
            # For matrix multiplication, a key insight is that the tensor has
            # a "slice" structure. We can try to find decompositions where
            # each rank-1 term covers a subset of the computation.
            
            # Approach 1: Random with constrained entries
            # Use entries from a small set to increase chance of integer solutions
            
            # Generate U and V with entries from {-1, 0, 1} with some structure
            U = np.zeros((nm, r), dtype=float)
            V = np.zeros((mp, r), dtype=float)
            
            # For each column c, assign random non-zero entries
            for c in range(r):
                # Choose which rows of U have non-zero entries in column c
                # Each row u corresponds to (i, j). For the tensor to be non-trivial,
                # we need coverage.
                
                # Random pattern: each entry independently non-zero with prob 0.5
                mask_u = np.random.rand(nm) < 0.6
                mask_v = np.random.rand(mp) < 0.6
                
                U[mask_u, c] = np.random.choice([-1, 1], size=np.sum(mask_u))
                V[mask_v, c] = np.random.choice([-1, 1], size=np.sum(mask_v))
            
            # For each w, solve for W[w, :]
            W = np.zeros((pn, r))
            valid = True
            
            for w in range(pn):
                # Build system: for all (u,v) where T[u,v,w] != 0,
                # sum_c U[u,c]*V[v,c]*W[w,c] = T[u,v,w]
                # And for (u,v) where T[u,v,w] == 0, the sum should be 0
                
                A = np.zeros((nm * mp, r))
                b = np.zeros(nm * mp)
                idx = 0
                for u in range(nm):
                    for v in range(mp):
                        A[idx, :] = U[u, :] * V[v, :]
                        tw = target_w[(u, v)]
                        b[idx] = 1.0 if (tw == w) else 0.0
                        idx += 1
                
                # Solve using least squares
                try:
                    W_w, residuals, rank_svd, sv = np.linalg.lstsq(A, b, rcond=None)
                except Exception:
                    valid = False
                    break
                
                W[w, :] = W_w
            
            if not valid:
                continue
            
            # Check if W has integer/half-integer entries
            W_scaled = W * 2
            W_rounded = np.round(W_scaled)
            if not np.allclose(W_scaled, W_rounded, atol=1e-6):
                # Try rounding and checking
                W_final = W_rounded / 2.0
                if is_valid_solution(U, V, W_final):
                    return {
                        "U": U.tolist(),
                        "V": V.tolist(),
                        "W": W_final.tolist()
                    }
                continue
            
            W_final = W_rounded / 2.0
            
            # Verify the full solution
            if is_valid_solution(U, V, W_final):
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W_final.tolist()
                }
        
        return None

    def try_rank_with_polish(r, num_trials=10):
        """
        Try to find a valid decomposition of rank r, with a polish step
        that refines near-integer solutions.
        """
        for trial in range(num_trials):
            if time.time() > deadline:
                return None

            U = np.random.randint(-1, 2, (nm, r)).astype(float)
            V = np.random.randint(-1, 2, (mp, r)).astype(float)
            
            W = np.zeros((pn, r))
            valid = True
            
            for w in range(pn):
                A = np.zeros((nm * mp, r))
                b = np.zeros(nm * mp)
                idx = 0
                for u in range(nm):
                    for v in range(mp):
                        A[idx, :] = U[u, :] * V[v, :]
                        tw = target_w[(u, v)]
                        b[idx] = 1.0 if (tw == w) else 0.0
                        idx += 1
                
                try:
                    W_w, residuals, rank_svd, sv = np.linalg.lstsq(A, b, rcond=None)
                except Exception:
                    valid = False
                    break
                
                W[w, :] = W_w
            
            if not valid:
                continue
            
            # Polish: round to nearest half-integer and verify
            W_scaled = W * 2
            W_rounded = np.round(W_scaled)
            W_final = W_rounded / 2.0
            
            if is_valid_solution(U, V, W_final):
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W_final.tolist()
                }
        
        return None

    # Try ranks from a reasonable lower bound up to schoolbook
    # Lower bound for matrix multiplication rank: at least max(n*p, n*m, m*p) in many cases
    # But for 4x4*4x9, the rank is known to be less than 144
    # Let's try from 50 up
    
    # First, try the structured approach for lower ranks
    for r in range(50, best_rank):
        if time.time() > deadline - 10:
            break
        
        time_left = deadline - time.time()
        if time_left < 3:
            break
        
        # Use more trials for lower ranks
        num_trials = min(30, max(5, int(time_left / 5)))
        
        result = try_rank_structured(r, num_trials)
        if result is not None:
            best_solution = result
            best_rank = r
            break
    
    # If we didn't find a better solution, try the polish approach for a few more ranks
    if best_rank == n * m * p:
        for r in range(60, best_rank):
            if time.time() > deadline - 5:
                break
            
            time_left = deadline - time.time()
            if time_left < 2:
                break
            
            num_trials = min(15, max(3, int(time_left / 4)))
            
            result = try_rank_with_polish(r, num_trials)
            if result is not None:
                best_solution = result
                best_rank = r
                break

    return best_solution
