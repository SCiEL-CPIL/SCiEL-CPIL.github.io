import numpy as np
import time
import sys
import random
from fractions import Fraction

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    # Schoolbook fallback
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1

    # Build the list of (i,j,k) triples
    triples = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    N_eq = len(triples)  # 24

    # Precompute the "index" mapping
    def ij(i, j): return i * m + j
    def jk(j, k): return j * p + k
    def ki(k, i): return k * n + i

    # Exact Gaussian elimination over rationals (fractions)
    def solve_linear_system(A, b):
        """Solve A x = b where A is (N_eq, rank) list of lists of Fractions, b is list of Fractions.
        Returns list of Fractions or None if no exact solution."""
        N = len(b)
        R = len(A[0])
        M = [row[:] + [b[i]] for i, row in enumerate(A)]
        rank = 0
        for col in range(R):
            # find pivot
            pivot = -1
            for r in range(rank, N):
                if M[r][col] != 0:
                    pivot = r
                    break
            if pivot < 0:
                continue
            M[rank], M[pivot] = M[pivot], M[rank]
            # normalize
            pv = M[rank][col]
            if pv != 1:
                M[rank] = [x / pv for x in M[rank]]
            # eliminate
            for r in range(N):
                if r != rank and M[r][col] != 0:
                    factor = M[r][col]
                    M[r] = [M[r][c] - factor * M[rank][c] for c in range(R + 1)]
            rank += 1
        # check consistency
        for r in range(rank, N):
            if M[r][R] != 0:
                return None
        # extract solution (free vars = 0)
        x = [Fraction(0)] * R
        for r in range(rank):
            # find pivot column
            for c in range(R):
                if M[r][c] != 0:
                    x[c] = M[r][R]
                    break
        return x

    def is_half_integer(fr):
        # fr is Fraction; check 2*fr is integer
        return (fr * 2).denominator == 1

    def try_rank(r, seed, time_budget):
        """Try to find a valid decomposition of rank r."""
        rng = random.Random(seed)
        # We'll try several random initializations of U, V, W with small integer entries
        # Then solve for coefficients x_c such that sum_c x_c U[i,j,c] V[j,k,c] W[k,i,c] = 1
        # Then final factors are U' = U * x (column-wise), etc.
        # For U',V',W' to be half-integers, we need x_c * U[i,j,c] etc to be half-integers.
        
        # Strategy: pick U,V,W with entries in {-1,0,1} (sparse), then the system A x = 1
        # where A[i,j,k, c] = U[i,j,c]*V[j,k,c]*W[k,i,c]
        # Solve for x. Then U'[:,c] = x_c * U[:,c]. For these to be half-integers,
        # x_c * U[i,j,c] must be half-integer for all i,j. Since U entries are in {-1,0,1},
        # we need x_c to be half-integer (if any U entry is +-1) or x_c*0=0 (always ok).
        
        # So: find x (solution to linear system) where each x_c is a half-integer.
        # Then the final factors U'=U*x, V'=V*x, W'=W*x are half-integers.
        
        # But we also need the system to be consistent (have a solution).
        # With random sparse U,V,W, the 24x r system may or may not be consistent.
        # Actually, the system is: for each (i,j,k), sum_c A_{ijk,c} x_c = 1.
        # This is 24 equations in r unknowns. For r >= 19, likely overdetermined but
        # may still have solutions if the equations are dependent.
        
        # Alternative: don't fix U,V,W. Instead, directly search for factor matrices
        # with half-integer entries by random generation + verification.
        
        # Let me try a different approach: generate random half-integer U,V,W and check.
        # This is essentially what the original does but with more clever search.
        
        # Actually, let me try the linear system approach properly:
        # Pick random sparse integer U,V,W (entries in {-1,0,1}).
        # Build A (24 x r) where A[eq, c] = U[i,j,c]*V[j,k,c]*W[k,i,c]
        # Solve A x = 1 (vector of ones, length 24)
        # If solution exists and all x_c are half-integers, we're done.
        
        max_attempts = 5000
        attempt = 0
        while attempt < max_attempts and time.time() < time_budget:
            attempt += 1
            
            # Generate sparse random integer matrices
            # Density: make it so that each equation has some non-zero terms
            # For rank r, each of the 24 equations needs enough non-zero A entries
            density = 0.5  # probability each entry is non-zero
            
            U = np.zeros((n * m, r), dtype=int)
            V = np.zeros((m * p, r), dtype=int)
            W = np.zeros((p * n, r), dtype=int)
            
            for idx in range(n * m):
                for c in range(r):
                    if rng.random() < density:
                        U[idx, c] = rng.choice([-1, 1])
            for idx in range(m * p):
                for c in range(r):
                    if rng.random() < density:
                        V[idx, c] = rng.choice([-1, 1])
            for idx in range(p * n):
                for c in range(r):
                    if rng.random() < density:
                        W[idx, c] = rng.choice([-1, 1])
            
            # Build linear system A x = 1
            # A has 24 rows, r columns
            A = []
            for (i, j, k) in triples:
                row = []
                for c in range(r):
                    val = U[ij(i,j), c] * V[jk(j,k), c] * W[ki(k,i), c]
                    row.append(Fraction(val))
                A.append(row)
            
            b = [Fraction(1)] * N_eq
            
            # Check if system is consistent and solve
            sol = solve_linear_system(A, b)
            if sol is None:
                continue
            
            # Check if all solution entries are half-integers
            valid = True
            for c in range(r):
                if not is_half_integer(sol[c]):
                    valid = False
                    break
            if not valid:
                continue
            
            # Construct final factors
            U_final = np.zeros((n * m, r))
            V_final = np.zeros((m * p, r))
            W_final = np.zeros((p * n, r))
            for c in range(r):
                xc = float(sol[c])
                U_final[:, c] = U[:, c] * xc
                V_final[:, c] = V[:, c] * xc
                W_final[:, c] = W[:, c] * xc
            
            # Verify
            ok = True
            for (i, j, k) in triples:
                val = 0.0
                for c in range(r):
                    val += U_final[ij(i,j), c] * V_final[jk(j,k), c] * W_final[ki(k,i), c]
                if abs(val - 1.0) > 1e-9:
                    ok = False
                    break
            if not ok:
                continue
            
            # Verify half-integer constraint
            for mat in [U_final, V_final, W_final]:
                for v in mat:
                    if abs(v * 2 - round(v * 2)) > 1e-9:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                continue
            
            return U_final, V_final, W_final
        
        return None

    # Try ranks from 19 down to 23
    for target_rank in [19, 20, 21, 22, 23]:
        if time.time() > deadline - 10:
            break
        
        time_budget = min(deadline - 5, time.time() + 25)
        result = try_rank(target_rank, seed=42 + target_rank, time_budget=time_budget)
        
        if result is not None:
            U, V, W = result
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
    
    # Return schoolbook
    return {
        "U": U_sb.tolist(),
        "V": V_sb.tolist(),
        "W": W_sb.tolist()
    }
