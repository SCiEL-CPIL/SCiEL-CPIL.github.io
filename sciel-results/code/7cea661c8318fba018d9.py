"""
Strategy: Sparse support enumeration and linear solving.

The current program hardcodes a rank-39 construction and fails to search for rank 38.
This new strategy:
1. Defines a theoretical lower bound (approx 2p+3 = 25, but we start higher to be safe, e.g., 38).
2. Enumerates sparse support patterns for U and V.
   - A support pattern specifies which (row, col) entries in U and V are non-zero.
   - We assume non-zero entries are 1.0 (Weyl scaling can adjust magnitudes later if needed, 
     but for rank-1 tensor sums with integer coefficients, 1.0 is a strong prior).
3. For each pattern, we set up a linear system for W:
   Sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1
   Since U and V are fixed (0 or 1), this is linear in W.
4. If the linear system has a unique or consistent solution, we verify the decomposition.
5. We search ranks from 38 downwards (or upwards from a lower bound if 38 fails).

To make enumeration feasible within 120s:
- We limit the sparsity of U and V. 
- We use a structured heuristic for supports: 
  - For each term r, U has non-zeros in at most 2 rows (Strassen-like).
  - For each term r, V has non-zeros in at most 2 rows (Strassen-like).
- We randomly sample such patterns and solve the linear system.
- We also try a deterministic "shifted Strassen" pattern for rank 38 to see if a simple 
  modification of the rank-39 construction works.

Key Insight: The rank-39 construction uses 7 terms per 2 columns (for 10 columns) + 4 terms for the last column.
7*5 + 4 = 39.
Can we do 7 terms for 11 columns? 7*11 = 77 terms? No.
Can we do 3.5 terms per column on average? 
Total rank R. 
If we can find a rank-38 solution, it would be an improvement.
"""

import time
import numpy as np
import random
from itertools import product

DIMENSIONS = [2, 2, 11]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 100  # 100s safety margin

    def _verify(U, V, W, rank):
        """Verify that the decomposition is exact."""
        # Efficient verification using tensor contraction
        # T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
        # We need T[i,j,k] == 1 for all i,j,k
        # Note: indices are flattened in the interface:
        # U: (n*m, rank) -> U[i*m+j, r]
        # V: (m*p, rank) -> V[j*p+k, r]
        # W: (p*n, rank) -> W[k*n+i, r]
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    def _make_schoolbook():
        rank = n * m * p  # 44
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W, rank

    def _make_strassen_rank39():
        """Strassen-based construction for rank 39."""
        rank = 39
        U = np.zeros((4, rank), dtype=float)
        V = np.zeros((22, rank), dtype=float)
        W = np.zeros((22, rank), dtype=float)
        
        col = 0
        for t in range(5):
            k0 = 2 * t
            k1 = 2 * t + 1
            
            # M1 = (A[0,0]+A[1,1]) * (B[0,k0]+B[1,k1])
            U[0, col] = 1; U[3, col] = 1
            V[k0, col] = 1; V[11 + k1, col] = 1
            W[2*k0, col] = 1; W[2*k0+1, col] = -1; W[2*k1+1, col] = 1
            col += 1
            
            # M2 = (A[1,0]+A[1,1]) * B[0,k0]
            U[2, col] = 1; U[3, col] = 1
            V[k0, col] = 1
            W[2*k0+1, col] = 1; W[2*k1+1, col] = -1
            col += 1
            
            # M3 = A[0,0] * (B[1,k0]-B[1,k1])
            U[0, col] = 1
            V[11 + k0, col] = 1; V[11 + k1, col] = -1
            W[2*k1, col] = 1; W[2*k1+1, col] = 1
            col += 1
            
            # M4 = A[1,1] * (B[0,k1]-B[0,k0])
            U[3, col] = 1
            V[k1, col] = 1; V[k0, col] = -1
            W[2*k0, col] = 1; W[2*k0+1, col] = 1
            col += 1
            
            # M5 = (A[0,0]+A[0,1]) * B[1,k1]
            U[0, col] = 1; U[1, col] = 1
            V[11 + k1, col] = 1
            W[2*k0, col] = -1; W[2*k1, col] = 1
            col += 1
            
            # M6 = (A[1,0]-A[0,0]) * (B[0,k0]+B[1,k0])
            U[2, col] = 1; U[0, col] = -1
            V[k0, col] = 1; V[11 + k0, col] = 1
            W[2*k1+1, col] = 1
            col += 1
            
            # M7 = (A[0,1]-A[1,1]) * (B[0,k1]+B[1,k1])
            U[1, col] = 1; U[3, col] = -1
            V[k1, col] = 1; V[11 + k1, col] = 1
            W[2*k0, col] = 1
            col += 1
        
        # Single column 10: need 4 terms
        k = 10
        U[0, col] = 1; V[k, col] = 1; W[2*k, col] = 1; col += 1
        U[1, col] = 1; V[11 + k, col] = 1; W[2*k, col] = 1; col += 1
        U[2, col] = 1; V[k, col] = 1; W[2*k + 1, col] = 1; col += 1
        U[3, col] = 1; V[11 + k, col] = 1; W[2*k + 1, col] = 1; col += 1
        
        assert col == rank, f"Expected {rank} columns, got {col}"
        return U, V, W, rank

    def _try_rank_via_linear_solve(R, max_attempts=200):
        """
        Try to find a rank-R decomposition by randomizing sparse U and V,
        then solving for W linearly.
        
        We restrict U and V to have at most 2 non-zeros per column (term).
        This mimics the Strassen structure.
        """
        if time.time() > deadline:
            return None
            
        for attempt in range(max_attempts):
            if time.time() > deadline:
                return None
                
            U = np.zeros((n * m, R), dtype=float)
            V = np.zeros((m * p, R), dtype=float)
            
            # Randomly assign non-zeros to U and V
            # Each term r gets 1 or 2 non-zeros in U and 1 or 2 in V
            for r in range(R):
                # U non-zeros: pick 1 or 2 distinct rows from [0, 1, 2, 3]
                num_u_nz = random.choice([1, 2])
                u_rows = random.sample(range(n * m), num_u_nz)
                for row in u_rows:
                    U[row, r] = 1.0
                    
                # V non-zeros: pick 1 or 2 distinct rows from [0, ..., 21]
                num_v_nz = random.choice([1, 2])
                v_rows = random.sample(range(m * p), num_v_nz)
                for row in v_rows:
                    V[row, r] = 1.0
            
            # Now solve for W
            # Equation: Sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            # Let's flatten W into a vector of size p*n*R = 22*R
            # The system is linear in W.
            
            # We have 44 equations (for each i, j, k)
            # Unknowns: W[k*n+i, r] for all k, i, r. Total 22*R unknowns.
            # Matrix A of size 44 x (22*R)
            # b of size 44, all ones.
            
            num_unknowns = p * n * R
            A = np.zeros((n * m * p, num_unknowns), dtype=float)
            b = np.ones(n * m * p, dtype=float)
            
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # Coefficient for W[k*n+i, r] is U[i*m+j, r] * V[j*p+k, r]
                        for r in range(R):
                            coeff = U[i * m + j, r] * V[j * p + k, r]
                            if coeff != 0:
                                # W index in flattened vector: (k*n+i) * R + r
                                w_idx = (k * n + i) * R + r
                                A[eq_idx, w_idx] = coeff
                        eq_idx += 1
            
            # Solve A * w_flat = b
            # Use least squares if underdetermined, or check consistency
            try:
                w_flat, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
                # Check if solution is exact
                if np.allclose(A @ w_flat, b, atol=1e-9):
                    W = w_flat.reshape((p * n, R))
                    # Verify to be sure
                    if _verify(U, V, W, R):
                        return U, V, W
            except Exception:
                pass

        return None

    # 1. Try known rank 39
    U, V, W, rank = _make_strassen_rank39()
    if _verify(U, V, W, rank):
        # We have a valid rank 39. Can we do better?
        best_U, best_V, best_W, best_rank = U, V, W, rank
    else:
        # Fallback to schoolbook if Strassen fails (shouldn't happen)
        U, V, W, rank = _make_schoolbook()
        if _verify(U, V, W, rank):
            best_U, best_V, best_W, best_rank = U, V, W, rank
        else:
            raise RuntimeError("Failed to construct valid decomposition")

    # 2. Try to find rank 38, 37, etc.
    # Start from 38 down to 30 (arbitrary lower bound for search)
    for R in range(38, 29, -1):
        if time.time() > deadline - 20:
            break
        result = _try_rank_via_linear_solve(R, max_attempts=100)
        if result is not None:
            U, V, W = result
            if _verify(U, V, W, R):
                best_U, best_V, best_W, best_rank = U, V, W, R
                break  # Found a lower rank, stop searching lower (unless we want to go even lower)
                # Actually, let's try one more lower rank if time permits?
                # No, break is fine for now, 38 is a significant improvement over 39.
                # If we find 38, we return it.
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
