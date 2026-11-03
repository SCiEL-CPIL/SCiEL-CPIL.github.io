"""Rewritten program using randomized algebraic geometry approach for low-rank tensor decomposition."""
import numpy as np
from fractions import Fraction
import random
import sys
import time

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    start_time = time.time()
    deadline = start_time + 140.0 # Leave 10s margin
    
    # Construct the target tensor T
    T_slices = []
    for k in range(pn):
        slice_mat = np.zeros((nm, mp), dtype=int)
        for i in range(n):
            for j in range(m):
                for k_orig in range(p):
                    a = i * m + j
                    b = j * p + k_orig
                    k_idx = k_orig * n + i
                    if k_idx == k:
                        slice_mat[a, b] = 1
        T_slices.append(slice_mat)
        
    def _schoolbook_decomposition(n, m, p):
        """Return the standard schoolbook decomposition of rank n*m*p."""
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

    def find_rank_decomposition(target_rank):
        """
        Try to find a decomposition of rank target_rank.
        """
        max_attempts = 200
        for attempt in range(max_attempts):
            if time.time() > deadline:
                return None
            
            # Generate dense random U and V with small integers
            U_rand = np.random.randint(-1, 2, size=(nm, target_rank))
            V_rand = np.random.randint(-1, 2, size=(mp, target_rank))
            
            # Construct B: (nm * mp) x target_rank
            B = np.zeros((nm * mp, target_rank), dtype=float)
            for c in range(target_rank):
                outer = np.outer(U_rand[:, c], V_rand[:, c])
                B[:, c] = outer.flatten()
            
            # Stack T into T_flat (nm*mp, pn)
            T_flat = np.zeros((nm * mp, pn), dtype=float)
            for k in range(pn):
                T_flat[:, k] = T_slices[k].flatten()
            
            try:
                # Check rank of B
                s = np.linalg.svd(B, compute_uv=False)
                if s[-1] < 1e-9:
                    continue
                
                # Solve B W = T_flat using least squares
                W_candidate, residuals, rank, s_vals = np.linalg.lstsq(B, T_flat, rcond=None)
                
                # Check residual
                Residual = np.abs(B @ W_candidate - T_flat)
                if np.max(Residual) > 1e-6:
                    continue
                
                # Exact verification
                B_frac = [[Fraction(int(x)) for x in row] for row in B]
                T_flat_frac = [[Fraction(int(x)) for x in col] for col in T_flat.T]
                
                valid = True
                W_out = []
                for k in range(pn):
                    w_row = W_candidate[k]
                    t_k = T_flat_frac[k]
                    is_valid_row = True
                    for i in range(nm * mp):
                        dot = Fraction(0)
                        for c in range(target_rank):
                            dot += B_frac[i][c] * Fraction(w_row[c])
                        if dot != t_k[i]:
                            is_valid_row = False
                            break
                    if not is_valid_row:
                        valid = False
                        break
                    W_out.append([int(x) if x == int(x) else Fraction(x) for x in w_row])
                
                if valid:
                    U_out = [[int(U_rand[i][c]) for c in range(target_rank)] for i in range(nm)]
                    V_out = [[int(V_rand[i][c]) for c in range(target_rank)] for i in range(mp)]
                    return {"U": U_out, "V": V_out, "W": W_out}
                
            except Exception:
                continue
                
        return None

    # Try ranks from lower bound up to schoolbook
    lower_bound = max(nm, mp, pn)
    schoolbook_rank = n * m * p
    
    for r in range(41, lower_bound - 1, -1):
        res = find_rank_decomposition(r)
        if res is not None:
            return res
            
    # Fallback to schoolbook
    return _schoolbook_decomposition(n, m, p)
# EVOLVE-BLOCK-END
