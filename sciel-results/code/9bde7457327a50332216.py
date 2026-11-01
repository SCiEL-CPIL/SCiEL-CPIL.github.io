"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    # Start with the known rank-53 solution as fallback
    rank = 7 * 7 + 4
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    U_strassen = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]
    V_strassen = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ]
    W_strassen = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]
    U_school = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]
    V_school = [
        [1, 0, 1, 0],
        [0, 1, 0, 1]
    ]
    W_school = [
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    
    col_offset = 0
    for block in range(7):
        k0 = 2 * block
        k1 = 2 * block + 1
        for r in range(7):
            c = col_offset + r
            for i in range(2):
                for j in range(2):
                    U[i * m + j][c] = U_strassen[i * 2 + j][r]
            for j in range(2):
                V[j * p + k0][c] = V_strassen[j * 2 + 0][r]
                V[j * p + k1][c] = V_strassen[j * 2 + 1][r]
            for k in range(2):
                kk = k0 + k
                for i in range(2):
                    W[kk * n + i][c] = W_strassen[k * 2 + i][r]
        col_offset += 7
    
    k_last = 14
    for r in range(4):
        c = col_offset + r
        for i in range(2):
            for j in range(2):
                U[i * m + j][c] = U_school[i * 2 + j][r]
        for j in range(2):
            V[j * p + k_last][c] = V_school[j][r]
        for i in range(2):
            W[k_last * n + i][c] = W_school[i][r]
    
    # Try to find a lower-rank solution by solving the linear system
    # The tensor T has entries T[i,j,k] = 1 if i==k else 0 for the multiplication
    # We need U, V, W such that sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = T[i,j,k]
    
    # Build the constraint matrix for rank r
    def try_rank(r):
        """Try to find a rank-r decomposition by solving linear system."""
        if time.time() > deadline - 5:
            return None
        
        # Number of constraints: n*m*p = 2*2*15 = 60
        # Unknowns: r * (n*m + m*p + p*n) = r * (4 + 30 + 30) = 64r
        # For r=52: 3328 unknowns, 60 constraints - underdetermined, always solvable
        # But we need the multiplicative structure, so this is nonlinear
        
        # Use a different approach: fix U and V, solve for W, then iterate
        # Or use the fact that for small n,m we can enumerate
        
        # For now, try to use a structured approach
        # Use 6 Strassen blocks (rank 42) + 1 block for last 3 columns
        # Try rank 10 for M_{2,2,3}
        
        if r < 42:
            return None
        
        remaining_rank = r - 42
        if remaining_rank < 10:
            return None
        
        # Build U, V, W for 6 Strassen blocks
        U_new = [[0] * r for _ in range(n * m)]
        V_new = [[0] * r for _ in range(m * p)]
        W_new = [[0] * r for _ in range(p * n)]
        
        for block in range(6):
            k0 = 2 * block
            k1 = 2 * block + 1
            for rr in range(7):
                c = block * 7 + rr
                for i in range(2):
                    for j in range(2):
                        U_new[i * m + j][c] = U_strassen[i * 2 + j][rr]
                for j in range(2):
                    V_new[j * p + k0][c] = V_strassen[j * 2 + 0][rr]
                    V_new[j * p + k1][c] = V_strassen[j * 2 + 1][rr]
                for k in range(2):
                    kk = k0 + k
                    for i in range(2):
                        W_new[kk * n + i][c] = W_strassen[k * 2 + i][rr]
        
        # Now handle last 3 columns (k=12,13,14) with remaining_rank
        # We need to find U_rem (4 x remaining_rank), V_rem (6 x remaining_rank), W_rem (6 x remaining_rank)
        # such that for i,j in {0,1}, k in {12,13,14}:
        # sum_rr U_rem[i,j,rr] * V_rem[j,k-12,rr] * W_rem[k-12,i,rr] = delta(i, k-12)
        
        # This is still nonlinear. Use a greedy/iterative approach.
        # For now, use schoolbook for last 3 columns if remaining_rank >= 12
        if remaining_rank >= 12:
            for kk in range(3):
                k_abs = 12 + kk
                for i in range(2):
                    for j in range(2):
                        c = 42 + kk * 4 + i * 2 + j
                        U_new[i * m + j][c] = 1
                        V_new[j * p + k_abs][c] = 1
                        W_new[k_abs * n + i][c] = 1
            return (U_new, V_new, W_new)
        
        # Try to find a rank-10 decomposition for M_{2,2,3}
        # Use a structured approach
        if remaining_rank == 10:
            # Try to construct manually
            # This is complex; fall back to higher rank
            return None
        
        return None
    
    # Try ranks from 52 down to 42
    for r in range(52, 41, -1):
        if time.time() > deadline - 10:
            break
        result = try_rank(r)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
    # Return the rank-53 solution
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
