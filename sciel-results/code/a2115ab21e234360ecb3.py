"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations

DIMENSIONS = [2, 5, 6]
_DEADLINE = None

# EVOLVE-BLOCK-START
def construct():
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + 120
    n, m, p = DIMENSIONS
    rank = 54
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Block 1: 2x1x6 (j=0)
    # i in {0,1}, j=0, k in {0..5}
    col = 0
    for i in range(2):
        for k in range(6):
            U[5*i + 0][col] = 1
            V[0*6 + k][col] = 1
            W[k*2 + i][col] = 1
            col += 1
    # col is now 12

    # Helper to add a Strassen block for 2x2x2
    # i_offset: 0 or 1 (not used directly, i is 0..1)
    # j_offset: 1 or 3 (for j in {1,2} or {3,4})
    # k_offset: 0, 2, 4 (for k pairs)
    # col_start: starting column index
    def add_strassen_block(j_offset, k_offset, col_start):
        # j' in {0,1} maps to j = j_offset + j'
        # k' in {0,1} maps to k = k_offset + k'
        # i in {0,1}
        
        # Strassen terms
        # r=1
        U[5*0 + j_offset + 0][col_start] += 1
        U[5*1 + j_offset + 1][col_start] += 1
        V[(j_offset+0)*6 + k_offset + 0][col_start] += 1
        V[(j_offset+1)*6 + k_offset + 1][col_start] += 1
        W[(k_offset+0)*2 + 0][col_start] += 1
        W[(k_offset+1)*2 + 1][col_start] += 1
        
        # r=2
        col_start += 1
        U[5*1 + j_offset + 0][col_start] += 1
        U[5*1 + j_offset + 1][col_start] += 1
        V[(j_offset+0)*6 + k_offset + 0][col_start] += 1
        W[(k_offset+0)*2 + 1][col_start] += 1
        W[(k_offset+1)*2 + 1][col_start] -= 1
        
        # r=3
        col_start += 1
        U[5*0 + j_offset + 0][col_start] += 1
        V[(j_offset+0)*6 + k_offset + 1][col_start] += 1
        V[(j_offset+1)*6 + k_offset + 1][col_start] -= 1
        W[(k_offset+1)*2 + 0][col_start] += 1
        W[(k_offset+1)*2 + 1][col_start] += 1
        
        # r=4
        col_start += 1
        U[5*1 + j_offset + 1][col_start] += 1
        V[(j_offset+1)*6 + k_offset + 0][col_start] += 1
        V[(j_offset+0)*6 + k_offset + 0][col_start] -= 1
        W[(k_offset+0)*2 + 0][col_start] += 1
        W[(k_offset+0)*2 + 1][col_start] += 1
        
        # r=5
        col_start += 1
        U[5*0 + j_offset + 0][col_start] += 1
        U[5*0 + j_offset + 1][col_start] += 1
        V[(j_offset+1)*6 + k_offset + 1][col_start] += 1
        W[(k_offset+0)*2 + 0][col_start] -= 1
        W[(k_offset+1)*2 + 0][col_start] += 1
        
        # r=6
        col_start += 1
        U[5*1 + j_offset + 0][col_start] += 1
        U[5*0 + j_offset + 0][col_start] -= 1
        V[(j_offset+0)*6 + k_offset + 0][col_start] += 1
        V[(j_offset+0)*6 + k_offset + 1][col_start] += 1
        W[(k_offset+1)*2 + 1][col_start] += 1
        
        # r=7
        col_start += 1
        U[5*0 + j_offset + 1][col_start] += 1
        U[5*1 + j_offset + 1][col_start] -= 1
        V[(j_offset+1)*6 + k_offset + 0][col_start] += 1
        V[(j_offset+1)*6 + k_offset + 1][col_start] += 1
        W[(k_offset+0)*2 + 0][col_start] += 1
        
        return col_start + 1

    # Block 2: j in {1,2}
    # k pairs: 0-1, 2-3, 4-5
    col = add_strassen_block(1, 0, 12)
    col = add_strassen_block(1, 2, col)
    col = add_strassen_block(1, 4, col)
    # col is now 12 + 21 = 33

    # Block 3: j in {3,4}
    col = add_strassen_block(3, 0, col)
    col = add_strassen_block(3, 2, col)
    col = add_strassen_block(3, 4, col)
    # col is now 33 + 21 = 54

    base_result = {"U": U, "V": V, "W": W}
    # Verify the base decomposition is correct
    Uv = [[U[r][c] for c in range(rank)] for r in range(n*m)]
    Vv = [[V[r][c] for c in range(rank)] for r in range(m*p)]
    Wv = [[W[r][c] for c in range(rank)] for r in range(p*n)]
    
    # Try to reduce rank by finding linear dependencies or cancellations
    # We'll use a simple approach: try removing each column and checking if the tensor is still decomposable
    # with the remaining columns (by solving for the missing column's contribution)
    
    # First, let's verify correctness of the base decomposition
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # Tensor: T[i,j,k] = 1 if the term exists, else 0
    # Decomposition: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    
    # Build the tensor T
    T = [[[0]*p for _ in range(m)] for _ in range(n)]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i][j][k] = 1  # matrix multiplication has all terms
    
    # Verify base decomposition
    def verify(Um, Vm, Wm, rnk):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rnk):
                        s += Um[i*m+j][r] * Vm[j*p+k][r] * Wm[k*n+i][r]
                    if abs(s - T[i][j][k]) > 1e-9:
                        return False
        return True
    
    if not verify(Uv, Vv, Wv, rank):
        # Base is broken, return it anyway (shouldn't happen)
        return base_result
    
    # Try to reduce rank: for each column r, check if we can express it as a linear combination
    # of other columns, or if removing it and adjusting others still works.
    # Simpler: try pairwise column removal - check if column r can be zeroed out
    # by adjusting other columns. This is complex, so let's try a different approach.
    
    # Approach: try to find if any column is linearly dependent in the tensor sense.
    # For each pair of columns (r1, r2), check if we can merge them.
    # A column r contributes U[:,r] ⊗ V[:,r] ⊗ W[:,r] to the tensor.
    # Two columns r1, r2 can be merged into one if their outer products sum to a valid rank-1 term,
    # which is generally not possible. But we can check if one column is redundant.
    
    # Let's try: for each column r, set it to zero and see if the tensor can still be represented
    # by the remaining rank-1 terms. This requires solving a linear system, which is expensive.
    
    # Instead, try a simpler heuristic: check if any column has all-zero entries in one of U, V, or W.
    # If U[:,r] is all zero, or V[:,r] is all zero, or W[:,r] is all zero, then that column is useless.
    for r in range(rank):
        if time.time() > _DEADLINE - 10:
            break
        u_zero = all(Uv[i*m+j][r] == 0 for i in range(n) for j in range(m))
        v_zero = all(Vv[j*p+k][r] == 0 for j in range(m) for k in range(p))
        w_zero = all(Wv[k*n+i][r] == 0 for k in range(p) for i in range(n))
        if u_zero or v_zero or w_zero:
            # This column is useless, remove it
            new_rank = rank - 1
            new_U = [[Uv[i*m+j][c] for c in range(rank) if c != r] for i in range(n) for j in range(m)]
            new_V = [[Vv[j*p+k][c] for c in range(rank) if c != r] for j in range(m) for k in range(p)]
            new_W = [[Wv[k*n+i][c] for c in range(rank) if c != r] for k in range(p) for i in range(n)]
            if verify(new_U, new_V, new_W, new_rank):
                return {"U": new_U, "V": new_V, "W": new_W}
    
    # Try to find linear dependencies: for each column r, check if U[:,r] is a linear combination
    # of other U columns, and similarly for V and W. If all three are linearly dependent in a compatible way,
    # we might be able to reduce rank. This is complex, so let's skip.
    
    return base_result
# EVOLVE-BLOCK-END
