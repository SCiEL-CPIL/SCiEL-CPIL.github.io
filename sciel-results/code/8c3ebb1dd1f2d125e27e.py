import numpy as np
import time

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Rank calculation:
    # Split p=7 into 2+2+2+1
    # 3 blocks of M_{2,4,2} (rank 14 each) + 1 block of M_{2,4,1} (rank 1)
    # Total rank = 3 * 14 + 1 = 43
    
    rank_total = 43
    U = np.zeros((nm, rank_total))
    V = np.zeros((mp, rank_total))
    W = np.zeros((pn, rank_total))
    
    col = 0
    
    # Helper to add a Strassen block for M_{2,2,2} embedded in M_{2,4,2}
    # offset_j: offset for the j index in U and V (0 or 2)
    # offset_k: offset for the k index in V and W (0, 2, or 4)
    def add_strassen_block(offset_j, offset_k):
        nonlocal col
        # 7 terms for Strassen
        # Term 1: (a+d)(e+h)
        U[0 * m + 0 + offset_j, col] = 1.0
        U[1 * m + 1 + offset_j, col] = 1.0
        V[0 * p + 0 + offset_k, col] = 1.0
        V[1 * p + 1 + offset_k, col] = 1.0
        W[0 * n + 0, col] = 1.0
        W[1 * n + 0, col] = 1.0
        W[0 * n + 1, col] = 1.0
        W[1 * n + 1, col] = 1.0
        col += 1
        
        # Term 2: (c+d)e
        U[1 * m + 0 + offset_j, col] = 1.0
        U[1 * m + 1 + offset_j, col] = 1.0
        V[0 * p + 0 + offset_k, col] = 1.0
        W[0 * n + 1, col] = 1.0
        col += 1
        
        # Term 3: a(f-h)
        U[0 * m + 0 + offset_j, col] = 1.0
        V[0 * p + 1 + offset_k, col] = 1.0
        V[1 * p + 1 + offset_k, col] = -1.0
        W[1 * n + 0, col] = 1.0
        col += 1
        
        # Term 4: d(g-e)
        U[1 * m + 1 + offset_j, col] = 1.0
        V[1 * p + 0 + offset_k, col] = 1.0
        V[0 * p + 0 + offset_k, col] = -1.0
        W[0 * n + 0, col] = 1.0
        W[0 * n + 1, col] = 1.0
        col += 1
        
        # Term 5: (a+b)h
        U[0 * m + 0 + offset_j, col] = 1.0
        U[0 * m + 1 + offset_j, col] = 1.0
        V[1 * p + 1 + offset_k, col] = 1.0
        W[0 * n + 0, col] = -1.0
        W[1 * n + 0, col] = 1.0
        col += 1
        
        # Term 6: (c-a)(e+f)
        U[1 * m + 0 + offset_j, col] = 1.0
        U[0 * m + 0 + offset_j, col] = -1.0
        V[0 * p + 0 + offset_k, col] = 1.0
        V[0 * p + 1 + offset_k, col] = 1.0
        W[1 * n + 1, col] = 1.0
        col += 1
        
        # Term 7: (b-d)(g+h)
        U[0 * m + 1 + offset_j, col] = 1.0
        U[1 * m + 1 + offset_j, col] = -1.0
        V[1 * p + 0 + offset_k, col] = 1.0
        V[1 * p + 1 + offset_k, col] = 1.0
        W[0 * n + 0, col] = 1.0
        W[1 * n + 1, col] = 1.0
        col += 1

    # Block 1: k in {0, 1}, j in {0, 1}
    add_strassen_block(0, 0)
    
    # Block 2: k in {0, 1}, j in {2, 3}
    # Note: The k index in W is shared, but the j index in U/V is different.
    # The k index in V is offset by 0? No, V index is j*p + k.
    # For block 2, j is 2,3. So V indices are 2*7+0, 2*7+1, 3*7+0, 3*7+1.
    # Wait, in the helper, V index is j*p + k.
    # If I call add_strassen_block(2, 0), it uses V[0*7+0], V[1*7+1] etc?
    # No, the helper uses local j 0,1.
    # I need to map local j to global j.
    # In helper: V[local_j * p + local_k, col].
    # If offset_j is 2, local_j 0 -> global j 2.
    # So V index should be (local_j + offset_j) * p + (local_k + offset_k).
    # My helper code above uses `V[0 * p + 0 + offset_k, col]`. This is wrong for V.
    # V index is `j * p + k`.
    # If j is offset, the term is `(j_local + offset_j) * p + k_local`.
    # Let's fix the helper to take offsets properly.
    
    # Reset col
    col = 0
    U[:] = 0.0
    V[:] = 0.0
    W[:] = 0.0

    def add_strassen_block_v2(offset_j, offset_k):
        nonlocal col
        # Term 1: (a+d)(e+h)
        U[0 * m + (0 + offset_j), col] = 1.0
        U[1 * m + (1 + offset_j), col] = 1.0
        V[(0 + offset_j) * p + (0 + offset_k), col] = 1.0
        V[(1 + offset_j) * p + (1 + offset_k), col] = 1.0
        W[0 * n + 0, col] = 1.0
        W[1 * n + 0, col] = 1.0
        W[0 * n + 1, col] = 1.0
        W[1 * n + 1, col] = 1.0
        col += 1
        
        # Term 2: (c+d)e
        U[1 * m + (0 + offset_j), col] = 1.0
        U[1 * m + (1 + offset_j), col] = 1.0
        V[(0 + offset_j) * p + (0 + offset_k), col] = 1.0
        W[0 * n + 1, col] = 1.0
        col += 1
        
        # Term 3: a(f-h)
        U[0 * m + (0 + offset_j), col] = 1.0
        V[(0 + offset_j) * p + (1 + offset_k), col] = 1.0
        V[(1 + offset_j) * p + (1 + offset_k), col] = -1.0
        W[1 * n + 0, col] = 1.0
        col += 1
        
        # Term 4: d(g-e)
        U[1 * m + (1 + offset_j), col] = 1.0
        V[(1 + offset_j) * p + (0 + offset_k), col] = 1.0
        V[(0 + offset_j) * p + (0 + offset_k), col] = -1.0
        W[0 * n + 0, col] = 1.0
        W[0 * n + 1, col] = 1.0
        col += 1
        
        # Term 5: (a+b)h
        U[0 * m + (0 + offset_j), col] = 1.0
        U[0 * m + (1 + offset_j), col] = 1.0
        V[(1 + offset_j) * p + (1 + offset_k), col] = 1.0
        W[0 * n + 0, col] = -1.0
        W[1 * n + 0, col] = 1.0
        col += 1
        
        # Term 6: (c-a)(e+f)
        U[1 * m + (0 + offset_j), col] = 1.0
        U[0 * m + (0 + offset_j), col] = -1.0
        V[(0 + offset_j) * p + (0 + offset_k), col] = 1.0
        V[(0 + offset_j) * p + (1 + offset_k), col] = 1.0
        W[1 * n + 1, col] = 1.0
        col += 1
        
        # Term 7: (b-d)(g+h)
        U[0 * m + (1 + offset_j), col] = 1.0
        U[1 * m + (1 + offset_j), col] = -1.0
        V[(1 + offset_j) * p + (0 + offset_k), col] = 1.0
        V[(1 + offset_j) * p + (1 + offset_k), col] = 1.0
        W[0 * n + 0, col] = 1.0
        W[1 * n + 1, col] = 1.0
        col += 1

    # Block 1: k offset 0, j offset 0
    add_strassen_block_v2(0, 0)
    
    # Block 2: k offset 0, j offset 2
    add_strassen_block_v2(2, 0)
    
    # Block 3: k offset 2, j offset 0
    add_strassen_block_v2(0, 2)
    
    # Block 4: k offset 2, j offset 2
    add_strassen_block_v2(2, 2)
    
    # Block 5: k offset 4, j offset 0
    add_strassen_block_v2(0, 4)
    
    # Block 6: k offset 4, j offset 2
    add_strassen_block_v2(2, 4)
    
    # Block 7: k offset 6 (size 1), j offset 0 and 2?
    # For p=1, we use rank 1.
    # U[i*4+j, col] = 1
    # V[j*1+0, col] = 1/4
    # W[0*2+i, col] = 1
    
    for i in range(n):
        for j in range(m):
            U[i * m + j, col] = 1.0
            V[j * p + 6, col] = 0.25
    W[0 * n + 0, col] = 1.0
    W[0 * n + 1, col] = 1.0
    col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
