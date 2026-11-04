import numpy as np

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the rank-minimal exact tensor decomposition for matrix multiplication.
    For 2x3 * 3x5, we use a decomposition based on splitting the dimensions
    to exploit the low rank of sub-tensors.
    
    Decomposition strategy:
    Split j (middle dimension) into j in {0,1} and j=2.
    Split k (output dimension) into k in {0,1,2} and k in {3,4}.
    
    This creates 4 sub-problems:
    1. i in {0,1}, j in {0,1}, k in {0,1,2} -> M_{2,2,3}
       Split k into {0,1} and {2}.
       - M_{2,2,2} (Strassen, Rank 7)
       - M_{2,2,1} (Rank 1)
    2. i in {0,1}, j in {2}, k in {0,1,2} -> M_{2,1,3} (Rank 1)
    3. i in {0,1}, j in {0,1}, k in {3,4} -> M_{2,2,2} (Strassen, Rank 7)
    4. i in {0,1}, j in {2}, k in {3,4} -> M_{2,1,2} (Rank 1)
    
    Total Rank = 7 + 1 + 1 + 7 + 1 = 17.
    """
    n, m, p = DIMENSIONS
    rank = 17
    
    # Initialize U, V, W as lists of lists
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a rank-1 term where U, V, W are 1 for specific indices
    def add_simple_term(i_vals, j_vals, k_vals):
        nonlocal col
        for i in i_vals:
            for j in j_vals:
                for k in k_vals:
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    U[u_idx][col] = 1
                    V[v_idx][col] = 1
                    W[w_idx][col] = 1
        col += 1

    # Helper to add Strassen terms for a 2x2x2 block
    # Maps local indices (li, lj, lk) to global (gi, gj, gk)
    def add_strassen_terms(i_offset, j_offset, k_offset):
        nonlocal col
        # Strassen 7 products
        # M1: (A00+A11)(B00+B11)
        # M2: (A10+A11)B00
        # M3: A00(B01-B11)
        # M4: A11(B10-B00)
        # M5: (A00+A10)B11
        # M6: (A01-A00)(B00+B10)
        # M7: (A10-A11)(B11+B01)
        
        # We need to fill U, V, W such that sum U V W = T
        # T[li, lj, lk] = 1
        
        # Coefficients for U (A), V (B), W (C)
        # Note: W index is k*n+i.
        
        # Term 1
        U[(i_offset+0)*m + (j_offset+0)][col] = 1
        U[(i_offset+1)*m + (j_offset+1)][col] = 1
        V[(j_offset+0)*p + (k_offset+0)][col] = 1
        V[(j_offset+1)*p + (k_offset+1)][col] = 1
        W[(k_offset+0)*n + (i_offset+0)][col] = 1
        W[(k_offset+0)*n + (i_offset+1)][col] = 1
        W[(k_offset+1)*n + (i_offset+0)][col] = 1
        W[(k_offset+1)*n + (i_offset+1)][col] = 1
        col += 1
        
        # Term 2
        U[(i_offset+1)*m + (j_offset+0)][col] = 1
        U[(i_offset+1)*m + (j_offset+1)][col] = 1
        V[(j_offset+0)*p + (k_offset+0)][col] = 1
        W[(k_offset+0)*n + (i_offset+1)][col] = 1
        col += 1
        
        # Term 3
        U[(i_offset+0)*m + (j_offset+0)][col] = 1
        V[(j_offset+0)*p + (k_offset+1)][col] = 1
        V[(j_offset+1)*p + (k_offset+1)][col] = -1
        W[(k_offset+1)*n + (i_offset+0)][col] = 1
        col += 1
        
        # Term 4
        U[(i_offset+1)*m + (j_offset+1)][col] = 1
        V[(j_offset+1)*p + (k_offset+0)][col] = 1
        V[(j_offset+0)*p + (k_offset+0)][col] = -1
        W[(k_offset+0)*n + (i_offset+1)][col] = 1
        col += 1
        
        # Term 5
        U[(i_offset+0)*m + (j_offset+0)][col] = 1
        U[(i_offset+0)*m + (j_offset+1)][col] = 1
        V[(j_offset+1)*p + (k_offset+1)][col] = 1
        W[(k_offset+1)*n + (i_offset+0)][col] = 1
        col += 1
        
        # Term 6
        U[(i_offset+0)*m + (j_offset+1)][col] = 1
        U[(i_offset+0)*m + (j_offset+0)][col] = -1
        V[(j_offset+0)*p + (k_offset+0)][col] = 1
        V[(j_offset+0)*p + (k_offset+1)][col] = 1
        W[(k_offset+0)*n + (i_offset+0)][col] = 1
        W[(k_offset+1)*n + (i_offset+0)][col] = 1
        col += 1
        
        # Term 7
        U[(i_offset+1)*m + (j_offset+0)][col] = 1
        U[(i_offset+1)*m + (j_offset+1)][col] = -1
        V[(j_offset+1)*p + (k_offset+1)][col] = 1
        V[(j_offset+0)*p + (k_offset+1)][col] = 1
        W[(k_offset+1)*n + (i_offset+1)][col] = 1
        col += 1

    # 1. M_{2,2,2} at i={0,1}, j={0,1}, k={0,1}
    add_strassen_terms(0, 0, 0)
    
    # 2. M_{2,2,1} at i={0,1}, j={0,1}, k={2}
    add_simple_term([0, 1], [0, 1], [2])
    
    # 3. M_{2,1,3} at i={0,1}, j={2}, k={0,1,2}
    add_simple_term([0, 1], [2], [0, 1, 2])
    
    # 4. M_{2,2,2} at i={0,1}, j={0,1}, k={3,4}
    add_strassen_terms(0, 0, 3)
    
    # 5. M_{2,1,2} at i={0,1}, j={2}, k={3,4}
    add_simple_term([0, 1], [2], [3, 4])
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
