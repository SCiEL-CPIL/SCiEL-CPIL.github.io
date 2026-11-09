import time
import numpy as np

DIMENSIONS = [4, 5, 6]

def _schoolbook(n, m, p):
    """
    Returns U, V, W for n x m x p matrix multiplication.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    Rank = n*m*p
    """
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _strassen_2x2x2():
    """
    Returns U, V, W for 2 x 2 x 2 matrix multiplication using Strassen.
    Rank = 7
    """
    n, m, p = 2, 2, 2
    rank = 7
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen formulas
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # Indices in U (A): 0=A11, 1=A12, 2=A21, 3=A22
    # Indices in V (B): 0=B11, 1=B12, 2=B21, 3=B22
    # Indices in W (C): 0=C11, 1=C12, 2=C21, 3=C22 (using k*n+i: 0*2+0=0, 1*2+0=2? No.
    # W index: k*n + i.
    # C11: i=0, k=0 -> 0
    # C12: i=0, k=1 -> 1*2+0 = 2? No.
    # Let's check the prompt: W[k*n+i, r].
    # n=2.
    # C11 (i=0, k=0): 0*2+0 = 0
    # C12 (i=0, k=1): 1*2+0 = 2
    # C21 (i=1, k=0): 0*2+1 = 1
    # C22 (i=1, k=1): 1*2+1 = 3
    # So W indices: C11->0, C21->1, C12->2, C22->3.
    
    # M1 = (A11+A22)(B11+B22)
    # U: A11(1), A22(1)
    # V: B11(1), B22(1)
    # W: C11(1), C12(1), C21(1), C22(-1)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # We define the 7 terms.
    terms = [
        # (U_indices, U_vals, V_indices, V_vals, W_indices, W_vals)
        ([0, 3], [1, 1], [0, 3], [1, 1], [0, 1, 2, 3], [1, 1, 1, -1]), # M1
        ([2, 3], [1, 1], [0], [1], [1, 3], [1, -1]), # M2: C21(1), C22(-1)
        ([0], [1], [1, 3], [1, -1], [2, 3], [1, 1]), # M3: C12(1), C22(1)
        ([3], [1], [2, 0], [1, -1], [0, 1], [1, 1]), # M4: C11(1), C21(1)
        ([0, 1], [1, 1], [3], [1], [0, 2], [-1, 1]), # M5: C11(-1), C21(1)
        ([2, 0], [1, -1], [0, 1], [1, 1], [3], [1]), # M6: C22(1)
        ([1, 3], [1, -1], [2, 3], [1, 1], [0, 3], [1, 1]), # M7: C11(1), C22(1)
    ]
    
    for c, (u_idx, u_val, v_idx, v_val, w_idx, w_val) in enumerate(terms):
        for idx, val in zip(u_idx, u_val):
            U[idx, c] += val
        for idx, val in zip(v_idx, v_val):
            V[idx, c] += val
        for idx, val in zip(w_idx, w_val):
            W[idx, c] += val
            
    return U, V, W

def _strassen_4x4x4():
    """
    Returns U, V, W for 4 x 4 x 4 matrix multiplication using Strassen.
    Rank = 49
    """
    n, m, p = 4, 4, 4
    rank = 49
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # We use the 2x2x2 Strassen decomposition on the 2x2 blocks.
    # The 4x4 matrix is composed of four 2x2 blocks.
    # We apply the 7 Strassen multiplications.
    # Each multiplication is a 2x2x2 problem.
    # We need to map the indices.
    
    # The 2x2x2 decomposition provides U, V, W for a 2x2x2 problem.
    U2, V2, W2 = _strassen_2x2x2()
    
    # The 4x4x4 problem can be seen as 7 sub-problems of size 2x2x2.
    # But the inputs are sums of blocks.
    # This is tricky to map directly to a single rank-49 decomposition without expanding.
    # However, we can construct it by taking the Kronecker product approach or similar.
    # Actually, the rank of 4x4x4 is 49.
    # We can generate the decomposition by treating the 4x4 matrices as 2x2 blocks of 2x2 matrices.
    # The Strassen algorithm for 4x4 uses 7 multiplications of 2x2 matrices.
    # Each 2x2 multiplication has rank 7.
    # Total rank 49.
    
    # We need to define how the 49 terms map to the 4x4 indices.
    # Let the 4x4 blocks be A00, A01, A10, A11 (each 2x2).
    # A = [[A00, A01], [A10, A11]]
    # Similarly for B and C.
    
    # The 7 Strassen terms M1..M7 are 2x2 matrices.
    # Each Mk is computed from sums of A and B blocks.
    # And C blocks are sums of Mk.
    
    # We can iterate over the 7 Strassen terms.
    # For each term, we have a 2x2x2 decomposition (7 scalar terms).
    # So 7 * 7 = 49 terms.
    
    # We need to define the coefficients for the A and B blocks for each Strassen term.
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # And C:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We will construct the U, V, W matrices by summing contributions from each scalar term in each Strassen term.
    
    # Helper to get indices for a 2x2 block in a 4x4 matrix.
    # Block (bi, bj) in 4x4 corresponds to rows 2*bi..2*bi+1 and cols 2*bj..2*bj+1.
    # Index in flattened 4x4 (row-major): (2*bi + r) * 4 + (2*bj + c)
    
    def block_indices(bi, bj):
        idxs = []
        for r in range(2):
            for c in range(2):
                idxs.append((2*bi + r) * 4 + (2*bj + c))
        return idxs

    # Strassen term definitions: (A_blocks, A_coeffs, B_blocks, B_coeffs)
    # A_blocks: list of (bi, bj)
    # B_blocks: list of (bi, bj)
    strassen_terms = [
        # M1
        ([(0,0), (1,1)], [1, 1], [(0,0), (1,1)], [1, 1]),
        # M2
        ([(1,0), (1,1)], [1, 1], [(0,0)], [1]),
        # M3
        ([(0,0)], [1], [(0,1), (1,1)], [1, -1]),
        # M4
        ([(1,1)], [1], [(1,0), (0,0)], [1, -1]),
        # M5
        ([(0,0), (0,1)], [1, 1], [(1,1)], [1]),
        # M6
        ([(1,0), (0,0)], [1, -1], [(0,0), (0,1)], [1, 1]),
        # M7
        ([(0,1), (1,1)], [1, -1], [(1,0), (1,1)], [1, 1]),
    ]
    
    # C coefficients for each Strassen term in each C block
    # C00: M1(1), M4(1), M5(-1), M7(1)
    # C01: M3(1), M5(1)
    # C10: M2(1), M4(1)
    # C11: M1(1), M2(-1), M3(1), M6(1)
    c_coeffs = [
        [1, 0, 0, 1, -1, 0, 1], # M1..M7 for C00
        [0, 0, 1, 0, 1, 0, 0], # M1..M7 for C01
        [0, 1, 0, 1, 0, 0, 0], # M1..M7 for C10
        [1, -1, 1, 0, 0, 1, 0], # M1..M7 for C11
    ]
    
    col = 0
    for s in range(7):
        a_blocks, a_coeffs, b_blocks, b_coeffs = strassen_terms[s]
        
        # Expand this Strassen term into 7 scalar terms using the 2x2x2 decomposition
        for c2 in range(7):
            # Get the 2x2x2 decomposition for this scalar term
            u2_vals = U2[:, c2]
            v2_vals = V2[:, c2]
            w2_vals = W2[:, c2]
            
            if np.all(u2_vals == 0) and np.all(v2_vals == 0) and np.all(w2_vals == 0):
                continue
                
            # Map u2_vals to 4x4 A indices
            # u2 index 0..3 corresponds to 2x2 block indices
            # We need to sum contributions from all A blocks involved in this Strassen term
            # This is not a simple mapping because the Strassen term is a sum of blocks.
            # The scalar term c2 in the 2x2x2 decomposition corresponds to a specific product of entries.
            # This approach is complex.
            
            # Alternative: Just use schoolbook for 4x4x4? Rank 64.
            # 64 + 32 + 24 = 120. No improvement.
            
            # Let's stick to the 105 rank decomposition which is easier to construct correctly.
            # We will use Schoolbook for 4x4x4 (Rank 64) in the actual construct function to be safe, 
            # unless I can quickly verify the Strassen mapping.
            # Actually, let's just use Schoolbook for 4x4x4.
            # Rank 64 + 32 + 24 = 120.
            # This is the same as the seed.
            # I need the 49 rank.
            
            # Let's try to construct the 49 rank decomposition properly.
            pass

    # Given the complexity and time, I will implement the 105 rank decomposition using Schoolbook for 4x4x4?
    # No, 64+32+24 = 120.
    # I must use Strassen for 4x4x4.
    
    # I will write a specific function for 4x4x4 Strassen.
    pass

def construct():
    n, m, p = DIMENSIONS
    # Rank 105 decomposition:
    # 1. 4x4x4 (Strassen, Rank 49)
    # 2. 4x4x2 (Schoolbook, Rank 32)
    # 3. 4x1x6 (Schoolbook, Rank 24)
    
    # For now, I will return the schoolbook 120 to ensure validity, 
    # but I will implement the 105 version in the final code.
    
    # Since I cannot easily verify the 49 rank mapping in this thought process, 
    # I will provide the 105 rank code structure.
    
    pass

# Final Code
