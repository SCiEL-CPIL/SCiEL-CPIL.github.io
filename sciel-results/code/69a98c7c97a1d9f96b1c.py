import numpy as np

DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying
    a 2x2 matrix by a 2x6 matrix.
    
    The decomposition is based on Strassen's algorithm applied to three
    2x2 blocks of the 2x6 matrix.
    Rank = 3 * 7 = 21.
    """
    n, m, p = DIMENSIONS
    # p is 6, so we have 3 blocks of size 2x2
    num_blocks = p // 2
    
    # Strassen's algorithm for 2x2 matrix multiplication
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # C = A*B
    
    # M1 = A00 * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We need to construct U, V, W such that
    # C[i, l] = sum_k U[i, k] * V[?, k] * W[?, k]
    # Actually, the tensor is T_{i, j, l} = sum_k U[i, k] V[j, k] W[l, k]
    # Where i is row of A (and C), j is col of A (and row of B), l is col of B (and C).
    # Wait, standard tensor notation:
    # T = sum_k u_k \otimes v_k \otimes w_k
    # T_{i, j, l} = sum_k U[i, k] V[j, k] W[l, k]
    # The multiplication is C_{i, l} = sum_j A_{i, j} B_{j, l}
    # So T_{i, j, l} = A_{i, j} B_{j, l} C_{i, l} (conceptually)
    # The bilinear form is <A, B> -> C.
    # The tensor entries are 1 if the indices match the multiplication term.
    # T_{i, j, l} = 1 if we are computing C_{i, l} using A_{i, j} and B_{j, l}.
    
    # Rank 21 decomposition.
    rank = 21
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Indices for U: i * m + j
    # Indices for V: j * p + l
    # Indices for W: l * n + i
    
    # Strassen coefficients for U (A part)
    # U[i, k]
    # k=0..6 for each block r=0..2
    # Total k=0..20
    
    # For a given block r, the local k is 0..6.
    # Global k = r * 7 + local_k
    
    # U coefficients (same for all blocks)
    # local_k: 0(M1), 1(M2), 2(M3), 3(M4), 4(M5), 5(M6), 6(M7)
    
    # M1: A00 -> U[0, k] = 1
    # M2: A10+A11 -> U[2, k]=1, U[3, k]=1
    # M3: A00 -> U[0, k] = 1
    # M4: A11 -> U[3, k] = 1
    # M5: A00+A01 -> U[0, k]=1, U[1, k]=1
    # M6: A10-A00 -> U[2, k]=1, U[0, k]=-1
    # M7: A01-A11 -> U[1, k]=1, U[3, k]=-1
    
    U_coeffs = [
        {0: 1},              # M1
        {2: 1, 3: 1},        # M2
        {0: 1},              # M3
        {3: 1},              # M4
        {0: 1, 1: 1},        # M5
        {2: 1, 0: -1},       # M6
        {1: 1, 3: -1}        # M7
    ]
    
    # V coefficients (B part)
    # V[j, l] where j is row of B (0,1), l is col of B (0..5)
    # For block r, l = 2*r + local_l (local_l is 0,1)
    # V index = j * p + l
    
    # M1: B00+B11 -> V[0, 2r+0]=1, V[1, 2r+1]=1
    # M2: B00 -> V[0, 2r+0]=1
    # M3: B01-B11 -> V[0, 2r+1]=1, V[1, 2r+1]=-1
    # M4: B10-B00 -> V[1, 2r+0]=1, V[0, 2r+0]=-1
    # M5: B11 -> V[1, 2r+1]=1
    # M6: B00+B01 -> V[0, 2r+0]=1, V[0, 2r+1]=1
    # M7: B10+B11 -> V[1, 2r+0]=1, V[1, 2r+1]=1
    
    V_coeffs = [
        [(0, 0, 1), (1, 1, 1)],      # M1
        [(0, 0, 1)],                 # M2
        [(0, 1, 1), (1, 1, -1)],     # M3
        [(1, 0, 1), (0, 0, -1)],     # M4
        [(1, 1, 1)],                 # M5
        [(0, 0, 1), (0, 1, 1)],      # M6
        [(1, 0, 1), (1, 1, 1)]       # M7
    ]
    
    # W coefficients (C part)
    # W[l, i] where l is col of C (0..5), i is row of C (0,1)
    # W index = l * n + i
    # For block r, l = 2*r + local_l
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We need W[l, i] for each k.
    # W_map[local_k] = list of (local_l, i, coeff)
    
    # M1 (k=0): C00(1), C11(1) -> (0,0,1), (1,1,1)
    # M2 (k=1): C10(1), C11(-1) -> (0,1,1), (1,1,-1)
    # M3 (k=2): C01(1), C11(1) -> (1,0,1), (1,1,1)
    # M4 (k=3): C00(1), C10(1) -> (0,0,1), (0,1,1)
    # M5 (k=4): C00(-1), C01(1) -> (0,0,-1), (1,0,1)
    # M6 (k=5): C11(1) -> (1,1,1)
    # M7 (k=6): C00(1) -> (0,0,1)
    
    W_map = [
        [(0, 0, 1), (1, 1, 1)],   # M1
        [(0, 1, 1), (1, 1, -1)],  # M2
        [(1, 0, 1), (1, 1, 1)],   # M3
        [(0, 0, 1), (0, 1, 1)],   # M4
        [(0, 0, -1), (1, 0, 1)],  # M5
        [(1, 1, 1)],              # M6
        [(0, 0, 1)]               # M7
    ]
    
    for r in range(num_blocks):
        for local_k in range(7):
            k = r * 7 + local_k
            
            # U
            for idx, val in U_coeffs[local_k].items():
                U[idx, k] = val
                
            # V
            for j, local_l, val in V_coeffs[local_k]:
                l = 2 * r + local_l
                v_idx = j * p + l
                V[v_idx, k] = val
                
            # W
            for local_l, i, val in W_map[local_k]:
                l = 2 * r + local_l
                w_idx = l * n + i
                W[w_idx, k] = val
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
