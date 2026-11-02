"""Strassen-based rank-49 tensor decomposition for 4x4 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """Construct a rank-49 exact tensor decomposition for 4x4 * 4x4 matrix multiplication.

    Uses the Kronecker product of two rank-7 Strassen decompositions for 2x2 matrices.
    The output factor W follows the convention W[k*n + i, r] with n=4.

    Returns:
        dict with keys 'U', 'V', 'W' where:
            U has shape (16, 49)  -- (n*m, rank)
            V has shape (16, 49)  -- (m*p, rank)
            W has shape (16, 49)  -- (p*n, rank)
    """
    deadline = time.time() + 120.0

    # Strassen 2x2 decomposition.
    # We multiply A (2x2) by B (2x2) to get C = A * B.
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]], C = [[C00, C01], [C10, C11]]
    #
    # Strassen's 7 products:
    #   M1 = (a+d)(e+h)
    #   M2 = (c+d)*e
    #   M3 = a*(f-h)
    #   M4 = d*(g-e)
    #   M5 = (a+b)*h
    #   M6 = (c-a)*(e+f)
    #   M7 = (b-d)*(g+h)
    #
    # C00 = M1 - M4 + M5 + M6
    # C01 = M3 + M4
    # C10 = M2 + M5
    # C11 = M1 - M3 + M2 - M7
    #
    # Tensor decomposition: C_{i,j} = sum_r U_{i,j,r} * (a*d stuff) * V_{...}
    # More precisely, for 2x2:
    #   C_{i,j} = sum_r U_{i,j,r} * M_r where M_r is a bilinear form in A and B.
    #   M_r = sum_{p,q} U^{A}_{p,q,r} * A_{p,q} * V^{B}_{p',q',r} * B_{p',q'}
    #
    # Let's define for each r=1..7:
    #   The coefficient vectors for A and B.
    #
    # Indexing: A = [[a, b], [c, d]] with a=(0,0), b=(0,1), c=(1,0), d=(1,1)
    #            B = [[e, f], [g, h]] with e=(0,0), f=(0,1), g=(1,0), h=(1,1)
    #
    # U for A (flattened row-major: index = i*2+j for 2x2):
    #   M1: a+d -> [1, 0, 0, 1]
    #   M2: c+d -> [0, 0, 1, 1]
    #   M3: a   -> [1, 0, 0, 0]
    #   M4: d   -> [0, 0, 0, 1]
    #   M5: a+b -> [1, 1, 0, 0]
    #   M6: c-a -> [-1, 0, 1, 0]
    #   M7: b-d -> [0, 1, 0, -1]
    #
    # V for B (flattened row-major):
    #   M1: e+h -> [1, 0, 0, 1]
    #   M2: e   -> [1, 0, 0, 0]
    #   M3: f-h -> [0, 1, 0, -1]
    #   M4: g-e -> [-1, 0, 1, 0]
    #   M5: h   -> [0, 0, 0, 1]
    #   M6: e+f -> [1, 1, 0, 0]
    #   M7: g+h -> [0, 0, 1, 1]
    #
    # W for C (flattened row-major: index = i*2+j for 2x2):
    #   M1: C00+1, C11+1 -> [1, 0, 0, 1]
    #   M2: C10+1, C11-1 -> [0, 0, 1, -1]
    #   M3: C01-1, C11-1 -> [0, -1, 0, -1]
    #   M4: C00-1, C01+1 -> [-1, 1, 0, 0]
    #   M5: C00+1, C10+1 -> [1, 0, 1, 0]
    #   M6: C00+1, C10-1 -> [1, 0, 1, 0]  -- wait, let me re-derive
    #
    # Let me be more careful.
    # C00 = M1 - M4 + M5 + M6
    # C01 = M3 + M4
    # C10 = M2 + M5
    # C11 = M1 - M3 + M2 - M7
    #
    # So W vectors (index = i*2+j, so C00=0, C01=1, C10=2, C11=3):
    #   M1: C00+1, C11+1 -> [1, 0, 0, 1]
    #   M2: C10+1, C11+1 -> [0, 0, 1, 1]
    #   M3: C01+1, C11-1 -> [0, 1, 0, -1]
    #   M4: C00-1, C01+1 -> [-1, 1, 0, 0]
    #   M5: C00+1, C10+1 -> [1, 0, 1, 0]
    #   M6: C00+1        -> [1, 0, 0, 0]  -- wait, C00 = M1 - M4 + M5 + M6, so M6 contributes +1 to C00 only
    #   M7: C11-1        -> [0, 0, 0, -1]  -- C11 = M1 - M3 + M2 - M7, so M7 contributes -1 to C11 only

    # Let me re-verify:
    # C00 = M1 - M4 + M5 + M6
    #   M1: +1 to C00
    #   M4: -1 to C00
    #   M5: +1 to C00
    #   M6: +1 to C00
    # C01 = M3 + M4
    #   M3: +1 to C01
    #   M4: +1 to C01
    # C10 = M2 + M5
    #   M2: +1 to C10
    #   M5: +1 to C10
    # C11 = M1 - M3 + M2 - M7
    #   M1: +1 to C11
    #   M3: -1 to C11
    #   M2: +1 to C11
    #   M7: -1 to C11

    # W vectors (flattened [C00, C01, C10, C11]):
    #   M1: [1, 0, 0, 1]
    #   M2: [0, 0, 1, 1]
    #   M3: [0, 1, 0, -1]
    #   M4: [-1, 1, 0, 0]
    #   M5: [1, 0, 1, 0]
    #   M6: [1, 0, 0, 0]
    #   M7: [0, 0, 0, -1]

    U_2x2 = [
        [1, 0, 0, 1],   # M1: a+d
        [0, 0, 1, 1],   # M2: c+d
        [1, 0, 0, 0],   # M3: a
        [0, 0, 0, 1],   # M4: d
        [1, 1, 0, 0],   # M5: a+b
        [-1, 0, 1, 0],  # M6: c-a
        [0, 1, 0, -1],  # M7: b-d
    ]

    V_2x2 = [
        [1, 0, 0, 1],   # M1: e+h
        [1, 0, 0, 0],   # M2: e
        [0, 1, 0, -1],  # M3: f-h
        [-1, 0, 1, 0],  # M4: g-e
        [0, 0, 0, 1],   # M5: h
        [1, 1, 0, 0],   # M6: e+f
        [0, 0, 1, 1],   # M7: g+h
    ]

    W_2x2 = [
        [1, 0, 0, 1],   # M1: C00+1, C11+1
        [0, 0, 1, 1],   # M2: C10+1, C11+1
        [0, 1, 0, -1],  # M3: C01+1, C11-1
        [-1, 1, 0, 0],  # M4: C00-1, C01+1
        [1, 0, 1, 0],   # M5: C00+1, C10+1
        [1, 0, 0, 0],   # M6: C00+1
        [0, 0, 0, -1],  # M7: C11-1
    ]

    # Convert to numpy arrays for Kronecker product
    U_arr = np.array(U_2x2, dtype=float)  # shape (7, 4)
    V_arr = np.array(V_2x2, dtype=float)  # shape (7, 4)
    W_arr = np.array(W_2x2, dtype=float)  # shape (7, 4)

    # For 4x4 multiplication via Kronecker product of two 2x2 Strassen:
    # A (4x4) is viewed as 2x2 block matrix of 2x2 blocks.
    # A_{I,J} where I,J in {0,1}, each block is 2x2.
    # Similarly for B.
    # C = A * B, where C_{I,J} = sum_K A_{I,K} * B_{K,J}.
    #
    # The Kronecker product decomposition:
    # For each pair (s, r) in 0..6 x 0..6:
    #   U_{sr} = kron(U_s, U_r) -- but we need to be careful with indexing.
    #
    # The 4x4 matrix A has entries A_{i,j} where i,j in 0..3.
    # View as A_{2I+i, 2J+j} = A_{I,J}^{(i,j)} where I,J in {0,1}, i,j in {0,1}.
    # Similarly B_{2K+k, 2L+l} = B_{K,L}^{(k,l)}.
    # C_{2I+i, 2J+j} = sum_{K,k} A_{2I+i, 2K+k} * B_{2K+k, 2J+j}
    #                = sum_{K,k} sum_{K',k'} A_{2I+i, 2K+k} B_{2K+k, 2J+j}
    #
    # Using Strassen on the block level:
    # C_{I,J} = sum_s U_s[I,K] * (A_block * B_block products) ... 
    #
    # Actually, the standard Kronecker product approach:
    # If we have a decomposition of 2x2 matmul with rank r,
    # then 4x4 matmul has rank r^2 via Kronecker product.
    #
    # The entries of the 4x4 matrices:
    # A is 4x4, indexed by (i, j) with i, j in 0..3.
    # We view i = 2*I + i', j = 2*J + j' where I, J, i', j' in {0, 1}.
    # So A[i, j] = A[2*I+i', 2*J+j'].
    #
    # U matrix for 4x4: U[i*m + j, r] where r = s*7 + r' (or s + 7*r')
    #   U[2*I+i', 2*J+j', s*7+r'] = U_2x2[s, 2*I+i'] * U_2x2[r', 2*J+j']
    #   Wait, that's not quite right. Let me think again.
    #
    # In the Kronecker product of tensor decompositions:
    # The tensor T for 4x4 matmul is the Kronecker product of the tensor for 2x2 matmul with itself.
    # T4[i, j, k] = T2[i1, j1, k1] * T2[i2, j2, k2] where i = 2*i1+i2, etc.
    #
    # In the decomposition: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # (using appropriate index conventions)
    #
    # For the Kronecker product:
    # U4[2*I+i', 2*J+j', s*7+r'] = U2[I, J, s] * U2[i', j', r']
    # V4[2*J+j', 2*K+k', s*7+r'] = V2[J, K, s] * V2[j', k', r']
    # W4[2*K+k', 2*I+i', s*7+r'] = W2[K, I, s] * W2[k', i', r']
    #
    # But we need to match the flattening conventions.
    # U is shape (n*m, rank) = (16, 49). Index = i*4 + j for 4x4.
    #   So U[i*4 + j, r] where i,j in 0..3.
    #   i = 2*I + i', j = 2*J + j'.
    #   U[2*I+i', 2*J+j', s*7+r'] = U2[I*2+J, s] * U2[i'*2+j', r']
    #   Wait, U2 is for 2x2, so U2 has shape (4, 7) with index = I*2+J.
    #   So U2[I*2+J, s].
    #
    # Let me define:
    # U2_flat[s, idx] where idx = I*2+J (row-major for 2x2)
    # V2_flat[s, idx] where idx = J*2+K
    # W2_flat[s, idx] where idx = K*2+I (following the convention W[k*n+i] for 2x2, n=2, so 2*K+I)
    #
    # For 4x4:
    # U4_flat[r, idx] where r = s*7+r', idx = i*4+j = (2*I+i')*4 + (2*J+j')
    #   = 8*I + 4*i' + 2*J + j'
    #   U4[idx, r] = U2[I*2+J, s] * U2[i'*2+j', r']
    #
    # V4_flat[r, idx] where idx = j*4+k = (2*J+j')*4 + (2*K+k')
    #   = 8*J + 4*j' + 2*K + k'
    #   V4[idx, r] = V2[J*2+K, s] * V2[j'*2+k', r']
    #
    # W4_flat[r, idx] where idx = k*4+i = (2*K+k')*4 + (2*I+i')
    #   = 8*K + 4*k' + 2*I + i'
    #   W4[idx, r] = W2[K*2+I, s] * W2[k'*2+i', r']

    rank = 49
    U4 = np.zeros((16, rank), dtype=int)
    V4 = np.zeros((16, rank), dtype=int)
    W4 = np.zeros((16, rank), dtype=int)

    for s in range(7):
        for rp in range(7):
            r = s * 7 + rp
            for I in range(2):
                for J in range(2):
                    for i_prime in range(2):
                        for j_prime in range(2):
                            # U
                            idx_u = (2 * I + i_prime) * 4 + (2 * J + j_prime)
                            U4[idx_u, r] = U_2x2[s][I * 2 + J] * U_2x2[rp][i_prime * 2 + j_prime]

            for J in range(2):
                for K in range(2):
                    for j_prime in range(2):
                        for k_prime in range(2):
                            # V
                            idx_v = (2 * J + j_prime) * 4 + (2 * K + k_prime)
                            V4[idx_v, r] = V_2x2[s][J * 2 + K] * V_2x2[rp][j_prime * 2 + k_prime]

            for K in range(2):
                for I in range(2):
                    for k_prime in range(2):
                        for i_prime in range(2):
                            # W
                            idx_w = (2 * K + k_prime) * 4 + (2 * I + i_prime)
                            W4[idx_w, r] = W_2x2[s][K * 2 + I] * W_2x2[rp][k_prime * 2 + i_prime]

    # Convert to list of lists of integers
    U_list = U4.tolist()
    V_list = V4.tolist()
    W_list = W4.tolist()

    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
