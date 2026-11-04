import numpy as np
import time

DIMENSIONS = [4, 4, 7]

def get_strassen_2x2x2():
    """
    Returns U, V, W matrices (4x7) for the rank-7 Strassen decomposition of 2x2x2 matrix multiplication.
    Indices: a=i*2+j, b=j*2+k, c=k*2+i for i,j,k in {0,1}.
    """
    U = np.zeros((4, 7), dtype=np.int8)
    V = np.zeros((4, 7), dtype=np.int8)
    W = np.zeros((4, 7), dtype=np.int8)
    
    # Strassen terms:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # We need to map these to vectors u, v, w of length 4.
    # u corresponds to A coefficients, v to B, w to C.
    # A = [[A00, A01], [A10, A11]] -> indices 0,1,2,3
    # B = [[B00, B01], [B10, B11]] -> indices 0,1,2,3
    # C = [[C00, C01], [C10, C11]] -> indices 0,1,2,3
    
    # Term 1: (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11
    # This is a sum of 4 rank-1 tensors? No, Strassen combines them into one multiplication.
    # The "multiplication" is the scalar product.
    # The tensor entry T[a,b,c] = 1 if a,b,c are consistent.
    # Strassen decomposes T into sum of 7 rank-1 tensors u v w.
    
    # Let's find the 7 vectors.
    # Reference: Standard Strassen decomposition vectors.
    
    # u1 = [1, 0, 1, 1] ? No.
    # Let's derive or look up.
    
    # u vectors (coefficients for A):
    u1 = [1, 0, 0, 1] # A00 + A11
    u2 = [0, 0, 1, 1] # A10 + A11
    u3 = [1, 0, 0, 0] # A00
    u4 = [0, 0, 0, 1] # A11
    u5 = [1, 1, 0, 0] # A00 + A01
    u6 = [-1, 0, 1, 0] # A10 - A00
    u7 = [0, 1, 0, -1] # A01 - A11
    
    # v vectors (coefficients for B):
    v1 = [1, 0, 0, 1] # B00 + B11
    v2 = [1, 0, 0, 0] # B00
    v3 = [0, 1, 0, -1] # B01 - B11
    v4 = [0, 0, 1, -1] # B10 - B00
    v5 = [0, 0, 0, 1] # B11
    v6 = [1, 1, 0, 0] # B00 + B01
    v7 = [0, 0, 1, 1] # B10 + B11
    
    # w vectors (coefficients for C):
    # C00 = M1 - M4 + M6 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M5 - M6
    
    w1 = [1, 0, 0, 1] # C00 + C11 (from M1) -> Wait, M1 contributes to C00 and C11.
    # Let's check the expansion:
    # M1 = (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11
    # C00 gets A00B00. C11 gets A11B11.
    # So M1 contributes +1 to C00 and +1 to C11?
    # No, the term is u1 * v1 * w1.
    # u1 = A00+A11, v1 = B00+B11.
    # The product is (A00+A11)(B00+B11).
    # This scalar multiplies w1.
    # w1 should be such that the sum of all 7 terms reconstructs C.
    
    # Let's use the standard w vectors for Strassen:
    w1 = [1, 0, 0, 1] # +1 for C00, +1 for C11?
    # Actually, let's verify the reconstruction.
    # C00 = M1 - M4 + M6 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M5 - M6
    
    # So:
    w1 = [1, 0, 0, 1] # M1: +C00, +C11
    w2 = [0, 0, 1, 0] # M2: +C10
    w3 = [0, 1, 0, 0] # M3: +C01
    w4 = [-1, 0, 1, 0] # M4: -C00, +C10
    w5 = [0, 1, 0, 1] # M5: +C01, +C11
    w6 = [1, 0, 0, -1] # M6: +C00, -C11
    w7 = [-1, 0, 0, 1] # M7: -C00, +C11
    
    # Let's double check M4.
    # M4 = A11(B10-B00).
    # C00 = ... - M4 ...
    # C10 = ... + M4 ...
    # So w4 should be [-1, 0, 1, 0]. Correct.
    
    # Let's double check M6.
    # M6 = (A10-A00)(B00+B01).
    # C00 = ... + M6 ...
    # C11 = ... - M6 ...
    # So w6 should be [1, 0, 0, -1]. Correct.
    
    # Let's double check M7.
    # M7 = (A01-A11)(B10+B11).
    # C00 = ... - M7 ...
    # C11 = ... + M7 ...
    # So w7 should be [-1, 0, 0, 1]. Correct.
    
    U = np.array([u1, u2, u3, u4, u5, u6, u7]).T
    V = np.array([v1, v2, v3, v4, v5, v6, v7]).T
    W = np.array([w1, w2, w3, w4, w5, w6, w7]).T
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    # Dimensions
    dim_u = n * m  # 16
    dim_v = m * p  # 28
    dim_w = p * n  # 28
    
    # 1. Get Strassen decomposition for 2x2x2
    U_2, V_2, W_2 = get_strassen_2x2x2()
    
    # 2. Construct 4x4x4 decomposition using Kronecker product
    # Rank is 7 * 7 = 49
    U_4 = []
    V_4 = []
    W_4 = []
    
    for r in range(7):
        for s in range(7):
            u = np.kron(U_2[:, r], U_2[:, s])
            v = np.kron(V_2[:, r], V_2[:, s])
            w = np.kron(W_2[:, r], W_2[:, s])
            U_4.append(u)
            V_4.append(v)
            W_4.append(w)
            
    # 3. Construct 4x4x3 schoolbook decomposition
    # This covers k in {4, 5, 6}
    # Indices: a = i*4+j, b = j*7+k, c = k*4+i
    U_3 = []
    V_3 = []
    W_3 = []
    
    for i in range(n):
        for j in range(m):
            for k in range(4, p): # k = 4, 5, 6
                u = np.zeros(dim_u)
                v = np.zeros(dim_v)
                w = np.zeros(dim_w)
                
                a = i * m + j
                b = j * p + k
                c = k * n + i
                
                u[a] = 1
                v[b] = 1
                w[c] = 1
                
                U_3.append(u)
                V_3.append(v)
                W_3.append(w)
                
    # Combine
    U_all = np.array(U_4 + U_3).T
    V_all = np.array(V_4 + V_3).T
    W_all = np.array(W_4 + W_3).T
    
    rank = U_all.shape[1]
    
    # Verify validity (optional but good for debugging in local runs)
    # T_target = np.zeros((dim_u, dim_v, dim_w), dtype=np.int8)
    # for i in range(n):
    #     for j in range(m):
    #         for k in range(p):
    #             a = i * m + j
    #             b = j * p + k
    #             c = k * n + i
    #             T_target[a, b, c] = 1
    #             
    # T_decomp = np.einsum('ar,br,cr->abc', U_all, V_all, W_all)
    # if not np.array_equal(T_target, T_decomp):
    #     print("Decomposition failed")
    #     # Fallback to schoolbook
    #     rank = n * m * p
    #     U = np.zeros((dim_u, rank))
    #     V = np.zeros((dim_v, rank))
    #     W = np.zeros((dim_w, rank))
    #     col = 0
    #     for i in range(n):
    #         for j in range(m):
    #             for k in range(p):
    #                 a = i * m + j
    #                 b = j * p + k
    #                 c = k * n + i
    #                 U[a, col] = 1
    #                 V[b, col] = 1
    #                 W[c, col] = 1
    #                 col += 1
    #     return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {
        "U": U_all.tolist(),
        "V": V_all.tolist(),
        "W": W_all.tolist()
    }
