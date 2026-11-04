import numpy as np

DIMENSIONS = [4, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    rank = 98
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    def strassen_2x2x2():
        forms = []
        u1 = np.zeros(4); u1[0] = 1; u1[3] = 1
        v1 = np.zeros(4); v1[0] = 1; v1[3] = 1
        w1 = np.zeros(4); w1[0] = 1; w1[3] = 1
        forms.append((u1, v1, w1))
        
        u2 = np.zeros(4); u2[2] = 1; u2[3] = 1
        v2 = np.zeros(4); v2[0] = 1
        w2 = np.zeros(4); w2[2] = 1; w2[3] = -1
        forms.append((u2, v2, w2))
        
        u3 = np.zeros(4); u3[0] = 1
        v3 = np.zeros(4); v3[1] = 1; v3[3] = -1
        w3 = np.zeros(4); w3[1] = 1; w3[3] = 1
        forms.append((u3, v3, w3))
        
        u4 = np.zeros(4); u4[3] = 1
        v4 = np.zeros(4); v4[2] = 1; v4[0] = -1
        w4 = np.zeros(4); w4[0] = 1; w4[2] = 1
        forms.append((u4, v4, w4))
        
        u5 = np.zeros(4); u5[0] = 1; u5[1] = 1
        v5 = np.zeros(4); v5[3] = 1
        w5 = np.zeros(4); w5[0] = -1; w5[1] = 1
        forms.append((u5, v5, w5))
        
        u6 = np.zeros(4); u6[2] = 1; u6[0] = -1
        v6 = np.zeros(4); v6[0] = 1; v6[1] = 1
        w6 = np.zeros(4); w6[3] = 1
        forms.append((u6, v6, w6))
        
        u7 = np.zeros(4); u7[1] = 1; u7[3] = -1
        v7 = np.zeros(4); v7[2] = 1; v7[3] = 1
        w7 = np.zeros(4); w7[0] = 1
        forms.append((u7, v7, w7))
        
        return forms

    strassen_forms = strassen_2x2x2()
    
    M_products = [
        ([((0,0), 1), ((1,1), 1)], [((0,0), 1), ((1,1), 1)]),
        ([((1,0), 1), ((1,1), 1)], [((0,0), 1)]),
        ([((0,0), 1)], [((0,1), 1), ((1,1), -1)]),
        ([((1,1), 1)], [((1,0), 1), ((0,0), -1)]),
        ([((0,0), 1), ((0,1), 1)], [((1,1), 1)]),
        ([((1,0), 1), ((0,0), -1)], [((0,0), 1), ((0,1), 1)]),
        ([((0,1), 1), ((1,1), -1)], [((1,0), 1), ((1,1), 1)])
    ]
    
    c_block_coeffs = np.zeros((2, 2, 7))
    c_block_coeffs[0, 0, 0] = 1
    c_block_coeffs[0, 0, 3] = 1
    c_block_coeffs[0, 0, 4] = -1
    c_block_coeffs[0, 0, 6] = 1
    
    c_block_coeffs[0, 1, 2] = 1
    c_block_coeffs[0, 1, 4] = 1
    
    c_block_coeffs[1, 0, 1] = 1
    c_block_coeffs[1, 0, 3] = 1
    
    c_block_coeffs[1, 1, 0] = 1
    c_block_coeffs[1, 1, 1] = -1
    c_block_coeffs[1, 1, 2] = 1
    c_block_coeffs[1, 1, 5] = 1
    
    col_idx = 0
    for m_idx, (m_a, m_b) in enumerate(M_products):
        for u_small, v_small, w_small in strassen_forms:
            u_full = np.zeros(16)
            v_full = np.zeros(16)
            
            for (bi, bj), coeff in m_a:
                idxs = [
                    2*bi*4 + 2*bj,
                    2*bi*4 + 2*bj + 1,
                    (2*bi+1)*4 + 2*bj,
                    (2*bi+1)*4 + 2*bj + 1
                ]
                u_full[idxs[0]] += coeff * u_small[0]
                u_full[idxs[1]] += coeff * u_small[1]
                u_full[idxs[2]] += coeff * u_small[2]
                u_full[idxs[3]] += coeff * u_small[3]
            
            for (bi, bj), coeff in m_b:
                idxs = [
                    2*bi*4 + 2*bj,
                    2*bi*4 + 2*bj + 1,
                    (2*bi+1)*4 + 2*bj,
                    (2*bi+1)*4 + 2*bj + 1
                ]
                v_full[idxs[0]] += coeff * v_small[0]
                v_full[idxs[1]] += coeff * v_small[1]
                v_full[idxs[2]] += coeff * v_small[2]
                v_full[idxs[3]] += coeff * v_small[3]
            
            w_full = np.zeros(16)
            for ci in range(2):
                for cj in range(2):
                    coeff = c_block_coeffs[ci, cj, m_idx]
                    if coeff != 0:
                        idxs = [
                            8*cj + 2*ci,
                            8*cj + 4 + 2*ci,
                            8*cj + 2*ci + 1,
                            8*cj + 4 + 2*ci + 1
                        ]
                        w_full[idxs[0]] += coeff * w_small[0]
                        w_full[idxs[1]] += coeff * w_small[1]
                        w_full[idxs[2]] += coeff * w_small[2]
                        w_full[idxs[3]] += coeff * w_small[3]
            
            U[:, col_idx] = u_full
            V[:, col_idx] = v_full
            W[:, col_idx] = w_full
            col_idx += 1
            
    for m_idx, (m_a, m_b) in enumerate(M_products):
        for u_small, v_small, w_small in strassen_forms:
            u_full = np.zeros(16)
            v_full = np.zeros(32)
            
            for (bi, bj), coeff in m_a:
                idxs = [
                    2*bi*4 + 2*bj,
                    2*bi*4 + 2*bj + 1,
                    (2*bi+1)*4 + 2*bj,
                    (2*bi+1)*4 + 2*bj + 1
                ]
                u_full[idxs[0]] += coeff * u_small[0]
                u_full[idxs[1]] += coeff * u_small[1]
                u_full[idxs[2]] += coeff * u_small[2]
                u_full[idxs[3]] += coeff * u_small[3]
            
            for (bi, bj), coeff in m_b:
                idxs = [
                    2*bi*8 + 4 + 2*bj,
                    2*bi*8 + 4 + 2*bj + 1,
                    (2*bi+1)*8 + 4 + 2*bj,
                    (2*bi+1)*8 + 4 + 2*bj + 1
                ]
                v_full[idxs[0]] += coeff * v_small[0]
                v_full[idxs[1]] += coeff * v_small[1]
                v_full[idxs[2]] += coeff * v_small[2]
                v_full[idxs[3]] += coeff * v_small[3]
            
            w_full = np.zeros(32)
            for ci in range(2):
                for cj in range(2):
                    coeff = c_block_coeffs[ci, cj, m_idx]
                    if coeff != 0:
                        idxs = [
                            16 + 8*cj + 2*ci,
                            16 + 8*cj + 4 + 2*ci,
                            16 + 8*cj + 2*ci + 1,
                            16 + 8*cj + 4 + 2*ci + 1
                        ]
                        w_full[idxs[0]] += coeff * w_small[0]
                        w_full[idxs[1]] += coeff * w_small[1]
                        w_full[idxs[2]] += coeff * w_small[2]
                        w_full[idxs[3]] += coeff * w_small[3]
            
            U[:, col_idx] = u_full
            V[:, col_idx] = v_full
            W[:, col_idx] = w_full
            col_idx += 1
            
    return {"U": U, "V": V, "W": W}
