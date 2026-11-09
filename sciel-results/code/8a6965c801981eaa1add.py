import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 100  # Safety margin
    
    n, m, p = 5, 5, 5
    # We will build a decomposition of rank 110.
    # Part 1: Strassen 4x4 for the top-left 4x4 block.
    # Part 2: Schoolbook for the cross terms.
    
    # Helper: Get Strassen 2x2 matrices
    def get_strassen_2x2():
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # P1
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        # P2
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        # P3
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        # P4
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        # P5
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        # P6
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        # P7
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        
        W[0, 0] = 1; W[0, 3] = 1; W[0, 4] = -1; W[0, 6] = 1
        W[2, 2] = 1; W[2, 4] = 1
        W[1, 1] = 1; W[1, 3] = 1
        W[3, 0] = 1; W[3, 1] = -1; W[3, 2] = 1; W[3, 5] = 1
        
        return U, V, W

    U_s, V_s, W_s = get_strassen_2x2()
    
    # Build Strassen 4x4
    # Block indices bi, bj, bk in {0, 1}
    # Scalar indices u, v, w in {0, 1}
    # Global i = 2*bi + u, j = 2*bj + v, k = 2*bk + w
    # U index: i*4 + j
    # V index: j*4 + k
    # W index: k*4 + i
    
    rank_4x4 = 49
    U_4 = np.zeros((16, rank_4x4))
    V_4 = np.zeros((16, rank_4x4))
    W_4 = np.zeros((16, rank_4x4))
    
    for rb in range(7):
        for rs in range(7):
            r = rb * 7 + rs
            for bi in range(2):
                for bj in range(2):
                    for bk in range(2):
                        # Coefficients from block Strassen
                        cb_u = U_s[bi*2 + bj, rb] # Wait, U_s index is i*2+j?
                        # In get_strassen_2x2, U is (4,7). Index is i*2+j.
                        # So U_s[bi*2+bj, rb] is correct for block U.
                        # But we need to combine with scalar U_s.
                        
                        # Actually, the block Strassen U_s maps block indices (bi, bj) to coefficient.
                        # The scalar Strassen U_s maps scalar indices (u, v) to coefficient.
                        
                        # The total coefficient is product.
                        # We iterate over all bi, bj, bk and u, v, w?
                        # No, the structure is separable.
                        # U[i, j, r] = sum over bi, bj, u, v ...
                        # It's a Kronecker product.
                        # U_4 = kron(U_block, U_scalar) ?
                        # The indices are interleaved.
                        
                        # Let's just loop over all combinations.
                        pass
                        
    # Efficient construction:
    # U_4[i, j, r] where i=2bi+u, j=2bj+v, r=7rb+rs
    # U_4[2bi+u, 2bj+v, 7rb+rs] = U_block[bi, bj, rb] * U_scalar[u, v, rs]
    
    # U_block is U_s but with indices bi, bj.
    # U_scalar is U_s with indices u, v.
    
    for bi in range(2):
        for bj in range(2):
            for rb in range(7):
                coeff_b = U_s[bi*2 + bj, rb]
                if coeff_b == 0: continue
                for u in range(2):
                    for v in range(2):
                        coeff_s = U_s[u*2 + v, rs] # Wait, rs is fixed in outer loop?
                        # No, rs is the scalar rank index.
                        # We need to loop over rs too.
                        
    # Let's rewrite the loop correctly.
    for rb in range(7):
        for rs in range(7):
            r = rb * 7 + rs
            for bi in range(2):
                for bj in range(2):
                    coeff_b = U_s[bi*2 + bj, rb]
                    if coeff_b == 0: continue
                    for u in range(2):
                        for v in range(2):
                            coeff_s = U_s[u*2 + v, rs]
                            if coeff_s == 0: continue
                            i = 2*bi + u
                            j = 2*bj + v
                            U_4[i*4 + j, r] += coeff_b * coeff_s
                            
            # Similarly for V and W
            for bj in range(2):
                for bk in range(2):
                    coeff_b = V_s[bj*2 + bk, rb]
                    if coeff_b == 0: continue
                    for v in range(2):
                        for w in range(2):
                            coeff_s = V_s[v*2 + w, rs]
                            if coeff_s == 0: continue
                            j = 2*bj + v
                            k = 2*bk + w
                            V_4[j*4 + k, r] += coeff_b * coeff_s
                            
            for bk in range(2):
                for bi in range(2):
                    coeff_b = W_s[bk*2 + bi, rb]
                    if coeff_b == 0: continue
                    for w in range(2):
                        for u in range(2):
                            coeff_s = W_s[w*2 + u, rs]
                            if coeff_s == 0: continue
                            k = 2*bk + w
                            i = 2*bi + u
                            W_4[k*4 + i, r] += coeff_b * coeff_s

    # Now assemble the 5x5 decomposition.
    # Rank 110.
    # First 49 columns: Strassen 4x4 for i,j,k in 0..3.
    # Map U_4 (16x49) to U (25x110).
    # U index: i*5 + j.
    # U_4 index: i*4 + j.
    # For i in 0..3, j in 0..3:
    # U[i*5 + j, 0:49] = U_4[i*4 + j, :]
    
    U = np.zeros((25, 110))
    V = np.zeros((25, 110))
    W = np.zeros((25, 110))
    
    # Copy Strassen part
    for i in range(4):
        for j in range(4):
            U[i*5 + j, 0:49] = U_4[i*4 + j, :]
    for j in range(4):
        for k in range(4):
            V[j*5 + k, 0:49] = V_4[j*4 + k, :]
    for k in range(4):
        for i in range(4):
            W[k*5 + i, 0:49] = W_4[k*4 + i, :]
            
    # Add schoolbook cross terms.
    # Term 1: A[i,4] B[4,k] for i in 0..3, k in 0..3. (16 terms)
    # U[i*5+4, r] = 1, V[4*5+k, r] = 1, W[k*5+i, r] = 1.
    r = 49
    for i in range(4):
        for k in range(4):
            U[i*5 + 4, r] = 1
            V[4*5 + k, r] = 1
            W[k*5 + i, r] = 1
            r += 1
            
    # Term 2: A[i,j] B[j,4] for i in 0..3, j in 0..3. (16 terms)
    # U[i*5+j, r] = 1, V[j*5+4, r] = 1, W[4*5+i, r] = 1.
    for i in range(4):
        for j in range(4):
            U[i*5 + j, r] = 1
            V[j*5 + 4, r] = 1
            W[4*5 + i, r] = 1
            r += 1
            
    # Term 3: A[4,j] B[j,k] for j in 0..3, k in 0..3. (16 terms)
    # U[4*5+j, r] = 1, V[j*5+k, r] = 1, W[k*5+4, r] = 1.
    for j in range(4):
        for k in range(4):
            U[4*5 + j, r] = 1
            V[j*5 + k, r] = 1
            W[k*5 + 4, r] = 1
            r += 1
            
    # Term 4: A[i,4] B[4,4] for i in 0..3. (4 terms)
    # U[i*5+4, r] = 1, V[4*5+4, r] = 1, W[4*5+i, r] = 1.
    for i in range(4):
        U[i*5 + 4, r] = 1
        V[4*5 + 4, r] = 1
        W[4*5 + i, r] = 1
        r += 1
        
    # Term 5: A[4,j] B[j,4] for j in 0..3. (4 terms)
    # U[4*5+j, r] = 1, V[j*5+4, r] = 1, W[4*5+4, r] = 1.
    for j in range(4):
        U[4*5 + j, r] = 1
        V[j*5 + 4, r] = 1
        W[4*5 + 4, r] = 1
        r += 1
        
    # Term 6: A[4,4] B[4,4]. (1 term)
    U[4*5 + 4, r] = 1
    V[4*5 + 4, r] = 1
    W[4*5 + 4, r] = 1
    r += 1
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

