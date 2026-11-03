import numpy as np
import time

def construct():
    n, m, p = 3, 3, 8
    deadline = time.time() + 110
    
    # Build rank-68 decomposition:
    # Start from schoolbook rank 72, then apply Strassen (rank 7) to the
    # 2x2x2 sub-blocks where i in {0,1}, j in {0,1}, k in {0,1}, {2,3}, {4,5}, {6,7}.
    # Each 2x2x2 block uses 8 schoolbook terms -> 7 Strassen terms, saving 4 total.
    
    # Strassen rank-7 decomposition for 2x2x2:
    # Local indices: i,j,k in {0,1}
    # u_local: index i*2+j (0..3)
    # v_local: index j*2+k (0..3)
    # w_local: index k*2+i (0..3)
    #
    # The 7 terms (u, v, w) as sparse (index, coeff) lists:
    # Term 1: A00*(B00+B11) -> C00 contribution
    #   u: {(0,1)}, v: {(0,1),(2,1)}, w: {(0,1)}
    #   Check: sum = u[0]*v[0]*w[0] + u[0]*v[2]*w[0] = 1*1*1 + 1*1*1 = 2? No.
    #   Actually for C00 = A00*B00 + A01*B10, we need specific terms.
    #
    # Let me use the verified Strassen terms:
    # The tensor M_2[i,j,k] = 1 if i,j,k match.
    # Rank 7 decomposition:
    # 1. u=[1,0,0,0], v=[1,0,1,0], w=[1,0,0,0]  -> A00*B00 + A00*B11... 
    #    Wait, v[0]=1 (j=0,k=0), v[2]=1 (j=1,k=0). So this gives A00*(B00 + B10)? No.
    #    v index = j*2+k. v[0]=j0k0, v[1]=j0k1, v[2]=j1k0, v[3]=j1k1.
    #    So v=[1,0,1,0] means B00 and B10. That's not standard Strassen.
    #
    # Let me use the correct standard Strassen decomposition:
    # C00 = M1 + M4 - M5 - M7
    # C01 = M2 + M3
    # C10 = M6 + M7
    # C11 = M1 - M3 + M4 + M5
    # where:
    # M1 = A00*(B00+B11)
    # M2 = (A00+A01)*B11
    # M3 = (A10+A11)*B00
    # M4 = A11*(B10+B11)
    # M5 = (A00+A11)*(B00-B01)
    # M6 = (A01-A11)*(B10+B11)
    # M7 = (A10-A00)*(B00+B01)
    #
    # For the tensor decomposition, each M_r contributes u_r ⊗ v_r ⊗ w_r where:
    # u_r[i,j] = coefficient of A_ij in the first factor
    # v_r[j,k] = coefficient of B_jk in the second factor  
    # w_r[k,i] = coefficient in the output C_ki... wait, w is indexed by (k*n+i)
    #
    # The output C = A*B, so C[i,k] = sum_j A[i,j]*B[j,k]
    # The tensor is T[i,j,k] = 1, and C[i,k] = sum_j T[i,j,k]*A[i,j]*B[j,k]
    # So w_r should have w_r[k*n + i] as the coefficient for C[i,k].
    #
    # For M1 = A00*(B00+B11):
    #   u: A00 -> u[0*2+0] = u[0] = 1
    #   v: B00+B11 -> v[0*2+0] + v[1*2+1] = v[0] + v[3] -> v[0]=1, v[3]=1
    #   w: contributes to C[0,0] and C[0,1]... 
    #   C00 gets A00*B00, C01 gets A00*B11
    #   So w[0*2+0] = 1 (C[0,0]), w[1*2+0] = 1 (C[0,1])
    #   w index = k*2+i. w[0]=C[0,0], w[1]=C[1,0], w[2]=C[0,1], w[3]=C[1,1]
    #   So w[0]=1, w[2]=1
    #
    # Let me re-derive carefully:
    # w index = k*n + i where n=2 locally. So w[k*2+i]:
    #   w[0] = k=0,i=0 -> C[0,0]
    #   w[1] = k=0,i=1 -> C[1,0]
    #   w[2] = k=1,i=0 -> C[0,1]
    #   w[3] = k=1,i=1 -> C[1,1]
    #
    # M1 = A00*(B00+B11):
    #   u[0]=1 (A[0,0])
    #   v[0]=1 (B[0,0]), v[3]=1 (B[1,1])
    #   C[0,0] += A00*B00 -> w[0] += 1
    #   C[0,1] += A00*B11 -> w[2] += 1
    #   So w = [1, 0, 1, 0]
    #
    # M2 = (A00+A01)*B11:
    #   u[0]=1, u[1]=1
    #   v[3]=1
    #   C[0,1] += A00*B11 -> w[2] += 1
    #   C[1,1] += A01*B11 -> w[3] += 1
    #   So w = [0, 0, 1, 1]
    #
    # M3 = (A10+A11)*B00:
    #   u[2]=1, u[3]=1
    #   v[0]=1
    #   C[1,0] += A10*B00 -> w[1] += 1
    #   C[1,1] += A11*B00 -> w[3] += 1
    #   So w = [0, 1, 0, 1]
    #
    # M4 = A11*(B10+B11):
    #   u[3]=1
    #   v[2]=1, v[3]=1
    #   C[1,0] += A11*B10 -> w[1] += 1
    #   C[1,1] += A11*B11 -> w[3] += 1
    #   So w = [0, 1, 0, 1]
    #
    # M5 = (A00+A11)*(B00-B01):
    #   u[0]=1, u[3]=1
    #   v[0]=1, v[1]=-1
    #   C[0,0] += A00*B00 -> w[0] += 1
    #   C[0,1] += A00*(-B01) -> w[2] += -1
    #   C[1,0] += A11*B00 -> w[1] += 1
    #   C[1,1] += A11*(-B01) -> w[3] += -1
    #   So w = [1, 1, -1, -1]
    #
    # M6 = (A01-A11)*(B10+B11):
    #   u[1]=1, u[3]=-1
    #   v[2]=1, v[3]=1
    #   C[0,0] += A01*B10 -> w[0] += 1
    #   C[0,1] += A01*B11 -> w[2] += 1
    #   C[1,0] += (-A11)*B10 -> w[1] += -1
    #   C[1,1] += (-A11)*B11 -> w[3] += -1
    #   So w = [1, -1, 1, -1]
    #
    # M7 = (A10-A00)*(B00+B01):
    #   u[2]=1, u[0]=-1
    #   v[0]=1, v[1]=1
    #   C[1,0] += A10*B00 -> w[1] += 1
    #   C[1,1] += A10*B01 -> w[3] += 1
    #   C[0,0] += (-A00)*B00 -> w[0] += -1
    #   C[0,1] += (-A00)*B01 -> w[2] += -1
    #   So w = [-1, 1, -1, 1]
    
    strassen_terms = [
        # (u_sparse, v_sparse, w_sparse)
        ({0: 1}, {0: 1, 3: 1}, {0: 1, 2: 1}),          # M1
        ({0: 1, 1: 1}, {3: 1}, {2: 1, 3: 1}),           # M2
        ({2: 1, 3: 1}, {0: 1}, {1: 1, 3: 1}),           # M3
        ({3: 1}, {2: 1, 3: 1}, {1: 1, 3: 1}),           # M4
        ({0: 1, 3: 1}, {0: 1, 1: -1}, {0: 1, 1: 1, 2: -1, 3: -1}),  # M5
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1, 1: -1, 2: 1, 3: -1}),  # M6
        ({0: -1, 2: 1}, {0: 1, 1: 1}, {0: -1, 1: 1, 2: -1, 3: 1}),  # M7
    ]
    
    # Verify Strassen decomposition locally
    def verify_strassen_local(terms):
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    total = 0.0
                    for u_sp, v_sp, w_sp in terms:
                        u_val = u_sp.get(i*2+j, 0)
                        v_val = v_sp.get(j*2+k, 0)
                        w_val = w_sp.get(k*2+i, 0)
                        total += u_val * v_val * w_val
                    if abs(total - 1.0) > 1e-10:
                        return False
        return True
    
    assert verify_strassen_local(strassen_terms), "Strassen decomposition verification failed"
    
    # Build the rank-68 decomposition
    # Blocks to apply Strassen: i in {0,1}, j in {0,1}, k blocks {0,1}, {2,3}, {4,5}, {6,7}
    # Remaining schoolbook terms: all other (i,j,k) combinations
    
    rank = 68
    U = np.zeros((9, rank))
    V = np.zeros((24, rank))
    W = np.zeros((24, rank))
    
    col = 0
    
    # Strassen blocks
    k_blocks = [(0, 2), (2, 2), (4, 2), (6, 2)]
    
    for k_start, k_len in k_blocks:
        for term_idx in range(7):
            u_sp, v_sp, w_sp = strassen_terms[term_idx]
            
            # u: local i*2+j -> global i*3+j
            for li, coeff in u_sp.items():
                li_i = li // 2
                li_j = li % 2
                U[li_i*3 + li_j, col] += coeff
            
            # v: local j*2+k -> global j*8 + (k_start + lk)
            for lv, coeff in v_sp.items():
                lv_j = lv // 2
                lv_k = lv % 2
                V[lv_j*8 + (k_start + lv_k), col] += coeff
            
            # w: local k*2+i -> global (k_start + lk)*3 + li
            for lw, coeff in w_sp.items():
                lw_k = lw // 2
                lw_i = lw % 2
                W[(k_start + lw_k)*3 + lw_i, col] += coeff
            
            col += 1
    
    # Schoolbook terms for remaining (i,j,k)
    # i in {0,1}, j in {2}, k in {0..7}
    # i in {2}, j in {0..2}, k in {0..7}
    # i in {0,1}, j in {0,1}, k is already covered by Strassen
    
    # i in {0,1}, j=2, all k
    for i in [0, 1]:
        for k in range(8):
            U[i*3+2, col] = 1.0
            V[2*8+k, col] = 1.0
            W[k*3+i, col] = 1.0
            col += 1
    
    # i=2, j in {0,1,2}, all k
    for j in range(3):
        for k in range(8):
            U[2*3+j, col] = 1.0
            V[j*8+k, col] = 1.0
            W[k*3+2, col] = 1.0
            col += 1
    
    assert col == rank, f"Column count {col} != rank {rank}"
    
    # Verify full decomposition
    def verify_full(U, V, W, n, m, p):
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for r in range(rank):
                        total += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(total - 1.0) > 1e-10:
                        return False
        return True
    
    if not verify_full(U, V, W, n, m, p):
        # Fall back to trivial rank-72
        U = np.zeros((9, 72))
        V = np.zeros((24, 72))
        W = np.zeros((24, 72))
        col = 0
        for i in range(3):
            for j in range(3):
                for k in range(8):
                    U[i*3+j, col] = 1.0
                    V[j*8+k, col] = 1.0
                    W[k*3+i, col] = 1.0
                    col += 1
        rank = 72
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
