import numpy as np
import time

DIMENSIONS = [4, 4, 6]

def _strassen_2x2():
    """
    Returns (U, V, W) for 2x2 * 2x2 -> 2x2 multiplication.
    U: (4, 7), V: (4, 7), W: (4, 7)
    Indexing:
    U row: i*2 + j (i,j in 0..1)
    V row: j*2 + k (j,k in 0..1)
    W row: k*2 + i (k,i in 0..1)
    """
    U = np.zeros((4, 7), dtype=int)
    V = np.zeros((4, 7), dtype=int)
    W = np.zeros((4, 7), dtype=int)
    
    # M1 = (A00 + A11) * (B00 + B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1 # C00
    W[3, 0] = 1 # C11
    
    # M2 = (A10 + A11) * B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1 # C10
    W[3, 1] = -1 # C11
    
    # M3 = A00 * (B01 - B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1 # C01
    W[3, 2] = 1 # C11
    
    # M4 = A11 * (B10 - B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1 # C00
    W[1, 3] = 1 # C10
    
    # M5 = (A00 + A01) * B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1 # C00
    W[2, 4] = 1 # C01
    
    # M6 = (A10 - A00) * (B00 + B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1 # C11
    
    # M7 = (A01 - A11) * (B10 + B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1 # C00
    
    return U, V, W

def _outer_hybrid():
    """
    Returns (U, V, W) for 2x2 * 2x3 -> 2x3 multiplication.
    U: (4, 11), V: (6, 11), W: (6, 11)
    Indexing:
    U row: i*2 + j (i,j in 0..1)  (A is 2x2 blocks)
    V row: j*3 + k (j in 0..1, k in 0..2) (B is 2x3 blocks)
    W row: k*2 + i (k in 0..2, i in 0..1) (C is 2x3 blocks, trace convention W[k*n+i] with n=2 for outer? 
                                            Wait, outer C is 2x3. Trace convention for outer is W_outer[k_out * 2 + i_out].)
    
    Structure:
    Cols 0-6: Strassen for B columns 0,1 (k_out in {0,1}).
    Cols 7-10: Schoolbook for B column 2 (k_out = 2).
    """
    R_out = 11
    U = np.zeros((4, R_out), dtype=int)
    V = np.zeros((6, R_out), dtype=int)
    W = np.zeros((6, R_out), dtype=int)
    
    # --- Part 1: Strassen for k_out in {0, 1} ---
    # We reuse the Strassen coefficients but map the V and W indices appropriately.
    # V index for outer: j*3 + k. j in {0,1}, k in {0,1,2}.
    # For Strassen part, k is 0 or 1.
    # W index for outer: k*2 + i. k in {0,1,2}, i in {0,1}.
    # For Strassen part, k is 0 or 1.
    
    # Copy Strassen U to U[0:4, 0:7]
    U_strassen = U[:, 0:7]
    # We need to fill V and W with correct indices.
    
    # M1: A(0,0)+A(1,1) * B(0,0)+B(1,1) -> C(0,0)+C(1,1)
    # U: (0,0), (1,1) -> Rows 0, 3
    U[0, 0] = 1; U[3, 0] = 1
    # V: (0,0), (1,1) -> Rows 0*3+0=0, 1*3+1=4
    V[0, 0] = 1; V[4, 0] = 1
    # W: (0,0), (1,1) -> Rows 0*2+0=0, 1*2+1=3
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2: (A(1,0)+A(1,1)) * B(0,0) -> C(1,0)-C(1,1)
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1 # C(1,0) -> k=0, i=1 -> 0*2+1=1
    W[3, 1] = -1 # C(1,1) -> k=1, i=1 -> 1*2+1=3
    
    # M3: A(0,0) * (B(0,1)-B(1,1)) -> C(0,1)+C(1,1)
    U[0, 2] = 1
    V[1, 2] = 1 # B(0,1) -> 0*3+1=1
    V[4, 2] = -1 # B(1,1) -> 1*3+1=4
    W[2, 2] = 1 # C(0,1) -> k=1, i=0 -> 1*2+0=2
    W[3, 2] = 1 # C(1,1) -> k=1, i=1 -> 1*2+1=3
    
    # M4: A(1,1) * (B(1,0)-B(0,0)) -> C(0,0)+C(1,0)
    U[3, 3] = 1
    V[3, 3] = 1 # B(1,0) -> 1*3+0=3
    V[0, 3] = -1 # B(0,0) -> 0*3+0=0
    W[0, 3] = 1 # C(0,0) -> k=0, i=0 -> 0
    W[1, 3] = 1 # C(1,0) -> k=0, i=1 -> 1
    
    # M5: (A(0,0)+A(0,1)) * B(1,1) -> -C(0,0)+C(0,1)
    U[0, 4] = 1; U[1, 4] = 1
    V[4, 4] = 1 # B(1,1) -> 1*3+1=4
    W[0, 4] = -1 # C(0,0)
    W[2, 4] = 1 # C(0,1)
    
    # M6: (A(1,0)-A(0,0)) * (B(0,0)+B(0,1)) -> C(1,1)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1 # B(0,0)
    V[1, 5] = 1 # B(0,1)
    W[3, 5] = 1 # C(1,1)
    
    # M7: (A(0,1)-A(1,1)) * (B(1,0)+B(1,1)) -> C(0,0)
    U[1, 6] = 1; U[3, 6] = -1
    V[3, 6] = 1 # B(1,0)
    V[4, 6] = 1 # B(1,1)
    W[0, 6] = 1 # C(0,0)
    
    # --- Part 2: Schoolbook for k_out = 2 ---
    # Terms: A(i,j) * B(j,2) -> C(i,2)
    # i,j in {0,1}
    # 4 terms.
    
    # Term 7: i=0, j=0
    # A(0,0) * B(0,2) -> C(0,2)
    U[0, 7] = 1
    V[2, 7] = 1 # B(0,2) -> 0*3+2=2
    W[4, 7] = 1 # C(0,2) -> k=2, i=0 -> 2*2+0=4
    
    # Term 8: i=0, j=1
    # A(0,1) * B(1,2) -> C(0,2)
    U[1, 8] = 1
    V[5, 8] = 1 # B(1,2) -> 1*3+2=5
    W[4, 8] = 1 # C(0,2)
    
    # Term 9: i=1, j=0
    # A(1,0) * B(0,2) -> C(1,2)
    U[2, 9] = 1
    V[2, 9] = 1
    W[5, 9] = 1 # C(1,2) -> k=2, i=1 -> 2*2+1=5
    
    # Term 10: i=1, j=1
    # A(1,1) * B(1,2) -> C(1,2)
    U[3, 10] = 1
    V[5, 10] = 1
    W[5, 10] = 1
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS # 4, 4, 6
    R_in = 7
    R_out = 11
    R_total = R_in * R_out
    
    U_in, V_in, W_in = _strassen_2x2()
    U_out, V_out, W_out = _outer_hybrid()
    
    U = np.zeros((n*m, R_total), dtype=int)
    V = np.zeros((m*p, R_total), dtype=int)
    W = np.zeros((p*n, R_total), dtype=int)
    
    for r_in in range(R_in):
        for r_out in range(R_out):
            r = r_in * R_out + r_out
            
            # Update U
            # U_in index: i_in * 2 + j_in
            # U_out index: i_out * 2 + j_out
            # Final U index: (2*i_out + i_in) * 4 + (2*j_out + j_in)
            for i_out in range(2):
                for j_out in range(2):
                    val_out = U_out[i_out*2 + j_out, r_out]
                    if val_out == 0: continue
                    for i_in in range(2):
                        for j_in in range(2):
                            val_in = U_in[i_in*2 + j_in, r_in]
                            if val_in == 0: continue
                            u_idx = (2*i_out + i_in) * 4 + (2*j_out + j_in)
                            U[u_idx, r] += val_out * val_in
            
            # Update V
            # V_in index: j_in * 2 + k_in
            # V_out index: j_out * 3 + k_out
            # Final V index: (2*j_out + j_in) * 6 + (2*k_out + k_in)
            for j_out in range(2):
                for k_out in range(3):
                    val_out = V_out[j_out*3 + k_out, r_out]
                    if val_out == 0: continue
                    for j_in in range(2):
                        for k_in in range(2):
                            val_in = V_in[j_in*2 + k_in, r_in]
                            if val_in == 0: continue
                            v_idx = (2*j_out + j_in) * 6 + (2*k_out + k_in)
                            V[v_idx, r] += val_out * val_in
            
            # Update W
            # W_in index: k_in * 2 + i_in
            # W_out index: k_out * 2 + i_out
            # Final W index: (2*k_out + k_in) * 4 + (2*i_out + i_in)
            for k_out in range(3):
                for i_out in range(2):
                    val_out = W_out[k_out*2 + i_out, r_out]
                    if val_out == 0: continue
                    for k_in in range(2):
                        for i_in in range(2):
                            val_in = W_in[k_in*2 + i_in, r_in]
                            if val_in == 0: continue
                            w_idx = (2*k_out + k_in) * 4 + (2*i_out + i_in)
                            W[w_idx, r] += val_out * val_in

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    # Simple verification
    res = construct()
    U = np.array(res['U'])
    V = np.array(res['V'])
    W = np.array(res['W'])
    
    n, m, p = 4, 4, 6
    np.random.seed(0)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    
    C_expected = A @ B
    
    C_decomp = np.zeros((n, p))
    rank = U.shape[1]
    for r in range(rank):
        # T[i,j,k] = U[i,j,r] V[j,k,r] W[k,i,r]
        # C[i,k] = sum_j A[i,j] B[j,k] T[i,j,k]
        # = sum_j sum_r A[i,j] B[j,k] U[i,j,r] V[j,k,r] W[k,i,r]
        # We can compute this more efficiently:
        # Let A_r[i,j] = A[i,j] * U[i,j,r]
        # Let B_r[j,k] = B[j,k] * V[j,k,r]
        # Then C_decomp += W[:,r] * (A_r @ B_r)? No.
        # C_decomp[i,k] += W[k*n+i, r] * sum_j A_r[i,j] B_r[j,k]
        
        A_r = A * U.reshape(n, m)
        B_r = B * V.reshape(m, p)
        M_r = A_r @ B_r
        # M_r[i,k] = sum_j A[i,j]U[i,j,r] B[j,k]V[j,k,r]
        # C_decomp[i,k] += W[k*n+i, r] * M_r[i,k]
        C_decomp += W.reshape(p, n).T * M_r # W.reshape(p,n) has [k,i]. W[k*n+i] is W.reshape(p,n)[k,i].
        # Wait, W.reshape(p,n)[k,i] corresponds to W[k*n+i] in flat index?
        # Flat index k*n+i. Row k, Col i. Yes.
        
    if not np.allclose(C_expected, C_decomp):
        print("Verification FAILED")
    else:
        print(f"Verification PASSED. Rank: {rank}")
