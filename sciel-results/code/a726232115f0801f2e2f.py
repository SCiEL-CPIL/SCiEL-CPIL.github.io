import numpy as np
import time

DIMENSIONS = [4, 5, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    n, m, p = DIMENSIONS
    
    # Partitions
    # n=4 -> 2, 2
    # m=5 -> 2, 3
    # p=5 -> 2, 3
    
    I1 = [0, 1]
    I2 = [2, 3]
    J1 = [0, 1]
    J2 = [2, 3, 4]
    K1 = [0, 1]
    K2 = [2, 3, 4]
    
    blocks = [
        (I1, J1, K1), # 2x2x2
        (I1, J1, K2), # 2x2x3
        (I1, J2, K1), # 2x3x2
        (I1, J2, K2), # 2x3x3
        (I2, J1, K1), # 2x2x2
        (I2, J1, K2), # 2x2x3
        (I2, J2, K1), # 2x3x2
        (I2, J2, K2), # 2x3x3
    ]
    
    U_terms = []
    V_terms = []
    W_terms = []
    
    def get_strassen_222():
        # Returns U, V, W for 2x2x2
        # U: 4x7, V: 4x7, W: 4x7
        # Indices: U[i*2+j], V[j*2+k], W[k*2+i]
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # Strassen terms
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # Coefficients for A (U)
        # A11: idx 0, A12: idx 1, A21: idx 2, A22: idx 3
        # B11: idx 0, B12: idx 1, B21: idx 2, B22: idx 3
        # C11: idx 0, C12: idx 1, C21: idx 2, C22: idx 3
        
        # M1
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1 # C11, C22? No, C indices are k*n+i.
        # W index is k*2+i.
        # C11 (i=0,k=0) -> 0. C12 (i=0,k=1) -> 1. C21 (i=1,k=0) -> 2. C22 (i=1,k=1) -> 3.
        
        # Let's use the standard Strassen combination formulas for C.
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # We need to express M_r as U_r * V_r.
        # Then C = sum_r W_r * M_r.
        # This means for each r, we add to U, V the coefficients for M_r.
        # And to W, we add the coefficients for how M_r contributes to C.
        
        # M1: A=(1,0,0,1), B=(1,0,0,1). C contrib: C11+=1, C22+=1.
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1 # C11
        W[3, 0] = 1 # C22
        
        # M2: A=(0,0,1,1), B=(1,0,0,0). C contrib: C21+=1.
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1 # C21
        
        # M3: A=(1,0,0,0), B=(0,1,0,-1). C contrib: C12+=1.
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1 # C12
        
        # M4: A=(0,0,0,1), B=(0,0,1,-1). C contrib: C11+=1, C21+=1.
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1 # C11
        W[2, 3] = 1 # C21
        
        # M5: A=(1,1,0,0), B=(0,0,0,1). C contrib: C11-=1, C12+=1.
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1 # C11
        W[1, 4] = 1 # C12
        
        # M6: A=(0,-1,1,0), B=(1,1,0,0). C contrib: C22+=1.
        U[0, 5] = -1; U[2, 5] = 1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1 # C22
        
        # M7: A=(1,0,0,-1), B=(0,0,1,1). C contrib: C11+=1, C22+=1.
        U[0, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1 # C11
        W[3, 6] = 1 # C22
        
        return U, V, W

    def get_schoolbook(n, m, p):
        U = np.zeros((n*m, n*m*p))
        V = np.zeros((m*p, n*m*p))
        W = np.zeros((p*n, n*m*p))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, c] = 1
                    V[j*p+k, c] = 1
                    W[k*n+i, c] = 1
                    c += 1
        return U, V, W

    def get_decomp(n, m, p):
        if n == 2 and m == 2 and p == 2:
            return get_strassen_222()
        elif n == 2 and m == 2 and p == 1:
            return get_schoolbook(2, 2, 1)
        elif n == 2 and m == 1 and p == 2:
            return get_schoolbook(2, 1, 2)
        elif n == 2 and m == 3 and p == 2:
            # Split m into 2, 1
            U1, V1, W1 = get_decomp(2, 2, 2)
            U2, V2, W2 = get_decomp(2, 1, 2)
            # Map indices
            # U: 2x3 -> 2x2 (cols 0,1) and 2x1 (col 2)
            # V: 3x2 -> 2x2 (rows 0,1) and 1x2 (row 2)
            # W: 2x2 -> same
            U = np.zeros((2*3, U1.shape[1] + U2.shape[1]))
            V = np.zeros((3*2, U1.shape[1] + U2.shape[1]))
            W = np.zeros((2*2, U1.shape[1] + U2.shape[1]))
            
            rank1 = U1.shape[1]
            # Part 1: m=2
            U[0:4, 0:rank1] = U1
            V[0:4, 0:rank1] = V1
            W[0:4, 0:rank1] = W1
            
            # Part 2: m=1 (index 2 in m)
            # U index: i*3+j. j=2.
            # U2 is 2x1 x rank2.
            # U2[i*1+j, c] -> U[i*3+2, rank1+c]
            for i in range(2):
                for c in range(U2.shape[1]):
                    U[i*3+2, rank1+c] = U2[i, c]
            # V index: j*2+k. j=2.
            # V2 is 1x2 x rank2.
            # V2[j*2+k, c] -> V[2*2+k, rank1+c]
            for k in range(2):
                for c in range(V2.shape[1]):
                    V[2*2+k, rank1+c] = V2[k, c]
            W[0:4, rank1:rank1+U2.shape[1]] = W2
            
            return U, V, W
        elif n == 2 and m == 2 and p == 3:
            # Split p into 2, 1
            U1, V1, W1 = get_decomp(2, 2, 2)
            U2, V2, W2 = get_decomp(2, 2, 1)
            rank1 = U1.shape[1]
            U = np.zeros((2*2, rank1 + U2.shape[1]))
            V = np.zeros((2*3, rank1 + U2.shape[1]))
            W = np.zeros((3*2, rank1 + U2.shape[1]))
            
            U[0:4, 0:rank1] = U1
            V[0:4, 0:rank1] = V1
            # W1: k*2+i. k in 0,1.
            for k in range(2):
                for i in range(2):
                    W[k*2+i, 0:rank1] = W1[k*2+i, 0:rank1]
            
            # Part 2: p=1 (index 2 in p)
            # U2 same as U1 shape? No, U2 is 2x2.
            U[0:4, rank1:rank1+U2.shape[1]] = U2
            # V2: j*3+k. k=2.
            for j in range(2):
                for c in range(V2.shape[1]):
                    V[j*3+2, rank1+c] = V2[j*1+0, c] # V2 is 2x1? No, V2 is (m*p) x rank. m=2, p=1.
            # W2: k*2+i. k=2.
            for i in range(2):
                for c in range(W2.shape[1]):
                    W[2*2+i, rank1+c] = W2[0*2+i, c]
            return U, V, W
        elif n == 2 and m == 3 and p == 3:
            # Split p into 2, 1
            U1, V1, W1 = get_decomp(2, 3, 2)
            U2, V2, W2 = get_decomp(2, 3, 1)
            rank1 = U1.shape[1]
            U = np.zeros((2*3, rank1 + U2.shape[1]))
            V = np.zeros((3*3, rank1 + U2.shape[1]))
            W = np.zeros((3*2, rank1 + U2.shape[1]))
            
            U[0:6, 0:rank1] = U1
            V[0:6, 0:rank1] = V1
            for k in range(2):
                for i in range(2):
                    W[k*2+i, 0:rank1] = W1[k*2+i, 0:rank1]
            
            U[0:6, rank1:rank1+U2.shape[1]] = U2
            for j in range(3):
                for c in range(V2.shape[1]):
                    V[j*3+2, rank1+c] = V2[j*1+0, c]
            for i in range(2):
                for c in range(W2.shape[1]):
                    W[2*2+i, rank1+c] = W2[0*2+i, c]
            return U, V, W
        else:
            return get_schoolbook(n, m, p)

    for I_set, J_set, K_set in blocks:
        n_b = len(I_set)
        m_b = len(J_set)
        p_b = len(K_set)
        
        U_b, V_b, W_b = get_decomp(n_b, m_b, p_b)
        
        # Map to global
        # U: i*m+j. i in I_set, j in J_set.
        # V: j*p+k. j in J_set, k in K_set.
        # W: k*n+i. k in K_set, i in I_set.
        
        for r in range(U_b.shape[1]):
            u_col = np.zeros(n*m)
            v_col = np.zeros(m*p)
            w_col = np.zeros(p*n)
            
            for i_local, i_global in enumerate(I_set):
                for j_local, j_global in enumerate(J_set):
                    u_col[i_global*m + j_global] = U_b[i_local*m_b + j_local, r]
            
            for j_local, j_global in enumerate(J_set):
                for k_local, k_global in enumerate(K_set):
                    v_col[j_global*p + k_global] = V_b[j_local*p_b + k_local, r]
            
            for k_local, k_global in enumerate(K_set):
                for i_local, i_global in enumerate(I_set):
                    w_col[k_global*n + i_global] = W_b[k_local*n_b + i_local, r]
            
            U_terms.append(u_col)
            V_terms.append(v_col)
            W_terms.append(w_col)
            
    U = np.array(U_terms).T
    V = np.array(V_terms).T
    W = np.array(W_terms).T
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    res = construct()
    rank = len(res["U"][0])
    print(f"Rank: {rank}")
