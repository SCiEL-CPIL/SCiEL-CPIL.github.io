import numpy as np
import time

DIMENSIONS = [4, 4, 5]

def _get_strassen_2x2():
    """
    Returns U, V, W for 2x2x2 Strassen decomposition.
    Shapes: (4, 7) each.
    U[i*m+j, c], V[j*p+k, c], W[k*n+i, c] for n=m=p=2.
    """
    U = np.zeros((4, 7), dtype=np.float64)
    V = np.zeros((4, 7), dtype=np.float64)
    W = np.zeros((4, 7), dtype=np.float64)
    
    # Term 0: (A00+A11)(B00+B11)
    # U: A00(0,0)=1, A11(1,1)=1 -> idx 0, 3
    # V: B00(0,0)=1, B11(1,1)=1 -> idx 0, 3
    # W: C00(0,0)=1, C11(1,1)=1 -> idx 0, 3
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # Term 1: (A10+A11)B00
    # U: A10(1,0)=1, A11(1,1)=1 -> idx 2, 3
    # V: B00(0,0)=1 -> idx 0
    # W: C10(1,0)=1, C11(1,1)=-1 -> idx 2, 3
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # Term 2: A00(B01-B11)
    # U: A00(0,0)=1 -> idx 0
    # V: B01(0,1)=1, B11(1,1)=-1 -> idx 1, 3
    # W: C01(0,1)=1, C11(1,1)=1 -> idx 1, 3
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # Term 3: A11(B10-B00)
    # U: A11(1,1)=1 -> idx 3
    # V: B10(1,0)=1, B00(0,0)=-1 -> idx 2, 0
    # W: C10(1,0)=1, C00(0,0)=1 -> idx 2, 0
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[2, 3] = 1; W[0, 3] = 1
    
    # Term 4: (A00+A01)B11
    # U: A00(0,0)=1, A01(0,1)=1 -> idx 0, 1
    # V: B11(1,1)=1 -> idx 3
    # W: C01(0,1)=-1, C11(1,1)=1 -> idx 1, 3
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[1, 4] = -1; W[3, 4] = 1
    
    # Term 5: (A10-A00)(B00+B01)
    # U: A10(1,0)=1, A00(0,0)=-1 -> idx 2, 0
    # V: B00(0,0)=1, B01(0,1)=1 -> idx 0, 1
    # W: C11(1,1)=1 -> idx 3
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # Term 6: (A01-A11)(B10+B11)
    # U: A01(0,1)=1, A11(1,1)=-1 -> idx 1, 3
    # V: B10(1,0)=1, B11(1,1)=1 -> idx 2, 3
    # W: C00(0,0)=1 -> idx 0
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    # n=4, m=4, p=5
    
    # Rank 49 for 4x4x4 part (k=0..3)
    # Rank 16 for 4x4x1 part (k=4)
    total_rank = 49 + 16
    
    U = np.zeros((n * m, total_rank), dtype=np.float64)
    V = np.zeros((m * p, total_rank), dtype=np.float64)
    W = np.zeros((p * n, total_rank), dtype=np.float64)
    
    # Get 2x2 Strassen factors
    S_U, S_V, S_W = _get_strassen_2x2()
    
    # Outer Strassen coefficients
    # L_t = sum c_{t, IJ} A_{IJ}
    # R_t = sum d_{t, IJ} B_{IJ}
    # C_{IJ} += e_{t, IJ} M_t
    
    # c coefficients for A blocks
    # t=0: A00+A11 -> c[0,0]=1, c[1,1]=1
    # t=1: A10+A11 -> c[1,0]=1, c[1,1]=1
    # t=2: A00 -> c[0,0]=1
    # t=3: A11 -> c[1,1]=1
    # t=4: A00+A01 -> c[0,0]=1, c[0,1]=1
    # t=5: A10-A00 -> c[1,0]=1, c[0,0]=-1
    # t=6: A01-A11 -> c[0,1]=1, c[1,1]=-1
    
    c_coeffs = np.zeros((7, 2, 2), dtype=np.float64)
    c_coeffs[0, 0, 0] = 1; c_coeffs[0, 1, 1] = 1
    c_coeffs[1, 1, 0] = 1; c_coeffs[1, 1, 1] = 1
    c_coeffs[2, 0, 0] = 1
    c_coeffs[3, 1, 1] = 1
    c_coeffs[4, 0, 0] = 1; c_coeffs[4, 0, 1] = 1
    c_coeffs[5, 1, 0] = 1; c_coeffs[5, 0, 0] = -1
    c_coeffs[6, 0, 1] = 1; c_coeffs[6, 1, 1] = -1
    
    # d coefficients for B blocks
    # t=0: B00+B11 -> d[0,0]=1, d[1,1]=1
    # t=1: B00 -> d[0,0]=1
    # t=2: B01-B11 -> d[0,1]=1, d[1,1]=-1
    # t=3: B10-B00 -> d[1,0]=1, d[0,0]=-1
    # t=4: B11 -> d[1,1]=1
    # t=5: B00+B01 -> d[0,0]=1, d[0,1]=1
    # t=6: B10+B11 -> d[1,0]=1, d[1,1]=1
    
    d_coeffs = np.zeros((7, 2, 2), dtype=np.float64)
    d_coeffs[0, 0, 0] = 1; d_coeffs[0, 1, 1] = 1
    d_coeffs[1, 0, 0] = 1
    d_coeffs[2, 0, 1] = 1; d_coeffs[2, 1, 1] = -1
    d_coeffs[3, 1, 0] = 1; d_coeffs[3, 0, 0] = -1
    d_coeffs[4, 1, 1] = 1
    d_coeffs[5, 0, 0] = 1; d_coeffs[5, 0, 1] = 1
    d_coeffs[6, 1, 0] = 1; d_coeffs[6, 1, 1] = 1
    
    # e coefficients for C blocks
    # C00 = M1 - M4 + M6 + M7 -> e[1,0,0]=1, e[4,0,0]=-1, e[6,0,0]=1, e[7,0,0]=1? No, indices 0..6
    # t=0 (M1): C00+=1, C11+=1
    # t=1 (M2): C10+=1
    # t=2 (M3): C01+=1
    # t=3 (M4): C10+=1, C00-=1
    # t=4 (M5): C01+=1, C11+=1
    # t=5 (M6): C00+=1, C11+=1? No.
    # Let's check Strassen reconstruction again.
    # C00 = M1 - M4 + M6 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M5 + M7
    
    e_coeffs = np.zeros((7, 2, 2), dtype=np.float64)
    # M1 (t=0)
    e_coeffs[0, 0, 0] = 1
    e_coeffs[0, 1, 1] = 1
    # M2 (t=1)
    e_coeffs[1, 1, 0] = 1
    # M3 (t=2)
    e_coeffs[2, 0, 1] = 1
    # M4 (t=3)
    e_coeffs[3, 1, 0] = 1
    e_coeffs[3, 0, 0] = -1
    # M5 (t=4)
    e_coeffs[4, 0, 1] = 1
    e_coeffs[4, 1, 1] = 1
    # M6 (t=5)
    e_coeffs[5, 0, 0] = 1
    e_coeffs[5, 1, 1] = 1
    # M7 (t=6)
    e_coeffs[6, 0, 0] = 1
    e_coeffs[6, 1, 1] = 1
    
    # Fill U, V, W for 4x4x4 part (columns 0..48)
    for t in range(7):
        for s in range(7):
            col = t * 7 + s
            
            # U factor
            for i in range(4):
                for j in range(4):
                    I = i // 2
                    J = j // 2
                    li = i % 2
                    lj = j % 2
                    coeff = c_coeffs[t, I, J] * S_U[li * 2 + lj, s]
                    if coeff != 0:
                        U[i * 4 + j, col] = coeff
                        
            # V factor
            for j in range(4):
                for k in range(4): # k=0..3
                    I = j // 2
                    J = k // 2
                    li = j % 2
                    lj = k % 2
                    coeff = d_coeffs[t, I, J] * S_V[li * 2 + lj, s]
                    if coeff != 0:
                        V[j * 5 + k, col] = coeff
                        
            # W factor
            for i in range(4):
                for k in range(4): # k=0..3
                    I = i // 2
                    J = k // 2
                    li = i % 2
                    lj = k % 2
                    coeff = e_coeffs[t, I, J] * S_W[li * 2 + lj, s]
                    if coeff != 0:
                        W[k * 4 + i, col] = coeff
                        
    # Fill U, V, W for 4x4x1 part (columns 49..64)
    # k=4
    for i in range(4):
        for j in range(4):
            col = 49 + (j * 4 + i) # 16 terms
            U[i * 4 + j, col] = 1
            V[j * 5 + 4, col] = 1
            W[4 * 4 + i, col] = 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    # Quick verification
    res = construct()
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    
    n, m, p = 4, 4, 5
    rank = U.shape[1]
    
    # Verify
    T = np.zeros((n, m, p), dtype=np.float64)
    for c in range(rank):
        u_c = U[:, c].reshape(n, m)
        v_c = V[:, c].reshape(m, p)
        w_c = W[:, c].reshape(p, n)
        for i in range(n):
            for j in range(m):
                T[i, j, :] += u_c[i, j] * v_c[j, :] * w_c[:, i]
                
    # Check if T is all ones
    if np.allclose(T, 1.0, atol=1e-9):
        print(f"Success! Rank {rank}")
    else:
        print("Verification failed")
        print(T)

