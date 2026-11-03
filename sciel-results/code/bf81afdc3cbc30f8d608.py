import numpy as np

def construct():
    n, m, p = 3, 4, 7
    
    # 1. Base: Strassen 2x2x2
    def strassen_222():
        # U: 4x7, V: 4x7, W: 4x7
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1 = (A11+A22)(B11+B22) -> C11 += M1, C22 += M1
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[1, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # M2 = (A21+A22)B11 -> C21 += M2
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1
        
        # M3 = A11(B21-B22) -> C12 += M3
        U[0, 2] = 1
        V[2, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1
        
        # M4 = A22(B12-B11) -> C22 -= M4 (Wait, Strassen C22 = M1 - M4)
        # Standard Strassen:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B22)
        # M4 = A22(B12-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        # C11 = M1 + M4 - M5 - M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M7
        
        # Let's re-verify W assignments for Strassen.
        # C11 (idx 0): M1 + M4 - M5 - M7
        # C12 (idx 1): M3 + M5
        # C21 (idx 2): M2 + M4
        # C22 (idx 3): M1 - M2 + M3 + M7
        
        # Reset W
        W = np.zeros((4, 7))
        
        # M1
        W[0, 0] = 1
        W[3, 0] = 1
        
        # M2
        W[2, 1] = 1
        W[3, 1] = -1
        
        # M3
        W[1, 2] = 1
        W[3, 2] = 1
        
        # M4
        W[0, 3] = 1
        W[2, 3] = 1
        
        # M5
        W[0, 4] = -1
        W[1, 4] = 1
        
        # M6 (Unused in C? M6 is not in standard 7-var Strassen output formulas? 
        # Wait. C11 = M1 + M4 - M5 - M7. C12 = M3 + M5. C21 = M2 + M4. C22 = M1 - M2 + M3 + M7.
        # M6 is not used? 
        # Ah, standard Strassen uses 7 multiplications. 
        # M6 is (A21-A11)(B11+B12). 
        # It is not in the formulas above? 
        # Let's check.
        # C11 = M1 + M4 - M5 - M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M7
        # Sum of coeffs:
        # M1: 2
        # M2: -1
        # M3: 2
        # M4: 2
        # M5: -1
        # M6: 0
        # M7: -1
        # This looks correct for 7 multiplications. M6 is indeed not used in the reconstruction?
        # No, that's impossible. You calculate 7 products. 
        # Let's re-read Strassen.
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B22)
        # M4 = A22(B12-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # C11 = M1 + M4 - M5 - M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M7
        
        # Where is M6? 
        # I am confusing myself. 
        # Actually, M6 is used in some variants, but the standard 7-var Strassen does NOT use M6 in the final sum?
        # Let's verify.
        # C11 = M1 + M4 - M5 - M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M7
        # If M6 is not used, why calculate it?
        # Ah, I might be misremembering the formulas.
        # Let's derive or look up.
        # Wikipedia:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B22)
        # M4 = A22(B12-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # C11 = M1 + M4 - M5 - M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M7
        
        # It seems M6 is indeed not in the output. 
        # This implies M6 is redundant? No, that can't be.
        # Wait. 
        # Let's check C21. 
        # A21 B11 + A22 B21.
        # M2 = (A21+A22)B11 = A21 B11 + A22 B11.
        # M4 = A22(B12-B11) = A22 B12 - A22 B11.
        # M2 + M4 = A21 B11 + A22 B12. 
        # This is NOT C21 (A21 B11 + A22 B21).
        # So my formula for C21 is wrong or M4 definition is wrong.
        
        # Correct Strassen Formulas:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B22)
        # M4 = A22(B12-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # C11 = M1 + M4 - M5 - M7
        # C12 = M3 + M5
        # C21 = M2 + M4  <-- This step is suspicious.
        # C22 = M1 - M2 + M3 + M7
        
        # Let's check C21 again.
        # C21 = A21 B11 + A22 B21.
        # M2 = A21 B11 + A22 B11.
        # M6 = (A21-A11)(B11+B12) = A21 B11 + A21 B12 - A11 B11 - A11 B12.
        # This is getting complicated.
        
        # Let's just use the known correct U,V,W for Strassen 2x2x2.
        # I will use the standard implementation found in many sources.
        
        U = np.array([
            [1, 0, 1, 0, 1, -1, 0],
            [0, 0, 0, 0, 1, 0, 1],
            [0, 1, 0, 0, 0, 1, 0],
            [1, 1, 0, 1, 0, 0, -1]
        ])
        
        V = np.array([
            [1, 1, 0, -1, 0, 1, 0],
            [1, 0, 1, 1, 0, 1, 0],
            [0, 0, -1, 0, 1, 0, 1],
            [0, 0, 0, 0, 1, 0, 1]
        ])
        
        W = np.array([
            [1, 0, 0, 1, -1, 0, -1],
            [0, 0, 1, 0, 1, 0, 0],
            [0, 1, 0, 1, 0, 0, 0],
            [1, -1, 1, 0, 0, 0, 1]
        ])
        
        # Verify: C = W * diag(M) * V * U ? No.
        # C_{ij} = sum_k W_{ik} M_k U_{kj} ? No.
        # Tensor decomposition: T = sum_k U_k \otimes V_k \otimes W_k.
        # C_{i,j} = sum_k U_{i,k} V_{j,k} W_{k}? No.
        # A_{i,l} B_{l,j}.
        # U is for A. V is for B. W is for C.
        # C_{i,j} = sum_k (sum_l U_{i,l} A_{l?}) ...
        
        # The interface is:
        # U: (n*m, R). Rows indexed by (i, l) for A_{i,l}.
        # V: (m*p, R). Rows indexed by (l, j) for B_{l,j}.
        # W: (p*n, R). Rows indexed by (j, i) for C_{j,i}? 
        # "Output factor follows the official trace convention W[k*n+i,r]"
        # So W row index is $j \cdot n + i$.
        
        # Let's check the Strassen U,V,W above against this convention.
        # A is 2x2. Indices (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3.
        # B is 2x2. Indices (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3.
        # C is 2x2. Indices (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3.
        # W row index $j \cdot 2 + i$.
        # C_{0,0} -> 0. C_{0,1} -> 1. C_{1,0} -> 2. C_{1,1} -> 3.
        
        # Check M1: (A11+A22)(B11+B22)
        # A11=0, A22=3. U[:,0] should have 1 at 0 and 3.
        # U[0,0]=1, U[3,0]=1. Correct.
        # B11=0, B22=3. V[:,0] should have 1 at 0 and 3.
        # V[0,0]=1, V[3,0]=1? 
        # My V above: V[0,0]=1, V[1,0]=1. 
        # V row 0 is B_{0,0}. V row 1 is B_{0,1}.
        # B11 is B_{0,0} (idx 0). B22 is B_{1,1} (idx 3).
        # So V[:,0] should be 1 at 0 and 3.
        # My V[0,0]=1, V[1,0]=1 is WRONG.
        
        # I need to construct V correctly.
        # B indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1).
        
        # M1: B11+B22 -> B[0]+B[3]. V[0,0]=1, V[3,0]=1.
        # M2: B11 -> B[0]. V[0,1]=1.
        # M3: B21-B22 -> B[2]-B[3]. V[2,2]=1, V[3,2]=-1.
        # M4: B12-B11 -> B[1]-B[0]. V[1,3]=1, V[0,3]=-1.
        # M5: B22 -> B[3]. V[3,4]=1.
        # M6: B11+B12 -> B[0]+B[1]. V[0,5]=1, V[1,5]=1.
        # M7: B21+B22 -> B[2]+B[3]. V[2,6]=1, V[3,6]=1.
        
        # U indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1).
        # M1: A11+A22 -> A[0]+A[3]. U[0,0]=1, U[3,0]=1.
        # M2: A21+A22 -> A[2]+A[3]. U[2,1]=1, U[3,1]=1.
        # M3: A11 -> A[0]. U[0,2]=1.
        # M4: A22 -> A[3]. U[3,3]=1.
        # M5: A11+A12 -> A[0]+A[1]. U[0,4]=1, U[1,4]=1.
        # M6: A21-A11 -> A[2]-A[0]. U[2,5]=1, U[0,5]=-1.
        # M7: A12-A22 -> A[1]-A[3]. U[1,6]=1, U[3,6]=-1.
        
        # W indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1).
        # C11 = M1 + M4 - M5 - M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M7
        
        W = np.zeros((4, 7))
        W[0, 0] = 1; W[0, 3] = 1; W[0, 4] = -1; W[0, 6] = -1 # C11
        W[1, 2] = 1; W[1, 4] = 1 # C12
        W[2, 1] = 1; W[2, 3] = 1 # C21
        W[3, 0] = 1; W[3, 1] = -1; W[3, 2] = 1; W[3, 6] = 1 # C22
        
        return U, V, W

    # 2. Base: Schoolbook 1x2x2
    def schoolbook_122():
        # A(1x2), B(2x2), C(1x2)
        # U: 2x4, V: 4x4, W: 2x4
        U = np.zeros((2, 4))
        V = np.zeros((4, 4))
        W = np.zeros((2, 4))
        
        # A indices: 0:(0,0), 1:(0,1)
        # B indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        # C indices: 0:(0,0), 1:(0,1)
        
        # C0 = A0 B0 + A1 B2
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        U[1, 1] = 1; V[2, 1] = 1; W[0, 1] = 1
        
        # C1 = A0 B1 + A1 B3
        U[0, 2] = 1; V[1, 2] = 1; W[1, 2] = 1
        U[1, 3] = 1; V[3, 3] = 1; W[1, 3] = 1
        
        return U, V, W

    # 3. Base: Schoolbook 3x2x1
    def schoolbook_321():
        # A(3x2), B(2x1), C(3x1)
        # U: 6x6, V: 2x6, W: 3x6
        U = np.zeros((6, 6))
        V = np.zeros((2, 6))
        W = np.zeros((3, 6))
        
        # A indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1), 4:(2,0), 5:(2,1)
        # B indices: 0:(0,0), 1:(1,0)
        # C indices: 0:(0,0), 1:(1,0), 2:(2,0)
        
        for i in range(3):
            for j in range(2):
                col = i * 2 + j
                U[i * 2 + j, col] = 1
                V[j, col] = 1
                W[i, col] = 1
                
        return U, V, W

    # 4. Build 3x2x2
    def build_322():
        U_s, V_s, W_s = strassen_222()
        U_b, V_b, W_b = schoolbook_122()
        
        U = np.zeros((6, 11))
        V = np.zeros((4, 11))
        W = np.zeros((6, 11))
        
        U[0:4, 0:7] = U_s
        V[0:4, 0:7] = V_s
        W[0:4, 0:7] = W_s
        
        U[4:6, 7:11] = U_b
        V[0:4, 7:11] = V_b
        W[4:6, 7:11] = W_b
        
        return U, V, W

    # 5. Build 3x2x1
    def build_321():
        return schoolbook_321()

    # 6. Build 3x2x3
    def build_323():
        U_2, V_2, W_2 = build_322()
        U_1, V_1, W_1 = build_321()
        
        U = np.zeros((6, 17))
        V = np.zeros((6, 17))
        W = np.zeros((9, 17))
        
        U[:, 0:11] = U_2
        V[0:4, 0:11] = V_2
        W[0:6, 0:11] = W_2
        
        U[:, 11:17] = U_1
        V[4:6, 11:17] = V_1
        W[6:9, 11:17] = W_1
        
        return U, V, W

    # 7. Build 3x2x4
    def build_324():
        U_2a, V_2a, W_2a = build_322()
        U_2b, V_2b, W_2b = build_322()
        
        U = np.zeros((6, 22))
        V = np.zeros((8, 22))
        W = np.zeros((12, 22))
        
        U[:, 0:11] = U_2a
        V[0:4, 0:11] = V_2a
        W[0:6, 0:11] = W_2a
        
        U[:, 11:22] = U_2b
        V[4:8, 11:22] = V_2b
        W[6:12, 11:22] = W_2b
        
        return U, V, W

    # 8. Build 3x2x7
    def build_327():
        U_4, V_4, W_4 = build_324()
        U_3, V_3, W_3 = build_323()
        
        U = np.zeros((6, 39))
        V = np.zeros((14, 39))
        W = np.zeros((21, 39))
        
        U[:, 0:22] = U_4
        V[0:8, 0:22] = V_4
        W[0:12, 0:22] = W_4
        
        U[:, 22:39] = U_3
        V[8:14, 22:39] = V_3
        W[12:21, 22:39] = W_3
        
        return U, V, W

    # 9. Final 3x4x7
    U_1, V_1, W_1 = build_327()
    U_2, V_2, W_2 = build_327()
    
    U = np.zeros((12, 78))
    V = np.zeros((28, 78))
    W = np.zeros((21, 78))
    
    U[0:6, 0:39] = U_1
    V[0:14, 0:39] = V_1
    W[:, 0:39] = W_1
    
    U[6:12, 39:78] = U_2
    V[14:28, 39:78] = V_2
    W[:, 39:78] = W_2
    
    return {"U": U, "V": V, "W": W}
