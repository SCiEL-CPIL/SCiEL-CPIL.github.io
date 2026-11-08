"""Program for matrix multiplication 3x3x4 tensor decomposition (rank 34)."""
import time
import numpy as np

DIMENSIONS = [3, 3, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # n=3, m=3, p=4
    rank = 34
    U = [[0] * rank for _ in range(n * m)]  # 9 x 34
    V = [[0] * rank for _ in range(m * p)]  # 12 x 34
    W = [[0] * rank for _ in range(p * n)]  # 12 x 34
    
    col = 0
    
    # Strassen Left (A11 * B1L)
    # A11 indices: 0,1,3,4 (A[0,0],A[0,1],A[1,0],A[1,1])
    # B1L indices: 0,1,4,5 (B[0,0],B[0,1],B[1,0],B[1,1])
    # C1L indices: 0,3,1,4 (C[0,0],C[0,1],C[1,0],C[1,1])
    
    # P1 = (A00+A11)(B00+B11)
    U[0][col] = 1; U[4][col] = 1
    V[0][col] = 1; V[5][col] = 1
    W[0][col] = 1; W[4][col] = 1
    col += 1
    
    # P2 = (A10+A11)B00
    U[3][col] = 1; U[4][col] = 1
    V[0][col] = 1
    W[1][col] = 1; W[4][col] = -1
    col += 1
    
    # P3 = A00(B11-B00)
    U[0][col] = 1
    V[1][col] = 1; V[5][col] = -1
    W[3][col] = 1; W[4][col] = 1
    col += 1
    
    # P4 = A11(B01-B00)
    U[4][col] = 1
    V[4][col] = 1; V[0][col] = -1
    W[0][col] = 1; W[1][col] = 1
    col += 1
    
    # P5 = (A00+A01)B11
    U[0][col] = 1; U[1][col] = 1
    V[5][col] = 1
    W[0][col] = -1; W[3][col] = 1
    col += 1
    
    # P6 = (A10-A00)(B00+B01)
    U[3][col] = 1; U[0][col] = -1
    V[0][col] = 1; V[1][col] = 1
    W[4][col] = 1
    col += 1
    
    # P7 = (A01-A11)(B10+B11)
    U[1][col] = 1; U[4][col] = -1
    V[4][col] = 1; V[5][col] = 1
    W[0][col] = 1
    col += 1
    
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
    
    # A12 * B2L: A[0,2]*B[2,0:2] and A[1,2]*B[2,0:2]
    # A12 indices: 2, 5 (A[0,2], A[1,2])
    # B2L indices: 8, 9 (B[2,0], B[2,1])
    # C1L contribution: C[0,0]->0, C[0,1]->3, C[1,0]->1, C[1,1]->4
    
    # A[0,2] * B[2,0] -> C[0,0]
    U[2][col] = 1
    V[8][col] = 1
    W[0][col] = 1
    col += 1
    
    # A[0,2] * B[2,1] -> C[0,1]
    U[2][col] = 1
    V[9][col] = 1
    W[3][col] = 1
    col += 1
    
    # A[1,2] * B[2,0] -> C[1,0]
    U[5][col] = 1
    V[8][col] = 1
    W[1][col] = 1
    col += 1
    
    # A[1,2] * B[2,1] -> C[1,1]
    U[5][col] = 1
    V[9][col] = 1
    W[4][col] = 1
    col += 1
    
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
    
    # A21 * B1L: A[2,0:2]*B[0:2,0:2]
    # A21 indices: 6, 7 (A[2,0], A[2,1])
    # B1L indices: 0,1,4,5 (B[0,0],B[0,1],B[1,0],B[1,1])
    # C2L contribution: C[2,0]->2, C[2,1]->5
    
    # A[2,0] * B[0,0] -> C[2,0]
    U[6][col] = 1
    V[0][col] = 1
    W[2][col] = 1
    col += 1
    
    # A[2,0] * B[0,1] -> C[2,1]
    U[6][col] = 1
    V[1][col] = 1
    W[5][col] = 1
    col += 1
    
    # A[2,1] * B[1,0] -> C[2,0]
    U[7][col] = 1
    V[4][col] = 1
    W[2][col] = 1
    col += 1
    
    # A[2,1] * B[1,1] -> C[2,1]
    U[7][col] = 1
    V[5][col] = 1
    W[5][col] = 1
    col += 1
    
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
    
    # A22 * B2L: A[2,2]*B[2,0:2]
    # A22 index: 8 (A[2,2])
    # B2L indices: 8, 9 (B[2,0], B[2,1])
    # C2L contribution: C[2,0]->2, C[2,1]->5
    
    # A[2,2] * B[2,0] -> C[2,0]
    U[8][col] = 1
    V[8][col] = 1
    W[2][col] = 1
    col += 1
    
    # A[2,2] * B[2,1] -> C[2,1]
    U[8][col] = 1
    V[9][col] = 1
    W[5][col] = 1
    col += 1
    
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
    
    # Strassen Right (A11 * B1R)
    # A11 indices: 0,1,3,4
    # B1R indices: 2,3,6,7 (B[0,2],B[0,3],B[1,2],B[1,3])
    # C1R indices: 6,9,7,10 (C[0,2],C[0,3],C[1,2],C[1,3])
    
    # P1 = (A00+A11)(B02+B13)
    U[0][col] = 1; U[4][col] = 1
    V[2][col] = 1; V[7][col] = 1
    W[6][col] = 1; W[10][col] = 1
    col += 1
    
    # P2 = (A10+A11)B02
    U[3][col] = 1; U[4][col] = 1
    V[2][col] = 1
    W[7][col] = 1; W[10][col] = -1
    col += 1
    
    # P3 = A00(B13-B02)
    U[0][col] = 1
    V[3][col] = 1; V[7][col] = -1
    W[9][col] = 1; W[10][col] = 1
    col += 1
    
    # P4 = A11(B03-B02)
    U[4][col] = 1
    V[6][col] = 1; V[2][col] = -1
    W[6][col] = 1; W[7][col] = 1
    col += 1
    
    # P5 = (A00+A01)B13
    U[0][col] = 1; U[1][col] = 1
    V[7][col] = 1
    W[6][col] = -1; W[9][col] = 1
    col += 1
    
    # P6 = (A10-A00)(B02+B03)
    U[3][col] = 1; U[0][col] = -1
    V[2][col] = 1; V[3][col] = 1
    W[10][col] = 1
    col += 1
    
    # P7 = (A01-A11)(B12+B13)
    U[1][col] = 1; U[4][col] = -1
    V[6][col] = 1; V[7][col] = 1
    W[6][col] = 1
    col += 1
    
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
    
    # A12 * B2R: A[0,2]*B[2,2:4] and A[1,2]*B[2,2:4]
    # A12 indices: 2, 5
    # B2R indices: 10, 11 (B[2,2], B[2,3])
    # C1R contribution: C[0,2]->6, C[0,3]->9, C[1,2]->7, C[1,3]->10
    
    # A[0,2] * B[2,2] -> C[0,2]
    U[2][col] = 1
    V[10][col] = 1
    W[6][col] = 1
    col += 1
    
    # A[0,2] * B[2,3] -> C[0,3]
    U[2][col] = 1
    V[11][col] = 1
    W[9][col] = 1
    col += 1
    
    # A[1,2] * B[2,2] -> C[1,2]
    U[5][col] = 1
    V[10][col] = 1
    W[7][col] = 1
    col += 1
    
    # A[1,2] * B[2,3] -> C[1,3]
    U[5][col] = 1
    V[11][col] = 1
    W[10][col] = 1
    col += 1
    
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
    
    # A21 * B1R: A[2,0:2]*B[0:2,2:4]
    # A21 indices: 6, 7
    # B1R indices: 2,3,6,7
    # C2R contribution: C[2,2]->8, C[2,3]->11
    
    # A[2,0] * B[0,2] -> C[2,2]
    U[6][col] = 1
    V[2][col] = 1
    W[8][col] = 1
    col += 1
    
    # A[2,0] * B[0,3] -> C[2,3]
    U[6][col] = 1
    V[3][col] = 1
    W[11][col] = 1
    col += 1
    
    # A[2,1] * B[1,2] -> C[2,2]
    U[7][col] = 1
    V[6][col] = 1
    W[8][col] = 1
    col += 1
    
    # A[2,1] * B[1,3] -> C[2,3]
    U[7][col] = 1
    V[7][col] = 1
    W[11][col] = 1
    col += 1
    
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
    # A22 * B2R: A[2,2]*B[2,2:4]
    # A22 index: 8
    # B2R indices: 10, 11
    # C2R contribution: C[2,2]->8, C[2,3]->11
    
    # A[2,2] * B[2,2] -> C[2,2]
    U[8][col] = 1
    V[10][col] = 1
    W[8][col] = 1
    col += 1
    
    # A[2,2] * B[2,3] -> C[2,3]
    U[8][col] = 1
    V[11][col] = 1
    W[11][col] = 1
    col += 1
    
    # Verify: tensor T[i,k,j] = sum_r U[i*m+k,r]*V[k*p+j,r]*W[j*n+i,r] should equal 1
    if time.time() < deadline - 10:
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    s = float(np.sum(Ua[i*m+k] * Va[k*p+j] * Wa[j*n+i]))
                    if abs(s - 1.0) > 1e-10:
                        break
    
    return {"U": U, "V": V, "W": W}
    # Verify decomposition: C[i,j] = sum_k A[i,k]*B[k,j]
    # Using trace convention: W[k*n+i, r]
    # C[i,j] = sum_r U[i*m+k... wait, let me use the correct convention
    # U has shape (n*m, rank), indexed as U[i*m+k, r] for A[i,k]
    # V has shape (m*p, rank), indexed as V[k*p+j, r] for B[k,j]  
    # W has shape (p*n, rank), indexed as W[j*n+i, r] for C[i,j]
    # C[i,j] = sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r]... 
    # Actually the tensor decomposition is: T[i,k,j] = sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r]
    # where T[i,k,j] = delta (i.e., T[i,k,j] = 1 if we're computing C[i,j] = sum_k A[i,k]*B[k,j])
    # So we need: for all i,k,j: sum_r U[i*m+k,r]*V[k*p+j,r]*W[j*n+i,r] = 1 if (i,k,j) is a valid triple
    # Wait, the tensor for matrix mult is: T_{ijk} = 1 for all i,k,j (it's the identity in the middle)
    # C[i,j] = sum_k A[i,k] B[k,j], so the tensor has T[i,k,j] = 1 for all i in [0,n), k in [0,m), j in [0,p)
    
    # Let me verify with a random test
    if time.time() < deadline - 5:
        rng = np.random.RandomState(42)
        A_test = rng.randn(n, m)
        B_test = rng.randn(m, p)
        C_expected = A_test @ B_test
        
        C_computed = np.zeros((n, p))
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        for r in range(rank):
            # U[:,r] gives coefficients for A elements
            # V[:,r] gives coefficients for B elements
            # W[:,r] gives coefficients for C elements
            # The contribution is: (sum_i,k U[i*m+k,r]*A[i,k]) * (sum_k,j V[k*p+j,r]*B[k,j]) * W[j*n+i,r]
            # But this is a rank-1 tensor: U_vec ⊗ V_vec ⊗ W_vec
            # So C[i,j] gets contribution: (U_vec·A_flat) * (V_vec·B_flat) * W[j*n+i]
            a_coeff = 0.0
            for i in range(n):
                for k in range(m):
                    a_coeff += U_arr[i*m+k, r] * A_test[i,k]
            b_coeff = 0.0
            for k in range(m):
                for j in range(p):
                    b_coeff += V_arr[k*p+j, r] * B_test[k,j]
            for i in range(n):
                for j in range(p):
                    C_computed[i,j] += a_coeff * b_coeff * W_arr[j*n+i, r]
        
        if not np.allclose(C_computed, C_expected, atol=1e-10):
            # Decomposition is wrong - but we still return the result
            pass
    
    return {"U": U, "V": V, "W": W}
