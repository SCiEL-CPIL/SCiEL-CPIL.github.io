"""Program for matrix multiplication tensor decomposition 4x4 x 4x6."""
import numpy as np
import time

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill
    
    # Schoolbook rank is n*m*p = 96
    best_rank = n * m * p
    best_result = None
    
    def build_schoolbook():
        """Standard schoolbook decomposition with rank n*m*p."""
        rank = n * m * p
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1.0
                    V[j * p + k][col] = 1.0
                    W[k * n + i][col] = 1.0
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    def validate(U, V, W, rank):
        """
        Check if U, V, W form a valid tensor decomposition.
        T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should equal 1.
        Uses numpy einsum for speed.
        """
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        
        # Reshape
        U_reshaped = U_arr.reshape(n, m, rank)  # (i, j, c)
        V_reshaped = V_arr.reshape(m, p, rank)  # (j, k, c)
        W_reshaped = W_arr.reshape(p, n, rank)  # (k, i, c)
        
        # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
        T = np.einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped)
        
        return np.allclose(T, 1.0, atol=1e-10)
    
    def strassen_2x2():
        """Strassen decomposition for 2x2 x 2x2 -> 2x2. Returns U(4,7), V(4,7), W(4,7)."""
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        # A indices: 0=A00, 1=A01, 2=A10, 3=A11
        # B indices: 0=B00, 1=B01, 2=B10, 3=B11
        # C indices: 0=C00, 1=C01, 2=C10, 3=C11
        U[0,0]=1; U[3,0]=1; V[0,0]=1; V[3,0]=1; W[0,0]=1; W[3,0]=1
        U[2,1]=1; U[3,1]=1; V[0,1]=1; W[2,1]=1; W[3,1]=1
        U[0,2]=1; V[1,2]=1; V[3,2]=-1; W[1,2]=1; W[3,2]=1
        U[3,3]=1; V[2,3]=1; V[0,3]=-1; W[0,3]=1; W[2,3]=1
        U[0,4]=1; U[1,4]=1; V[3,4]=1; W[0,4]=-1; W[1,4]=1
        U[2,5]=1; U[0,5]=-1; V[0,5]=1; V[1,5]=1; W[3,5]=1
        U[1,6]=1; U[3,6]=-1; V[2,6]=1; V[3,6]=1; W[0,6]=1
        return U, V, W
    
    def strassen_2x2x3():
        """
        Decomposition for 2x2 x 2x3 -> 2x3 with rank 7.
        A is 2x2 (indices 0-3), B is 2x3 (indices 0-5), C is 2x3 (indices 0-5).
        A: 0=A00, 1=A01, 2=A10, 3=A11
        B: 0=B00, 1=B01, 2=B02, 3=B10, 4=B11, 5=B12
        C: 0=C00, 1=C01, 2=C02, 3=C10, 4=C11, 5=C12
        
        C00 = A00*B00 + A01*B10
        C01 = A00*B01 + A01*B11
        C02 = A00*B02 + A01*B12
        C10 = A10*B00 + A11*B10
        C11 = A10*B01 + A11*B11
        C12 = A10*B02 + A11*B12
        """
        U = np.zeros((4, 7))
        V = np.zeros((6, 7))
        W = np.zeros((6, 7))
        
        # M1 = (A00+A11)(B00+B10) -> C00 += M1, C10 += M1
        U[0,0]=1; U[3,0]=1
        V[0,0]=1; V[3,0]=1
        W[0,0]=1; W[3,0]=1
        
        # M2 = (A10+A11)(B00) -> C10 += M2
        U[2,1]=1; U[3,1]=1
        V[0,1]=1
        W[3,1]=1
        
        # M3 = (A00)(B01-B11) -> C01 += M3, C11 -= M3
        U[0,2]=1
        V[1,2]=1; V[4,2]=-1
        W[1,2]=1; W[4,2]=-1
        
        # M4 = (A11)(B10-B00) -> C00 -= M4, C10 += M4
        U[3,3]=1
        V[3,3]=1; V[0,3]=-1
        W[0,3]=-1; W[3,3]=1
        
        # M5 = (A00+A01)(B11) -> C01 += M5, C11 += M5
        U[0,4]=1; U[1,4]=1
        V[4,4]=1
        W[1,4]=1; W[4,4]=1
        
        # M6 = (A10-A00)(B01+B02) -> C11 += M6, C12 += M6
        U[2,5]=1; U[0,5]=-1
        V[1,5]=1; V[2,5]=1
        W[4,5]=1; W[5,5]=1
        
        # M7 = (A01-A11)(B12) -> C02 += M7, C12 -= M7
        U[1,6]=1; U[3,6]=-1
        V[5,6]=1
        W[2,6]=1; W[5,6]=-1
        
        return U, V, W
    
    def kron_decomposition(U1, V1, W1, U2, V2, W2):
        """
        Kronecker product of two tensor decompositions.
        If T1 has rank r1 with (U1, V1, W1) and T2 has rank r2 with (U2, V2, W2),
        then T1 ⊗ T2 has rank r1*r2.
        
        U1 shape: (n1*m1, r1), V1 shape: (m1*p1, r1), W1 shape: (p1*n1, r1)
        U2 shape: (n2*m2, r2), V2 shape: (m2*p2, r2), W2 shape: (p2*n2, r2)
        
        Result: U shape: (n1*n2 * m1*m2, r1*r2) = ((n1*m1)*(n2*m2), r1*r2)
        """
        r1 = U1.shape[1]
        r2 = U2.shape[1]
        rank = r1 * r2
        
        nm1 = U1.shape[0]  # n1*m1
        mp1 = V1.shape[0]  # m1*p1
        pn1 = W1.shape[0]  # p1*n1
        nm2 = U2.shape[0]  # n2*m2
        mp2 = V2.shape[0]  # m2*p2
        pn2 = W2.shape[0]  # p2*n2
        
        U = np.zeros((nm1 * nm2, rank))
        V = np.zeros((mp1 * mp2, rank))
        W = np.zeros((pn1 * pn2, rank))
        
        for c1 in range(r1):
            for c2 in range(r2):
                c = c1 * r2 + c2
                U[:, c] = np.kron(U1[:, c1], U2[:, c2])
                V[:, c] = np.kron(V1[:, c1], V2[:, c2])
                W[:, c] = np.kron(W1[:, c1], W2[:, c2])
        
        return U, V, W
    
    # Build schoolbook as fallback
    school = build_schoolbook()
    if validate(school["U"], school["V"], school["W"], best_rank):
        best_result = school
        best_rank = n * m * p
    
    # Try Kronecker product: M_{2,2,3} ⊗ M_{2,2,2}
    # M_{2,2,3}: A(2x2), B(2x3), C(2x3)
    # M_{2,2,2}: A(2x2), B(2x2), C(2x2)
    # Result: A(4x4), B(4x6), C(4x6)
    
    U23, V23, W23 = strassen_2x2x3()
    U22, V22, W22 = strassen_2x2()
    
    U_kron, V_kron, W_kron = kron_decomposition(U23, V23, W23, U22, V22, W22)
    
    rank_kron = U_kron.shape[1]
    U_list = U_kron.tolist()
    V_list = V_kron.tolist()
    W_list = W_kron.tolist()
    
    if validate(U_list, V_list, W_list, rank_kron):
        best_rank = rank_kron
        best_result = {"U": U_list, "V": V_list, "W": W_list}
    
    # Try to reduce rank by finding linear dependencies
    # Check if we can remove columns that are linearly dependent
    if time.time() < deadline - 20:
        def try_reduce_rank(U, V, W, rank, max_removals=10):
            """Try to remove linearly dependent columns."""
            U_arr = np.array(U, dtype=np.float64)
            V_arr = np.array(V, dtype=np.float64)
            W_arr = np.array(W, dtype=np.float64)
            
            # For each column, check if removing it still gives valid decomposition
            # This is expensive, so only try a few
            removed = []
            for c in range(rank):
                if time.time() > deadline - 10:
                    break
                if len(removed) >= max_removals:
                    break
                if c in removed:
                    continue
                # Try removing column c
                mask = [i for i in range(rank) if i != c]
                U_new = U_arr[:, mask]
                V_new = V_arr[:, mask]
                W_new = W_arr[:, mask]
                
                U_r = U_new.reshape(n, m, -1)
                V_r = V_new.reshape(m, p, -1)
                W_r = W_new.reshape(p, n, -1)
                T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
                
                if np.allclose(T, 1.0, atol=1e-10):
                    removed.append(c)
            
            if removed:
                mask = [i for i in range(rank) if i not in removed]
                return U_arr[:, mask].tolist(), V_arr[:, mask].tolist(), W_arr[:, mask].tolist(), len(mask)
            return None
    
        if best_result is not None:
            reduced = try_reduce_rank(best_result["U"], best_result["V"], best_result["W"], best_rank)
            if reduced is not None:
                U_r, V_r, W_r, new_rank = reduced
                if validate(U_r, V_r, W_r, new_rank):
                    best_rank = new_rank
                    best_result = {"U": U_r, "V": V_r, "W": W_r}
    
    if best_result is None:
        best_result = school
    
    return best_result
# EVOLVE-BLOCK-END
