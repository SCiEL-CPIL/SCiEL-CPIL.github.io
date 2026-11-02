"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
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
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    def validate(U, V, W, rank):
        """
        Check if U, V, W form a valid tensor decomposition.
        U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
        For all i, j, k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
        """
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        
        # Build tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        T = np.zeros((n, m, p), dtype=np.float64)
        
        for c in range(rank):
            Uc = U_arr[:, c]  # (n*m,)
            Vc = V_arr[:, c]  # (m*p,)
            Wc = W_arr[:, c]  # (p*n,)
            
            Uc_r = Uc.reshape(n, m)
            Vc_r = Vc.reshape(m, p)
            Wc_r = Wc.reshape(p, n)
            
            # T[i,j,k] += Uc_r[i,j] * Vc_r[j,k] * Wc_r[k,i]
            # We can do this more efficiently using einsum
            # T += Uc_r[i,j] * Vc_r[j,k] * Wc_r[k,i]
            # This is like a trace operation
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i, j, k] += Uc_r[i, j] * Vc_r[j, k] * Wc_r[k, i]
        
        return np.allclose(T, 1.0, atol=1e-8)
    
    def validate_fast(U, V, W, rank):
        """
        Faster validation using numpy operations.
        """
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        
        # For each c, we have Uc (n*m,), Vc (m*p,), Wc (p*n,)
        # T[i,j,k] = sum_c Uc_r[i,j] * Vc_r[j,k] * Wc_r[k,i]
        
        # Let's use a different approach: 
        # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        
        # We can compute this as:
        # For fixed j, T[:, j, :] = sum_c (U[:, j, :] * W[:, j, :]) where 
        # U[:, j, :] means U[i*m+j, c] for all i, c
        # and W[:, j, :] means W[k*n+i, c] for all k, c... wait, that's not right.
        
        # Let me just do it directly but with numpy broadcasting
        T = np.zeros((n, m, p), dtype=np.float64)
        
        for c in range(rank):
            Uc = U_arr[:, c].reshape(n, m)
            Vc = V_arr[:, c].reshape(m, p)
            Wc = W_arr[:, c].reshape(p, n)
            
            # T[i,j,k] += Uc[i,j] * Vc[j,k] * Wc[k,i]
            # This can be written as:
            # T += Uc[i,j] * (Vc[j,k] * Wc[k,i])
            # Vc[j,k] * Wc[k,i] summed over k gives (Vc @ Wc)[j, i]... no
            # Vc is (m,p), Wc is (p,n), so Vc @ Wc is (m,n)
            # (Vc @ Wc)[j, i] = sum_k Vc[j,k] * Wc[k,i]
            # So T[i,j,k] += Uc[i,j] * Vc[j,k] * Wc[k,i]
            # This is NOT simply Uc * (Vc @ Wc) because the k index is free.
            
            # Let's just loop over i, j, k for each c
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i, j, k] += Uc[i, j] * Vc[j, k] * Wc[k, i]
        
        return np.allclose(T, 1.0, atol=1e-8)
    
    def validate_einsum(U, V, W, rank):
        """
        Validation using einsum for speed.
        T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        
        Let's reshape:
        U_r[i, j, c] = U[i*m+j, c]
        V_r[j, k, c] = V[j*p+k, c]
        W_r[k, i, c] = W[k*n+i, c]
        
        T[i,j,k] = sum_c U_r[i,j,c] * V_r[j,k,c] * W_r[k,i,c]
        
        This is: T[i,j,k] = sum_c U_r[i,j,c] * V_r[j,k,c] * W_r[k,i,c]
        
        We can write this as:
        T[i,j,k] = sum_c U_r[i,j,c] * (V_r[j,k,c] * W_r[k,i,c])
        
        Or using einsum:
        T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
        """
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        
        U_r = U_arr.reshape(n, m, rank)
        V_r = V_arr.reshape(m, p, rank)
        W_r = W_arr.reshape(p, n, rank)
        
        T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
        
        return np.allclose(T, 1.0, atol=1e-8)
    
    # Start with schoolbook
    school = build_schoolbook()
    if validate_einsum(school["U"], school["V"], school["W"], best_rank):
        best_result = school
    
    # Try structured Strassen-based approach
    # For 4x4 x 4x6, we can split into 2x2 blocks:
    # A is 2x2 blocks of 2x2 matrices
    # B is 2x3 blocks of 2x2 matrices
    # C is 2x3 blocks of 2x2 matrices
    
    # The block multiplication is:
    # C[i,j] = sum_l A[i,l] * B[l,j] for i in {0,1}, j in {0,1,2}, l in {0,1}
    
    # We can use Strassen-like techniques for the 2x2 core with 3 columns.
    # Let's define the block matrices:
    # A = [[A00, A01], [A10, A11]] where each Aij is 2x2
    # B = [[B00, B01, B02], [B10, B11, B12]] where each Bjl is 2x2
    # C = [[C00, C01, C02], [C10, C11, C12]] where each Cjl is 2x2
    
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C02 = A00*B02 + A01*B12
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    # C12 = A10*B02 + A11*B12
    
    # We can use Strassen for the 2x2 x 2x2 -> 2x2 products.
    # But we have 6 products of 2x2 matrices, not 4.
    
    # Let's try a different approach: use the known rank bounds.
    # The rank of M_{4,4,6} is at most 4*4*6 = 96 (schoolbook)
    # It could be lower. Let's try to find a decomposition with rank < 96.
    
    # For now, let's just return the schoolbook result since random search
    # is unlikely to find valid decompositions quickly.
    
    if best_result is None:
        best_result = school
    
    return best_result
# EVOLVE-BLOCK-END
