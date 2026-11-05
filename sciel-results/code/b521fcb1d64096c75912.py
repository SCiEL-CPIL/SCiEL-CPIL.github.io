"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    
    n, m, p = DIMENSIONS  # 2, 4, 7
    
    def add_strassen(U, V, W, i0, j0, k0, col):
        # Standard Strassen 2x2x2
        # M1 = (A11+A22)(B11+B22)
        U[i0*m+j0, col] += 1; U[(i0+1)*m+(j0+1), col] += 1
        V[j0*p+k0, col] += 1; V[(j0+1)*p+(k0+1), col] += 1
        W[k0*n+i0, col] += 1; W[(k0+1)*n+(i0+1), col] += 1
        col += 1
        # M2 = (A21+A22)B11
        U[(i0+1)*m+j0, col] += 1; U[(i0+1)*m+(j0+1), col] += 1
        V[j0*p+k0, col] += 1
        W[k0*n+(i0+1), col] += 1; W[(k0+1)*n+(i0+1), col] -= 1
        col += 1
        # M3 = A11(B12-B22)
        U[i0*m+j0, col] += 1
        V[j0*p+(k0+1), col] += 1; V[(j0+1)*p+(k0+1), col] -= 1
        W[(k0+1)*n+i0, col] += 1; W[(k0+1)*n+(i0+1), col] += 1
        col += 1
        # M4 = A22(B21-B11)
        U[(i0+1)*m+(j0+1), col] += 1
        V[(j0+1)*p+k0, col] += 1; V[j0*p+k0, col] -= 1
        W[k0*n+(i0+1), col] += 1; W[k0*n+i0, col] += 1
        col += 1
        # M5 = (A11+A12)B22
        U[i0*m+j0, col] += 1; U[i0*m+(j0+1), col] += 1
        V[(j0+1)*p+(k0+1), col] += 1
        W[k0*n+i0, col] -= 1; W[(k0+1)*n+i0, col] += 1
        col += 1
        # M6 = (A21-A11)(B11+B12)
        U[(i0+1)*m+j0, col] += 1; U[i0*m+j0, col] -= 1
        V[j0*p+k0, col] += 1; V[j0*p+(k0+1), col] += 1
        W[(k0+1)*n+(i0+1), col] += 1
        col += 1
        # M7 = (A12-A22)(B21+B22)
        U[i0*m+(j0+1), col] += 1; U[(i0+1)*m+(j0+1), col] -= 1
        V[(j0+1)*p+k0, col] += 1; V[(j0+1)*p+(k0+1), col] += 1
        W[k0*n+i0, col] += 1
        col += 1
        return col
    
    def build_rank46():
        rank = 46
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        # 6 Strassen blocks for 2x4x6
        col = add_strassen(U, V, W, 0, 0, 0, col)
        col = add_strassen(U, V, W, 0, 0, 2, col)
        col = add_strassen(U, V, W, 0, 2, 0, col)
        col = add_strassen(U, V, W, 0, 2, 2, col)
        col = add_strassen(U, V, W, 0, 0, 4, col)
        col = add_strassen(U, V, W, 0, 2, 4, col)
        
        # 4 terms for 2x4x1 (column 6)
        # C[0,6] = A[0,0]B[0,6] + A[0,1]B[1,6] + A[0,2]B[2,6] + A[0,3]B[3,6]
        # C[1,6] = A[1,0]B[0,6] + A[1,1]B[1,6] + A[1,2]B[2,6] + A[1,3]B[3,6]
        
        # Term 1: (A[0,0]+A[1,0]) * B[0,6] * (C[0,6]+C[1,6])
        U[0*m+0, col] += 1; U[1*m+0, col] += 1
        V[0*p+6, col] += 1
        W[6*n+0, col] += 1; W[6*n+1, col] += 1
        col += 1
        
        # Term 2: (A[0,0]-A[1,0]) * B[0,6] * (C[0,6]-C[1,6])
        U[0*m+0, col] += 1; U[1*m+0, col] -= 1
        V[0*p+6, col] += 1
        W[6*n+0, col] += 1; W[6*n+1, col] -= 1
        col += 1
        
        # Term 3: (A[0,1]+A[1,1]) * B[1,6] * (C[0,6]+C[1,6])
        U[0*m+1, col] += 1; U[1*m+1, col] += 1
        V[1*p+6, col] += 1
        W[6*n+0, col] += 1; W[6*n+1, col] += 1
        col += 1
        
        # Term 4: (A[0,1]-A[1,1]) * B[1,6] * (C[0,6]-C[1,6])
        U[0*m+1, col] += 1; U[1*m+1, col] -= 1
        V[1*p+6, col] += 1
        W[6*n+0, col] += 1; W[6*n+1, col] -= 1
        col += 1
        
        # Wait, this only covers j=0,1. We need j=2,3.
        # This decomposition is for 2x2x1.
        # We need 2x4x1.
        # So we need 2 more terms for j=2,3?
        # Or can we combine?
        # The rank of 2x4x1 is 4.
        # So we need 4 terms total.
        # The above 4 terms only cover j=0,1.
        # We need to cover j=2,3 as well.
        # So we need 8 terms?
        # If R(2x4x1) = 4, then 4 terms are sufficient.
        # How?
        # Maybe the terms are not per j.
        
        # Let's verify R(2x4x1).
        # If R(2x2x1) = 2, then R(2x4x1) = 4.
        # So we need 4 terms.
        # The above construction uses 4 terms but only covers j=0,1.
        # So it's incorrect for 2x4x1.
        
        # Let's try to construct 4 terms for 2x4x1.
        # We need to represent sum_{j=0}^3 B_j (A_0j C_0 + A_1j C_1).
        # This is sum_{j=0}^3 (A_0j C_0 + A_1j C_1) B_j.
        # Let v_j = e_j in B-space.
        # We need u_j in A-space, w_j in C-space.
        # sum_j u_j v_j w_j.
        # u_j must be such that u_j . A = A_0j C_0 + A_1j C_1? No.
        # u_j . A is a scalar.
        # We need the sum to be C.
        
        # It is known that R(2x4x1) = 4.
        # Let's use 4 terms.
        # Term 1: j=0,1 combined?
        # Actually, if we use 4 terms, we can assign one term per j?
        # Term j: A_0j B_j C_0 + A_1j B_j C_1.
        # This requires u_j = e_0j + e_1j? No.
        # u_j . A = A_0j + A_1j?
        # Then (A_0j+A_1j) B_j (C_0+C_1).
        # This gives A_0j B_j C_0 + A_0j B_j C_1 + A_1j B_j C_0 + A_1j B_j C_1.
        # We want A_0j B_j C_0 + A_1j B_j C_1.
        # We have extra terms.
        
        # So we need to cancel them.
        # This implies we need more terms.
        # If R(2x4x1) = 4, there must be a clever combination.
        
        # For now, let's stick with 8 terms for the last column to be safe?
        # Rank 50 is current best.
        # If I can't find rank 46, I should return rank 50.
        
        return None

    def build_rank50():
        rank = 50
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        col = add_strassen(U, V, W, 0, 0, 0, col)
        col = add_strassen(U, V, W, 0, 0, 2, col)
        col = add_strassen(U, V, W, 0, 2, 0, col)
        col = add_strassen(U, V, W, 0, 2, 2, col)
        col = add_strassen(U, V, W, 0, 0, 4, col)
        col = add_strassen(U, V, W, 0, 2, 4, col)
        for i in range(2):
            for j in range(4):
                U[i*m + j, col] = 1
                V[j*p + 6, col] = 1
                W[6*n + i, col] = 1
                col += 1
        return U, V, W
    
    def verify(U, V, W):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        C_computed = np.zeros((n, p))
        for r in range(U.shape[1]):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        C_computed[i, k] += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        return np.allclose(C_computed, C_expected, atol=1e-10)
    
    # Try rank 46?
    # Since I'm not sure about the rank 4 construction, I will return rank 50.
    U, V, W = build_rank50()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
