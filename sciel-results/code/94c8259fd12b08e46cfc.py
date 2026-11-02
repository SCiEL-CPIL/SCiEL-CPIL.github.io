"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import lsq_linear

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Attempt to find a rank-19 decomposition using numerical optimization
    # The matrix multiplication tensor constraint:
    # For all i,j,k: sum_r U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] = 1 if j==j' else 0
    # Actually, the constraint is that the bilinear form reproduces matrix multiplication.
    
    # Let's use a known rank-19 decomposition structure.
    # We'll use Strassen's 7 terms for the 2x2 block, plus 12 more terms.
    
    rank = 19
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # First 7 terms: Strassen for top-left 2x2
    # M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[4, 0] = 1
    V[0, 0] = 1; V[4, 0] = 1
    W[0, 0] = 1; W[4, 0] = 1
    
    # M2 = (A10+A11)B00
    U[3, 1] = 1; U[4, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[4, 1] = -1
    
    # M3 = A00(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[4, 2] = -1
    W[3, 2] = 1; W[4, 2] = 1
    
    # M4 = A11(B10-B00)
    U[4, 3] = 1
    V[3, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    # M5 = (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[4, 4] = 1
    W[0, 4] = -1; W[3, 4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    U[3, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[4, 5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    U[1, 6] = 1; U[4, 6] = -1
    V[3, 6] = 1; V[4, 6] = 1
    W[0, 6] = 1
    
    # Now we need 12 more terms for the remaining products.
    # The remaining products involve indices with at least one coordinate = 2.
    
    # We'll set up a system of equations to solve for the remaining 12 terms.
    # For each term r (7 to 18), we have variables U[i*3+j, r], V[j*3+k, r], W[k*3+i, r]
    
    # The constraint is: for all (i, j, k, i', k') where the product A[i,j]*B[j,k] should contribute to C[i',k']:
    # The contribution is 1 if i==i' and k==k' and the j index matches, else 0.
    
    # Actually, let's just verify by testing with random matrices.
    # For now, let's try a simple construction and see if it works.
    
    # Terms 8-19: Handle the boundary terms
    # We need to compute:
    # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2] + A[0,2]*B[2,2]
    # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2] + A[1,2]*B[2,2]
    # C[2,0] = A[2,0]*B[0,0] + A[2,1]*B[1,0] + A[2,2]*B[2,0]
    # C[2,1] = A[2,0]*B[0,1] + A[2,1]*B[1,1] + A[2,2]*B[2,1]
    # C[2,2] = A[2,0]*B[0,2] + A[2,1]*B[1,2] + A[2,2]*B[2,2]
    
    # Let's use a simple approach: direct products for now (will be rank 15 for these)
    # But we only have 12 slots. Let's try to be clever.
    
    # For now, let's just use direct products and accept rank 22, then optimize.
    # Actually, let's just return a valid (if not optimal) decomposition first.
    
    # Let me try a different known decomposition.
    # I'll use the fact that we can share some multiplications.
    
    # Term 8: A[0,2]*B[2,0] -> C[2,2]... wait, no.
    # A[0,2]*B[2,k] contributes to C[0,k] for k=0,1,2
    # Specifically A[0,2]*B[2,0] -> C[0,0], A[0,2]*B[2,1] -> C[0,1], A[0,2]*B[2,2] -> C[0,2]
    
    # Let me just use a straightforward approach with rank 27 (schoolbook) to make sure it's valid,
    # then try to optimize.
    
    # Actually, let me just keep the rank-26 solution but verify it's correct.
    # Let me test it.
    
    # Convert to lists for the return format
    U_list = [[0] * rank for _ in range(n * m)]
    V_list = [[0] * rank for _ in range(m * p)]
    W_list = [[0] * rank for _ in range(p * n)]
    
    for i in range(n * m):
        for r in range(rank):
            U_list[i][r] = U[i, r]
    for i in range(m * p):
        for r in range(rank):
            V_list[i][r] = V[i, r]
    for i in range(p * n):
        for r in range(rank):
            W_list[i][r] = W[i, r]
    
    # Verify the decomposition
    def verify(U, V, W, n, m, p):
        import numpy as np
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        rank = U.shape[1]
        
        # Test with random matrices
        for trial in range(5):
            A = np.random.randn(n, m)
            B = np.random.randn(m, p)
            C_expected = A @ B
            
            C_computed = np.zeros((n, p))
            for r in range(rank):
                # For each rank term, compute the contribution
                # a_i = sum_j U[i*m+j, r] * A[i,j]
                # b_k = sum_j V[j*p+k, r] * B[j,k]
                # C[i,k] += W[k*n+i, r] * a_i * b_k
                a = np.zeros(n)
                for i in range(n):
                    for j in range(m):
                        a[i] += U[i*m+j, r] * A[i,j]
                
                b = np.zeros(p)
                for k in range(p):
                    for j in range(m):
                        b[k] += V[j*p+k, r] * B[j,k]
                
                for i in range(n):
                    for k in range(p):
                        C_computed[i,k] += W[k*n+i, r] * a[i] * b[k]
            
            if not np.allclose(C_computed, C_expected, atol=1e-10):
                return False, C_computed, C_expected
        return True, None, None
    
    valid, _, _ = verify(U_list, V_list, W_list, n, m, p)
    if not valid:
        # Fall back to schoolbook rank 27
        rank = 27
        U_list = [[0] * rank for _ in range(n * m)]
        V_list = [[0] * rank for _ in range(m * p)]
        W_list = [[0] * rank for _ in range(p * n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx = i*m*j + j*p*k  # This is wrong, let me fix
                    # Actually, for schoolbook:
                    # C[i,k] = sum_j A[i,j] B[j,k]
                    # Each term is A[i,j] * B[j,k] -> C[i,k]
                    # U index: i*m+j, V index: j*p+k, W index: k*n+i
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    U_list[u_idx][i*m*j + j*p*k] = 1  # This is getting messy
                    break
                break
            break
        # Let me just do it properly
        rank = 27
        U_list = [[0] * rank for _ in range(n * m)]
        V_list = [[0] * rank for _ in range(m * p)]
        W_list = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    U_list[u_idx][col] = 1
                    V_list[v_idx][col] = 1
                    W_list[w_idx][col] = 1
                    col += 1
        return {"U": U_list, "V": V_list, "W": W_list}
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
