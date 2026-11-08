"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 7]
N, M, P = DIMENSIONS
RANK = N * M * P

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = 145.0  # safety margin before 150s kill

    # Schoolbook decomposition as baseline
    U = np.zeros((N * M, RANK), dtype=float)
    V = np.zeros((M * P, RANK), dtype=float)
    W = np.zeros((P * N, RANK), dtype=float)
    col = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U[i * M + j, col] = 1.0
                V[j * P + k, col] = 1.0
                W[k * N + i, col] = 1.0
                col += 1

    def verify(U_mat, V_mat, W_mat):
        """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
        for i in range(N):
            for j in range(M):
                for k in range(P):
                    a = i * M + j
                    b = j * P + k
                    c = k * N + i
                    val = 0.0
                    for r in range(U_mat.shape[1]):
                        val += U_mat[a, r] * V_mat[b, r] * W_mat[c, r]
                    if abs(val - 1.0) > 1e-8:
                        return False
        # Also check that off-diagonal entries are zero
        for i in range(N):
            for j in range(M):
                for k in range(P):
                    a = i * M + j
                    b = j * P + k
                    c = k * N + i
                    # Check a few off-diagonal positions
                    for i2 in range(N):
                        for j2 in range(M):
                            for k2 in range(P):
                                if (i2, j2, k2) == (i, j, k):
                                    continue
                                a2 = i2 * M + j2
                                b2 = j2 * P + k2
                                c2 = k2 * N + i2
                                if a2 == a and b2 == b:
                                    val = 0.0
                                    for r in range(U_mat.shape[1]):
                                        val += U_mat[a, r] * V_mat[b, r] * W_mat[c2, r]
                                    if abs(val) > 1e-8:
                                        return False
                                elif a2 == a and c2 == c:
                                    val = 0.0
                                    for r in range(U_mat.shape[1]):
                                        val += U_mat[a, r] * V_mat[b2, r] * W_mat[c, r]
                                    if abs(val) > 1e-8:
                                        return False
                                elif b2 == b and c2 == c:
                                    val = 0.0
                                    for r in range(U_mat.shape[1]):
                                        val += U_mat[a2, r] * V_mat[b, r] * W_mat[c, r]
                                    if abs(val) > 1e-8:
                                        return False
        return True

    best_rank = RANK
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

    # Try to find lower-rank decompositions using structured approaches
    # Strategy: exploit the small n=2 dimension
    
    # Approach 1: Try rank 21 (theoretical lower bound = max(n*m, m*p, p*n) = 21)
    # For rank 21, we need U: 6x21, V: 21x21, W: 14x21
    # This is highly underdetermined: 6*21 + 21*21 + 14*21 = 126 + 441 + 294 = 861 unknowns
    # vs 42 equations (the nonzero tensor entries)
    
    # Let's try a systematic search using the structure
    # Key insight: C[i,k] = sum_j A[i,j] B[j,k]
    # For n=2: C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k] + A[0,2]*B[2,k]
    #           C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k] + A[1,2]*B[2,k]
    
    # Try to find a decomposition where we share multiplications across k values
    # For each j, the products A[i,j]*B[j,k] for all k form a rank-1 matrix
    # A[i,j] * B[j,:] (a row vector scaled by scalar)
    
    # Total schoolbook: 2*3*7 = 42 rank-1 terms
    # Can we do better by combining terms?
    
    # Let's try a randomized search for rank < 42
    def try_rank(R, n_trials=50):
        """Try to find a rank-R decomposition using random initialization + ALS."""
        for trial in range(n_trials):
            if time.time() > deadline - 10:
                break
            
            np.random.seed(42 + trial)
            U_c = np.random.randn(N * M, R)
            V_c = np.random.randn(M * P, R)
            W_c = np.random.randn(P * N, R)
            
            # Normalize to avoid numerical issues
            for r in range(R):
                norm_u = np.linalg.norm(U_c[:, r])
                norm_v = np.linalg.norm(V_c[:, r])
                norm_w = np.linalg.norm(W_c[:, r])
                if norm_u > 1e-10 and norm_v > 1e-10 and norm_w > 1e-10:
                    U_c[:, r] /= norm_u
                    V_c[:, r] /= norm_v
                    W_c[:, r] /= norm_w
                    # Redistribute scale
                    scale = (norm_u * norm_v * norm_w) ** (1.0/3.0)
                    U_c[:, r] *= scale
                    V_c[:, r] *= scale
                    W_c[:, r] *= scale
            
            # ALS iterations
            lr = 0.1
            converged = False
            for it in range(200):
                if time.time() > deadline - 5:
                    break
                
                # Update U
                for a in range(N * M):
                    grad = np.zeros(R)
                    for b in range(M * P):
                        for c in range(P * N):
                            # T[a,b,c] is 1 if (a,b,c) corresponds to a valid (i,j,k)
                            # a = i*M+j, b = j*P+k, c = k*N+i
                            # Check if this is a valid tensor entry
                            i = a // M
                            j = a % M
                            j_b = b // P
                            k = b % P
                            k_c = c // N
                            i_c = c % N
                            if j == j_b and k == k_c and i == i_c:
                                target = 1.0
                            else:
                                target = 0.0
                            # Gradient of ||sum_r U[a,r]*V[b,r]*W[c,r] - target||^2 w.r.t. U[a,r]
                            for r in range(R):
                                val = 0.0
                                for r2 in range(R):
                                    val += U_c[a, r2] * V_c[b, r2] * W_c[c, r2]
                                grad[r] += 2 * (val - target) * V_c[b, r] * W_c[c, r]
                    U_c[a] -= lr * grad
                
                # Update V
                for b in range(M * P):
                    grad = np.zeros(R)
                    for a in range(N * M):
                        for c in range(P * N):
                            i = a // M
                            j = a % M
                            j_b = b // P
                            k = b % P
                            k_c = c // N
                            i_c = c % N
                            if j == j_b and k == k_c and i == i_c:
                                target = 1.0
                            else:
                                target = 0.0
                            for r in range(R):
                                val = 0.0
                                for r2 in range(R):
                                    val += U_c[a, r2] * V_c[b, r2] * W_c[c, r2]
                                grad[r] += 2 * (val - target) * U_c[a, r] * W_c[c, r]
                    V_c[b] -= lr * grad
                
                # Update W
                for c in range(P * N):
                    grad = np.zeros(R)
                    for a in range(N * M):
                        for b in range(M * P):
                            i = a // M
                            j = a % M
                            j_b = b // P
                            k = b % P
                            k_c = c // N
                            i_c = c % N
                            if j == j_b and k == k_c and i == i_c:
                                target = 1.0
                            else:
                                target = 0.0
                            for r in range(R):
                                val = 0.0
                                for r2 in range(R):
                                    val += U_c[a, r2] * V_c[b, r2] * W_c[c, r2]
                                grad[r] += 2 * (val - target) * U_c[a, r] * V_c[b, r]
                    W_c[c] -= lr * grad
                
                # Check convergence
                if it % 50 == 0:
                    max_err = 0.0
                    for i in range(N):
                        for j in range(M):
                            for k in range(P):
                                a = i * M + j
                                b = j * P + k
                                c = k * N + i
                                val = 0.0
                                for r in range(R):
                                    val += U_c[a, r] * V_c[b, r] * W_c[c, r]
                                max_err = max(max_err, abs(val - 1.0))
                    if max_err < 1e-6:
                        if verify(U_c, V_c, W_c):
                            return U_c, V_c, W_c
                        elif max_err < 1e-4:
                            # Try more iterations with smaller lr
                            lr = 0.01
                            for it2 in range(100):
                                if time.time() > deadline - 5:
                                    break
                                # Just check again
                                max_err = 0.0
                                for i in range(N):
                                    for j in range(M):
                                        for k in range(P):
                                            a = i * M + j
                                            b = j * P + k
                                            c = k * N + i
                                            val = 0.0
                                            for r in range(R):
                                                val += U_c[a, r] * V_c[b, r] * W_c[c, r]
                                            max_err = max(max_err, abs(val - 1.0))
                                if max_err < 1e-8:
                                    if verify(U_c, V_c, W_c):
                                        return U_c, V_c, W_c
                            break
        
        return None

    # Try ranks from 21 up to 41
    for R in range(21, RANK):
        if time.time() > deadline - 15:
            break
        
        result = try_rank(R, n_trials=10)
        if result is not None:
            U_c, V_c, W_c = result
            if verify(U_c, V_c, W_c):
                best_rank = R
                best_U, best_V, best_W = U_c.copy(), V_c.copy(), W_c.copy()
                break

    if best_rank < RANK:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
