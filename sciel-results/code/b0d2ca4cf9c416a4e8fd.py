import numpy as np
import time

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 130  # 20s margin before 150s kill

    # Build the target tensor T as a 3D array of shape (n*m, m*p, p*n)
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    # Fallback: Schoolbook rank 56
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W, rank

    # Rank 50 Strassen-based decomposition
    def strassen_rank50():
        rank = 50
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        
        def add_strassen(i0, j0, k0):
            nonlocal col
            # M1: (A11+A22)(B11+B22) -> C11 += M1, C22 += M1
            U[i0*m + j0, col] += 1
            U[(i0+1)*m + (j0+1), col] += 1
            V[j0*p + k0, col] += 1
            V[(j0+1)*p + (k0+1), col] += 1
            W[k0*n + i0, col] += 1
            W[(k0+1)*n + (i0+1), col] += 1
            col += 1
            
            # M2: (A21+A22)B11 -> C21 += M2, C22 -= M2
            U[(i0+1)*m + j0, col] += 1
            U[(i0+1)*m + (j0+1), col] += 1
            V[j0*p + k0, col] += 1
            W[k0*n + (i0+1), col] += 1
            W[(k0+1)*n + (i0+1), col] -= 1
            col += 1
            
            # M3: A11(B12-B22) -> C12 += M3, C22 += M3
            U[i0*m + j0, col] += 1
            V[j0*p + (k0+1), col] += 1
            V[(j0+1)*p + (k0+1), col] -= 1
            W[(k0+1)*n + i0, col] += 1
            W[(k0+1)*n + (i0+1), col] += 1
            col += 1
            
            # M4: A22(B21-B11) -> C11 += M4, C21 += M4
            U[(i0+1)*m + (j0+1), col] += 1
            V[(j0+1)*p + k0, col] += 1
            V[j0*p + k0, col] -= 1
            W[k0*n + i0, col] += 1
            W[k0*n + (i0+1), col] += 1
            col += 1
            
            # M5: (A11+A12)B22 -> C11 -= M5, C12 += M5
            U[i0*m + j0, col] += 1
            U[i0*m + (j0+1), col] += 1
            V[(j0+1)*p + (k0+1), col] += 1
            W[k0*n + i0, col] -= 1
            W[(k0+1)*n + i0, col] += 1
            col += 1
            
            # M6: (A21-A11)(B11+B12) -> C22 += M6
            U[(i0+1)*m + j0, col] += 1
            U[i0*m + j0, col] -= 1
            V[j0*p + k0, col] += 1
            V[j0*p + (k0+1), col] += 1
            W[(k0+1)*n + (i0+1), col] += 1
            col += 1
            
            # M7: (A12-A22)(B21+B22) -> C11 += M7
            U[i0*m + (j0+1), col] += 1
            U[(i0+1)*m + (j0+1), col] -= 1
            V[(j0+1)*p + k0, col] += 1
            V[(j0+1)*p + (k0+1), col] += 1
            W[k0*n + i0, col] += 1
            col += 1

        # Part 1: 2x4x4 (k=0..3) - 4 Strassen blocks
        add_strassen(0, 0, 0)
        add_strassen(0, 0, 2)
        add_strassen(0, 2, 0)
        add_strassen(0, 2, 2)
        
        # Part 2: 2x4x2 (k=4..5) - 2 Strassen blocks
        add_strassen(0, 0, 4)
        add_strassen(0, 2, 4)
        
        # Part 3: 2x4x1 (k=6) - 8 rank-1 terms
        for i in range(2):
            for j in range(4):
                U[i*m + j, col] = 1
                V[j*p + 6, col] = 1
                W[6*n + i, col] = 1
                col += 1
                
        return U, V, W, rank

    # Verify a decomposition
    def verify(U, V, W):
        result = np.zeros((n*m, m*p, p*n))
        for c in range(U.shape[1]):
            result += np.einsum('i,j,k->ijk', U[:,c], V[:,c], W[:,c])
        return np.allclose(result, T, atol=1e-10)

    # Start with rank 50 Strassen
    U_s, V_s, W_s, rank_s = strassen_rank50()
    if not verify(U_s, V_s, W_s):
        U_s, V_s, W_s, rank_s = schoolbook()
    
    best_U, best_V, best_W, best_rank = U_s, V_s, W_s, rank_s

    # Try to reduce rank by checking if any column can be removed
    for target_rank in range(best_rank - 1, max(best_rank - 8, 35), -1):
        if time.time() > deadline:
            break
        
        if target_rank == best_rank - 1:
            # Try removing each column
            for col_to_remove in range(best_rank):
                if time.time() > deadline:
                    break
                U_test = np.delete(best_U, col_to_remove, axis=1)
                V_test = np.delete(best_V, col_to_remove, axis=1)
                W_test = np.delete(best_W, col_to_remove, axis=1)
                if verify(U_test, V_test, W_test):
                    best_U, best_V, best_W, best_rank = U_test, V_test, W_test, target_rank
                    break
            continue
        
        # Try random restarts with ALS for lower ranks
        if time.time() > deadline:
            break
        
        for restart in range(20):
            if time.time() > deadline:
                break
            
            np.random.seed(300 + restart)
            U_try = np.random.randn(n*m, target_rank) * 0.1
            V_try = np.random.randn(m*p, target_rank) * 0.1
            W_try = np.random.randn(p*n, target_rank) * 0.1
            
            # ALS iterations
            for iteration in range(40):
                if time.time() > deadline:
                    break
                
                # Update U
                for c in range(target_rank):
                    residual = T.copy()
                    for c2 in range(target_rank):
                        if c2 != c:
                            residual -= np.einsum('i,j,k->ijk', U_try[:,c2], V_try[:,c2], W_try[:,c2])
                    
                    vw = V_try[:,c] * W_try[:,c]
                    denom = np.sum(vw * vw)
                    if abs(denom) > 1e-12:
                        U_try[:,c] = np.einsum('abk,bk->a', residual, vw) / denom
                    else:
                        U_try[:,c] = 0
                
                # Update V
                for c in range(target_rank):
                    residual = T.copy()
                    for c2 in range(target_rank):
                        if c2 != c:
                            residual -= np.einsum('i,j,k->ijk', U_try[:,c2], V_try[:,c2], W_try[:,c2])
                    
                    uw = U_try[:,c] * W_try[:,c]
                    denom = np.sum(uw * uw)
                    if abs(denom) > 1e-12:
                        V_try[:,c] = np.einsum('abk,ak->b', residual, uw) / denom
                    else:
                        V_try[:,c] = 0
                
                # Update W
                for c in range(target_rank):
                    residual = T.copy()
                    for c2 in range(target_rank):
                        if c2 != c:
                            residual -= np.einsum('i,j,k->ijk', U_try[:,c2], V_try[:,c2], W_try[:,c2])
                    
                    uv = U_try[:,c] * V_try[:,c]
                    denom = np.sum(uv * uv)
                    if abs(denom) > 1e-12:
                        W_try[:,c] = np.einsum('abk,ab->k', residual, uv) / denom
                    else:
                        W_try[:,c] = 0
                
                # Check convergence
                result = np.zeros((n*m, m*p, p*n))
                for c in range(target_rank):
                    result += np.einsum('i,j,k->ijk', U_try[:,c], V_try[:,c], W_try[:,c])
                if np.linalg.norm(result - T) < 1e-8:
                    break
            
            if verify(U_try, V_try, W_try):
                best_U, best_V, best_W, best_rank = U_try, V_try, W_try, target_rank
                break

    U_out = best_U.tolist()
    V_out = best_V.tolist()
    W_out = best_W.tolist()
    
    return {"U": U_out, "V": V_out, "W": W_out}
