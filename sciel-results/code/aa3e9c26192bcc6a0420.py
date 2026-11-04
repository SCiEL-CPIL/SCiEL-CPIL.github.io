import numpy as np
import time

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140  # 10s margin before 150s kill

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

    # Fast verification using vectorized einsum
    def verify(U, V, W):
        R = U.shape[1]
        # result[a,b,c] = sum_c U[a,c]*V[b,c]*W[c,c]
        # Use batched outer products
        result = np.einsum('ac,bc,dc->abd', U, V, W)
        return np.allclose(result, T, atol=1e-9)

    def verify_fast(U, V, W):
        """Faster check: only check non-zero entries of T"""
        R = U.shape[1]
        # For each non-zero entry T[a,b,c]=1, check sum_c U[a,c]*V[b,c]*W[c,c] == 1
        # And for zero entries, check sum == 0
        # We'll just do the full computation but with early exit
        result = np.einsum('ac,bc,dc->abd', U, V, W)
        return np.allclose(result, T, atol=1e-9)

    # Start with rank 50 Strassen
    U_s, V_s, W_s, rank_s = strassen_rank50()
    if not verify(U_s, V_s, W_s):
        U_s, V_s, W_s, rank_s = schoolbook()
    
    best_U, best_V, best_W, best_rank = U_s, V_s, W_s, rank_s

    # Try to reduce rank by checking if any column can be removed
    # Use fast verification
    for target_rank in range(best_rank - 1, max(best_rank - 10, 30), -1):
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
        
        # Try random restarts with improved ALS for lower ranks
        if time.time() > deadline:
            break
        
        for restart in range(30):
            if time.time() > deadline:
                break
            
            np.random.seed(500 + restart * 7)
            
            # Initialize from Strassen structure with perturbation
            if best_rank >= target_rank:
                # Take first target_rank columns from best solution and perturb
                U_try = best_U[:, :target_rank].copy() + np.random.randn(n*m, target_rank) * 0.05
                V_try = best_V[:, :target_rank].copy() + np.random.randn(m*p, target_rank) * 0.05
                W_try = best_W[:, :target_rank].copy() + np.random.randn(p*n, target_rank) * 0.05
            else:
                U_try = np.random.randn(n*m, target_rank) * 0.1
                V_try = np.random.randn(m*p, target_rank) * 0.1
                W_try = np.random.randn(p*n, target_rank) * 0.1
            
            # Fast ALS using matrix operations
            for iteration in range(60):
                if time.time() > deadline:
                    break
                
                # Update U: for each c, U[:,c] = argmin ||T - sum U[:,c']*V[:,c']*W[:,c]||^2
                # = argmin ||T - U[:,c] outer (V[:,c]*W[:,c])||^2
                # Solution: U[:,c] = einsum(T, VW) / ||VW||^2 where VW = V[:,c]*W[:,c]
                # But we need to subtract other terms
                # Build the "other" tensor once
                other = np.zeros((n*m, m*p, p*n))
                for c2 in range(target_rank):
                    other += np.einsum('i,j,k->ijk', U_try[:,c2], V_try[:,c2], W_try[:,c2])
                
                for c in range(target_rank):
                    # Remove c-th term from other
                    other -= np.einsum('i,j,k->ijk', U_try[:,c], V_try[:,c], W_try[:,c])
                    residual = T - other
                    
                    vw = V_try[:,c] * W_try[:,c]
                    denom = np.sum(vw * vw)
                    if abs(denom) > 1e-12:
                        U_try[:,c] = np.einsum('abk,bk->a', residual, vw) / denom
                    else:
                        U_try[:,c] = 0
                    # Add back
                    other += np.einsum('i,j,k->ijk', U_try[:,c], V_try[:,c], W_try[:,c])
                
                # Update V
                for c in range(target_rank):
                    other -= np.einsum('i,j,k->ijk', U_try[:,c], V_try[:,c], W_try[:,c])
                    residual = T - other
                    
                    uw = U_try[:,c] * W_try[:,c]
                    denom = np.sum(uw * uw)
                    if abs(denom) > 1e-12:
                        V_try[:,c] = np.einsum('abk,ak->b', residual, uw) / denom
                    else:
                        V_try[:,c] = 0
                    other += np.einsum('i,j,k->ijk', U_try[:,c], V_try[:,c], W_try[:,c])
                
                # Update W
                for c in range(target_rank):
                    other -= np.einsum('i,j,k->ijk', U_try[:,c], V_try[:,c], W_try[:,c])
                    residual = T - other
                    
                    uv = U_try[:,c] * V_try[:,c]
                    denom = np.sum(uv * uv)
                    if abs(denom) > 1e-12:
                        W_try[:,c] = np.einsum('abk,ab->k', residual, uv) / denom
                    else:
                        W_try[:,c] = 0
                    other += np.einsum('i,j,k->ijk', U_try[:,c], V_try[:,c], W_try[:,c])
                
                # Check convergence
                if iteration % 5 == 4:
                    result = np.einsum('ac,bc,dc->abd', U_try, V_try, W_try)
                    if np.linalg.norm(result - T) < 1e-8:
                        break
            
            if verify(U_try, V_try, W_try):
                best_U, best_V, best_W, best_rank = U_try, V_try, W_try, target_rank
                break

    # Final: try removing more columns from best solution
    for _ in range(5):
        if time.time() > deadline:
            break
        removed = False
        for col_to_remove in range(best_rank):
            if time.time() > deadline:
                break
            U_test = np.delete(best_U, col_to_remove, axis=1)
            V_test = np.delete(best_V, col_to_remove, axis=1)
            W_test = np.delete(best_W, col_to_remove, axis=1)
            if verify(U_test, V_test, W_test):
                best_U, best_V, best_W, best_rank = U_test, V_test, W_test, best_rank - 1
                removed = True
                break
        if not removed:
            break

    U_out = best_U.tolist()
    V_out = best_V.tolist()
    W_out = best_W.tolist()
    
    return {"U": U_out, "V": V_out, "W": W_out}
