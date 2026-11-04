import numpy as np
import time

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 145  # 5s margin before 150s kill

    # Build the target tensor T
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

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

    def strassen_rank50():
        rank = 50
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        
        def add_strassen(i0, j0, k0):
            nonlocal col
            # M1
            U[i0*m + j0, col] += 1
            U[(i0+1)*m + (j0+1), col] += 1
            V[j0*p + k0, col] += 1
            V[(j0+1)*p + (k0+1), col] += 1
            W[k0*n + i0, col] += 1
            W[(k0+1)*n + (i0+1), col] += 1
            col += 1
            # M2
            U[(i0+1)*m + j0, col] += 1
            U[(i0+1)*m + (j0+1), col] += 1
            V[j0*p + k0, col] += 1
            W[k0*n + (i0+1), col] += 1
            W[(k0+1)*n + (i0+1), col] -= 1
            col += 1
            # M3
            U[i0*m + j0, col] += 1
            V[j0*p + (k0+1), col] += 1
            V[(j0+1)*p + (k0+1), col] -= 1
            W[(k0+1)*n + i0, col] += 1
            W[(k0+1)*n + (i0+1), col] += 1
            col += 1
            # M4
            U[(i0+1)*m + (j0+1), col] += 1
            V[(j0+1)*p + k0, col] += 1
            V[j0*p + k0, col] -= 1
            W[k0*n + i0, col] += 1
            W[k0*n + (i0+1), col] += 1
            col += 1
            # M5
            U[i0*m + j0, col] += 1
            U[i0*m + (j0+1), col] += 1
            V[(j0+1)*p + (k0+1), col] += 1
            W[k0*n + i0, col] -= 1
            W[(k0+1)*n + i0, col] += 1
            col += 1
            # M6
            U[(i0+1)*m + j0, col] += 1
            U[i0*m + j0, col] -= 1
            V[j0*p + k0, col] += 1
            V[j0*p + (k0+1), col] += 1
            W[(k0+1)*n + (i0+1), col] += 1
            col += 1
            # M7
            U[i0*m + (j0+1), col] += 1
            U[(i0+1)*m + (j0+1), col] -= 1
            V[(j0+1)*p + k0, col] += 1
            V[(j0+1)*p + (k0+1), col] += 1
            W[k0*n + i0, col] += 1
            col += 1

        add_strassen(0, 0, 0)
        add_strassen(0, 0, 2)
        add_strassen(0, 2, 0)
        add_strassen(0, 2, 2)
        add_strassen(0, 0, 4)
        add_strassen(0, 2, 4)
        
        for i in range(2):
            for j in range(4):
                U[i*m + j, col] = 1
                V[j*p + 6, col] = 1
                W[6*n + i, col] = 1
                col += 1
                
        return U, V, W, rank

    def verify(U, V, W):
        result = np.zeros((n*m, m*p, p*n))
        for c in range(U.shape[1]):
            result += np.einsum('i,j,k->ijk', U[:,c], V[:,c], W[:,c])
        return np.allclose(result, T, atol=1e-8)

    def round_half(U, V, W):
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        return U_r, V_r, W_r

    def verify_exact_half(U, V, W):
        # Check if entries are half-integers
        if not np.allclose(U * 2, np.round(U * 2), atol=1e-6): return False
        if not np.allclose(V * 2, np.round(V * 2), atol=1e-6): return False
        if not np.allclose(W * 2, np.round(W * 2), atol=1e-6): return False
        
        U_r, V_r, W_r = round_half(U, V, W)
        result = np.zeros((n*m, m*p, p*n))
        for c in range(U.shape[1]):
            result += np.einsum('i,j,k->ijk', U_r[:,c], V_r[:,c], W_r[:,c])
        return np.allclose(result, T, atol=1e-6)

    def fast_als(U_init, V_init, W_init, target_rank, max_iters=100, tol=1e-9):
        U = U_init.copy()
        V = V_init.copy()
        W = W_init.copy()
        
        R = np.zeros_like(T)
        for c in range(target_rank):
            R += np.einsum('i,j,k->ijk', U[:,c], V[:,c], W[:,c])
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Update U
            for c in range(target_rank):
                R -= np.einsum('i,j,k->ijk', U[:,c], V[:,c], W[:,c])
                vw = V[:,c] * W[:,c]
                denom = np.sum(vw * vw)
                if denom > 1e-12:
                    U_new = np.einsum('abk,bk->a', T + R, vw) / denom
                else:
                    U_new = U[:,c]
                R += np.einsum('i,j,k->ijk', U_new, V[:,c], W[:,c])
                U[:,c] = U_new
            
            # Update V
            for c in range(target_rank):
                R -= np.einsum('i,j,k->ijk', U[:,c], V[:,c], W[:,c])
                uw = U[:,c] * W[:,c]
                denom = np.sum(uw * uw)
                if denom > 1e-12:
                    V_new = np.einsum('abk,ak->b', T + R, uw) / denom
                else:
                    V_new = V[:,c]
                R += np.einsum('i,j,k->ijk', U[:,c], V_new, W[:,c])
                V[:,c] = V_new
            
            # Update W
            for c in range(target_rank):
                R -= np.einsum('i,j,k->ijk', U[:,c], V[:,c], W[:,c])
                uv = U[:,c] * V[:,c]
                denom = np.sum(uv * uv)
                if denom > 1e-12:
                    W_new = np.einsum('abk,ab->k', T + R, uv) / denom
                else:
                    W_new = W[:,c]
                R += np.einsum('i,j,k->ijk', U[:,c], V[:,c], W_new)
                W[:,c] = W_new
            
            err = np.linalg.norm(T - R)
            if err < tol:
                break
        
        return U, V, W, R

    U_s, V_s, W_s, rank_s = strassen_rank50()
    if not verify(U_s, V_s, W_s):
        U_s, V_s, W_s, rank_s = schoolbook()
    
    best_U, best_V, best_W, best_rank = U_s, V_s, W_s, rank_s

    # Try to reduce rank
    for target_rank in range(best_rank - 1, max(best_rank - 15, 30), -1):
        if time.time() > deadline:
            break
        
        success = False
        for restart in range(10):
            if time.time() > deadline:
                break
            
            if restart == 0:
                U_init = best_U[:, :target_rank]
                V_init = best_V[:, :target_rank]
                W_init = best_W[:, :target_rank]
            elif restart == 1:
                idx = np.random.choice(best_rank, target_rank, replace=False)
                U_init = best_U[:, idx]
                V_init = best_V[:, idx]
                W_init = best_W[:, idx]
            else:
                np.random.seed(42 + restart)
                U_init = np.random.randn(n*m, target_rank)
                V_init = np.random.randn(m*p, target_rank)
                W_init = np.random.randn(p*n, target_rank)
                
            U_try, V_try, W_try, R_try = fast_als(U_init, V_init, W_try, target_rank, max_iters=80, tol=1e-10)
            
            if np.linalg.norm(T - R_try) < 1e-8:
                # Check if it can be rounded to half-integers
                if verify_exact_half(U_try, V_try, W_try):
                    best_U, best_V, best_W, best_rank = U_try, V_try, W_try, target_rank
                    success = True
                    break
                # If not half-integer, we might still accept it if the problem allows floats?
                # "integer/half-integer entries" is explicit. So we must reject non-integer.
                # However, if we find a float solution, it proves rank <= target_rank.
                # But we need to output the matrices.
                # If we can't round it, we can't use it.
                # But maybe we can try to snap it?
                U_r, V_r, W_r = round_half(U_try, V_try, W_try)
                if verify(U_r, V_r, W_r):
                    best_U, best_V, best_W, best_rank = U_r, V_r, W_r, target_rank
                    success = True
                    break

    U_out = best_U.tolist()
    V_out = best_V.tolist()
    W_out = best_W.tolist()
    
    return {"U": U_out, "V": V_out, "W": W_out}
