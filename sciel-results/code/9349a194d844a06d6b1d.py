import numpy as np
import time
import random

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130.0
    rng = np.random.RandomState(42)

    # Target tensor T[a,b,c] where a=i*m+j, b=j*p+k, c=k*n+i
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0

    def verify(U, V, W, atol=1e-9):
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T_recon, T, atol=atol)

    def verify_rounded(U, V, W, atol=1e-9):
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        T_recon = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
        return np.allclose(T_recon, T, atol=atol), U_r, V_r, W_r

    # ---- Schoolbook decomposition (rank 30) as baseline ----
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, c] = 1.0
                    V[j*p+k, c] = 1.0
                    W[k*n+i, c] = 1.0
                    c += 1
        return U, V, W

    # ---- Strassen 2x2x2 block ----
    def strassen_block(n2, m2, p2):
        U = np.zeros((n2*m2, 7))
        V = np.zeros((m2*p2, 7))
        W = np.zeros((p2*n2, 7))
        a00, a01, a10, a11 = 0, 1, 2, 3
        b00, b01, b10, b11 = 0, 1, 2, 3
        c00, c01, c10, c11 = 0, 1, 2, 3
        U[a00,0]+=1; U[a11,0]+=1; V[b00,0]+=1; V[b11,0]+=1
        W[c00,0]+=1; W[c11,0]+=1
        U[a10,1]+=1; U[a11,1]+=1; V[b00,1]+=1
        W[c10,1]+=1; W[c11,1]-=1
        U[a00,2]+=1; V[b01,2]+=1; V[b11,2]-=1
        W[c01,2]+=1; W[c11,2]+=1
        U[a11,3]+=1; V[b10,3]+=1; V[b00,3]-=1
        W[c00,3]+=1; W[c10,3]+=1
        U[a00,4]+=1; U[a01,4]+=1; V[b11,4]+=1
        W[c00,4]-=1; W[c01,4]+=1
        U[a10,5]+=1; U[a00,5]-=1; V[b00,5]+=1; V[b01,5]+=1
        W[c11,5]+=1
        U[a01,6]+=1; U[a11,6]-=1; V[b10,6]+=1; V[b11,6]+=1
        W[c00,6]+=1
        return U, V, W

    # ---- Schoolbook for nx1xp (rank n*p) ----
    def schoolbook_block(n2, p2):
        rank = n2 * p2
        U = np.zeros((n2*1, rank))
        V = np.zeros((1*p2, rank))
        W = np.zeros((p2*n2, rank))
        c = 0
        for i in range(n2):
            for k in range(p2):
                U[i, c] = 1.0
                V[k, c] = 1.0
                W[k*n2+i, c] = 1.0
                c += 1
        return U, V, W

    # ---- Block decomposition: Strassen(2x2x5) + Schoolbook(2x1x5) = rank 7+10=17 ----
    U1, V1, W1 = strassen_block(n, 2, p)
    U2, V2, W2 = schoolbook_block(n, p)

    U_block = np.zeros((n*m, 17))
    V_block = np.zeros((m*p, 17))
    W_block = np.zeros((p*n, 17))
    U_block[:4, :7] = U1
    V_block[:10, :7] = V1
    W_block[:, :7] = W1
    U_block[4:6, 7:] = U2
    V_block[10:15, 7:] = V2
    W_block[:, 7:] = W2

    if not verify(U_block, V_block, W_block):
        Us, Vs, Ws = schoolbook()
        return {"U": Us.tolist(), "V": Vs.tolist(), "W": Ws.tolist()}

    best_U, best_V, best_W = U_block.copy(), V_block.copy(), W_block.copy()
    best_rank = 17

    # ---- Try to reduce rank using structured approaches ----
    
    # Approach: Try rank 16 by finding a linear dependency
    # The idea: if there exists a non-trivial vector (alpha_0, ..., alpha_15) such that
    # sum_r alpha_r * U[:,r] V[:,r] W[:,r] = 0 (as a tensor), then we can eliminate one term.
    
    # Try a more efficient optimization for rank 16
    def try_optimize(target_rank, max_time=20.0, n_restarts=5):
        start = time.time()
        for restart in range(n_restarts):
            if time.time() > deadline or time.time() - start > max_time:
                break

            # Initialize near the block decomposition
            U = np.zeros((n*m, target_rank))
            V = np.zeros((m*p, target_rank))
            W = np.zeros((p*n, target_rank))

            if target_rank <= 17:
                U[:, :target_rank] = U_block[:, :target_rank]
                V[:, :target_rank] = V_block[:, :target_rank]
                W[:, :target_rank] = W_block[:, :target_rank]
            else:
                U = rng.randn(n*m, target_rank) * 0.5
                V = rng.randn(m*p, target_rank) * 0.5
                W = rng.randn(p*n, target_rank) * 0.5

            U += rng.randn(*U.shape) * 0.01
            V += rng.randn(*V.shape) * 0.01
            W += rng.randn(*W.shape) * 0.01

            # Adam-like optimizer
            lr = 0.1
            beta1, beta2, eps = 0.9, 0.999, 1e-8
            m_U = np.zeros_like(U)
            v_U = np.zeros_like(U)
            m_V = np.zeros_like(V)
            v_V = np.zeros_like(V)
            m_W = np.zeros_like(W)
            v_W = np.zeros_like(W)
            t = 0

            for step in range(500):
                if time.time() > deadline or time.time() - start > max_time:
                    break

                T_recon = np.einsum('ar,br,cr->abc', U, V, W)
                R = T_recon - T
                err = np.sum(R**2)

                if err < 1e-12:
                    ok, U_r, V_r, W_r = verify_rounded(U, V, W)
                    if ok:
                        return U_r, V_r, W_r
                    U_i = np.round(U)
                    V_i = np.round(V)
                    W_i = np.round(W)
                    T_check = np.einsum('ar,br,cr->abc', U_i, V_i, W_i)
                    if np.allclose(T_check, T, atol=1e-9):
                        return U_i, V_i, W_i
                    break

                grad_U = 2 * np.einsum('abc,br,cr->ar', R, V, W)
                grad_V = 2 * np.einsum('abc,ar,cr->br', R, U, W)
                grad_W = 2 * np.einsum('abc,ar,br->cr', R, U, V)

                t += 1
                m_U = beta1 * m_U + (1 - beta1) * grad_U
                v_U = beta2 * v_U + (1 - beta2) * grad_U**2
                m_U_hat = m_U / (1 - beta1**t)
                v_U_hat = v_U / (1 - beta2**t)
                U -= lr * m_U_hat / (np.sqrt(v_U_hat) + eps)

                m_V = beta1 * m_V + (1 - beta1) * grad_V
                v_V = beta2 * v_V + (1 - beta2) * grad_V**2
                m_V_hat = m_V / (1 - beta1**t)
                v_V_hat = v_V / (1 - beta2**t)
                V -= lr * m_V_hat / (np.sqrt(v_V_hat) + eps)

                m_W = beta1 * m_W + (1 - beta1) * grad_W
                v_W = beta2 * v_W + (1 - beta2) * grad_W**2
                m_W_hat = m_W / (1 - beta1**t)
                v_W_hat = v_W / (1 - beta2**t)
                W -= lr * m_W_hat / (np.sqrt(v_W_hat) + eps)

                lr *= 0.995

        return None

    # Try rank 16
    if time.time() < deadline:
        result = try_optimize(16, max_time=15.0, n_restarts=5)
        if result is not None:
            U_r, V_r, W_r = result
            if verify(U_r, V_r, W_r):
                best_U, best_V, best_W = U_r, V_r, W_r
                best_rank = 16

    # Try rank 15
    if time.time() < deadline and best_rank > 15:
        result = try_optimize(15, max_time=15.0, n_restarts=5)
        if result is not None:
            U_r, V_r, W_r = result
            if verify(U_r, V_r, W_r):
                best_U, best_V, best_W = U_r, V_r, W_r
                best_rank = 15

    # Final verification
    if not verify(best_U, best_V, best_W):
        best_U, best_V, best_W = U_block.copy(), V_block.copy(), W_block.copy()
        best_rank = 17
        if not verify(best_U, best_V, best_W):
            Us, Vs, Ws = schoolbook()
            return {"U": Us.tolist(), "V": Vs.tolist(), "W": Ws.tolist()}

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
