"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 140  # safety margin before 150s limit

    n, m, p = DIMENSIONS
    nm = n * m  # 16
    mp = m * p  # 36
    pn = p * n  # 36

    # Schoolbook decomposition
    rank_schoolbook = n * m * p  # 144
    U_fb = np.zeros((nm, rank_schoolbook))
    V_fb = np.zeros((mp, rank_schoolbook))
    W_fb = np.zeros((pn, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j, col] = 1
                V_fb[j * p + k, col] = 1
                W_fb[k * n + i, col] = 1
                col += 1

    best_rank = rank_schoolbook
    best_U = U_fb.copy()
    best_V = V_fb.copy()
    best_W = W_fb.copy()

    def tensor_residual(U, V, W):
        """Compute the residual tensor R[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] - 1"""
        rank = U.shape[1]
        U_3d = U.reshape(n, m, rank)
        V_3d = V.reshape(m, p, rank)
        W_3d = W.reshape(p, n, rank)

        # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        UV = np.einsum('ijr,jkr->ijkr', U_3d, V_3d)
        T = np.einsum('ijkr,kir->ijk', UV, W_3d)
        return T - 1.0

    def tensor_grads(U, V, W):
        """Compute gradients of loss = sum_{i,j,k} (T[i,j,k]-1)^2 with respect to U, V, W"""
        rank = U.shape[1]
        U_3d = U.reshape(n, m, rank)
        V_3d = V.reshape(m, p, rank)
        W_3d = W.reshape(p, n, rank)

        # T[i,j,k] = sum_r U[i,j,r]*V[j,k,r]*W[k,i,r]
        UV = np.einsum('ijr,jkr->ijkr', U_3d, V_3d)
        T = np.einsum('ijkr,kir->ijk', UV, W_3d)
        err = T - 1.0

        # grad_U[i,j,r] = 2 * sum_k err[i,j,k]*V[j,k,r]*W[k,i,r]
        grad_U = 2.0 * np.einsum('ijk,jkr,kir->ijr', err, V_3d, W_3d)

        # grad_V[j,k,r] = 2 * sum_i err[i,j,k]*U[i,j,r]*W[k,i,r]
        grad_V = 2.0 * np.einsum('ijk,ijr,kir->jkr', err, U_3d, W_3d)

        # grad_W[k,i,r] = 2 * sum_j err[i,j,k]*U[i,j,r]*V[j,k,r]
        grad_W = 2.0 * np.einsum('ijk,ijr,jkr->kir', err, U_3d, V_3d)

        return grad_U.reshape(nm, rank), grad_V.reshape(mp, rank), grad_W.reshape(pn, rank)

    # Strassen-based initialization for 4x4 * 4x4 case (then extend to 4x9)
    # For 4x4*4x4, we can use 2x2 blocks with Strassen (rank 7 per block pair)
    # 4x4 = 4 blocks of 2x2, so 4*7 = 28 multiplications for the 4x4*4x4 part
    # For 4x9, we can partition as 4x(2+2+2+2+1) or use other decompositions
    
    # Let's try a block-based approach:
    # Partition A (4x4) into 2x2 blocks: A11, A12, A21, A22
    # Partition B (4x9) into 2x9 blocks: B1, B2 (each 2x9)
    # Or partition B into 2x3 blocks: B11, B12, B13, B21, B22, B23 (each 2x3)
    
    # Actually, let's use a simpler approach: try to find low-rank decompositions
    # using random restarts with good initializations
    
    # Try structured initializations based on Kronecker products
    # For 4x4 * 4x9, we can think of it as (2x2 ⊗ 2) * (2x2 ⊗ 2x(9/2)) but 9 is odd
    
    # Let's try rank values from a reasonable lower bound
    # Theoretical lower bound is max(n*m, m*p, p*n) = max(16, 36, 36) = 36
    # But we can potentially do better with clever decompositions
    
    # Try ranks from 30 to 100
    for target_rank in range(30, min(rank_schoolbook + 1, 100)):
        if time.time() > deadline:
            break

        # Try several random initializations with different strategies
        for trial in range(30):
            if time.time() > deadline:
                break

            # Different initialization strategies
            if trial < 10:
                # Small random initialization
                U = np.random.randn(nm, target_rank) * 0.1
                V = np.random.randn(mp, target_rank) * 0.1
                W = np.random.randn(pn, target_rank) * 0.1
            elif trial < 20:
                # Larger random initialization
                U = np.random.randn(nm, target_rank) * 0.5
                V = np.random.randn(mp, target_rank) * 0.5
                W = np.random.randn(pn, target_rank) * 0.5
            else:
                # Sparse initialization
                U = np.zeros((nm, target_rank))
                V = np.zeros((mp, target_rank))
                W = np.zeros((pn, target_rank))
                for r in range(target_rank):
                    # Randomly assign 1s to some entries
                    U[np.random.choice(nm, 4, replace=False), r] = np.random.choice([-1, 1], 4)
                    V[np.random.choice(mp, 4, replace=False), r] = np.random.choice([-1, 1], 4)
                    W[np.random.choice(pn, 4, replace=False), r] = np.random.choice([-1, 1], 4)

            # Adam optimizer with adaptive learning rate
            lr = 0.01
            beta1 = 0.9
            beta2 = 0.999
            eps = 1e-8
            m_U = np.zeros_like(U)
            v_U = np.zeros_like(U)
            m_V = np.zeros_like(V)
            v_V = np.zeros_like(V)
            m_W = np.zeros_like(W)
            v_W = np.zeros_like(W)
            t = 0

            max_steps = 1500
            for step in range(max_steps):
                if time.time() > deadline:
                    break

                t += 1
                gU, gV, gW = tensor_grads(U, V, W)

                # Adam update
                m_U = beta1 * m_U + (1 - beta1) * gU
                v_U = beta2 * v_U + (1 - beta2) * gU**2
                m_U_hat = m_U / (1 - beta1**t)
                v_U_hat = v_U / (1 - beta2**t)
                U -= lr * m_U_hat / (np.sqrt(v_U_hat) + eps)

                m_V = beta1 * m_V + (1 - beta1) * gV
                v_V = beta2 * v_V + (1 - beta2) * gV**2
                m_V_hat = m_V / (1 - beta1**t)
                v_V_hat = v_V / (1 - beta2**t)
                V -= lr * m_V_hat / (np.sqrt(v_V_hat) + eps)

                m_W = beta1 * m_W + (1 - beta1) * gW
                v_W = beta2 * v_W + (1 - beta2) * gW**2
                m_W_hat = m_W / (1 - beta1**t)
                v_W_hat = v_W / (1 - beta2**t)
                W -= lr * m_W_hat / (np.sqrt(v_W_hat) + eps)

                # Check convergence every 50 steps
                if step % 50 == 0:
                    T = tensor_residual(U, V, W)
                    max_err = np.max(np.abs(T))
                    if max_err < 1e-6:
                        # Found a valid decomposition
                        if target_rank < best_rank:
                            best_rank = target_rank
                            best_U = U.copy()
                            best_V = V.copy()
                            best_W = W.copy()
                        break

            if target_rank < best_rank:
                break

    # Final polish: try to improve the best solution found
    if best_rank < rank_schoolbook and time.time() < deadline:
        U = best_U.copy()
        V = best_V.copy()
        W = best_W.copy()
        
        # Additional refinement with smaller learning rate
        lr = 0.001
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8
        m_U = np.zeros_like(U)
        v_U = np.zeros_like(U)
        m_V = np.zeros_like(V)
        v_V = np.zeros_like(V)
        m_W = np.zeros_like(W)
        v_W = np.zeros_like(W)
        t = 0

        for step in range(500):
            if time.time() > deadline:
                break

            t += 1
            gU, gV, gW = tensor_grads(U, V, W)

            m_U = beta1 * m_U + (1 - beta1) * gU
            v_U = beta2 * v_U + (1 - beta2) * gU**2
            m_U_hat = m_U / (1 - beta1**t)
            v_U_hat = v_U / (1 - beta2**t)
            U -= lr * m_U_hat / (np.sqrt(v_U_hat) + eps)

            m_V = beta1 * m_V + (1 - beta1) * gV
            v_V = beta2 * v_V + (1 - beta2) * gV**2
            m_V_hat = m_V / (1 - beta1**t)
            v_V_hat = v_V / (1 - beta2**t)
            V -= lr * m_V_hat / (np.sqrt(v_V_hat) + eps)

            m_W = beta1 * m_W + (1 - beta1) * gW
            v_W = beta2 * v_W + (1 - beta2) * gW**2
            m_W_hat = m_W / (1 - beta1**t)
            v_W_hat = v_W / (1 - beta2**t)
            W -= lr * m_W_hat / (np.sqrt(v_W_hat) + eps)

        # Check if the refined solution is valid
        T = tensor_residual(U, V, W)
        max_err = np.max(np.abs(T))
        if max_err < 1e-6:
            best_U = U.copy()
            best_V = V.copy()
            best_W = W.copy()

    # Return the best found
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
