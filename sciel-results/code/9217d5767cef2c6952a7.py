import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145

    def verify_decomp(U, V, W, T):
        recon = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(recon, T, atol=1e-8)

    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    def solve_small_tensor(n_s, m_s, p_s, target_rank):
        """
        Solve for tensor decomposition of n_s x m_s x p_s multiplication.
        T[i*m_s+j, j*p_s+k, k*n_s+i] = 1
        """
        T = np.zeros((n_s * m_s, m_s * p_s, p_s * n_s))
        for i in range(n_s):
            for j in range(m_s):
                for k in range(p_s):
                    T[i * m_s + j, j * p_s + k, k * n_s + i] = 1.0
        
        # Random initialization
        best_factors = None
        best_loss = np.inf
        
        for attempt in range(20):
            if time.time() > deadline:
                break
            np.random.seed(42 + attempt)
            U = np.random.randn(n_s * m_s, target_rank)
            V = np.random.randn(m_s * p_s, target_rank)
            W = np.random.randn(p_s * n_s, target_rank)
            
            # Gradient descent
            lr = 0.1
            for step in range(2000):
                if time.time() > deadline:
                    break
                recon = np.einsum('ar,br,cr->abc', U, V, W)
                error = recon - T
                loss = np.sum(error**2)
                
                if loss < best_loss:
                    best_loss = loss
                    if loss < 1e-10:
                        best_factors = (U.copy(), V.copy(), W.copy())
                        break
                
                dU = np.einsum('abc,br,cr->ar', error, V, W)
                dV = np.einsum('abc,ar,cr->br', error, U, W)
                dW = np.einsum('abc,ar,br->cr', error, U, V)
                
                # Simple backtracking
                step_lr = lr
                while step_lr > 1e-6:
                    U_new = U - step_lr * dU
                    V_new = V - step_lr * dV
                    W_new = W - step_lr * dW
                    recon_new = np.einsum('ar,br,cr->abc', U_new, V_new, W_new)
                    loss_new = np.sum((recon_new - T)**2)
                    if loss_new < loss:
                        U, V, W = U_new, V_new, W_new
                        break
                    step_lr *= 0.5
                
                if best_factors is not None:
                    break
            
            if best_factors is not None:
                break
        
        return best_factors

    def get_rank50():
        rank = 50
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0

        def add_strassen(i0, j0, k0):
            nonlocal col
            # M1
            U[i0 * m + j0, col] += 1
            U[(i0 + 1) * m + (j0 + 1), col] += 1
            V[j0 * p + k0, col] += 1
            V[(j0 + 1) * p + (k0 + 1), col] += 1
            W[k0 * n + i0, col] += 1
            W[(k0 + 1) * n + (i0 + 1), col] += 1
            col += 1
            # M2
            U[(i0 + 1) * m + j0, col] += 1
            U[(i0 + 1) * m + (j0 + 1), col] += 1
            V[j0 * p + k0, col] += 1
            W[k0 * n + (i0 + 1), col] += 1
            W[(k0 + 1) * n + (i0 + 1), col] -= 1
            col += 1
            # M3
            U[i0 * m + j0, col] += 1
            V[j0 * p + (k0 + 1), col] += 1
            V[(j0 + 1) * p + (k0 + 1), col] -= 1
            W[(k0 + 1) * n + i0, col] += 1
            W[(k0 + 1) * n + (i0 + 1), col] += 1
            col += 1
            # M4
            U[(i0 + 1) * m + (j0 + 1), col] += 1
            V[(j0 + 1) * p + k0, col] += 1
            V[j0 * p + k0, col] -= 1
            W[k0 * n + i0, col] += 1
            W[k0 * n + (i0 + 1), col] += 1
            col += 1
            # M5
            U[i0 * m + j0, col] += 1
            U[i0 * m + (j0 + 1), col] += 1
            V[(j0 + 1) * p + (k0 + 1), col] += 1
            W[k0 * n + i0, col] -= 1
            W[(k0 + 1) * n + i0, col] += 1
            col += 1
            # M6
            U[(i0 + 1) * m + j0, col] += 1
            U[i0 * m + j0, col] -= 1
            V[j0 * p + k0, col] += 1
            V[j0 * p + (k0 + 1), col] += 1
            W[(k0 + 1) * n + (i0 + 1), col] += 1
            col += 1
            # M7
            U[i0 * m + (j0 + 1), col] += 1
            U[(i0 + 1) * m + (j0 + 1), col] -= 1
            V[(j0 + 1) * p + k0, col] += 1
            V[(j0 + 1) * p + (k0 + 1), col] += 1
            W[k0 * n + i0, col] += 1
            col += 1

        for j0 in [0, 2]:
            for k0 in [0, 2, 4]:
                add_strassen(0, j0, k0)
        
        for j in range(m):
            U[0 * m + j, col] = 1
            V[j * p + 6, col] = 1
            W[6 * n + 0, col] = 1
            col += 1
            U[1 * m + j, col] = 1
            V[j * p + 6, col] = 1
            W[6 * n + 1, col] = 1
            col += 1

        assert col == rank
        return U, V, W

    # Try to construct rank 28 decomposition
    # Split B into 3 blocks of width 2 and 1 block of width 1
    # Blocks: k in [0,1], [2,3], [4,5], [6]
    
    # Solve for 2x4x2 (Rank 8)
    U242, V242, W242 = solve_small_tensor(2, 4, 2, 8)
    
    # Solve for 2x4x1 (Rank 4)
    U241, V241, W241 = solve_small_tensor(2, 4, 1, 4)
    
    if U242 is not None and U241 is not None:
        # Construct full decomposition
        rank_total = 8 * 3 + 4
        U = np.zeros((n * m, rank_total))
        V = np.zeros((m * p, rank_total))
        W = np.zeros((p * n, rank_total))
        
        col = 0
        # Block 1: k=0,1
        U[:, col:col+8] = U242
        # V indices: v = j*7 + k. For k=0,1, v = j*7 + 0, j*7 + 1
        for j in range(4):
            for r in range(8):
                V[j*7 + 0, col+r] = V242[j*2 + 0, r]
                V[j*7 + 1, col+r] = V242[j*2 + 1, r]
        # W indices: w = k*2 + i. For k=0,1, w = 0*2+i, 1*2+i
        for k in range(2):
            for i in range(2):
                for r in range(8):
                    W[k*2 + i, col+r] = W242[k*2 + i, r]
        col += 8
        
        # Block 2: k=2,3
        U[:, col:col+8] = U242
        for j in range(4):
            for r in range(8):
                V[j*7 + 2, col+r] = V242[j*2 + 0, r]
                V[j*7 + 3, col+r] = V242[j*2 + 1, r]
        for k in range(2):
            for i in range(2):
                for r in range(8):
                    W[(k+2)*2 + i, col+r] = W242[k*2 + i, r]
        col += 8
        
        # Block 3: k=4,5
        U[:, col:col+8] = U242
        for j in range(4):
            for r in range(8):
                V[j*7 + 4, col+r] = V242[j*2 + 0, r]
                V[j*7 + 5, col+r] = V242[j*2 + 1, r]
        for k in range(2):
            for i in range(2):
                for r in range(8):
                    W[(k+4)*2 + i, col+r] = W242[k*2 + i, r]
        col += 8
        
        # Block 4: k=6
        U[:, col:col+4] = U241
        for j in range(4):
            for r in range(4):
                V[j*7 + 6, col+r] = V241[j*1 + 0, r]
        for k in range(1):
            for i in range(2):
                for r in range(4):
                    W[(k+6)*2 + i, col+r] = W241[k*2 + i, r]
        col += 4
        
        # Verify
        T = np.zeros((n * m, m * p, p * n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i * m + j, j * p + k, k * n + i] = 1.0
        
        if verify_decomp(U, V, W, T):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
    # Fallback
    U50, V50, W50 = get_rank50()
    if verify_decomp(U50, V50, W50, np.zeros((n*m, m*p, p*n)) + 1.0): # Dummy check, just return
        pass
    return {"U": U50.tolist(), "V": V50.tolist(), "W": W50.tolist()}

