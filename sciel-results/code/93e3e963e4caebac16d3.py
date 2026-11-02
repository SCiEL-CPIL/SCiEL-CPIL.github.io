import time
import numpy as np

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110
    
    def schoolbook():
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
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    best_result = schoolbook()
    best_rank = n * m * p
    
    def is_valid(U, V, W, rank):
        # Check entries are integer or half-integer
        for mat in [U, V, W]:
            mat = np.array(mat)
            if not np.allclose(mat * 2, np.round(mat * 2), atol=1e-9):
                return False
        # Check tensor decomposition
        # T_ijk = sum_c U_ij_c V_jk_c W_ki_c
        # U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
        # Need to reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
        U_t = np.array(U).reshape(n, m, rank)
        V_t = np.array(V).reshape(m, p, rank)
        W_t = np.array(W).reshape(p, n, rank)
        
        # Compute T
        # T[i,j,k] = sum_c U_t[i,j,c] * V_t[j,k,c] * W_t[k,i,c]
        # Note: W_t is indexed by [k, i, c] where k is row (p dim), i is col (n dim)
        # In the problem statement: W[k*n+i, r]. So W_t[k, i, c] is correct.
        T = np.einsum('ijc,jkc,kic->ijk', U_t, V_t, W_t)
        
        if not np.allclose(T, 1.0, atol=1e-9):
            return False
        return True

    def optimize(rank):
        np.random.seed(rank * 100) # Fixed seed for reproducibility in search, but we vary trial
        # Actually, let's use a global trial counter for seed
        
        U = np.random.randn(n * m, rank) * 0.5
        V = np.random.randn(m * p, rank) * 0.5
        W = np.random.randn(p * n, rank) * 0.5
        
        lr = 0.01
        for it in range(200):
            if time.time() > deadline - 10:
                return None
            
            U_t = U.reshape(n, m, rank)
            V_t = V.reshape(m, p, rank)
            W_t = W.reshape(p, n, rank)
            
            T = np.einsum('ijc,jkc,kic->ijk', U_t, V_t, W_t)
            err = T - 1.0
            
            # Gradients
            # dL/dU_ijk = 2 * err_ijk * V_jk * W_ki
            grad_U = np.einsum('ijk,jkc,kic->ijc', err, V_t, W_t) * 2
            grad_V = np.einsum('ijk,ijc,kic->jkc', err, U_t, W_t) * 2
            grad_W = np.einsum('ijk,ijc,jkc->kic', err, U_t, V_t) * 2
            
            U -= lr * grad_U.reshape(n*m, rank)
            V -= lr * grad_V.reshape(m*p, rank)
            W -= lr * grad_W.reshape(p*n, rank)
            
            # Optional: check convergence or just run fixed steps
            
        # Snap to half-integers
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        
        if is_valid(U_r, V_r, W_r, rank):
            return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
        return None

    # Search for lower ranks
    # Try ranks from 59 down to 45
    for rank in range(59, 44, -1):
        if time.time() > deadline - 5:
            break
            
        found = False
        for trial in range(5):
            if time.time() > deadline - 5:
                break
            
            np.random.seed(rank * 1000 + trial)
            U = np.random.randn(n * m, rank) * 0.5
            V = np.random.randn(m * p, rank) * 0.5
            W = np.random.randn(p * n, rank) * 0.5
            
            lr = 0.005
            # Adaptive learning rate or just fixed
            for it in range(100):
                if time.time() > deadline - 5:
                    break
                    
                U_t = U.reshape(n, m, rank)
                V_t = V.reshape(m, p, rank)
                W_t = W.reshape(p, n, rank)
                
                T = np.einsum('ijc,jkc,kic->ijk', U_t, V_t, W_t)
                err = T - 1.0
                
                grad_U = np.einsum('ijk,jkc,kic->ijc', err, V_t, W_t) * 2
                grad_V = np.einsum('ijk,ijc,kic->jkc', err, U_t, W_t) * 2
                grad_W = np.einsum('ijk,ijc,jkc->kic', err, U_t, V_t) * 2
                
                U -= lr * grad_U.reshape(n*m, rank)
                V -= lr * grad_V.reshape(m*p, rank)
                W -= lr * grad_W.reshape(p*n, rank)
                
            # Snap and check
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2
            
            if is_valid(U_r, V_r, W_r, rank):
                best_rank = rank
                best_result = {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                found = True
                break
        
        if found:
            # If we found a rank, we could stop, but maybe we can find even lower?
            # The prompt says "rank-minimal". So we should try to go lower.
            # But the loop goes down. So we continue to next rank.
            pass

    return best_result
# EVOLVE-BLOCK-END
