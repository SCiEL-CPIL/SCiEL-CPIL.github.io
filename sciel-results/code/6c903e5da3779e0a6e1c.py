"""Matrix multiplication 3x4x7 tensor decomposition search."""
DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 125.0
    
    # Schoolbook rank = n*m*p = 84
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1.0
                V_school[j * p + k, col] = 1.0
                W_school[k * n + i, col] = 1.0
                col += 1
    
    def make_tensor_indices():
        """Precompute all (i,j,k) indices and their positions."""
        triples = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    triples.append((i * m + j, j * p + k, k * n + i))
        return triples
    
    triples = make_tensor_indices()
    idx_u = np.array([t[0] for t in triples])
    idx_v = np.array([t[1] for t in triples])
    idx_w = np.array([t[2] for t in triples])
    
    def compute_error(U, V, W):
        """Compute sum of squared errors for the decomposition."""
        # For each triple (i,j,k), we need U[i*m+j, :] . (V[j*p+k, :] * W[k*n+i, :])
        errors = np.zeros(len(triples))
        for c in range(U.shape[1]):
            uc = U[:, c]
            vc = V[:, c]
            wc = W[:, c]
            for t in range(len(triples)):
                errors[t] += uc[idx_u[t]] * vc[idx_v[t]] * wc[idx_w[t]]
        # The target is 1.0 for each entry
        return np.sum((errors - 1.0) ** 2)
    
    def compute_error_fast(U, V, W):
        """Faster error computation using vectorization."""
        r = U.shape[1]
        # For each rank component c, compute the outer product contribution
        # sum_c U[idx_u, c] * V[idx_v, c] * W[idx_w, c]
        result = np.zeros(len(triples))
        for c in range(r):
            uc = U[idx_u, c]
            vc = V[idx_v, c]
            wc = W[idx_w, c]
            result += uc * vc * wc
        return np.sum((result - 1.0) ** 2)
    
    def try_rank(rank, max_trials=50, seed_base=0):
        """Try to find a rank decomposition with given rank."""
        best_err = float('inf')
        best_solution = None
        
        for trial in range(max_trials):
            if time.time() > deadline - 15:
                break
            
            seed = seed_base + trial * 10000
            np.random.seed(seed)
            
            # Initialize with small random values
            U = np.random.randn(n * m, rank) * 0.5
            V = np.random.randn(m * p, rank) * 0.5
            W = np.random.randn(p * n, rank) * 0.5
            
            # Gradient-based optimization
            lr = 0.01
            momentum = 0.9
            vU = np.zeros_like(U)
            vV = np.zeros_like(V)
            vW = np.zeros_like(W)
            
            for it in range(200):
                if time.time() > deadline - 10:
                    break
                
                # Compute gradient
                # error = sum_t (sum_c U[idx_u[t],c] * V[idx_v[t],c] * W[idx_w[t],c] - 1)^2
                # dE/dU[idx_u[t],c] = 2 * (sum_c' U[idx_u[t],c'] * V[idx_v[t],c'] * W[idx_w[t],c'] - 1) * V[idx_v[t],c] * W[idx_w[t],c]
                
                result = np.zeros(len(triples))
                for c in range(rank):
                    result += U[idx_u, c] * V[idx_v, c] * W[idx_w, c]
                err_t = 2.0 * (result - 1.0)  # 2*(sum - 1)
                
                dU = np.zeros_like(U)
                dV = np.zeros_like(V)
                dW = np.zeros_like(W)
                
                for c in range(rank):
                    # dU[idx_u[t], c] += err_t[t] * V[idx_v[t], c] * W[idx_w[t], c]
                    contrib = err_t * V[idx_v, c] * W[idx_w, c]
                    # Scatter to dU
                    for t in range(len(triples)):
                        dU[idx_u[t], c] += contrib[t]
                    
                    contrib = err_t * U[idx_u, c] * W[idx_w, c]
                    for t in range(len(triples)):
                        dV[idx_v[t], c] += contrib[t]
                    
                    contrib = err_t * U[idx_u, c] * V[idx_v, c]
                    for t in range(len(triples)):
                        dW[idx_w[t], c] += contrib[t]
                
                # Momentum update
                vU = momentum * vU + lr * dU
                vV = momentum * vV + lr * dV
                vW = momentum * vW + lr * dW
                
                U -= vU
                V -= vV
                W -= vW
                
                # Check convergence
                if it % 20 == 0:
                    current_err = compute_error_fast(U, V, W)
                    if current_err < best_err:
                        best_err = current_err
                        best_solution = (U.copy(), V.copy(), W.copy())
                    if current_err < 1e-10:
                        return best_solution, current_err
                
                # Adaptive learning rate
                if it > 50 and lr > 0.001:
                    lr *= 0.95
        
        # Final check
        if best_solution is not None:
            final_err = compute_error_fast(best_solution[0], best_solution[1], best_solution[2])
            if final_err < 1e-6:
                return best_solution, final_err
        
        return None, best_err
    
    # Try ranks from low to high
    # Lower bounds: For matrix multiplication of n x m and m x p, the rank is at least max(n*m, m*p, p*n)
    # Actually, the lower bound is max(n*m, m*p, p*n) = max(12, 28, 21) = 28
    # But better lower bounds exist. For rectangular matrix mult, rank >= max(n*m, m*p, p*n)
    # Also rank >= ceil(n*p * min(n,m,p) / max(n,m,p)) type bounds
    # Let's start from a reasonable lower bound
    
    # Known: for 3x4 x 4x7, the schoolbook is 84
    # The bilinear complexity is at least max(n*m, m*p, p*n) = 28
    # Let's try from 30 upward
    
    for rank in range(30, rank_school + 1):
        if time.time() > deadline - 30:
            break
        
        solution, err = try_rank(rank, max_trials=30, seed_base=rank * 1000)
        if solution is not None:
            U, V, W = solution
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
# EVOLVE-BLOCK-END
