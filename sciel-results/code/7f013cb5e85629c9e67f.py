import numpy as np
import time
import random
from itertools import product

def construct():
    start_time = time.time()
    
    def time_left():
        return 150 - (time.time() - start_time)
        
    def schoolbook(n, m, p):
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, c] = 1
                    V[j * p + k, c] = 1
                    W[k * n + i, c] = 1
                    c += 1
        return U, V, W

    def check_decomp(U, V, W, n, m, p, tol=1e-9):
        # Check if U, V, W decompose the tensor
        # T[i, j, k] = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
        # Should be 1 for all i,j,k
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for c in range(rank):
                        val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    if abs(val - 1) > tol:
                        return False
        return True

    def tensor_error(U, V, W, n, m, p):
        """Calculate the maximum absolute error in the tensor reconstruction."""
        rank = U.shape[1]
        max_err = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for c in range(rank):
                        val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    err = abs(val - 1)
                    if err > max_err:
                        max_err = err
        return max_err

    def random_init(n, m, p, rank, scale=1.0):
        U = np.random.randn(n * m, rank) * scale
        V = np.random.randn(m * p, rank) * scale
        W = np.random.randn(p * n, rank) * scale
        return U, V, W

    def local_search(U, V, W, n, m, p, max_iter=1000, step_size=0.1, lr_decay=0.99):
        """
        Perform a simple gradient descent-like local search to minimize tensor error.
        This is a heuristic and may not converge to global minimum, but can polish.
        """
        rank = U.shape[1]
        current_err = tensor_error(U, V, W, n, m, p)
        
        # Precompute indices for efficiency
        indices = list(product(range(n), range(m), range(p)))
        
        for it in range(max_iter):
            if time_left() < 5:
                break
                
            # Calculate gradients
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            for i, j, k in indices:
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                # Current value at (i,j,k)
                val = 0
                for c in range(rank):
                    val += U[u_idx, c] * V[v_idx, c] * W[w_idx, c]
                
                err = val - 1
                
                for c in range(rank):
                    # d/dU[u_idx,c] = V[v_idx,c] * W[w_idx,c]
                    grad_U[u_idx, c] += err * V[v_idx, c] * W[w_idx, c]
                    # d/dV[v_idx,c] = U[u_idx,c] * W[w_idx,c]
                    grad_V[v_idx, c] += err * U[u_idx, c] * W[w_idx, c]
                    # d/dW[w_idx,c] = U[u_idx,c] * V[v_idx,c]
                    grad_W[w_idx, c] += err * U[u_idx, c] * V[v_idx, c]
            
            # Update
            U -= step_size * grad_U
            V -= step_size * grad_V
            W -= step_size * grad_W
            
            new_err = tensor_error(U, V, W, n, m, p)
            if new_err < current_err:
                current_err = new_err
                step_size *= lr_decay
            else:
                # Revert if error increased
                U += step_size * grad_U
                V += step_size * grad_V
                W += step_size * grad_W
                step_size *= 0.9
                if step_size < 1e-4:
                    break
            
            if current_err < 1e-9:
                break
                
        return U, V, W, current_err

    def search_decomp(n, m, p, target_rank, max_time=10):
        """
        Try to find a decomposition of rank target_rank using random init + local search.
        """
        best_U, best_V, best_W = None, None, None
        best_err = float('inf')
        
        # Number of random restarts
        n_restarts = 20
        for r in range(n_restarts):
            if time_left() < max_time:
                break
                
            U, V, W = random_init(n, m, p, target_rank, scale=0.5)
            
            # Run local search
            U, V, W, err = local_search(U, V, W, n, m, p, max_iter=500, step_size=0.05)
            
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
                if best_err < 1e-9:
                    break
        
        if best_err < 1e-9:
            return best_U, best_V, best_W
        else:
            return None

    n, m, p = 3, 5, 6
    
    # Schoolbook rank is 90
    school_U, school_V, school_W = schoolbook(n, m, p)
    
    # Try to find a better decomposition
    # Known lower bounds suggest rank could be significantly lower.
    # For 3x5x6, schoolbook is 90.
    # Let's try ranks from 80 down to 70.
    # This is a heuristic search.
    
    best_result = (school_U, school_V, school_W)
    
    # We will attempt to find a decomposition with rank < 90.
    # Start with a few random attempts at lower ranks.
    for rank in range(85, 60, -5):
        if time_left() < 20:
            break
            
        res = search_decomp(n, m, p, rank, max_time=10)
        if res is not None:
            U, V, W = res
            best_result = (U, V, W)
            print(f"Found decomposition with rank {rank}")
            break
        else:
            print(f"Failed to find decomposition with rank {rank} in time")

    U, V, W = best_result
    
    # Final verification
    if not check_decomp(U, V, W, n, m, p):
        # Fallback to schoolbook if verification fails
        U, V, W = schoolbook(n, m, p)

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    res = construct()
    U, V, W = res['U'], res['V'], res['W']
    n, m, p = 3, 5, 6
    rank = len(U[0])
    print(f"Rank: {rank}")
