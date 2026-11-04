"""Matrix multiplication 3,3,8 rank-minimal exact tensor decomposition."""
import time
import numpy as np
from scipy.optimize import linprog

DIMENSIONS = [3, 3, 8]

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix by a 3x8 matrix.
    
    The tensor T[i,j,k] = 1 for all (i,j,k) in [0,3) x [0,3) x [0,8) is the multiplication tensor.
    We seek rank r and matrices U (9 x r), V (24 x r), W (24 x r) such that:
      T = sum_c U[:,c] * V[:,c] * W[:,c]  (tensor outer product)
    
    The output factor W follows the official trace convention W[k*n+i, r].
    
    For matrix multiplication A (n x m) * B (m x p) = C (n x p):
    - U has shape (n*m, rank) = (9, rank)
    - V has shape (m*p, rank) = (24, rank)
    - W has shape (p*n, rank) = (24, rank)
    
    The multiplication tensor T is a 3D tensor of shape (n, m, p) = (3, 3, 8) where T[i,j,k] = 1.
    A rank-r decomposition means T[i,j,k] = sum_{c=1}^{r} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    """
    n, m, p = DIMENSIONS
    nm = n * m  # 9
    mp = m * p  # 24
    pn = p * n  # 24
    
    # The multiplication tensor T[i,j,k] = 1 for all i,j,k
    # We need to find rank-r decomposition
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # This is equivalent to: for each (i,j,k), the equation is
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # Let's try to find a low-rank decomposition using linear programming.
    # The constraint is that the tensor must be exactly decomposed.
    
    # Strategy: Try ranks from the theoretical minimum upward.
    # Theoretical minimum for matrix multiplication is at least max(n*m, m*p, p*n) / something
    # For 3x3 * 3x8, the tensor has 72 entries.
    
    # Let's use a different approach: we know that the standard algorithm uses rank 72.
    # But we can potentially do better.
    
    # For small matrices, we can try to find exact decompositions using LP.
    # The key insight: if we fix the structure, we can solve for the coefficients.
    
    # Let's try a systematic search for low-rank decompositions.
    # We'll use the fact that for matrix multiplication, there are known lower bounds.
    
    # For 3x3 * 3x3, the minimum rank is 19 (Strassen-like).
    # For 3x3 * 3x8, we need to find the minimum.
    
    # Let's try to use a linear programming approach.
    # We'll parameterize the decomposition and solve for the coefficients.
    
    # Approach: Use the fact that the multiplication tensor can be decomposed as
    # T = sum_c u_c * v_c * w_c
    # where u_c is in R^{n*m}, v_c is in R^{m*p}, w_c is in R^{p*n}
    
    # We can set up an LP to find if a rank-r decomposition exists.
    # But this is nonlinear due to the product structure.
    
    # Alternative: Use the known result that for 3x3 matrix multiplication, rank 19 works.
    # For 3x3 * 3x8, we can potentially use a similar structure.
    
    # Let's try a heuristic approach: start with a known decomposition and try to reduce rank.
    
    # First, let's create the standard schoolbook decomposition as a baseline.
    def make_schoolbook():
        rank = n * m * p  # 72
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W, rank
    
    U, V, W, rank = make_schoolbook()
    
    # Now try to reduce the rank by finding linear dependencies.
    # We can try to find a smaller set of rank-1 tensors that sum to T.
    
    # For the 3x3 case, we know that rank 19 is achievable.
    # Let's try to adapt known decompositions.
    
    # Actually, let's try a different approach: use random initialization and gradient-based optimization
    # to find a low-rank decomposition.
    
    # But first, let's check if we can find an exact decomposition with a smaller rank.
    
    # The multiplication tensor for 3x3*3x8 has a specific structure.
    # Let's try to use the fact that we can share computation across the 8 columns.
    
    # For each output column k, we need to compute A * B[:,k] which is a 3x3 * 3x1 multiplication.
    # If we have a rank-r decomposition for 3x3*3x1, we can use it 8 times.
    
    # For 3x3*3x1, the minimum rank is 7 (I think).
    # So for 3x3*3x8, we might need 8*7 = 56, but we can share some computation.
    
    # Let's try to find a good decomposition using optimization.
    
    # We'll use a random restart approach with local optimization.
    
    def try_decomposition(target_rank, max_iters=1000):
        """Try to find a rank-target_rank decomposition using optimization."""
        # Initialize randomly
        np.random.seed(42)
        U = np.random.randn(nm, target_rank)
        V = np.random.randn(mp, target_rank)
        W = np.random.randn(pn, target_rank)
        
        # Scale to get the right magnitude
        U /= np.sqrt(target_rank)
        V /= np.sqrt(target_rank)
        W /= np.sqrt(target_rank)
        
        best_loss = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iters):
            # Compute the current tensor
            T_current = np.zeros((n, m, p))
            for c in range(target_rank):
                T_current += np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
            
            # Compute loss
            T_target = np.ones((n, m, p))
            loss = np.sum((T_current - T_target) ** 2)
            
            if loss < best_loss:
                best_loss = loss
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if loss < 1e-10:
                break
            
            # Gradient computation
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            for c in range(target_rank):
                # dT/dU[:,c] = V[:,c] * W[:,c] (outer product in j,k)
                diff = T_target - T_current
                grad_U[:, c] = -2 * np.einsum('ijk,j,k->i', diff, V[:, c], W[:, c])
                grad_V[:, c] = -2 * np.einsum('ijk,i,k->j', diff, U[:, c], W[:, c])
                grad_W[:, c] = -2 * np.einsum('ijk,i,j->k', diff, U[:, c], V[:, c])
            
            # Gradient descent with adaptive learning rate
            lr = 0.01 / (1 + it * 0.01)
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
        
        return best_U, best_V, best_W, best_loss
    
    # Try different ranks
    for target_rank in [24, 32, 40, 48, 56, 64, 72]:
        U_cand, V_cand, W_cand, loss = try_decomposition(target_rank, max_iters=200)
        
        if loss < 1e-8:
            # Verify the decomposition
            T_check = np.zeros((n, m, p))
            for c in range(target_rank):
                T_check += np.einsum('i,j,k->ijk', U_cand[:, c], V_cand[:, c], W_cand[:, c])
            
            if np.allclose(T_check, np.ones((n, m, p)), atol=1e-6):
                # Round to exact values if possible
                U_exact = np.round(U_cand * 2) / 2
                V_exact = np.round(V_cand * 2) / 2
                W_exact = np.round(W_cand * 2) / 2
                
                # Verify exactness
                T_exact = np.zeros((n, m, p))
                for c in range(target_rank):
                    T_exact += np.einsum('i,j,k->ijk', U_exact[:, c], V_exact[:, c], W_exact[:, c])
                
                if np.allclose(T_exact, np.ones((n, m, p)), atol=1e-6):
                    return {
                        "U": U_exact.tolist(),
                        "V": V_exact.tolist(),
                        "W": W_exact.tolist()
                    }
    
    # If we couldn't find a better decomposition, return the schoolbook one
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
