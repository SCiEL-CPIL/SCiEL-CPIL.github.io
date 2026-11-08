"""Matrix multiplication 3x3x4 with a search for low-rank exact decomposition."""
import time
import itertools
import numpy as np
from scipy import linalg

DIMENSIONS = [3, 3, 4]

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T with T[i, j, k] = 1 if i==j and j==k... wait.
    The tensor for matrix multiplication C = A * B where A is n x m, B is m x p, C is n x p.
    C[i, k] = sum_j A[i, j] * B[j, k].
    The tensor T has entries T[i, j, k] = 1 if the term A[i,j]*B[j,k] appears in C[i,k].
    Actually, the standard form is that the tensor is the coefficient of A[i,j] B[j,k] in C[i,k].
    So T[i, k, j] = 1? Or T[i, j, k]?
    The decomposition is T = sum_r U[:,r] V[:,r] W[:,r] (outer product).
    C[i,k] = sum_r U[i,r] V[k,r] W[?,r]?
    Let's check the interface: U is (n*m, rank), V is (m*p, rank), W is (p*n, rank).
    The tensor is a 3-tensor of size (n*m, m*p, p*n)?
    The product is sum_r U[i, r] V[j, r] W[k, r].
    Here indices for U are i in n*m, for V are j in m*p, for W are k in p*n.
    This corresponds to the flattening of the tensor.
    The tensor T is defined by T[I, J, K] = 1 if I=(i,j), J=(j,k), K=(k,i) ... no.
    Let's derive from the schoolbook:
    U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1.
    The term is U[I, c] * V[J, c] * W[K, c].
    This contributes to the tensor entry T[I, J, K].
    For a given (i, j, k), I = i*m+j, J = j*p+k, K = k*n+i.
    The tensor T has non-zero entries at T[i*m+j, j*p+k, k*n+i] = 1.
    """
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0
    return T

def _verify(U, V, W):
    """Verify if U, V, W form a valid decomposition."""
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Compute the reconstructed tensor
    # T_recon[I, J, K] = sum_r U[I, r] V[J, r] W[K, r]
    T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
    
    # Target tensor
    T_target = _build_tensor(n, m, p)
    
    return np.allclose(T_recon, T_target, atol=1e-9)

def _search_rank(n, m, p, target_rank, time_limit=145.0):
    """Try to find a decomposition of the given rank."""
    start_time = time.time()
    T = _build_tensor(n, m, p)
    dim1, dim2, dim3 = n*m, m*p, p*n
    
    # We want to find U, V, W such that sum_r U[:,r] V[:,r] W[:,r] = T
    # This is equivalent to finding a rank-r decomposition of the tensor.
    
    # One approach: Fix V and W, solve for U?
    # Or fix U and V, solve for W?
    # T is dim1 x dim2 x dim3.
    # Flattening along 3: T[:,:,r] ? No.
    # Let's flatten T into a matrix T_mat of size (dim1*dim2, dim3).
    # T_mat[I*dim2+J, K] = T[I, J, K]
    # Then we want T_mat = M * N where M is (dim1*dim2, rank) and N is (rank, dim3).
    # But M and N are not independent. M rows are vectors of dim2, N rows are vectors of dim1.
    # Actually, U is (dim1, rank). V is (dim2, rank). W is (dim3, rank).
    # The decomposition is T = sum_r U[:,r] V[:,r] W[:,r].
    # In matrix form, if we flatten U and V into a matricization?
    # Let's try a different approach: Random initialization + gradient descent on the manifold?
    # Or: Fix one factor, e.g., W, and solve for U, V?
    # If W is fixed, we have a system of linear equations for U and V?
    # No, it's bilinear in U and V.
    
    # Let's try: Fix V. Then for each r, U[:,r] and W[:,r] must satisfy:
    # sum_r U[:,r] V[:,r] W[:,r] = T
    # This is hard.
    
    # Alternative: Use the fact that we can permute the factors.
    # Let's try to find a decomposition where one factor has a simple structure, e.g., V has standard basis vectors?
    # If V has standard basis vectors (i.e., V is a permutation of identity with some scaling), then rank >= dim2.
    # dim2 = m*p = 12. So rank >= 12.
    # If we can find a decomposition with rank 20 where V has many zeros, that would be great.
    
    # Let's try a random search with local optimization.
    # Initialize U, V, W randomly.
    # Define error E = || T - sum_r U[:,r] V[:,r] W[:,r] ||_F^2
    # Minimize E using gradient descent.
    
    # However, gradient descent might get stuck in local minima.
    # We will try multiple random initializations.
    
    best_score = float('inf')
    best_U, best_V, best_W = None, None, None
    
    for attempt in range(100):
        if time.time() - start_time > time_limit:
            break
            
        # Random initialization
        U = np.random.randn(dim1, target_rank)
        V = np.random.randn(dim2, target_rank)
        W = np.random.randn(dim3, target_rank)
        
        # Normalize to prevent divergence
        for r in range(target_rank):
            norm = np.linalg.norm(U[:, r]) * np.linalg.norm(V[:, r]) * np.linalg.norm(W[:, r])
            if norm > 0:
                scale = 1.0 / (norm ** (1.0/3.0))
                U[:, r] *= scale
                V[:, r] *= scale
                W[:, r] *= scale
        
        # Gradient descent
        lr = 0.01
        prev_err = float('inf')
        stuck_count = 0
        
        for step in range(1000):
            if time.time() - start_time > time_limit:
                break
                
            # Compute T_recon and Error
            T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
            error = T_recon - T
            err_val = np.linalg.norm(error)
            
            if err_val < best_score:
                best_score = err_val
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
            if err_val < 1e-9:
                return U.copy(), V.copy(), W.copy()
                
            if abs(prev_err - err_val) < 1e-12:
                stuck_count += 1
            else:
                stuck_count = 0
                
            if stuck_count > 50:
                break
                
            prev_err = err_val
            
            # Compute gradients
            # dE/dU[:, r] = sum_{j,k} V[j, r] W[k, r] * error[:, j, k] ? No.
            # E = sum_{i,j,k} (T_recon[i,j,k] - T[i,j,k])^2
            # T_recon[i,j,k] = sum_r U[i,r] V[j,r] W[k,r]
            # dE/dU[i,r] = 2 * sum_{j,k} (T_recon[i,j,k] - T[i,j,k]) * V[j,r] W[k,r]
            # dE/dV[j,r] = 2 * sum_{i,k} (T_recon[i,j,k] - T[i,j,k]) * U[i,r] W[k,r]
            # dE/dW[k,r] = 2 * sum_{i,j} (T_recon[i,j,k] - T[i,j,k]) * U[i,r] V[j,r]
            
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            # Efficient computation using einsum
            # dU[i, r] = 2 * sum_{j,k} error[i, j, k] V[j, r] W[k, r]
            dU = 2.0 * np.einsum('ijk,jr,kr->ir', error, V, W)
            dV = 2.0 * np.einsum('ijk,ir,kr->jr', error, U, W)
            dW = 2.0 * np.einsum('ijk,ir,jr->kr', error, U, V)
            
            # Update
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
            # Renormalize to maintain scale
            for r in range(target_rank):
                norm = np.linalg.norm(U[:, r]) * np.linalg.norm(V[:, r]) * np.linalg.norm(W[:, r])
                if norm > 0:
                    scale = 1.0 / (norm ** (1.0/3.0))
                    U[:, r] *= scale
                    V[:, r] *= scale
                    W[:, r] *= scale
        
        if best_score < 1e-9:
            return best_U, best_V, best_W

    if best_U is not None and best_score < 1e-9:
        return best_U, best_V, best_W
        
    return None

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Try ranks from 20 up to 35
    for target_rank in [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]:
        if time.time() - start_time > 145.0:
            break
            
        result = _search_rank(n, m, p, target_rank, time_limit=145.0)
        
        if result is not None:
            U, V, W = result
            
            # Convert to list of lists for output, ensuring integers if close
            U_out = []
            for row in U:
                U_out.append([int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row])
            V_out = []
            for row in V:
                V_out.append([int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row])
            W_out = []
            for row in W:
                W_out.append([int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row])
                
            # Verify one last time
            if _verify(U_out, V_out, W_out):
                return {"U": U_out, "V": V_out, "W": W_out}
            
    # Fallback to schoolbook
    return _schoolbook()

# EVOLVE-BLOCK-START
# (The construct function is defined above)
# EVOLVE-BLOCK-END
