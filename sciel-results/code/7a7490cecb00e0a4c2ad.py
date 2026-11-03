"""Matrix multiplication 2,4,7 tensor decomposition using linear algebra solver."""
import time
import numpy as np

DIMENSIONS = [2, 4, 7]
DEADLINE = 140  # Safety margin

def _build_multiplication_tensor(n, m, p):
    """
    Build the multiplication tensor T of shape (n, m, p) where T[i, j, k] = 1.
    This represents the identity map in the tensor decomposition context.
    """
    T = np.ones((n, m, p), dtype=np.float64)
    return T

def _try_decomposition(n, m, p, rank, max_iter=100):
    """
    Try to find a rank-r decomposition for the n x m x p multiplication tensor.
    
    We seek U (n*m x r), V (m*p x r), W (p*n x r) such that:
    sum_{l=1}^r U[i, j, l] * V[j, k, l] * W[k, i, l] = 1 for all i, j, k
    
    To solve this, we fix V to be a random matrix (or identity-like) to break gauge symmetry,
    then solve for U and W. However, this is a quadratic problem in U and W.
    
    A better approach: 
    The tensor T can be written as sum_l u_l v_l^T w_l^T (in some basis).
    Actually, the standard decomposition is:
    T_{ijk} = sum_l U_{ij}^l V_{jk}^l W_{ki}^l
    
    Let's vectorize.
    Fix V. Then for each l, we have a term.
    This is still non-linear.
    
    Alternative: Use the fact that if we fix U and V, W is determined linearly?
    No, it's trilinear.
    
    Let's use a different strategy: 
    Iterate.
    1. Initialize U, V, W randomly.
    2. Fix U and V, solve for W (linear system).
    3. Fix V and W, solve for U (linear system).
    4. Fix U and W, solve for V (linear system).
    Repeat until convergence.
    """
    # Dimensions
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    # Initialize
    U = np.random.randn(dim_u, rank)
    V = np.random.randn(dim_v, rank)
    W = np.random.randn(dim_w, rank)
    
    # Target tensor T[i, j, k] = 1
    # We want: sum_l U[i*m+j, l] * V[j*p+k, l] * W[k*n+i, l] = 1
    
    # Precompute indices
    indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                indices.append((i, j, k))
    
    n_eq = len(indices)
    
    for it in range(max_iter):
        # Solve for W given U, V
        # For each (i, j, k), sum_l U[i,j,l] V[j,k,l] W[k,i,l] = 1
        # This is not a simple linear system for W because W indices are permuted.
        # Let's build the matrix A_W such that A_W @ W_flat = b
        # W_flat is (p*n*rank).
        # Actually, let's solve for W column by column? No.
        
        # Let's use the gradient descent or simple alternating least squares.
        # For fixed U, V, the equation for W is:
        # sum_l W[k*n+i, l] * (U[i*m+j, l] * V[j*p+k, l]) = 1
        # This couples different l's? No, for a fixed (i,j,k), the sum is over l.
        # But W[k*n+i, l] appears in many equations.
        
        # Let's construct the linear system for W.
        # Variables: W[0,0], W[0,1], ..., W[p*n-1, rank-1]
        # Equation for (i,j,k): sum_{l=0}^{rank-1} W[k*n+i, l] * U[i*m+j, l] * V[j*p+k, l] = 1
        
        A_W = np.zeros((n_eq, dim_w * rank))
        b_W = np.ones(n_eq)
        
        for idx, (i, j, k) in enumerate(indices):
            for l in range(rank):
                w_idx = (k * n + i) * rank + l
                u_val = U[i * m + j, l]
                v_val = V[j * p + k, l]
                A_W[idx, w_idx] = u_val * v_val
        
        # Solve A_W @ W_flat = b_W
        try:
            W_flat, _, _, _ = np.linalg.lstsq(A_W, b_W, rcond=None)
            W = W_flat.reshape(dim_w, rank)
        except:
            break
            
        # Solve for U given V, W
        # Equation: sum_l U[i*m+j, l] * V[j*p+k, l] * W[k*n+i, l] = 1
        A_U = np.zeros((n_eq, dim_u * rank))
        b_U = np.ones(n_eq)
        
        for idx, (i, j, k) in enumerate(indices):
            for l in range(rank):
                u_idx = (i * m + j) * rank + l
                v_val = V[j * p + k, l]
                w_val = W[k * n + i, l]
                A_U[idx, u_idx] = v_val * w_val
                
        try:
            U_flat, _, _, _ = np.linalg.lstsq(A_U, b_U, rcond=None)
            U = U_flat.reshape(dim_u, rank)
        except:
            break
            
        # Solve for V given U, W
        # Equation: sum_l U[i*m+j, l] * V[j*p+k, l] * W[k*n+i, l] = 1
        A_V = np.zeros((n_eq, dim_v * rank))
        b_V = np.ones(n_eq)
        
        for idx, (i, j, k) in enumerate(indices):
            for l in range(rank):
                v_idx = (j * p + k) * rank + l
                u_val = U[i * m + j, l]
                w_val = W[k * n + i, l]
                A_V[idx, v_idx] = u_val * w_val
                
        try:
            V_flat, _, _, _ = np.linalg.lstsq(A_V, b_V, rcond=None)
            V = V_flat.reshape(dim_v, rank)
        except:
            break
            
        # Check convergence
        # Compute residual
        residual = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for l in range(rank):
                        val += U[i * m + j, l] * V[j * p + k, l] * W[k * n + i, l]
                    residual += (val - 1.0) ** 2
                    if time.time() > DEADLINE:
                        return None
        
        if residual < 1e-10:
            # Verify exactness
            if _verify(U, V, W, n, m, p):
                return U, V, W
                
    return None

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify if the decomposition is exact."""
    rank = U.shape[1]
    result = np.zeros((n, m, p), dtype=np.float64)
    for r in range(rank):
        U_r = U.reshape(n, m, rank)[:, :, r]
        V_r = V.reshape(m, p, rank)[:, :, r]
        W_r = W.reshape(p, n, rank)[:, :, r]
        result += np.einsum('ij,jk,ki->ijk', U_r, V_r, W_r)
    
    if np.max(np.abs(result - 1.0)) > tol:
        return False
    return True

def _construct_schoolbook(n, m, p):
    """Return the schoolbook (rank n*m*p) decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
                
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    best_U, best_V, best_W = None, None, None
    best_rank = n * m * p
    
    # Try ranks from lower to higher
    # Lower bound is max(n*m, m*p, p*n) = max(8, 28, 14) = 28
    # Upper bound is schoolbook = 56
    # Strassen gave 50.
    
    min_rank = max(n * m, m * p, p * n)
    
    # Try a few specific ranks
    # 28 is the theoretical lower bound.
    # 32, 36, 40, 44, 48, 50, 56
    
    candidate_ranks = [28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56]
    
    # Filter out ranks > schoolbook
    candidate_ranks = [r for r in candidate_ranks if r <= best_rank]
    
    # Sort by rank
    candidate_ranks.sort()
    
    # To speed up, we can try to find a solution for the smallest rank first.
    # But ALS might be slow.
    
    # Let's try to solve for rank = 28 first.
    # If it fails, try 32, etc.
    
    for rank in candidate_ranks:
        if time.time() > DEADLINE - 5:
            break
            
        # Try multiple initializations for the current rank
        found = False
        for init in range(5):
            if time.time() > DEADLINE - 5:
                break
                
            np.random.seed(42 + rank + init)
            res = _try_decomposition(n, m, p, rank, max_iter=50)
            if res is not None:
                U, V, W = res
                if U.shape[1] < best_rank:
                    best_rank = U.shape[1]
                    best_U, best_V, best_W = U, V, W
                found = True
                break # Found a solution for this rank, move to next (lower) rank? No, we are going up.
                # Actually we want the MINIMAL rank. So if we find one at rank R, we should check if R-2 works?
                # We are iterating from low to high. So the first one found is the minimal.
        
        if found:
            break
            
    if best_U is None:
        # Fallback to schoolbook
        best_U, best_V, best_W = _construct_schoolbook(n, m, p)
        best_rank = n * m * p
        
    # Convert to lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
