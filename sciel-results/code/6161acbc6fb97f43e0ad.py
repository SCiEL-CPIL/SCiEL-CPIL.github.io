"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START

def _is_half_integer(x):
    """Check if x is a half-integer (multiple of 0.5) within tolerance."""
    return np.isclose(2 * x, np.round(2 * x), atol=1e-9)

def _verify_decomposition(U, V, W, n=3, m=3, p=3):
    """
    Verify that U, V, W form a valid tensor decomposition for n x m x p matrix multiplication.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    Convention: W[k*n + i, r]
    T[i, j, k] = sum_r U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
    We require T[i, j, k] = 1 if (i,j,k) corresponds to a term in standard matrix mult, else 0.
    Standard matrix mult: (AB)[i, k] = sum_j A[i, j] B[j, k]
    The tensor T is defined such that T[i, j, k] = 1 for all i, j, k? 
    Wait, the decomposition is for the *multiplication tensor*.
    The multiplication tensor M of shape (n*m, m*p, p*n) is defined by:
    M[i*m + j, j*p + k, k*n + i] = 1? No, that's the index mapping.
    
    The tensor being decomposed is the matrix multiplication tensor.
    Let the indices be (row_in, col_in, col_in, row_out).
    Actually, the standard definition is:
    We have tensor T of shape (n*m, m*p, p*n).
    T[a, b, c] = 1 if and only if the indices correspond to a valid term in the product.
    Let a = i*m + j (row i, col j of first matrix)
    Let b = j*p + k (row j, col k of second matrix)
    Let c = k*n + i (row k, col i of result... wait, result is (n, p). Index is k*n + i? No, result is n x p. Index is i*p + k?
    
    Let's check the convention given in the prompt:
    "output factor follows the official trace convention W[k*n+i,r]"
    This implies the third dimension of the tensor is indexed by c = k*n + i.
    Here i is the row of the result, k is the column of the result.
    So c = k*n + i.
    
    The first dimension is U. Shape (n*m, rank). Index a = i*m + j.
    The second dimension is V. Shape (m*p, rank). Index b = j*p + k.
    
    So the tensor T has entries T[a, b, c] where:
    a = i*m + j
    b = j*p + k
    c = k*n + i
    
    For the matrix multiplication to be correct, we must have:
    Sum over j of A[i, j] B[j, k] = C[i, k].
    The tensor T represents the bilinear map.
    T[a, b, c] should be 1 if the indices (i, j) from a, (j, k) from b, and (k, i) from c are consistent.
    i.e., if we expand a=i*m+j, b=j*p+k, c=k*n+i, then T is non-zero (specifically 1) for these triples.
    All other entries of the tensor T are 0.
    
    So verification:
    Reconstruct T from U, V, W.
    T_recon[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
    
    Check 1: For all i, j, k in range(n):
       a = i*m + j
       b = j*p + k
       c = k*n + i
       T_recon[a, b, c] must be 1.
       
    Check 2: For all other (a, b, c), T_recon[a, b, c] must be 0.
    """
    rank = U.shape[1]
    # Efficient reconstruction using Einstein summation or matrix multiplication
    # T[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
    # This is a contraction of three matrices.
    # We can compute it as: for each r, outer product of U[:, r], V[:, r], W[:, r]
    # Or: T = sum_r np.einsum('ar,br,cr->abc', U, V, W)
    
    # Given small dimensions (9, 9, 9) and rank <= 27, this is fast.
    T_recon = np.zeros((n*m, m*p, p*n), dtype=float)
    for r in range(rank):
        u = U[:, r]
        v = V[:, r]
        w = W[:, r]
        T_recon += np.einsum('a,b,c->abc', u, v, w)
    
    # Check required 1s
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                if abs(T_recon[a, b, c] - 1.0) > 1e-9:
                    return False
    
    # Check required 0s
    # Create a mask of valid indices
    mask = np.zeros((n*m, m*p, p*n), dtype=bool)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                mask[a, b, c] = True
    
    # Check that all entries where mask is False are 0
    if not np.allclose(T_recon[~mask], 0.0, atol=1e-9):
        return False
        
    return True

def _check_half_ints(U, V, W):
    """Check if all entries are half-integers."""
    if not np.all(_is_half_integer(U)): return False
    if not np.all(_is_half_integer(V)): return False
    if not np.all(_is_half_integer(W)): return False
    return True

def _round_to_half_int(x):
    """Round a matrix to the nearest half-integer."""
    return np.round(2 * x) / 2.0

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    timeout = 140  # 10s margin
    
    # 1. Start with Schoolbook (Rank 27)
    rank_sb = n * m * p
    U_sb = np.zeros((n*m, rank_sb))
    V_sb = np.zeros((m*p, rank_sb))
    W_sb = np.zeros((p*n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m + j, col] = 1
                V_sb[j*p + k, col] = 1
                W_sb[k*n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = rank_sb
    found_valid = True
    
    # 2. Try to find Rank 23 (Winograd 3x3)
    # We use a randomized local search / gradient descent approach.
    # We initialize with random half-integer matrices of rank 23.
    # Then we use gradient descent on the squared error, with rounding to half-integers.
    # This is tricky. Instead, we can try to find a decomposition over reals first, then round.
    
    # Let's try a specific known rank 23 structure or a search.
    # Since random search in 23*9*3 = 621 dimensions is hard, we might not succeed.
    # However, the "Winograd 3x3" algorithm is a specific rank 23 scheme.
    # Coefficients are typically integers or half-integers.
    
    # Let's try a heuristic search for rank 23.
    # We will attempt to optimize the tensor decomposition.
    # Objective: Minimize || T_target - T_decomp ||^2
    # Subject to: U, V, W having half-integer entries.
    
    # We'll run a few random restarts.
    # Each restart:
    # 1. Initialize U, V, W with random small integers/half-integers.
    # 2. Perform gradient steps on real values.
    # 3. Round to half-integers.
    # 4. Check validity.
    
    # To save time, we'll limit the number of restarts and steps.
    
    num_restarts = 5
    steps_per_restart = 100
    
    # Target Tensor
    T_target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T_target[a, b, c] = 1.0
                
    T_flat = T_target.flatten()
    
    def compute_error(U, V, W):
        """Compute squared error between decomposition and target tensor."""
        # T_recon = sum_r U[:,r] V[:,r] W[:,r]
        # Efficient way:
        # M1 = U @ V.T  -> (n*m, m*p)
        # Then for each c, we need to combine with W.
        # Actually, T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        # Let's construct it.
        # This is a 3-way contraction.
        # We can do:
        # For each r, add outer(u_r, v_r, w_r)
        # Or:
        # A = U @ V.T  (shape n*m, m*p)
        # Then T[:,:,r] = A * W[r, :]? No.
        # T[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
        # Let's just use the loop since rank is small (23) and dims are 9.
        T_recon = np.zeros((n*m, m*p, p*n))
        for r in range(U.shape[1]):
            T_recon += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        return np.sum((T_recon - T_target)**2)
    
    def compute_gradient(U, V, W):
        """Compute gradient of error w.r.t U, V, W."""
        # Error E = sum_{a,b,c} (T[a,b,c] - sum_r U[a,r]V[b,r]W[c,r])^2
        # dE/dU[a,r] = 2 * sum_{b,c} (T[a,b,c] - T_recon[a,b,c]) * V[b,r] * W[c,r]
        # Let R = T_target - T_recon
        # grad_U[a,r] = 2 * sum_{b,c} R[a,b,c] V[b,r] W[c,r]
        # This is equivalent to:
        # For each r: grad_U[:, r] = 2 * (R @ (V[:, r] * W[:, r].reshape... no))
        # Let's vectorize.
        # R is (n*m, m*p, p*n)
        # V is (m*p, rank)
        # W is (p*n, rank)
        
        T_recon = np.zeros((n*m, m*p, p*n))
        for r in range(U.shape[1]):
            T_recon += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        
        R = T_target - T_recon
        
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        for r in range(U.shape[1]):
            # grad_U[:, r] = 2 * sum_{b,c} R[:, b, c] V[b, r] W[c, r]
            # This is: 2 * einsum('abc,br,cr->ar', R, V, W)[:, r]
            # But we are in a loop over r, so we can just do it simply.
            # For fixed r:
            # term_U = sum_{b,c} R[:, b, c] * V[b, r] * W[c, r]
            # We can precompute M_r = R * (V[:, r, None] * W[None, :, r])? No, indices differ.
            # R[a, b, c] * V[b, r] * W[c, r]
            # Let's just use einsum for the whole thing if possible, or loop.
            # Given small sizes, a direct loop or simple einsum is fine.
            
            # Using einsum for the entire gradient calculation at once is faster.
            pass # We will use a vectorized approach below.
            
        # Vectorized Gradient Calculation:
        # grad_U[a, r] = 2 * sum_{b, c} R[a, b, c] V[b, r] W[c, r]
        # This is: 2 * einsum('abc,br,cr->ar', R, V, W)
        
        # grad_V[b, r] = 2 * sum_{a, c} R[a, b, c] U[a, r] W[c, r]
        # This is: 2 * einsum('abc,ar,cr->br', R, U, W)
        
        # grad_W[c, r] = 2 * sum_{a, b} R[a, b, c] U[a, r] V[b, r]
        # This is: 2 * einsum('abc,ar,br->cr', R, U, V)
        
        grad_U = 2 * np.einsum('abc,br,cr->ar', R, V, W)
        grad_V = 2 * np.einsum('abc,ar,cr->br', R, U, W)
        grad_W = 2 * np.einsum('abc,ar,br->cr', R, U, V)
        
        return grad_U, grad_V, grad_W

    for restart in range(num_restarts):
        if time.time() - start_time > timeout:
            break
            
        # Initialize randomly
        # To help convergence, start with small values
        scale = 1.0
        U = np.random.uniform(-scale, scale, (n*m, 23))
        V = np.random.uniform(-scale, scale, (m*p, 23))
        W = np.random.uniform(-scale, scale, (p*n, 23))
        
        # Add some structure? Random is hard.
        # Maybe initialize with rank 27 and remove 4 columns?
        # Let's try that.
        U = U_sb.copy()
        V = V_sb.copy()
        W = W_sb.copy()
        
        # Remove 4 random columns to get rank 23
        cols_to_keep = list(range(rank_sb))
        random.shuffle(cols_to_keep)
        cols_to_keep = cols_to_keep[:23]
        U = U[:, cols_to_keep]
        V = V[:, cols_to_keep]
        W = W[:, cols_to_keep]
        
        # Perturb slightly to escape local minima
        U += np.random.uniform(-0.5, 0.5, U.shape)
        V += np.random.uniform(-0.5, 0.5, V.shape)
        W += np.random.uniform(-0.5, 0.5, W.shape)
        
        lr = 0.1
        for step in range(steps_per_restart):
            if time.time() - start_time > timeout:
                break
                
            grad_U, grad_V, grad_W = compute_gradient(U, V, W)
            
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            # Periodically round to half-integers to snap to the grid
            if step % 10 == 0:
                U_rounded = _round_to_half_int(U)
                V_rounded = _round_to_half_int(V)
                W_rounded = _round_to_half_int(W)
                
                # Check if rounded version is valid
                if _verify_decomposition(U_rounded, V_rounded, W_rounded):
                    if _check_half_ints(U_rounded, V_rounded, W_rounded):
                        best_U, best_V, best_W = U_rounded, V_rounded, W_rounded
                        best_rank = 23
                        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
                
                # Update U, V, W to rounded values to stay on the manifold?
                # Or just continue with real values?
                # Usually, alternating between real optimization and projection is better.
                U = U_rounded
                V = V_rounded
                W = W_rounded

    # If rank 23 not found, return schoolbook
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

# EVOLVE-BLOCK-END
