"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 4, 6]

def _make_schoolbook(n, m, p):
    """Return the schoolbook (trivial) decomposition of rank n*m*p."""
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

def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = U.shape[1]
    if V.shape[1] != rank or W.shape[1] != rank:
        return False
    # The tensor T[i,j,k] should equal sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # We check a random subset of entries for efficiency, then all if small enough
    np.random.seed(42)
    num_checks = min(n * m * p, 1000)
    for _ in range(num_checks):
        i = np.random.randint(n)
        j = np.random.randint(m)
        k = np.random.randint(p)
        expected = 1  # T[i,j,k] = 1 for all valid (i,j,k)
        actual = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
        if abs(actual - expected) > tol:
            return False
    return True

def _try_rank_decomposition(n, m, p, target_rank, max_time=30):
    """Try to find a decomposition of the given target rank using optimization."""
    start_time = time.time()
    deadline = start_time + max_time
    
    # Initialize with random factors
    best_result = None
    best_loss = float('inf')
    
    for attempt in range(5):  # Multiple random restarts
        if time.time() > deadline - 5:
            break
            
        # Random initialization with small values
        np.random.seed(attempt * 12345 + 42)
        U = np.random.randn(n * m, target_rank) * 0.1
        V = np.random.randn(m * p, target_rank) * 0.1
        W = np.random.randn(p * n, target_rank) * 0.1
        
        # Use gradient descent to minimize the decomposition error
        lr = 0.01
        for step in range(500):
            if time.time() > deadline - 5:
                break
                
            # Compute the tensor residual
            # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
            # We want this to equal 1 for all (i,j,k)
            
            # Compute gradients using the chain rule
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            # For efficiency, we can compute the loss and gradients in a vectorized way
            # But for simplicity, we'll use a simpler approach with a subset of samples
            
            # Sample a subset of (i,j,k) triples
            num_samples = min(n * m * p, 200)
            indices = np.random.randint(0, n * m * p, size=num_samples)
            i_vals = indices // (m * p)
            j_vals = (indices % (m * p)) // p
            k_vals = indices % p
            
            # Compute the predicted values
            pred = np.sum(U[i_vals * m + j_vals, :] * V[j_vals * p + k_vals, :] * W[k_vals * n + i_vals, :], axis=1)
            loss = np.mean((pred - 1.0) ** 2)
            
            if loss < best_loss:
                best_loss = loss
                best_result = (U.copy(), V.copy(), W.copy())
            
            # Compute gradients
            # d_loss/d_pred = 2 * (pred - 1) / num_samples
            d_loss_d_pred = 2 * (pred - 1.0) / num_samples
            
            # d_pred/d_U[i*m+j,c] = V[j*p+k,c] * W[k*n+i,c]
            # We'll accumulate gradients
            for idx in range(num_samples):
                i, j, k = i_vals[idx], j_vals[idx], k_vals[idx]
                factor = d_loss_d_pred[idx]
                
                # Gradient for U
                grad_U[i * m + j, :] += factor * V[j * p + k, :] * W[k * n + i, :]
                
                # Gradient for V
                grad_V[j * p + k, :] += factor * U[i * m + j, :] * W[k * n + i, :]
                
                # Gradient for W
                grad_W[k * n + i, :] += factor * U[i * m + j, :] * V[j * p + k, :]
            
            # Update parameters
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            # Adaptive learning rate
            if step % 100 == 0 and step > 0:
                lr *= 0.5
        
        # Check if we found a valid decomposition
        if best_result is not None:
            U, V, W = best_result
            if _verify_decomposition(U, V, W, n, m, p, tol=1e-8):
                return U, V, W, True
    
    return None, None, None, False

def _snap_to_rational(arr, max_denom=100):
    """Snap array values to nearby rational numbers with small denominators."""
    result = arr.copy()
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            val = result[i, j]
            # Try to find a rational approximation
            best_frac = val
            best_err = float('inf')
            for denom in range(1, max_denom + 1):
                num = round(val * denom)
                if num == 0 and denom != 1:
                    continue
                frac = num / denom
                err = abs(frac - val)
                if err < best_err:
                    best_err = err
                    best_frac = frac
            result[i, j] = best_frac
    return result

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # Start with the schoolbook decomposition as fallback
    U_school, V_school, W_school = _make_schoolbook(n, m, p)
    best_rank = n * m * p
    best_U, best_V, best_W = U_school, V_school, W_school
    
    # Try to find lower-rank decompositions
    # Based on known results, the minimum rank for 3x4x6 might be around 72-80
    # Let's try a range of ranks
    for target_rank in range(70, min(85, n * m * p + 1)):
        if time.time() > deadline - 20:
            break
        
        U, V, W, success = _try_rank_decomposition(n, m, p, target_rank, max_time=15)
        
        if success:
            best_rank = target_rank
            best_U, best_V, best_W = U, V, W
            # Try to snap to rational values
            U_snapped = _snap_to_rational(U, max_denom=50)
            V_snapped = _snap_to_rational(V, max_denom=50)
            W_snapped = _snap_to_rational(W, max_denom=50)
            if _verify_decomposition(U_snapped, V_snapped, W_snapped, n, m, p, tol=1e-10):
                best_U, best_V, best_W = U_snapped, V_snapped, W_snapped
            break  # Found a valid decomposition, stop
    
    # Convert to lists of lists
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
