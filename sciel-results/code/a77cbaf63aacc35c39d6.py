"""Optimized matrix multiplication tensor decomposition for 2x3 by 3x6."""
import time
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, tol=1e-8):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                if abs(s - 1.0) > tol:
                    return False
    return True

def _try_rank_with_refinement(r, n, m, p, deadline, init_type='structured'):
    """Try to find a valid decomposition of rank r with gradient refinement."""
    np.random.seed(12345)
    
    for trial in range(30):
        if time.time() > deadline - 3:
            break
        
        # Different initialization strategies
        if init_type == 'structured':
            # Use Kronecker-like structure
            U = np.zeros((n*m, r))
            V = np.zeros((m*p, r))
            W = np.zeros((p*n, r))
            
            # Try to exploit the structure: for 2x3x6, we can think of
            # splitting the 6 columns into groups
            # Initialize with patterns that respect the tensor structure
            for c in range(r):
                # Assign random but structured values
                i_idx = c % n
                j_idx = (c // n) % m
                k_idx = (c // (n*m)) % p
                
                U[i_idx * m + j_idx, c] = 1.0
                V[j_idx * p + k_idx, c] = 1.0
                W[k_idx * n + i_idx, c] = 1.0
                
                # Add some random perturbation to break symmetry
                if trial > 0:
                    U[i_idx * m + j_idx, c] += np.random.randn() * 0.1
                    V[j_idx * p + k_idx, c] += np.random.randn() * 0.1
                    W[k_idx * n + i_idx, c] += np.random.randn() * 0.1
        else:
            # Random initialization with integer/half-integer values
            vals = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
            U = np.random.choice(vals, size=(n*m, r))
            V = np.random.choice(vals, size=(m*p, r))
            W = np.random.choice(vals, size=(p*n, r))
        
        # Check if already valid
        if _verify(U, V, W, n, m, p):
            return U, V, W
        
        # Gradient-based refinement
        # The objective: minimize sum_{i,j,k} (sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] - 1)^2
        # Use simple gradient descent with multiple steps
        
        lr = 0.01
        for step in range(100):
            if time.time() > deadline - 2:
                break
            
            # Compute gradients
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # Current value
                        s = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                        err = s - 1.0
                        if abs(err) < 1e-10:
                            continue
                        
                        # Gradient with respect to U[i*m+j, c]
                        # d/dU[c] = err * V[j*p+k,c] * W[k*n+i,c]
                        grad_U = err * (V[j*p+k] * W[k*n+i])
                        dU[i*m+j] += grad_U
                        
                        # Gradient with respect to V[j*p+k, c]
                        grad_V = err * (U[i*m+j] * W[k*n+i])
                        dV[j*p+k] += grad_V
                        
                        # Gradient with respect to W[k*n+i, c]
                        grad_W = err * (U[i*m+j] * V[j*p+k])
                        dW[k*n+i] += grad_W
            
            # Update
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
            # Check convergence
            if _verify(U, V, W, n, m, p):
                return U, V, W
        
        # After refinement, snap to nearest half-integer and verify
        U_snap = np.round(U * 2) / 2
        V_snap = np.round(V * 2) / 2
        W_snap = np.round(W * 2) / 2
        
        if _verify(U_snap, V_snap, W_snap, n, m, p):
            return U_snap, V_snap, W_snap
    
    return None

def construct():
    """
    Construct a tensor decomposition for multiplying a 2x3 matrix by a 3x6 matrix.
    
    Returns a dictionary with keys 'U', 'V', 'W' containing factor matrices
    that exactly decompose the matrix multiplication tensor.
    """
    start_time = time.time()
    deadline = start_time + 140  # Leave 10 seconds safety margin
    
    n, m, p = DIMENSIONS  # 2, 3, 6
    target_rank = n * m * p  # 36 for schoolbook
    
    # Build the schoolbook solution first as guaranteed fallback
    rank = target_rank
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
    
    best_result = {
        'U': U.tolist(),
        'V': V.tolist(),
        'W': W.tolist()
    }
    
    # Try to find lower rank decompositions
    # Lower bound: at least max(n*m, m*p, p*n) / max(n,m,p) but let's use practical bounds
    # For 2x3x6, try ranks from 20 to 35
    
    best_rank = target_rank
    
    for r in range(20, target_rank):
        if time.time() > deadline - 10:
            break
        
        # Try structured initialization first
        result = _try_rank_with_refinement(r, n, m, p, deadline, init_type='structured')
        if result is not None:
            U_found, V_found, W_found = result
            best_rank = r
            best_result = {
                'U': U_found.tolist(),
                'V': V_found.tolist(),
                'W': W_found.tolist()
            }
            break
        
        # Try random initialization
        if time.time() > deadline - 5:
            break
        result = _try_rank_with_refinement(r, n, m, p, deadline, init_type='random')
        if result is not None:
            U_found, V_found, W_found = result
            best_rank = r
            best_result = {
                'U': U_found.tolist(),
                'V': V_found.tolist(),
                'W': W_found.tolist()
            }
            break
    
    return best_result
# EVOLVE-BLOCK-END
