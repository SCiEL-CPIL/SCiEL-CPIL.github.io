"""Matrix multiplication 2x3x9 rank-minimal decomposition using Kronecker structure."""
import time
import numpy as np

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x3 matrix by a 3x9 matrix.
    
    Strategy: Model the 3x9 multiplication as matrix multiplication over the polynomial ring
    F[x]/(x^3-1). The 9 columns are packed into 3 groups of 3, representing polynomial
    coefficients. This gives a structured decomposition we can optimize.
    
    The multiplication tensor for (2,3,9) has T[i,j,k] = 1 for all i in [0,2), j in [0,3), k in [0,9).
    We need: sum_r U[i*3+j, r] * V[j*9+k, r] * W[k*2+i, r] = 1 for all i,j,k.
    """
    n, m, p = DIMENSIONS  # n=2, m=3, p=9
    deadline = time.time() + 115  # Safety margin before 120s limit
    
    # Lower bound on rank: max(n*m, m*p, p*n) = max(6, 27, 18) = 27
    # Try ranks from 27 upward
    
    for rank in range(27, 55):
        if time.time() > deadline - 10:
            break
        
        # Try to find a decomposition of this rank
        found = _try_decompose(n, m, p, rank, deadline)
        if found is not None:
            return found
    
    # Fallback: schoolbook decomposition with rank 54
    return _schoolbook(n, m, p)

def _try_decompose(n, m, p, rank, deadline):
    """Try to find a rank decomposition using polynomial ring structure."""
    # Since p = 9 = 3*3, we can exploit the cyclic convolution structure
    # The 3x9 multiplication can be seen as 2x3 multiplication over F[x]/(x^3-1)
    
    # Base decomposition: 2x3x3 matrix multiplication has rank 6
    # Cyclic convolution of length 3 has rank 3 (using DFT)
    # Total rank: 6 * 3 = 18? But we need to be careful about the structure.
    
    # Let's use a direct approach: initialize with the Kronecker product structure
    # and optimize
    
    np.random.seed(42)
    
    # Create initial structured decomposition based on polynomial ring
    U_init, V_init, W_init = _init_polynomial_ring(n, m, p, rank)
    
    if U_init is None:
        return None
    
    # Optimize using gradient descent
    best_residual = float('inf')
    best_U, best_V, best_W = U_init.copy(), V_init.copy(), W_init.copy()
    
    lr = 0.01
    for it in range(300):
        if time.time() > deadline - 15:
            break
        
        # Compute residual
        res = _compute_residual_fast(U_init, V_init, W_init, n, m, p)
        residual_norm = np.max(np.abs(res))
        
        if residual_norm < 1e-10:
            return _finalize(U_init, V_init, W_init, n, m, p)
        
        if residual_norm < best_residual:
            best_residual = residual_norm
            best_U, best_V, best_W = U_init.copy(), V_init.copy(), W_init.copy()
        
        # Compute gradients efficiently
        dU, dV, dW = _compute_gradients(U_init, V_init, W_init, res, n, m, p)
        
        # Adaptive learning rate
        grad_norm = np.sqrt(np.sum(dU**2) + np.sum(dV**2) + np.sum(dW**2))
        if grad_norm > 1e-12:
            effective_lr = lr / (1 + grad_norm * 0.1)
            U_init -= effective_lr * dU
            V_init -= effective_lr * dV
            W_init -= effective_lr * dW
        
        # Decay learning rate
        lr *= 0.995
    
    if best_residual < 1e-8:
        return _finalize(best_U, best_V, best_W, n, m, p)
    
    return None

def _init_polynomial_ring(n, m, p, rank):
    """Initialize decomposition using polynomial ring structure F[x]/(x^3-1)."""
    # The 3x9 matrix B can be written as [B_0 | B_1 | B_2] where each B_k is 3x3
    # This corresponds to polynomial B(x) = B_0 + B_1*x + B_2*x^2 in F[x]/(x^3-1)
    
    # For 2x3 matrix multiplication over this ring:
    # (A * B)[i,k] = sum_j A[i,j] * B_j(x) mod (x^3-1)
    
    # The base 2x3x3 multiplication has rank 6 with standard decomposition
    # The cyclic convolution of length 3 can be done with rank 3 using DFT
    
    # Combined rank: 6 * 3 = 18, but we need rank >= 27
    # So we may need to use a less efficient convolution or add redundancy
    
    # Let's start with a structured initialization
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Use the fact that we can decompose into 3 blocks
    # Each block handles one coefficient of the polynomial
    for coeff in range(3):
        # For each coefficient, we need a 2x3x3 decomposition
        # Standard rank-6 decomposition for 2x3x3
        for i in range(n):
            for j in range(m):
                for k in range(3):
                    idx = coeff * 6 + i * m + j * 3 + k
                    if idx < rank:
                        U[i * m + j, idx] = 1.0
                        V[j * 3 + k, idx] = 1.0  # This indexing needs adjustment
                        W[(coeff * 3 + k) * n + i, idx] = 1.0
    
    # This is a rough initialization; let's use a better one
    # Use random initialization with structure
    np.random.seed(123)
    U = np.random.randn(n * m, rank) * 0.3
    V = np.random.randn(m * p, rank) * 0.3
    W = np.random.randn(p * n, rank) * 0.3
    
    return U, V, W

def _compute_residual_fast(U, V, W, n, m, p):
    """Compute the residual tensor efficiently."""
    res = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                res[i, j, k] = np.sum(U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :]) - 1.0
    return res

def _compute_gradients(U, V, W, res, n, m, p):
    """Compute gradients for optimization."""
    rank = U.shape[1]
    dU = np.zeros_like(U)
    dV = np.zeros_like(V)
    dW = np.zeros_like(W)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = res[i, j, k]
                if abs(r) < 1e-12:
                    continue
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                dU[u_idx, :] += r * V[v_idx, :] * W[w_idx, :]
                dV[v_idx, :] += r * U[u_idx, :] * W[w_idx, :]
                dW[w_idx, :] += r * U[u_idx, :] * V[v_idx, :]
    
    return dU, dV, dW

def _finalize(U, V, W, n, m, p):
    """Finalize and verify the decomposition."""
    # Verify
    res = _compute_residual_fast(U, V, W, n, m, p)
    if np.max(np.abs(res)) < 1e-8:
        # Try to round to half-integers
        U_rounded = np.round(U * 2) / 2
        V_rounded = np.round(V * 2) / 2
        W_rounded = np.round(W * 2) / 2
        
        res_rounded = _compute_residual_fast(U_rounded, V_rounded, W_rounded, n, m, p)
        if np.max(np.abs(res_rounded)) < 1e-6:
            return {
                "U": U_rounded.tolist(),
                "V": V_rounded.tolist(),
                "W": W_rounded.tolist()
            }
        
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    
    return None

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition with rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
