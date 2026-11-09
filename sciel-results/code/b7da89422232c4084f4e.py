"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def _check_decomposition(U, V, W, n, m, p):
    """
    Verify that U, V, W exactly decompose the matrix multiplication tensor.
    U: (n*m, r), V: (m*p, r), W: (p*n, r)
    Convention: W[k*n + i, c]
    The tensor T[i,j,k] = 1 should equal sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    """
    r = U.shape[1]
    # Build the tensor via einsum
    # U_ijk where U is indexed by (i*m+j), V by (j*p+k), W by (k*n+i)
    # We need: T[i,j,k] = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
    U_reshaped = U.reshape(n, m, r)
    V_reshaped = V.reshape(m, p, r)
    W_reshaped = W.reshape(p, n, r)
    
    # T[i,j,k] = sum_c U_reshaped[i,j,c] * V_reshaped[j,k,c] * W_reshaped[k,i,c]
    # Use einsum: 'ijc,jkc,kic->ijk'
    T = np.einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped, optimize=True)
    
    # Check if T is exactly all ones (within floating point tolerance for half-integers)
    # Since entries are integer/half-integer, the sum should be exactly 1.0
    # Use a small tolerance for floating point
    if not np.allclose(T, 1.0, atol=1e-9, rtol=1e-9):
        return False
    return True

def _make_schoolbook(n, m, p):
    """Create the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
    return U.tolist(), V.tolist(), W.tolist()

def _try_random_rank(n, m, p, r, max_iter=500, seed=0):
    """
    Try to find a rank-r decomposition using random initialization and gradient descent.
    Returns (U, V, W) as lists if found, else None.
    """
    rng = np.random.RandomState(seed)
    
    # Random initialization with half-integer values
    # Values in {-1, -0.5, 0, 0.5, 1}
    U = rng.choice([-1, -0.5, 0, 0.5, 1], size=(n * m, r))
    V = rng.choice([-1, -0.5, 0, 0.5, 1], size=(m * p, r))
    W = rng.choice([-1, -0.5, 0, 0.5, 1], size=(p * n, r))
    
    U_reshaped = U.reshape(n, m, r)
    V_reshaped = V.reshape(m, p, r)
    W_reshaped = W.reshape(p, n, r)
    
    for it in range(max_iter):
        # Compute current tensor
        T = np.einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped, optimize=True)
        
        # Check if we've converged
        if np.allclose(T, 1.0, atol=1e-9, rtol=1e-9):
            # Verify with exact check
            if _check_decomposition(U, V, W, n, m, p):
                return U.tolist(), V.tolist(), W.tolist()
        
        # Compute gradients
        # T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
        # dT/dU[i,j,c] = V[j,k,c] W[k,i,c] summed over k
        # dT/dV[j,k,c] = U[i,j,c] W[k,i,c] summed over i
        # dT/dW[k,i,c] = U[i,j,c] V[j,k,c] summed over j
        
        # Error E[i,j,k] = T[i,j,k] - 1
        E = T - 1.0
        
        # Gradient for U: sum_k E[i,j,k] * V[j,k,c] * W[k,i,c]
        # grad_U[i,j,c] = sum_k E[i,j,k] V[j,k,c] W[k,i,c]
        # Using einsum: 'ijk,jkc,kic->ijc'
        grad_U = np.einsum('ijk,jkc,kic->ijc', E, V_reshaped, W_reshaped, optimize=True)
        
        # Gradient for V: sum_i E[i,j,k] U[i,j,c] W[k,i,c]
        # grad_V[j,k,c] = sum_i E[i,j,k] U[i,j,c] W[k,i,c]
        grad_V = np.einsum('ijk,ijc,kic->jkc', E, U_reshaped, W_reshaped, optimize=True)
        
        # Gradient for W: sum_j E[i,j,k] U[i,j,c] V[j,k,c]
        # grad_W[k,i,c] = sum_j E[i,j,k] U[i,j,c] V[j,k,c]
        grad_W = np.einsum('ijk,ijc,jkc->kic', E, U_reshaped, V_reshaped, optimize=True)
        
        # Project to nearest half-integer
        # For each element, find the nearest value in {-1, -0.5, 0, 0.5, 1}
        def project_half_integer(x, grad, lr=0.1):
            x_new = x - lr * grad
            # Project to nearest half-integer
            # Round to nearest 0.5
            x_rounded = np.round(x_new * 2) / 2
            # Clip to [-1, 1]
            x_rounded = np.clip(x_rounded, -1, 1)
            return x_rounded
        
        U_reshaped = project_half_integer(U_reshaped, grad_U)
        V_reshaped = project_half_integer(V_reshaped, grad_V)
        W_reshaped = project_half_integer(W_reshaped, grad_W)
        
        U = U_reshaped.reshape(n * m, r)
        V = V_reshaped.reshape(m * p, r)
        W = W_reshaped.reshape(p * n, r)
    
    # Final check
    if _check_decomposition(U, V, W, n, m, p):
        return U.tolist(), V.tolist(), W.tolist()
    return None

def _try_known_constructions(n, m, p, r):
    """
    Try some known constructions for specific dimensions.
    For 3x5x7, the schoolbook is 105. Known lower bounds are higher.
    We try to see if there's a known rank.
    """
    # For now, just return None
    return None

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3 by 5 matrix 
    by a 5 by 7 matrix.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140  # 10s safety margin
    
    # Start with schoolbook
    U, V, W = _make_schoolbook(n, m, p)
    best_rank = n * m * p
    best_result = {"U": U, "V": V, "W": W}
    
    # Try to improve using random search for lower ranks
    # Theoretical lower bound: max(n*p, m*p, n*m) = max(21, 35, 15) = 35
    # But for matrix multiplication, the lower bound is at least n*p = 21 (actually higher)
    # Let's try ranks from a reasonable lower bound up to schoolbook
    
    # Known: rank(A*B) for n x m times m x p is at least max(n, m, p) * min(n, m, p) ? No.
    # The trivial lower bound is max(n*p, m*p, n*m) = 35 for 3,5,7
    # But actually, the lower bound for matrix multiplication is at least n*p = 21? 
    # No, the rank must be at least max(n, m, p) = 7? That's too weak.
    # The rank is at least the dimension of the output, which is n*p = 21.
    # But also at least the "slice rank" which is at least max(n, m, p) * min(n, m, p)? 
    # Let's just try from 35 upwards since that's a reasonable lower bound.
    
    # Actually, let's be more careful. The rank must be at least n*p = 21.
    # Let's try a few ranks below schoolbook.
    
    ranks_to_try = []
    # Try a few promising ranks
    for r in range(35, min(105, 50)):  # Try ranks from 35 to 49
        if time.time() > deadline - 30:
            break
        ranks_to_try.append(r)
    
    # Also try some specific ranks
    for r in [21, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90]:
        if r < best_rank and r not in ranks_to_try:
            ranks_to_try.append(r)
    
    ranks_to_try.sort()
    
    for r in ranks_to_try:
        if time.time() > deadline - 10:
            break
        if r >= best_rank:
            break
        
        # Try multiple random seeds
        found = False
        for seed in range(20):
            if time.time() > deadline - 10:
                break
            result = _try_random_rank(n, m, p, r, max_iter=200, seed=seed)
            if result is not None:
                U, V, W = result
                best_rank = r
                best_result = {"U": U, "V": V, "W": W}
                found = True
                break
        
        if found:
            # Found a better rank, continue searching for even lower
            continue
    
    return best_result
# EVOLVE-BLOCK-END
