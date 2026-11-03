"""Matrix multiplication 3x4x7 - rank-minimal exact tensor decomposition search."""
import time
import itertools
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 7]

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W exactly decompose the matmul tensor."""
    rank = U.shape[1]
    # Build the tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # and check it equals delta_{i,i} * delta_{j,j} * delta_{k,k} = 1 for all (i,j,k)
    # Actually for matmul: T[i,j,k] should be 1 if the indices correspond to the same
    # (i,j,k) position, i.e., the tensor is the identity in the appropriate sense.
    
    # The matmul tensor T_{ijk} satisfies: T_{ijk} = 1 if (i,j,k) is a valid index triple,
    # and the tensor decomposition requires:
    # T_{ijk} = sum_c U_{i,j,c} * V_{j,k,c} * W_{k,i,c}
    
    # For matmul A (n x m) * B (m x p) = C (n x p):
    # C_{i,k} = sum_j A_{i,j} B_{j,k}
    # The tensor T_{i,j,k} = delta_{i,i'} delta_{j,j'} delta_{k,k'} ... 
    # Actually the matmul tensor has T_{i,j,k} = 1 for all valid (i,j,k) where
    # i in [0,n), j in [0,m), k in [0,p). No wait.
    
    # Let me reconsider. The matmul tensor M_{i,j,k} is defined such that
    # C_{ik} = sum_j M_{i,j,k} A_{ij} B_{jk}
    # So M_{i,j,k} = 1 for all valid (i,j,k).
    
    # The decomposition requires: M_{i,j,k} = sum_c U_{ij,c} V_{jk,c} W_{ki,c}
    
    # Check:
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m + j, c] * V[j*p + k, c] * W[k*n + i, c]
                if abs(val - 1.0) > tol:
                    return False
    return True

def _verify_numpy(U, V, W, n, m, p, tol=1e-9):
    """Fast verification using numpy."""
    rank = U.shape[1]
    U_mat = np.array(U, dtype=np.float64)
    V_mat = np.array(V, dtype=np.float64)
    W_mat = np.array(W, dtype=np.float64)
    
    # Reshape
    U3 = U_mat.reshape(n, m, rank)
    V3 = V_mat.reshape(m, p, rank)
    W3 = W_mat.reshape(p, n, rank)
    
    # Compute tensor: T[i,j,k] = sum_c U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
    # Use einsum
    T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    
    # T should be all ones
    if np.max(np.abs(T - 1.0)) > tol:
        return False
    return True

def _try_fixed_rank(rank, n, m, p, time_limit=10.0, seed=42):
    """Try to find a decomposition of given rank using gradient descent."""
    start = time.time()
    
    # Initialize with schoolbook-like random structure
    # Use the schoolbook decomposition and try to merge terms
    # Start from a random initialization with the target rank
    
    np.random.seed(seed)
    
    # Initialize U, V, W with small random values
    U = np.random.randn(n * m, rank) * 0.5
    V = np.random.randn(m * p, rank) * 0.5
    W = np.random.randn(p * n, rank) * 0.5
    
    # Reshape for computation
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    
    # Target tensor: all ones
    target = np.ones((n, m, p))
    
    # Loss function: ||T - target||^2
    def loss_and_grad(U_flat, V_flat, W_flat):
        U3 = U_flat.reshape(n, m, rank)
        V3 = V_flat.reshape(m, p, rank)
        W3 = W_flat.reshape(p, n, rank)
        
        T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
        loss = np.sum((T - target) ** 2)
        
        # Gradients
        # dL/dU3[i,j,c] = 2 * sum_{k} (T[i,j,k] - 1) * V3[j,k,c] * W3[k,i,c]
        # dL/dV3[j,k,c] = 2 * sum_{i} (T[i,j,k] - 1) * U3[i,j,c] * W3[k,i,c]
        # dL/dW3[k,i,c] = 2 * sum_{j} (T[i,j,k] - 1) * U3[i,j,c] * V3[j,k,c]
        
        err = T - target
        
        dU3 = np.einsum('ijk,jkc,kic->ijc', err, V3, W3) * 2
        dV3 = np.einsum('ijk,ijc,kic->jkc', err, U3, W3) * 2
        dW3 = np.einsum('ijk,ijc,jkc->kic', err, U3, V3) * 2
        
        return loss, (dU3.ravel(), dV3.ravel(), dW3.ravel())
    
    # Adam optimizer
    lr = 0.01
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    
    m_U = np.zeros_like(U)
    v_U = np.zeros_like(U)
    m_V = np.zeros_like(V)
    v_V = np.zeros_like(V)
    m_W = np.zeros_like(W)
    v_W = np.zeros_like(W)
    t = 0
    
    best_loss = float('inf')
    best_params = (U.copy(), V.copy(), W.copy())
    
    max_iters = 5000
    while t < max_iters:
        if time.time() - start > time_limit - 2.0:
            break
        
        loss, (dU, dV, dW) = loss_and_grad(U, V, W)
        
        if loss < best_loss:
            best_loss = loss
            best_params = (U.copy(), V.copy(), W.copy())
        
        if loss < 1e-10:
            break
        
        t += 1
        b1 = 1 - beta1 ** t
        b2 = 1 - beta2 ** t
        
        m_U = beta1 * m_U + (1 - beta1) * dU
        v_U = beta2 * v_U + (1 - beta2) * dU ** 2
        m_V = beta1 * m_V + (1 - beta1) * dV
        v_V = beta2 * v_V + (1 - beta2) * dV ** 2
        m_W = beta1 * m_W + (1 - beta1) * dW
        v_W = beta2 * v_W + (1 - beta2) * dW ** 2
        
        m_U_hat = m_U / b1
        v_U_hat = v_U / b2
        m_V_hat = m_V / b1
        v_V_hat = v_V / b2
        m_W_hat = m_W / b1
        v_W_hat = v_W / b2
        
        U -= lr * m_U_hat / (np.sqrt(v_U_hat) + eps)
        V -= lr * m_V_hat / (np.sqrt(v_V_hat) + eps)
        W -= lr * m_W_hat / (np.sqrt(v_W_hat) + eps)
    
    # Check if we found a valid solution
    U_final, V_final, W_final = best_params
    if _verify_numpy(U_final, V_final, W_final, n, m, p, tol=1e-6):
        return U_final, V_final, W_final
    return None

def _schoolbook_decomposition(n, m, p):
    """Return the schoolbook decomposition."""
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
    
    return U, V, W, rank

def _try_rank_reduction_by_merging(n, m, p, time_limit=8.0):
    """Try to reduce rank by identifying and merging near-dependent terms."""
    start = time.time()
    U, V, W, rank = _schoolbook_decomposition(n, m, p)
    
    # Try to find pairs of columns that can be merged
    # This is a heuristic approach
    # For now, just return the schoolbook solution
    return U, V, W, rank

def construct():
    start = time.time()
    n, m, p = DIMENSIONS
    
    # First, get the schoolbook solution as a fallback
    U_sb, V_sb, W_sb, rank_sb = _schoolbook_decomposition(n, m, p)
    
    # Try to find lower rank solutions
    # Known lower bounds for matmul rank:
    # For 3x4x7, the schoolbook rank is 84
    # The Strassen rank for 2x2x2 is 7, for 3x3x3 it's known to be at most 23
    # For 3x4x7, we need to search
    
    # Try ranks from a reasonable lower bound up to schoolbook
    # Lower bound: max(n*m, m*p, p*n) = max(12, 28, 21) = 28
    # But actual lower bounds are higher
    
    # Try a few candidate ranks
    candidate_ranks = [28, 30, 32, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]
    
    best_solution = None
    best_rank = rank_sb
    
    for target_rank in candidate_ranks:
        if time.time() - start > 120 - 20:  # Safety margin
            break
        
        if target_rank >= best_rank:
            break
        
        # Try multiple seeds
        found = False
        for seed in range(5):
            if time.time() - start > 120 - 20:
                break
            
            result = _try_fixed_rank(target_rank, n, m, p, time_limit=15.0, seed=seed)
            if result is not None:
                U_r, V_r, W_r = result
                best_solution = (U_r, V_r, W_r)
                best_rank = target_rank
                found = True
                break
        
        if found:
            break
    
    if best_solution is not None:
        U_out, V_out, W_out = best_solution
        # Convert to list of lists
        U_list = U_out.tolist()
        V_list = V_out.tolist()
        W_list = W_out.tolist()
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        # Fallback to schoolbook
        U_list = U_sb.tolist()
        V_list = V_sb.tolist()
        W_list = W_sb.tolist()
        return {"U": U_list, "V": V_list, "W": W_list}
