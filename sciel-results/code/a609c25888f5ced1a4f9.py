"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    nm = n * m  # 9
    mp = m * p  # 21
    np_ = n * p  # 21
    
    # Build the target tensor entries
    # T[i*m+j, j*p+k, k*n+i] = 1
    tensor_entries = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_a = i * m + j
                idx_b = j * p + k
                idx_c = k * n + i
                tensor_entries.append((idx_a, idx_b, idx_c))
                
    tensor_entries = np.array(tensor_entries)
    A_idx = tensor_entries[:, 0]
    B_idx = tensor_entries[:, 1]
    C_idx = tensor_entries[:, 2]
    
    best_rank = n * m * p  # 63
    best_decomp = None
    
    # Try ranks from a reasonable lower bound up to schoolbook
    # Lower bound is max(nm, mp, np_) = 21
    # Known upper bounds for similar problems suggest trying from 40 upwards
    for r in range(40, best_rank + 1):
        if time.time() > deadline - 5:
            break
            
        # Try to find a rank-r decomposition
        decomp = _find_cp_decomposition_fast(A_idx, B_idx, C_idx, nm, mp, np_, r, deadline - 10)
        if decomp is not None:
            best_rank = r
            best_decomp = decomp
            if r <= 45: # If we find a very good one, stop early
                break
    
    if best_decomp is not None:
        U, V, W = best_decomp
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
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

def _find_cp_decomposition_fast(A_idx, B_idx, C_idx, nm, mp, np_, r, deadline):
    """
    Find a rank-r CP decomposition using gradient descent.
    """
    num_restarts = 10
    max_iters = 1000
    lr = 0.01
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            return None
        
        rng = np.random.default_rng(seed=42 + restart)
        U = rng.standard_normal((nm, r))
        V = rng.standard_normal((mp, r))
        W = rng.standard_normal((np_, r))
        
        # Normalize
        for c in range(r):
            norm = max(np.linalg.norm(U[:, c]), np.linalg.norm(V[:, c]), np.linalg.norm(W[:, c]), 1e-10)
            U[:, c] /= norm
            V[:, c] /= norm
            W[:, c] /= norm
            
        prev_error = float('inf')
        for it in range(max_iters):
            if time.time() > deadline:
                return None
                
            # Compute error and gradients
            # T_approx[a,b,c] = sum_d U[a,d] V[b,d] W[c,d]
            # We only care about the entries in A_idx, B_idx, C_idx
            # Error = sum (T_approx - 1)^2
            
            # Efficient computation:
            # For each d, compute U[:,d] * V[:,d] * W[:,d]
            # This is a rank-1 tensor.
            # We need the sum over d.
            
            # Let's compute the gradients directly.
            # dE/dU[a,d] = 2 * sum_{(a,b,c) in support} (T_approx[a,b,c] - 1) * V[b,d] * W[c,d]
            
            # We can vectorize this.
            # Let M = V * W (element-wise product, shape mp x np_ x r)
            # Then for each a, grad_U[a,d] = 2 * sum_{b,c} (T_approx[a,b,c] - 1) * M[b,c,d]
            # But T_approx is not explicitly computed for all a,b,c.
            
            # Instead, we can use the fact that the support is sparse.
            # For each (a,b,c) in support:
            #   err = sum_d U[a,d] V[b,d] W[c,d] - 1
            #   grad_U[a,d] += 2 * err * V[b,d] * W[c,d]
            #   grad_V[b,d] += 2 * err * U[a,d] * W[c,d]
            #   grad_W[c,d] += 2 * err * U[a,d] * V[b,d]
            
            grad_U = np.zeros((nm, r))
            grad_V = np.zeros((mp, r))
            grad_W = np.zeros((np_, r))
            
            # This loop is over 63 entries, which is fast.
            for idx in range(len(A_idx)):
                a = A_idx[idx]
                b = B_idx[idx]
                c = C_idx[idx]
                
                # Compute approx value
                approx = np.dot(U[a, :], V[b, :] * W[c, :])
                err = approx - 1.0
                
                # Update gradients
                grad_U[a, :] += 2.0 * err * (V[b, :] * W[c, :])
                grad_V[b, :] += 2.0 * err * (U[a, :] * W[c, :])
                grad_W[c, :] += 2.0 * err * (U[a, :] * V[b, :])
            
            # Update parameters
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            # Compute error for convergence check
            error = 0.0
            for idx in range(len(A_idx)):
                a = A_idx[idx]
                b = B_idx[idx]
                c = C_idx[idx]
                approx = np.dot(U[a, :], V[b, :] * W[c, :])
                error += (approx - 1.0) ** 2
            error = np.sqrt(error / len(A_idx))
            
            if abs(prev_error - error) < 1e-8:
                break
            prev_error = error
        
        # Final check
        max_err = 0.0
        for idx in range(len(A_idx)):
            a = A_idx[idx]
            b = B_idx[idx]
            c = C_idx[idx]
            approx = np.dot(U[a, :], V[b, :] * W[c, :])
            max_err = max(max_err, abs(approx - 1.0))
            
        if max_err < 1e-5:
            # Check non-support entries (sample)
            sampled_err = 0.0
            for _ in range(100):
                a = rng.integers(0, nm)
                b = rng.integers(0, mp)
                c = rng.integers(0, np_)
                # Check if (a,b,c) is in support
                # This is O(63) check, which is fine
                in_support = False
                for idx in range(len(A_idx)):
                    if A_idx[idx] == a and B_idx[idx] == b and C_idx[idx] == c:
                        in_support = True
                        break
                if not in_support:
                    approx = np.dot(U[a, :], V[b, :] * W[c, :])
                    sampled_err = max(sampled_err, abs(approx))
            
            if sampled_err < 1e-5:
                U_list = [[float(U[a, c]) for c in range(r)] for a in range(nm)]
                V_list = [[float(V[b, c]) for c in range(r)] for b in range(mp)]
                W_list = [[float(W[c, c]) for c in range(r)] for c in range(np_)]
                return (U_list, V_list, W_list)
    
    return None
# EVOLVE-BLOCK-END
