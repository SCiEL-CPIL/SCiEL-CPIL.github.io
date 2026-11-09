import numpy as np
import time
import scipy.optimize as opt

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110
    
    # Trivial decomposition
    rank0 = n * m * p
    U0 = np.zeros((n*m, rank0))
    V0 = np.zeros((m*p, rank0))
    W0 = np.zeros((p*n, rank0))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i*m+j, col] = 1
                V0[j*p+k, col] = 1
                W0[k*n+i, col] = 1
                col += 1
    
    best_rank = rank0
    best_U, best_V, best_W = U0.copy(), V0.copy(), W0.copy()
    
    # Target tensor in flattened form
    # The tensor T_{ijk} is 1 if the decomposition matches the identity multiplication
    # Actually, the decomposition is:
    # (A B)_{ik} = sum_j A_{ij} B_{jk}
    # The tensor is defined by the coefficients in:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # This sum must equal 1 if the indices correspond to the standard multiplication?
    # No, the standard multiplication is:
    # C_{ik} = sum_j A_{ij} B_{jk}
    # This can be written as a tensor contraction.
    # The "tensor" of the multiplication algorithm is T such that:
    # T_{ijk} = 1 if the term A_{ij} B_{jk} is used to compute C_{ik}?
    # Yes, for the trivial algorithm, T_{ijk} = 1 for all i,j,k.
    # Wait, the trace convention W[k*n+i, r] implies the output is indexed by k, i.
    # The input matrices are A (n x m) and B (m x p).
    # The term U[i*m+j, c] picks A_{ij}.
    # The term V[j*p+k, c] picks B_{jk}.
    # The term W[k*n+i, c] picks the coefficient for C_{ki} (which is C_{ik}^T? No, C is n x p).
    # Let's check the index of W. W is (p*n, rank). Index is k*n + i.
    # So W contributes to the element at row i, col k of the result?
    # Usually C = A B. C_{ik} = sum_j A_{ij} B_{jk}.
    # If we sum over c: sum_c U_{ij,c} V_{jk,c} W_{ki,c}
    # This results in a value associated with (i, k).
    # For the result to be C_{ik}, the sum must equal 1 if j is the intermediate index?
    # No, the sum over c produces a single number for each (i, j, k) triple?
    # No, U, V, W are fixed. The sum is over c.
    # The result of the sum is a number for each (i, j, k).
    # But we want the result to be the matrix product.
    # The matrix product C_{ik} depends on j.
    # The decomposition is a sum of rank-1 tensors.
    # Each rank-1 tensor is U^{(c)} \otimes V^{(c)} \otimes W^{(c)}.
    # The sum of these tensors must equal the multiplication tensor T.
    # The multiplication tensor T_{ijk} is 1 if the standard algorithm uses A_{ij} and B_{jk} to contribute to C_{ik}.
    # In the standard algorithm, C_{ik} = sum_j A_{ij} B_{jk}.
    # So T_{ijk} = 1 for all i, j, k.
    # Yes, the trivial tensor is all ones.
    
    T_target = np.ones((n, m, p))
    T_flat = T_target.flatten()
    
    def tensor_to_vec(U, V, W):
        # U: (n*m, r), V: (m*p, r), W: (p*n, r)
        # Compute sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
        # This is equivalent to:
        # Reshape U to (n, m, r), V to (m, p, r), W to (p, n, r)
        # Sum over r: U[i,j,c] * V[j,k,c] * W[k,i,c]
        # Result shape (n, m, p)
        r = U.shape[1]
        U_r = U.reshape(n, m, r)
        V_r = V.reshape(m, p, r)
        W_r = W.reshape(p, n, r)
        
        # Efficiently compute the sum
        # This is a tensor contraction.
        # We can do:
        # Temp[j, k, i] = sum_c V[j, k, c] * W[k, i, c] ? No, W is (p, n, r).
        # Let's use einsum or explicit loops for clarity if small, but vectorized is better.
        # sum_c U[i,j,c] V[j,k,c] W[k,i,c]
        # = sum_c U[i,j,c] * (V[j,k,c] * W[k,i,c])
        
        # Let's construct the result array
        result = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                # U[i, j, :] is a vector of length r
                u_vec = U[i*m+j, :]
                # We need sum_c u_vec[c] * V[j, k, c] * W[k, i, c]
                # Let M[j, k, i] = sum_c V[j, k, c] * W[k, i, c]
                # Then result[i, j, k] = sum_c u_vec[c] * M[j, k, c] ... wait, M doesn't have c.
                
        # Alternative:
        # Sum over c:
        # R[i, j, k] = sum_c U[i, j, c] * V[j, k, c] * W[k, i, c]
        
        # We can compute this as:
        # 1. Multiply U and V element-wise along c? No.
        # 2. Let's use np.einsum
        # 'ijc,jkc,kic->ijk'
        # U is (n, m, r), V is (m, p, r), W is (p, n, r)
        # Note: W index in einsum should match the output k, i order.
        # W is stored as (p*n, r). Reshape to (p, n, r).
        # So W[k, i, c].
        
        return np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)

    def loss_and_grad(params):
        r = len(params) // (n*m + m*p + p*n)
        # This is inefficient if r changes. We will fix r for each optimization call.
        pass

    def optimize_rank(r, init_type='trivial_remove'):
        if time.time() > deadline:
            return None
            
        # Initialize U, V, W
        U = np.zeros((n*m, r))
        V = np.zeros((m*p, r))
        W = np.zeros((p*n, r))
        
        if init_type == 'trivial_remove':
            # Copy from trivial
            # We have 84 terms in trivial. We want r terms.
            # Keep the first r terms? Or random r terms?
            # Let's keep random r terms from the trivial 84.
            indices = np.random.choice(rank0, r, replace=False)
            U[:, indices] = U0[:, indices]
            V[:, indices] = V0[:, indices]
            W[:, indices] = W0[:, indices]
            # Add small noise to break symmetry
            U += np.random.randn(n*m, r) * 0.01
            V += np.random.randn(m*p, r) * 0.01
            W += np.random.randn(p*n, r) * 0.01
        elif init_type == 'random':
            U = np.random.randn(n*m, r) * 0.1
            V = np.random.randn(m*p, r) * 0.1
            W = np.random.randn(p*n, r) * 0.1
            
        # Flatten
        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        def objective(x):
            U_flat = x[:n*m*r]
            V_flat = x[n*m*r : n*m*r + m*p*r]
            W_flat = x[n*m*r + m*p*r :]
            
            U_curr = U_flat.reshape(n*m, r)
            V_curr = V_flat.reshape(m*p, r)
            W_curr = W_flat.reshape(p*n, r)
            
            T_curr = tensor_to_vec(U_curr, V_curr, W_curr)
            diff = T_curr.flatten() - T_flat
            return np.sum(diff**2)
            
        # Use L-BFGS-B for speed
        res = opt.minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-15})
        
        if res.fun < 1e-10:
            # Check if we can round to integers/half-integers
            U_sol = res.x[:n*m*r].reshape(n*m, r)
            V_sol = res.x[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
            W_sol = res.x[n*m*r + m*p*r :].reshape(p*n, r)
            
            # Try rounding to nearest 0.5
            U_int = np.round(U_sol * 2) / 2
            V_int = np.round(V_sol * 2) / 2
            W_int = np.round(W_sol * 2) / 2
            
            # Verify
            T_int = tensor_to_vec(U_int, V_int, W_int)
            if np.allclose(T_int.flatten(), T_flat, atol=1e-9):
                return U_int, V_int, W_int
            else:
                # Maybe integers?
                U_int = np.round(U_sol)
                V_int = np.round(V_sol)
                W_int = np.round(W_sol)
                T_int = tensor_to_vec(U_int, V_int, W_int)
                if np.allclose(T_int.flatten(), T_flat, atol=1e-9):
                    return U_int, V_int, W_int
        return None

    # Try ranks from 83 down to 80
    for r in range(83, 79, -1):
        if time.time() > deadline:
            break
        for trial in range(5):
            if time.time() > deadline:
                break
            init_type = 'trivial_remove' if trial < 3 else 'random'
            sol = optimize_rank(r, init_type)
            if sol is not None:
                U_sol, V_sol, W_sol = sol
                best_rank = r
                best_U, best_V, best_W = U_sol, V_sol, W_sol
                break # Found valid for this rank, try lower
        if best_rank < 84:
             # If we found one, we might want to try to go lower, but time is limited.
             # Let's try one more step down if time permits.
             pass

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
