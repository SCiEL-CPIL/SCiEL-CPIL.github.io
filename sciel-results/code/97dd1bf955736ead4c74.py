import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    # Tensor T of shape (n, m, p)
    # T[i, j, k] = 1 if this term is in the product
    # Actually, the multiplication tensor has T[i, j, k] = 1 for all i, j, k?
    # No, C_ik = sum_j A_ij B_jk.
    # So the term A_ij B_jk contributes to C_ik.
    # The tensor index is (i, j, k) for the factor A_ij, B_jk, C_ik.
    # Wait, the standard tensor is in V^* x W^* x U.
    # V is space of A (dim nm), W is space of B (dim mp), U is space of C (dim pn).
    # T_{ (i,j), (j,k), (i,k) } = 1.
    # Let's flatten indices.
    # A index: i*m + j
    # B index: j*p + k
    # C index: k*n + i (as per task description: W[k*n+i, r])
    
    # We want to find U (nm x r), V (mp x r), W (pn x r)
    # such that sum_l U[ai, l] V[bi, l] W[ci, l] = T[ai, bi, ci]
    # where T[ai, bi, ci] = 1 if ai, bi, ci correspond to a valid multiplication term, else 0.
    # ai = i*m + j
    # bi = j*p + k
    # ci = k*n + i
    
    # Let's build the tensor T as a list of non-zero entries.
    # T is sparse.
    # Non-zero entries:
    # for i in range(n):
    #   for j in range(m):
    #     for k in range(p):
    #       ai = i*m + j
    #       bi = j*p + k
    #       ci = k*n + i
    #       T[ai, bi, ci] = 1
    
    # We can represent the tensor as a 3D array or a list of tuples.
    # For least squares, we need a matrix form.
    # vec(T) = (U^T V^T) * W^T ? No.
    # T_{ai, bi, ci} = sum_l U_{ai, l} V_{bi, l} W_{ci, l}
    # This is a quadratic equation in U, V, W.
    
    # We can fix one matrix, say W, and solve for U and V?
    # Or flatten U and V.
    # Let's try to optimize all three.
    
    def loss(params):
        # params: flat array of U, V, W
        # U: nm * r
        # V: mp * r
        # W: pn * r
        size_U = n * m * r
        size_V = m * p * r
        size_W = p * n * r
        
        U_flat = params[:size_U]
        V_flat = params[size_U:size_U+size_V]
        W_flat = params[size_U+size_V:size_U+size_V+size_W]
        
        U = U_flat.reshape(n*m, r)
        V = V_flat.reshape(m*p, r)
        W = W_flat.reshape(p*n, r)
        
        errors = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    ai = i*m + j
                    bi = j*p + k
                    ci = k*n + i
                    # T[ai, bi, ci] = 1
                    val = np.sum(U[ai, :] * V[bi, :] * W[ci, :])
                    errors.append(val - 1.0)
                    
        # We also need to ensure that other entries are 0?
        # The tensor T is zero for other indices.
        # So we need to add constraints for all (ai, bi, ci) not in the support.
        # Total number of entries is (nm)*(mp)*(pn) = 6*9*6 = 324.
        # Support size is 18.
        # So we have 306 zero constraints.
        
        # To be efficient, we can precompute the list of all indices and their target values.
        # But iterating 324 times in Python is slow if done in a loop inside loss.
        # We can vectorize.
        
        # Let's create a tensor T of shape (nm, mp, pn).
        # T = np.zeros((n*m, m*p, p*n))
        # for i in range(n):
        #   for j in range(m):
        #     for k in range(p):
        #       T[i*m+j, j*p+k, k*n+i] = 1
        
        # We want sum_l U[:, l] V[:, l] W[:, l] = T
        # This is equivalent to:
        # For each l, U[:, l] V[:, l] is a matrix of size nm x mp.
        # We need sum_l (U[:, l] V[:, l])_{ai, bi} W_{ci, l} = T_{ai, bi, ci}
        
        # This is hard to vectorize directly for least squares without unfolding.
        # Let's stick to the loop but optimize it.
        
        # Actually, we can use the fact that T is sparse.
        # We only need to check the 18 non-zero entries and a subset of zero entries?
        # No, we must satisfy all.
        
        # Let's try a different approach:
        # Fix W. Then the equation is linear in U and V?
        # sum_l (U_{ai, l} V_{bi, l}) W_{ci, l} = T_{ai, bi, ci}
        # Let M_{ai, bi} = sum_l U_{ai, l} V_{bi, l} W_{ci, l} ? No, ci is fixed in T.
        # For a fixed ci, we have a matrix equation:
        # sum_l (U[:, l] V[:, l]) W_{ci, l} = T[:, :, ci]
        # Let A_l = U[:, l] V[:, l].
        # We want sum_l A_l W_{ci, l} = T_c.
        # This is a linear combination of rank-1 matrices.
        
        # This seems complicated to solve directly.
        
        # Let's just use the loop and hope it's fast enough for small r.
        # 324 iterations * 15 multiplications * 3 arrays = small.
        # Python loop overhead might be significant.
        
        # Let's precompute the indices.
        # And use numpy operations inside the loop.
        
        # Even better:
        # Construct the full tensor product.
        # T_hat[ai, bi, ci] = sum_l U[ai, l] V[bi, l] W[ci, l]
        # We can compute this using einsum.
        # T_hat = np.einsum('il,jl,kl->ijk', U, V, W)
        # This creates a 6x9x6 array.
        # Then we compute the error against T.
        
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i*m+j, j*p+k, k*n+i] = 1.0
        
        T_hat = np.einsum('il,jl,kl->ijk', U, V, W)
        err = T_hat - T
        return err.flatten()

    # Try ranks from 15 to 18
    best_result = None
    start_time = time.time()
    deadline = start_time + 100  # 100s limit
    
    for r in range(15, 19):
        if time.time() > deadline - 10:
            break
            
        # Try multiple random restarts
        for restart in range(20):
            if time.time() > deadline - 5:
                break
                
            np.random.seed(restart * 1000 + r)
            # Initialize U, V, W
            # Scale appropriately
            U0 = np.random.randn(n*m, r)
            V0 = np.random.randn(m*p, r)
            W0 = np.random.randn(p*n, r)
            
            x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
            
            try:
                res = least_squares(loss, x0, method='lm', max_nfev=1000)
                if res.success or res.cost < 1e-10:
                    # Verify
                    U = res.x[:n*m*r].reshape(n*m, r)
                    V = res.x[n*m*r:n*m*r+m*p*r].reshape(m*p, r)
                    W = res.x[n*m*r+m*p*r:].reshape(p*n, r)
                    
                    # Check if entries are close to integers/half-integers
                    # The task says "integer/half-integer entries".
                    # We should round them and verify.
                    U_round = np.round(U * 2) / 2
                    V_round = np.round(V * 2) / 2
                    W_round = np.round(W * 2) / 2
                    
                    T_hat = np.einsum('il,jl,kl->ijk', U_round, V_round, W_round)
                    if np.allclose(T_hat, T, atol=1e-9):
                        # Found a valid decomposition
                        return {
                            "U": U_round.tolist(),
                            "V": V_round.tolist(),
                            "W": W_round.tolist()
                        }
            except Exception:
                pass

    # If no lower rank found, return schoolbook
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
