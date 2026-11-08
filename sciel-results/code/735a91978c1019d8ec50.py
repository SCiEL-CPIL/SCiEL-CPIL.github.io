import numpy as np
import itertools
import time

DIMENSIONS = [3, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    # Target rank is 19.
    # We will attempt to find a rank 19 decomposition.
    # If it fails, we fallback to rank 27 (schoolbook) or best found.
    
    # The tensor T is defined by T[i, j, k] = 1 if i==j and k==? No.
    # Matrix multiplication tensor: (A * B)_{i, k} = sum_j A_{i, j} B_{j, k}.
    # Tensor indices: U index (i, j), V index (j, k), W index (k, i) ??
    # Contract: U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Sum over r.
    # We want sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = delta_{i,k} ? No.
    # The tensor for matrix multiplication is T_{i, j, k} where the multiplication is (AB)_{i,k} = sum_j A_{i,j} B_{j,k}.
    # So the tensor has indices i (row of A), j (col of A / row of B), k (col of B).
    # The value is 1 for all i, j, k? No.
    # The tensor is the identity in the "multiplication" sense.
    # Actually, the tensor is $T = \sum_{i,j,k} e_{i,j} \otimes e_{j,k} \otimes e_{k,i}$?
    # Let's check the convention: W[k*n+i, r].
    # The term in the product is U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r].
    # Summing over r gives the coefficient for the basis element corresponding to (i, j, k) in the input and (i, k) in the output?
    # No, the decomposition reconstructs the *tensor* of the bilinear map.
    # The bilinear map is $\mu: A \otimes B \to C$.
    # $\mu(A, B) = AB$.
    # The tensor associated with $\mu$ is $T = \sum_{i,j,k} \mu(e_{i,j}, e_{j,k}) \otimes e_{i,k}^*$.
    # $\mu(e_{i,j}, e_{j,k}) = e_{i,k}$ if the indices match?
    # $e_{i,j} e_{j,k} = \delta_{j,j} e_{i,k} = e_{i,k}$.
    # So $T = \sum_{i,j,k} e_{i,j} \otimes e_{j,k} \otimes e_{i,k}^*$.
    # In the factorization: $U_{i,j, r} V_{j,k, r} W_{i,k, r}$.
    # Wait, the W index is $k*n+i$. This corresponds to output index $(i, k)$?
    # Usually output is $(i, k)$. Index $i*n+k$.
    # Here it is $k*n+i$. This means output row is $k$ and col is $i$? Or output is transposed?
    # "output factor follows the official trace convention W[k*n+i,r]".
    # This implies the output matrix $C$ is indexed by $C_{k, i} = (AB)_{i, k} = (AB)^T_{k, i}$.
    # So we are computing the transpose of the product?
    # Or the indices are just swapped in the W matrix layout.
    # Let's assume the standard tensor $T_{i,j,k} = 1$ for all $i,j,k$ is WRONG.
    # The tensor is sparse.
    # $T_{i,j,k}$ is non-zero only if the multiplication $e_{i,j} \cdot e_{j,k}$ is non-zero.
    # $e_{i,j} e_{j,k} = e_{i,k}$.
    # So for every $i, j, k$, the term $e_{i,j} \otimes e_{j,k} \otimes e_{i,k}^*$ is present.
    # So the tensor has $3 \times 3 \times 3 = 27$ non-zero entries, all equal to 1.
    # Wait, $e_{i,j}$ is the matrix with 1 at $(i,j)$.
    # $e_{i,j} e_{j,k}$ is the matrix with 1 at $(i,k)$ if the inner index $j$ matches.
    # Yes, for any $i, j, k$, the product is $e_{i,k}$.
    # So the tensor is $\sum_{i,j,k} e_{i,j} \otimes e_{j,k} \otimes e_{i,k}^*$.
    # This is a tensor of rank at most 27 (schoolbook).
    # The question is the *minimal* rank.
    
    # The schoolbook decomposition uses 27 terms.
    # We want to find a decomposition with rank 19.
    
    # I will use a randomized search.
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    # Define the target tensor T (flattened)
    # T_flat[i*m+j, j*p+k, k*n+i] = 1 ?
    # No, the tensor is a 3D array $T_{a, b, c}$ where $a \in [0, 9)$, $b \in [0, 9)$, $c \in [0, 9)$.
    # $a = i*m+j$, $b = j*p+k$, $c = k*n+i$.
    # The value is 1 if the indices are consistent with a single multiplication term.
    # Actually, for a fixed $r$, the term is $u_r \otimes v_r \otimes w_r$.
    # The sum is $\sum_r u_r \otimes v_r \otimes w_r$.
    # This sum must equal the matrix multiplication tensor.
    # The matrix multiplication tensor $M$ has entries $M_{a, b, c} = 1$ if $a=(i,j), b=(j,k), c=(k,i)$?
    # Let's check the W index: $W[k*n+i]$.
    # If $c = k*n+i$, then $c$ corresponds to output $(k, i)$.
    # The product $(AB)_{i, k}$ goes to output position $(k, i)$?
    # This means the output matrix is the transpose of the standard product.
    # Or the indices are just labeled differently.
    # Let's assume the target tensor $M$ is defined by:
    # For all $i, j, k \in \{0, 1, 2\}$:
    # $M[i*m+j, j*p+k, k*n+i] = 1$.
    # All other entries are 0.
    
    M = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                M[i*m+j, j*p+k, k*n+i] = 1.0

    # We want to find U, V, W of shape (9, R), (9, R), (9, R)
    # such that sum_r U[:, r] outer V[:, r] outer W[:, r] == M.
    
    # Strategy: Random search for R=19.
    # Since exact integer/half-integer is required, we can search over a grid?
    # Too large.
    # Use optimization to find real solution, then round.
    
    best_rank = n*m*p
    best_U = None
    best_V = None
    best_W = None
    
    # Fallback: Schoolbook
    U_school = np.zeros((n*m, best_rank))
    V_school = np.zeros((m*p, best_rank))
    W_school = np.zeros((p*n, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    best_U, best_V, best_W = U_school, V_school, W_school

    # Try to find rank 19
    # Use scipy.optimize to minimize error.
    from scipy.optimize import minimize
    
    def error_fn(params, R):
        # params: flattened U, V, W
        size_u = n*m * R
        size_v = m*p * R
        size_w = p*n * R
        
        U = params[:size_u].reshape(n*m, R)
        V = params[size_u:size_u+size_v].reshape(m*p, R)
        W = params[size_u+size_v:].reshape(p*n, R)
        
        # Compute tensor product sum
        # M_approx = sum_r U[:, r] outer V[:, r] outer W[:, r]
        # This is expensive if done naively.
        # M_approx[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
        # We can compute this using einsum: 'ar, br, cr -> abc'
        M_approx = np.einsum('ar,br,cr->abc', U, V, W)
        
        # Error
        err = np.sum((M_approx - M)**2)
        return err

    # Random restarts
    num_restarts = 50
    for _ in range(num_restarts):
        if time.time() > deadline:
            break
            
        R = 19
        # Initialize with small random values
        init_U = np.random.randn(n*m, R) * 0.1
        init_V = np.random.randn(m*p, R) * 0.1
        init_W = np.random.randn(p*n, R) * 0.1
        
        x0 = np.concatenate([init_U.flatten(), init_V.flatten(), init_W.flatten()])
        
        try:
            res = minimize(error_fn, x0, args=(R,), method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-10})
            if res.success or res.fun < 1e-6:
                # Extract solution
                size_u = n*m * R
                size_v = m*p * R
                size_w = p*n * R
                U_sol = res.x[:size_u].reshape(n*m, R)
                V_sol = res.x[size_u:size_u+size_v].reshape(m*p, R)
                W_sol = res.x[size_u+size_v:].reshape(p*n, R)
                
                # Check if it's close to integer/half-integer
                # Multiply by 2, round, divide by 2
                U_int = np.round(U_sol * 2) / 2
                V_int = np.round(V_sol * 2) / 2
                W_int = np.round(W_sol * 2) / 2
                
                # Verify exactness
                M_check = np.einsum('ar,br,cr->abc', U_int, V_int, W_int)
                if np.allclose(M_check, M, atol=1e-9):
                    # Found a valid rank 19 decomposition!
                    best_rank = 19
                    best_U = U_int
                    best_V = V_int
                    best_W = W_int
                    break
        except Exception:
            pass

    # Convert to lists of lists as required
    # Ensure entries are int or float (half-integer)
    def to_list(mat):
        return [[float(x) for x in row] for row in mat]

    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
