import time
import numpy as np
from scipy.optimize import minimize

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying 
    a 2 by 3 matrix by a 3 by 5 matrix.
    
    Strategy:
    1. Attempt to find a decomposition of rank 26, 27, 28, 29, 30.
    2. Use numerical optimization (L-BFGS-B) to minimize the squared error 
       of the tensor decomposition.
    3. If the error is sufficiently small, round to nearest half-integers 
       and verify exactness.
    4. Return the lowest rank that yields an exact verification.
    5. Fallback to schoolbook rank 30 if optimization fails.
    """
    n, m, p = 2, 3, 5
    nm, mp, pn = n * m, m * p, p * n  # 6, 15, 10
    
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    # Construct the target matrix multiplication tensor T_target
    # T_target[i, j, k] = 1 if it corresponds to a valid multiplication term?
    # Actually, the tensor M_{n,m,p} is defined such that:
    # (A B)_{i, k} = \sum_j A_{i, j} B_{j, k}
    # The decomposition is: (A B)_{i, k} = \sum_t (\sum_j U_{i, j, t} A_{i, j}) (\sum_k V_{j, k, t} B_{j, k}) W_{k, i, t} ?
    # No, the standard tensor decomposition form is:
    # \sum_j A_{i, j} B_{j, k} = \sum_{t=1}^r U_{i, j, t} V_{j, k, t} W_{k, i, t}
    # Wait, the indices in the task are:
    # U: (n*m, rank) -> U[i*m + j, t]
    # V: (m*p, rank) -> V[j*p + k, t]
    # W: (p*n, rank) -> W[k*n + i, t]
    # The equation is:
    # \sum_j A_{i, j} B_{j, k} = \sum_t U_{i*m+j, t} V_{j*p+k, t} W_{k*n+i, t}
    # Let's verify the indices.
    # LHS: Sum over j.
    # RHS: Sum over t. The term is U[i,j,t] * V[j,k,t] * W[k,i,t].
    # This matches the tensor structure of matrix multiplication.
    
    # We need to construct the target tensor T_target of shape (n, m, p) ?
    # No, the "tensor" we are decomposing is the multiplication tensor.
    # The entries of the multiplication tensor T are:
    # T[i, j, k] = 1 if ... ?
    # Actually, the multiplication tensor is a 3-tensor $M_{n,m,p} \in \mathbb{R}^{n \times m \times p}$.
    # The definition is: $\langle M, A \otimes B \rangle = \text{trace}(A B)$? No.
    # The standard definition: $M_{n,m,p}$ is the tensor such that
    # $\sum_{i,j,k} M_{i,j,k} A_{i,j} B_{j,k} = \text{sum of all entries of } AB$?
    # No.
    # The matrix multiplication map is a bilinear map $A \times B \to C$.
    # The tensor $T$ represents this map.
    # $T_{i, j, k} = 1$ if the term $A_{i,j} B_{j,k}$ contributes to $C_{i,k}$.
    # In the decomposition $\sum_t U_{i,j,t} V_{j,k,t} W_{k,i,t}$,
    # the coefficient of $A_{i,j} B_{j,k}$ in the expansion of $\sum_t (\dots)$ is:
    # $\sum_t U_{i,j,t} V_{j,k,t} W_{k,i,t}$.
    # We want this to equal 1 if $A_{i,j} B_{j,k}$ is a valid term in the product, and 0 otherwise?
    # Wait. The product $C_{i,k} = \sum_j A_{i,j} B_{j,k}$.
    # The full tensor equation is:
    # $\sum_{i,k} C_{i,k} X_{i,k} = \sum_{i,j,k} A_{i,j} B_{j,k} X_{i,k}$
    # This is linear in A and B.
    # The tensor $T$ has components $T_{i,j,k}$.
    # The decomposition is $T = \sum_t u_t \otimes v_t \otimes w_t$.
    # Where $u_t$ is a vector of size $n \times m$, $v_t$ of size $m \times p$, $w_t$ of size $p \times n$.
    # The entry $T_{i,j,k}$ is 1 if $j$ is the intermediate index?
    # Actually, for matrix multiplication, $T_{i,j,k} = 1$ for all valid $i,j,k$?
    # No. The map is $(A,B) \to AB$.
    # The tensor $T$ is defined by $T_{i, j, k} = 1$ if the term $A_{i,j} B_{j,k}$ appears in the sum for $C_{i,k}$.
    # Since $C_{i,k}$ sums over $j$, the term $A_{i,j} B_{j,k}$ appears exactly once for each $j$.
    # So, for a fixed $i, k$, we sum over $j$.
    # The tensor $T$ is such that:
    # $\sum_{j} A_{i,j} B_{j,k} = \sum_t (\sum_j U_{i,j,t} A_{i,j}) (\sum_k V_{j,k,t} B_{j,k}) W_{k,i,t}$?
    # No, the grouping is different.
    # The standard tensor decomposition for matrix multiplication $M_{n,m,p}$ is:
    # $M_{n,m,p} = \sum_{t=1}^r u_t \otimes v_t \otimes w_t$
    # where $u_t \in \mathbb{R}^{n \times m}$, $v_t \in \mathbb{R}^{m \times p}$, $w_t \in \mathbb{R}^{p \times n}$.
    # And the contraction is:
    # $\sum_{i,j,k} M_{i,j,k} A_{i,j} B_{j,k} = \sum_{i,k} (AB)_{i,k}$.
    # Wait, the LHS is a scalar if we contract with A and B.
    # The RHS is $\sum_{i,k} (AB)_{i,k}$.
    # So the tensor $M$ satisfies:
    # $M_{i,j,k} = 1$ for all $i,j,k$?
    # If $M_{i,j,k} = 1$, then $\sum_{i,j,k} A_{i,j} B_{j,k} = \sum_i \sum_k \sum_j A_{i,j} B_{j,k} = \sum_{i,k} (AB)_{i,k}$.
    # Yes! The tensor $M_{n,m,p}$ has all entries equal to 1.
    # 
    # So we want to decompose the tensor $T$ where $T[i, j, k] = 1$ for all $i \in \{0..n-1\}, j \in \{0..m-1\}, k \in \{0..p-1\}$.
    # The decomposition is:
    # $T[i, j, k] = \sum_{t=1}^r U[i*m+j, t] V[j*p+k, t] W[k*n+i, t]$
    
    # Target tensor T is all ones.
    # T_target shape: (n, m, p) -> flattened?
    # We will work with the flattened indices for the optimization.
    
    def get_target(i, j, k):
        return 1.0

    def residual_flat(params, r):
        """
        params: flattened array of size (nm + mp + pn) * r
        Returns sum of squared errors.
        """
        U_flat = params[:nm * r]
        V_flat = params[nm * r : nm * r + mp * r]
        W_flat = params[nm * r + mp * r : ]
        
        U = U_flat.reshape(nm, r)
        V = V_flat.reshape(mp, r)
        W = W_flat.reshape(pn, r)
        
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    # sum_t U[u_idx, t] * V[v_idx, t] * W[w_idx, t]
                    s = 0.0
                    for t in range(r):
                        s += U[u_idx, t] * V[v_idx, t] * W[w_idx, t]
                    diff = s - 1.0
                    err += diff * diff
        return err

    def try_rank(r, max_iters=5000):
        """
        Try to find a decomposition of rank r.
        Returns (U, V, W) if successful, else None.
        """
        if time.time() > deadline:
            return None
            
        # Initialize with random small values
        # We need to satisfy the "integer/half-integer" constraint eventually.
        # Random initialization in [-1, 1] is a good start.
        
        best_err = float('inf')
        best_params = None
        
        # Number of restarts
        num_restarts = 5
        if r <= 27:
            num_restarts = 10
            
        for _ in range(num_restarts):
            if time.time() > deadline:
                break
                
            params0 = np.random.uniform(-1, 1, size=(nm + mp + pn) * r)
            
            # Use L-BFGS-B
            # Gradient computation is expensive, so we might use numerical gradients or analytic.
            # For speed, let's try a few iterations of a simple gradient step or just L-BFGS-B with finite diff (slow).
            # Better: Use scipy.optimize.minimize with method='L-BFGS-B'.
            
            # To speed up, we can compute the gradient analytically.
            # dE/dU[u, t] = 2 * (S[u, v, w] - 1) * V[v, t] * W[w, t]
            # where S[u, v, w] = sum_t' U[u, t'] V[v, t'] W[w, t']
            
            def objective_and_grad(params):
                U_flat = params[:nm * r]
                V_flat = params[nm * r : nm * r + mp * r]
                W_flat = params[nm * r + mp * r : ]
                
                U = U_flat.reshape(nm, r)
                V = V_flat.reshape(mp, r)
                W = W_flat.reshape(pn, r)
                
                # Compute S[i, j, k] for all i, j, k
                # S is a tensor of size n x m x p
                S = np.zeros((n, m, p))
                for t in range(r):
                    # u_t is vector of size nm (n*m)
                    # v_t is vector of size mp (m*p)
                    # w_t is vector of size pn (p*n)
                    u_t = U[:, t]
                    v_t = V[:, t]
                    w_t = W[:, t]
                    
                    # u_t reshaped to (n, m)
                    u_mat = u_t.reshape(n, m)
                    # v_t reshaped to (m, p)
                    v_mat = v_t.reshape(m, p)
                    # w_t reshaped to (p, n)
                    w_mat = w_t.reshape(p, n)
                    
                    # S[i, j, k] += u_mat[i, j] * v_mat[j, k] * w_mat[k, i]
                    # We can do this efficiently?
                    # For each t, we add u_mat[i,j] * v_mat[j,k] * w_mat[k,i]
                    # This is a trilinear form.
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                S[i, j, k] += u_mat[i, j] * v_mat[j, k] * w_mat[k, i]
                
                # Error
                err = np.sum((S - 1.0) ** 2)
                
                # Gradient
                # dE/dU[i*m+j, t] = 2 * (S[i, j, k] - 1) * V[j*p+k, t] * W[k*n+i, t]
                # Note: S[i, j, k] depends on all t.
                # The term (S[i, j, k] - 1) is the same for all t in the gradient w.r.t U[i*m+j, t]?
                # Yes, S is the sum. The derivative of S w.r.t U[i*m+j, t] is V[j*p+k, t] * W[k*n+i, t].
                # So grad_U[i*m+j, t] = 2 * (S[i, j, k] - 1) * V[j*p+k, t] * W[k*n+i, t]
                
                grad_U = np.zeros_like(U)
                grad_V = np.zeros_like(V)
                grad_W = np.zeros_like(W)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            diff = 2.0 * (S[i, j, k] - 1.0)
                            u_idx = i * m + j
                            v_idx = j * p + k
                            w_idx = k * n + i
                            
                            # grad_U[u_idx, :] += diff * V[v_idx, :] * W[w_idx, :]
                            grad_U[u_idx, :] += diff * V[v_idx, :] * W[w_idx, :]
                            # grad_V[v_idx, :] += diff * U[u_idx, :] * W[w_idx, :]
                            grad_V[v_idx, :] += diff * U[u_idx, :] * W[w_idx, :]
                            # grad_W[w_idx, :] += diff * U[u_idx, :] * V[v_idx, :]
                            grad_W[w_idx, :] += diff * U[u_idx, :] * V[v_idx, :]
                
                grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
                return err, grad

            res = minimize(objective_and_grad, params0, jac=True, method='L-BFGS-B', 
                           options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
            
            if res.fun < best_err:
                best_err = res.fun
                best_params = res.x
                
            # Check if we can round to half-integers
            if best_err < 1e-6:
                # Try rounding
                U_flat = best_params[:nm * r]
                V_flat = best_params[nm * r : nm * r + mp * r]
                W_flat = best_params[nm * r + mp * r : ]
                
                U = U_flat.reshape(nm, r)
                V = V_flat.reshape(mp, r)
                W = W_flat.reshape(pn, r)
                
                # Round to nearest half-integer
                U_r = np.round(U * 2) / 2.0
                V_r = np.round(V * 2) / 2.0
                W_r = np.round(W * 2) / 2.0
                
                # Verify exactness
                if verify(U_r, V_r, W_r, r):
                    return U_r, V_r, W_r

        return None

    def verify(U, V, W, r):
        """Verify exact decomposition with integer/half-integer entries."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    s = 0.0
                    for t in range(r):
                        s += U[u_idx, t] * V[v_idx, t] * W[w_idx, t]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    # Try ranks in increasing order
    for r in [26, 27, 28, 29, 30]:
        if time.time() > deadline:
            break
            
        result = try_rank(r)
        if result:
            U, V, W = result
            # Convert to lists of lists as required by interface
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }

    # Fallback: Schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res["U"][0]))
