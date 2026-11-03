"""Matrix multiplication 2,4,7 tensor decomposition using rigorous linear algebraic rank reduction."""
import time
import numpy as np

DIMENSIONS = [2, 4, 7]
DEADLINE = 130  # Safety margin before 150s kill


def _verify(U, V, W, n, m, p, tol=1e-7):
    """Verify if the decomposition is exact using the tensor identity."""
    rank = U.shape[1]
    if rank == 0:
        return False
        
    # Efficient check using einsum
    # U: (n*m, rank) -> (n, m, rank)
    # V: (m*p, rank) -> (m, p, rank)
    # W: (p*n, rank) -> (p, n, rank)
    
    U_ = U.reshape(n, m, rank)
    V_ = V.reshape(m, p, rank)
    W_ = W.reshape(p, n, rank)
    
    # We want T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # Note: W is indexed by (k, i) in the tensor product for C[i,k]
    # The problem statement says W[k*n+i, r]. So W_ is (p, n, rank) where first index is k, second is i.
    # So W_[k, i, r] is correct.
    
    # T[i,j,k] = sum_r U_[i,j,r] * V_[j,k,r] * W_[k,i,r]
    # Let's transpose W_ to (n, p, rank) -> W_t[i, k, r] = W_[k, i, r]
    W_t = W_.transpose(1, 0, 2)
    
    T = np.einsum('ijr,jkr,ikr->ijk', U_, V_, W_t)
    
    if np.max(np.abs(T - 1.0)) > tol:
        return False
    return True


def _construct_schoolbook(n, m, p):
    """Return the schoolbook (rank n*m*p) decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    return U, V, W


def _construct_block_decomposition():
    """
    Construct 2x4x7 decomposition using block structure.
    This provides a starting point of rank 50.
    """
    n, m, p = 2, 4, 7
    
    # We will build this by combining smaller decompositions.
    # For now, let's just use the schoolbook if block construction is complex to verify quickly,
    # but the current program has a working one. I will include a simplified version or use schoolbook
    # to ensure correctness, then reduce.
    # Actually, the current program's block decomposition is valid. 
    # Let's try to construct a better initial seed if possible, or just use schoolbook (rank 56).
    # Schoolbook is safer to start with for the algebraic reduction.
    return _construct_schoolbook(n, m, p)


def _reduce_rank(U, V, W, n, m, p, tol=1e-9):
    """
    Iteratively reduce the rank of the decomposition by projecting onto column spaces.
    
    The tensor T is defined by the triplet (U, V, W).
    We can perform "Schmidt-like" orthogonalization:
    1. View U as a matrix of vectors u_r in C^{nm}.
    2. View V as a matrix of vectors v_r in C^{mp}.
    3. View W as a matrix of vectors w_r in C^{pn}.
    
    The decomposition is valid if T = sum_r u_r \otimes v_r \otimes w_r equals the target tensor.
    
    If we replace u_r with u_r' = u_r - proj_S(u_r) where S is the span of {u_s | s != r},
    the tensor changes unless v_r \otimes w_r is orthogonal to the "shadow" of the removed part.
    
    A more robust algebraic method:
    The set of valid decompositions of a fixed tensor T is a variety.
    We can minimize the rank by finding a basis for the subspace spanned by the terms.
    
    Algorithm:
    1. Start with current (U, V, W).
    2. Compute the "flattened" tensor matrices.
    3. Check for linear dependencies in the rank-1 components.
    
    Specifically, if we have a decomposition of rank R, we can check if the vectors 
    {u_r \otimes v_r \otimes w_r} are linearly independent.
    If not, we can combine them.
    
    However, a more direct geometric approach:
    Let S_U = col(U), S_V = col(V), S_W = col(W).
    The tensor T lies in S_U \otimes S_V \otimes S_W.
    
    We can iteratively project:
    - Project columns of U onto the subspace spanned by the "contractions" of V and W?
    
    Let's use the standard "peeling" via SVD of the flattened tensor.
    Actually, the most effective numerical method for reducing tensor rank of a specific decomposition
    is to check if any term is redundant.
    
    Term r is redundant if there exist coefficients c_s (s!=r) such that:
    u_r \otimes v_r \otimes w_r = sum_{s!=r} c_s (u_s \otimes v_s \otimes w_s)
    
    This is a system of linear equations.
    Let's flatten the tensor space.
    T_flat is a vector of size n*m*p.
    Each term t_r is a vector of size n*m*p.
    If the set {t_r} is linearly dependent, we can reduce the rank.
    
    We compute the Gram matrix G where G_{r,s} = <t_r, t_s> = (u_r^T u_s) (v_r^T v_s) (w_r^T w_s).
    If G is singular, the terms are linearly dependent.
    
    We can find the null space of G.
    Let z be a non-zero vector in null(G).
    Then sum_r z_r t_r = 0.
    This implies we can modify the decomposition.
    Specifically, if we add z_r * (something) to the other factors?
    
    Actually, if sum z_r t_r = 0, it doesn't directly mean we can remove a term.
    It means the terms are linearly dependent.
    We can replace the set of terms with a basis of their span.
    
    Let M be the matrix where column r is t_r (flattened).
    Rank(M) = Tensor Rank (for this specific decomposition).
    We can compute the SVD of M.
    If Rank(M) < R, we can reconstruct the decomposition with Rank(M) terms.
    
    Reconstruction:
    Let M = U_M S V_M^T.
    The non-zero singular values give the new rank R'.
    The new terms are the columns of M corresponding to non-zero singular values.
    But we need to factor these new terms back into U', V', W'.
    
    This is the "tensor rank decomposition" problem from a matrix.
    It is not trivial to factor a generic rank-R' tensor back into U', V', W' of size R'.
    However, if we start with a valid decomposition, and we just want to remove linear dependencies 
    among the *existing* terms, we can do this:
    
    1. Compute M (n*m*p, R).
    2. Compute Q, R_qr = qr(M). Q is (n*m*p, R_qr) orthonormal basis.
    3. M = Q R_qr.
    4. Now M is represented in a basis of size R_qr.
    5. But we need U', V', W' such that sum_{r'=1}^{R_qr} u'_{r'} \otimes v'_{r'} \otimes w'_{r'} = M.
    
    This step is hard.
    
    Alternative:
    Use the fact that the current program's "peeling" works by checking if removing a term still works.
    My improvement: Use the Gram matrix to identify *which* terms are dependent, and then solve for the
    coefficients to express the dependent term as a combination of others, then "absorb" it.
    
    If t_r = sum_{s!=r} c_s t_s, we can't just remove r.
    But we can update the factors.
    Let u_r, v_r, w_r be the factors for term r.
    We want to modify u_s, v_s, w_s for s!=r to include the contribution of r.
    This is generally impossible with a fixed number of terms unless we allow changes to all factors.
    
    However, if we allow the rank to decrease, we can simply check:
    Is the rank of the matrix M (flattened terms) less than R?
    If yes, the current representation is not minimal.
    
    Let's try a different structural change:
    **Randomized SVD based reduction**.
    1. Generate random vectors a in C^{nm}, b in C^{mp}, c in C^{pn}.
    2. Project the tensor onto these?
    
    Let's stick to the most robust mathematical fact:
    The minimal rank is the rank of the tensor.
    For a specific decomposition, if the terms are linearly dependent, we can reduce the count.
    
    Let's implement a check for linear dependence of the terms t_r.
    If dependent, we can reduce R by 1.
    How to update U, V, W?
    If t_r is in the span of others, we can't just drop it.
    But we can replace the set of terms with a basis.
    
    Wait, the "score" is the rank R.
    If I have a decomposition with R terms, and they are linearly dependent, the *effective* rank is lower.
    But I must return U, V, W with shape (., R').
    If I just drop the column, the tensor changes!
    
    So I must *transform* the decomposition.
    
    Let's use the following theorem:
    If we have a decomposition T = sum_{r=1}^R u_r \otimes v_r \otimes w_r,
    and we apply a linear transformation T_r' = sum_s M_{rs} u_s, etc., the tensor is preserved.
    
    If we can find a transformation that makes one row of the transformation matrix zero,
    we can drop that term.
    
    This is equivalent to finding a vector z in the null space of the "coefficient matrix" of the dependencies.
    
    Let's try a simpler heuristic that is structurally different:
    **Optimization based on Random Projections**.
    
    Actually, looking at the current program, it uses `_try_rank_reduction` which just tries to *remove* columns.
    This only works if the column was exactly zero in the tensor contribution (which is rare) or if the verification logic is flawed (it isn't).
    Wait, `_verify` checks if the *remaining* columns sum to 1.
    If I remove a column, the sum changes. So `_verify` will fail unless the removed column was 0.
    So the current program's reduction logic is likely failing to reduce rank at all, or only reducing if there are zeros.
    
    My strategy: **Correct Rank Reduction via Linear Algebra**.
    
    1. Compute the flattened tensor matrix M (n*m*p, R).
    2. Compute the rank R_eff of M.
    3. If R_eff < R, we can reduce the rank.
    4. To do this, we find a basis for the column space of M.
    5. We express the original columns as linear combinations of the basis.
    6. We need to factor the new basis vectors back into U', V', W'.
    
    This factorization is the hard part.
    
    Alternative Strategy: **Start from a different construction**.
    The current program starts with Rank 56 (schoolbook) or 50 (block).
    The theoretical lower bound is 28.
    
    Let's try to construct a decomposition of rank 28 directly?
    Lower bound is max(n*m, m*p, p*n) = max(8, 28, 14) = 28.
    
    Is it possible to achieve rank 28?
    This would require the tensor to be "rank 28".
    
    Let's try to find a decomposition of rank 28 using a structured search.
    We can parameterize U, V, W with 28 columns.
    We want to solve for U, V, W such that T = 1.
    
    This is a system of non-linear equations.
    We can use `scipy.optimize.least_squares` or `minimize`.
    
    Variables:
    U: 8 * 28 = 224
    V: 28 * 28 = 784
    W: 14 * 28 = 392
    Total: 1400 variables.
    
    Constraints:
    T[i,j,k] = 1 for all i,j,k (56 constraints).
    
    This is a highly underdetermined system (1400 vars, 56 eqs).
    We can easily find a solution if one exists.
    
    Let's try to solve for a rank 28 decomposition.
    If it fails, try 29, 30, etc.
    
    This is a fundamentally different approach: **Direct Parameterization and Optimization**.
    
    We will define a function that takes a rank R, initializes U, V, W randomly (or structured),
    and uses gradient descent to minimize the error ||T - 1||^2.
    
    Since the problem is non-convex, we need multiple restarts.
    We have 120s.
    
    Let's implement this.
"""

import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 4, 7]
DEADLINE = 130


def _compute_error(params, n, m, p, rank):
    """
    Compute the error for a given parameter vector.
    params: flattened [U, V, W]
    """
    size_U = n * m * rank
    size_V = m * p * rank
    
    U_flat = params[:size_U]
    V_flat = params[size_U:size_U + size_V]
    W_flat = params[size_U + size_V:]
    
    U = U_flat.reshape(n, m, rank)
    V = V_flat.reshape(m, p, rank)
    W = W_flat.reshape(p, n, rank)
    
    # Compute T[i,j,k]
    W_t = W.transpose(1, 0, 2) # (n, p, rank)
    T = np.einsum('ijr,jkr,ikr->ijk', U, V, W_t)
    
    # Error is sum of squares of (T - 1)
    diff = T - 1.0
    return np.sum(diff * diff)


def _compute_error_and_grad(params, n, m, p, rank):
    """
    Compute error and gradient using finite differences or analytical derivation.
    For speed, let's use finite differences if analytical is too complex to code quickly,
    but analytical is better.
    
    T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    
    dT[i,j,k] / dU[i',j',r'] = delta_{ii'} delta_{jj'} V[j,k,r'] W[k,i,r']
    dT[i,j,k] / dV[j',k',r'] = U[i,j',r'] delta_{jj'} W[k',i,r'] delta_{kk'}
    dT[i,j,k] / dW[k',i',r'] = U[i,j,r'] V[j,k',r'] delta_{kk'} delta_{ii'}
    
    Gradient of Error E = sum_{i,j,k} (T[i,j,k] - 1)^2
    dE/dU[i',j',r'] = 2 * sum_{i,j,k} (T[i,j,k] - 1) * dT[i,j,k]/dU[i',j',r']
    = 2 * sum_k (T[i',j',k] - 1) * V[j',k,r'] * W[k,i',r']
    
    Similarly for V and W.
    """
    size_U = n * m * rank
    size_V = m * p * rank
    
    U_flat = params[:size_U]
    V_flat = params[size_U:size_U + size_V]
    W_flat = params[size_U + size_V:]
    
    U = U_flat.reshape(n, m, rank)
    V = V_flat.reshape(m, p, rank)
    W = W_flat.reshape(p, n, rank)
    
    W_t = W.transpose(1, 0, 2)
    T = np.einsum('ijr,jkr,ikr->ijk', U, V, W_t)
    diff = T - 1.0
    error = np.sum(diff * diff)
    
    # Gradient for U
    # dE/dU[i,j,r] = 2 * sum_k diff[i,j,k] * V[j,k,r] * W[k,i,r]
    # Note: W[k,i,r] is W_[k,i,r] in the original layout.
    # In our W array (p,n,rank), W[k,i,r] is directly accessible.
    
    # einsum for grad_U:
    # diff[i,j,k] * V[j,k,r] * W[k,i,r]
    # Indices: i,j,k,r
    # Result: i,j,r
    grad_U = 2.0 * np.einsum('ijk,jkr,kr->ijr', diff, V, W)
    
    # Gradient for V
    # dE/dV[j,k,r] = 2 * sum_i diff[i,j,k] * U[i,j,r] * W[k,i,r]
    # Wait, check derivative of T wrt V[j',k',r']:
    # T[i,j,k] depends on V[j,k,r] only if j'=j, k'=k.
    # dT[i,j,k]/dV[j,k,r] = U[i,j,r] * W[k,i,r]
    # So dE/dV[j,k,r] = 2 * sum_i diff[i,j,k] * U[i,j,r] * W[k,i,r]
    
    # einsum for grad_V:
    # diff[i,j,k] * U[i,j,r] * W[k,i,r]
    # Indices: i,j,k,r
    # Result: j,k,r
    grad_V = 2.0 * np.einsum('ijk,ijr,kr->jkr', diff, U, W)
    
    # Gradient for W
    # dE/dW[k,i,r] = 2 * sum_j diff[i,j,k] * U[i,j,r] * V[j,k,r]
    # Check derivative of T wrt W[k',i',r']:
    # T[i,j,k] depends on W[k,i,r] only if k'=k, i'=i.
    # dT[i,j,k]/dW[k,i,r] = U[i,j,r] * V[j,k,r]
    # So dE/dW[k,i,r] = 2 * sum_j diff[i,j,k] * U[i,j,r] * V[j,k,r]
    
    # einsum for grad_W:
    # diff[i,j,k] * U[i,j,r] * V[j,k,r]
    # Indices: i,j,k,r
    # Result: k,i,r
    grad_W = 2.0 * np.einsum('ijk,ijr,jkr->kir', diff, U, V)
    
    # Flatten gradients
    grad_flat = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
    
    return error, grad_flat


def _try_find_decomposition(n, m, p, rank, time_budget=10):
    """
    Try to find a decomposition of the given rank using gradient descent.
    Returns (U, V, W) if found, else None.
    """
    start_time = time.time()
    rng = np.random.default_rng(42 + rank)
    
    size_U = n * m * rank
    size_V = m * p * rank
    size_W = p * n * rank
    total_size = size_U + size_V + size_W
    
    best_error = float('inf')
    best_params = None
    
    # Multiple restarts
    max_restarts = 5
    for r in range(max_restarts):
        if time.time() - start_time > time_budget:
            break
        
        # Initialize parameters
        # Scale initialization to help convergence
        # U, V, W should be O(1)
        params = rng.standard_normal(total_size) * 1.0
        
        # Use L-BFGS-B
        try:
            res = minimize(
                _compute_error,
                params,
                jac=lambda x: _compute_error_and_grad(x, n, m, p, rank)[1],
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 200}
            )
            
            if res.success or res.fun < best_error:
                if res.fun < 1e-8:
                    # Found a solution!
                    best_error = res.fun
                    best_params = res.x
                    break # Stop early if we found a perfect solution
                else:
                    best_error = res.fun
                    best_params = res.x
        except Exception:
            continue
    
    if best_params is None:
        return None
        
    if best_error > 1e-6:
        return None
        
    # Extract U, V, W
    U_flat = best_params[:size_U]
    V_flat = best_params[size_U:size_U + size_V]
    W_flat = best_params[size_U + size_V:]
    
    U = U_flat.reshape(n * m, rank)
    V = V_flat.reshape(m * p, rank)
    W = W_flat.reshape(p * n, rank)
    
    # Verify
    if _verify(U, V, W, n, m, p):
        return U, V, W
    else:
        return None


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Lower bound is max(n*m, m*p, p*n) = 28
    # Upper bound is 56 (schoolbook)
    
    # Try ranks from lower bound upwards
    # We have ~120s. 
    # Let's allocate time per rank attempt.
    
    # First, get a valid fallback (Schoolbook)
    U_fallback, V_fallback, W_fallback = _construct_schoolbook(n, m, p)
    
    best_U, best_V, best_W = U_fallback, V_fallback, W_fallback
    best_rank = U_fallback.shape[1]
    
    # Try to find lower rank decompositions
    # Start from 28, go up to best_rank - 1
    # We can parallelize this in our head, but we run sequentially.
    
    # Time budget per rank: 
    # If we try 28 ranks, and have 100s, that's ~3s per rank.
    # L-BFGS-B might be slow. Let's limit iterations.
    
    for rank in range(28, best_rank):
        if time.time() - start_time > DEADLINE - 10:
            break
            
        # Allocate time based on how many ranks are left
        ranks_left = best_rank - rank
        time_per_rank = min(5.0, (DEADLINE - 10 - (time.time() - start_time)) / max(1, ranks_left))
        
        if time_per_rank < 1.0:
            break
            
        result = _try_find_decomposition(n, m, p, rank, time_budget=time_per_rank)
        
        if result is not None:
            U, V, W = result
            if _verify(U, V, W, n, m, p):
                best_U, best_V, best_W = U, V, W
                best_rank = rank
                # If we found the lower bound, we can't do better
                if rank == 28:
                    break
    
    # Final safety check
    if not _verify(best_U, best_V, best_W, n, m, p):
        best_U, best_V, best_W = U_fallback, V_fallback, W_fallback
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }


if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
