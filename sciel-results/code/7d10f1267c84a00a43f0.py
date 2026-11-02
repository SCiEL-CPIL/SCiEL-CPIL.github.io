"""Matrix multiplication 3x3x5 rank search via numerical optimization."""
import numpy as np
from scipy.optimize import minimize
import time

DIMENSIONS = [3, 3, 5]
DEADLINE = 120.0
START_TIME = None

def _tensor_entry(i, j, k, n, m, p):
    """Return the tensor entry T[i,j,k] for matrix multiplication.
    T is defined such that C[i,k] = sum_j A[i,j] * B[j,k].
    The tensor for matmul is T[i,j,k] = 1 if the index k in B matches j in A...
    
    Actually, the standard tensor for matmul (n x m) * (m x p) -> (n x p) is:
    T[i,j,k] where i in [0,n), j in [0,m), k in [0,p)
    T[i,j,k] = 1 if j == j (the shared index), but that's not a tensor decomposition.
    
    Let me reconsider. The matrix multiplication tensor M is a 3-tensor with entries:
    M[i,j,k] = 1 if the multiplication involves A[i,j] * B[j,k] -> C[i,k]
    
    For a decomposition U (n*m x r), V (m*p x r), W (p*n x r) with entries
    U[i*m+j, r], V[j*p+k, r], W[k*n+i, r], we need:
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = M[i,j,k]
    
    where M[i,j,k] = 1 if the indices (i,j,k) correspond to a valid multiplication
    term. Actually, the tensor M is:
    M[i,j,k] = 1 if j == j (always true?) No.
    
    Let me think more carefully. The matrix multiplication map is:
    (A, B) -> A*B
    A is n x m, B is m x p, C = A*B is n x p.
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    As a bilinear map, this is a tensor in (R^n x R^p) * (R^m)^2... 
    Actually, the tensor for matmul is in (R^n * R^m) * (R^m * R^p) * (R^n * R^p)*
    
    The standard formulation: define tensor T in R^{n*m} x R^{m*p} x R^{n*p}
    T[i*m+j, j'*p+k, (i' * p + k')] = 1 if i=i', j=j', k=k'
    
    Wait, that's not right either. Let me use the convention from the task:
    U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    
    The decomposition is: for all i,j,k:
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = M[i,j,k]
    
    where M[i,j,k] = 1 if (i,j,k) is a "valid" triple for matrix multiplication.
    
    For matrix multiplication (n x m) * (m x p) -> (n x p):
    The relevant triples are those where the j index in A matches the j index in B.
    So M[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    
    Wait, that means M is just the all-ones tensor? That can't be right.
    
    Let me re-read: "The output factor follows the official trace convention W[k*n+i,r]."
    
    I think the tensor is: M[i,j,k] = 1 if and only if the multiplication term A[i,j]*B[j,k] contributes to C[i,k]. But that's always the case for valid i,j,k.
    
    Actually, I think the correct interpretation is:
    M is a tensor in R^{(n*m)} x R^{(m*p)} x R^{(n*p)}
    M[a, b, c] where a = i*m+j, b = j'*p+k, c = i'*p + k'... 
    
    No, let me use the indexing from the task:
    U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    
    So the decomposition gives:
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i,j,k]
    
    For this to represent matrix multiplication, we need T[i,j,k] = 1 when the
    multiplication A[i,j] * B[j,k] -> C[i,k] is valid, which is always.
    
    Hmm, but that would make T the all-ones tensor, which has rank 1. That's clearly wrong.
    
    Let me look at the schoolbook decomposition:
    For each (i,j,k), create a rank-1 component with U[i*m+j]=1, V[j*p+k]=1, W[k*n+i]=1.
    
    This gives: sum over (i,j,k) of U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] 
    = sum over (i,j,k) of delta(i*m+j, i*m+j) * delta(j*p+k, j*p+k) * delta(k*n+i, k*n+i)
    
    Wait, each rank-1 component is sparse. Let me trace through:
    For a specific (i0,j0,k0), the component has:
    U[i0*m+j0, r] = 1, V[j0*p+k0, r] = 1, W[k0*n+i0, r] = 1
    
    So the product U[a,r]*V[b,r]*W[c,r] is nonzero only when:
    a = i0*m+j0, b = j0*p+k0, c = k0*n+i0
    
    So the full sum gives T[a,b,c] = 1 if there exists (i0,j0,k0) such that:
    a = i0*m+j0, b = j0*p+k0, c = k0*n+i0
    
    In terms of indices: a = i*m+j, b = j'*p+k, c = k'*n+i'
    We need j = j' and i = i' and k = k'.
    So T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k, and 0 otherwise.
    
    This is the correct matrix multiplication tensor!
    
    So T[i*m+j, j'*p+k, k'*n+i'] = 1 if j=j', i=i', k=k', else 0.
    """
    return 0  # We'll build the target tensor directly

def _build_target_tensor(n, m, p):
    """Build the target matrix multiplication tensor.
    T has shape (n*m, m*p, n*p) where
    T[i*m+j, j'*p+k, k'*n+i'] = 1 if i==i', j==j', k==k' else 0
    """
    T = np.zeros((n*m, m*p, n*p), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0
    return T

def _objective(params, U_shape, V_shape, W_shape, T_flat):
    """Compute the squared error of the decomposition."""
    n, m, p = U_shape[0] // U_shape[1], U_shape[1], W_shape[0] // U_shape[0]
    # Actually, let's pass n,m,p separately
    pass

def _decompose_via_optimization(n, m, p, rank, max_time=60.0):
    """Try to find a rank-rank decomposition via numerical optimization."""
    U_shape = (n*m, rank)
    V_shape = (m*p, rank)
    W_shape = (p*n, rank)
    
    T = _build_target_tensor(n, m, p)
    T_flat = T.flatten()
    
    def objective(params):
        U = params[:U_shape[0]*U_shape[1]].reshape(U_shape)
        V = params[U_shape[0]*U_shape[1] : U_shape[0]*U_shape[1] + V_shape[0]*V_shape[1]].reshape(V_shape)
        W = params[U_shape[0]*U_shape[1] + V_shape[0]*V_shape[1] :].reshape(W_shape)
        
        # Compute the reconstructed tensor
        # T_recon[i*m+j, j'*p+k, k'*n+i'] = sum_r U[i*m+j,r] * V[j'*p+k,r] * W[k'*n+i',r]
        # We only need to check entries where i==i', j==j', k==k'
        # Actually, we need ALL entries to match: T_recon should equal T
        # T is zero except at (i*m+j, j*p+k, k*n+i) where it's 1
        
        # Efficient computation: for each r, compute outer product
        T_recon = np.zeros((n*m, m*p, n*p), dtype=np.float64)
        for r in range(rank):
            T_recon += np.einsum('a,r->ar', U[:, r], np.ones(m*p))  # This is wrong
            # Let's do it properly
            T_recon_r = np.zeros((n*m, m*p, n*p), dtype=np.float64)
            for i in range(n):
                for j in range(m):
                    a = i*m + j
                    for j2 in range(m):
                        for k in range(p):
                            b = j2*p + k
                            for k2 in range(p):
                                for i2 in range(n):
                                    c = k2*n + i2
                                    T_recon_r[a, b, c] += U[a, r] * V[b, r] * W[c, r]
            T_recon += T_recon_r
        
        return np.sum((T_recon - T) ** 2)
    
    # This is way too slow. Let me use a different approach.
    # Use the fact that we only care about the specific entries.
    
    def objective_fast(params):
        U = params[:U_shape[0]*U_shape[1]].reshape(U_shape)
        V = params[U_shape[0]*U_shape[1] : U_shape[0]*U_shape[1] + V_shape[0]*V_shape[1]].reshape(V_shape)
        W = params[U_shape[0]*U_shape[1] + V_shape[0]*V_shape[1] :].reshape(W_shape)
        
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i*m + j
                    b = j*p + k
                    c = k*n + i
                    s = np.sum(U[a, :] * V[b, :] * W[c, :])
                    err += (s - 1.0) ** 2
        
        # Also need off-diagonal entries to be zero
        # T[a,b,c] = 0 unless a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k
        # with i==i', j==j', k==k'
        # So for a=i*m+j, b=j2*p+k, c=k2*n+i2, we need sum_r U[a,r]*V[b,r]*W[c,r] = 0
        # unless j==j2 and k==k2 and i==i2
        
        for i in range(n):
            for j in range(m):
                a = i*m + j
                for j2 in range(m):
                    for k in range(p):
                        b = j2*p + k
                        for i2 in range(n):
                            for k2 in range(p):
                                c = k2*n + i2
                                if j == j2 and k == k2 and i == i2:
                                    continue
                                s = np.sum(U[a, :] * V[b, :] * W[c, :])
                                err += s * s
                                if time.time() - START_TIME > DEADLINE - 5:
                                    return err
        return err
    
    # Random initialization
    np.random.seed(42)
    U_init = np.random.randn(*U_shape) * 0.1
    V_init = np.random.randn(*V_shape) * 0.1
    W_init = np.random.randn(*W_shape) * 0.1
    params0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
    
    best_err = objective_fast(params0)
    best_params = params0.copy()
    
    deadline = time.time() + max_time
    while time.time() < deadline:
        result = minimize(objective_fast, best_params, method='L-BFGS-B', 
                         options={'maxiter': 100, 'maxfun': 5000})
        if result.fun < best_err:
            best_err = result.fun
            best_params = result.x.copy()
        
        if best_err < 1e-10:
            break
        
        # Perturb
        np.random.seed(int(time.time() * 1000) % 100000)
        U_pert = best_params[:U_shape[0]*U_shape[1]].reshape(U_shape) + np.random.randn(*U_shape) * 0.1
        V_pert = best_params[U_shape[0]*U_shape[1] : U_shape[0]*U_shape[1] + V_shape[0]*V_shape[1]].reshape(V_shape) + np.random.randn(*V_shape) * 0.1
        W_pert = best_params[U_shape[0]*U_shape[1] + V_shape[0]*V_shape[1] :].reshape(W_shape) + np.random.randn(*W_shape) * 0.1
        params_pert = np.concatenate([U_pert.flatten(), V_pert.flatten(), W_pert.flatten()])
        pert_err = objective_fast(params_pert)
        if pert_err < best_err:
            best_err = pert_err
            best_params = params_pert.copy()
    
    if best_err > 1e-8:
        return None
    
    # Extract and round to rational
    U = best_params[:U_shape[0]*U_shape[1]].reshape(U_shape)
    V = best_params[U_shape[0]*U_shape[1] : U_shape[0]*U_shape[1] + V_shape[0]*V_shape[1]].reshape(V_shape)
    W = best_params[U_shape[0]*U_shape[1] + V_shape[0]*V_shape[1] :].reshape(W_shape)
    
    # Try to find rational approximations
    from fractions import Fraction
    
    def to_rational(x, max_denom=100):
        if abs(x) < 1e-10:
            return 0.0
        f = Fraction(x).limit_denominator(max_denom)
        return float(f)
    
    U_r = np.array([[to_rational(x) for x in row] for row in U])
    V_r = np.array([[to_rational(x) for x in row] for row in V])
    W_r = np.array([[to_rational(x) for x in row] for row in W])
    
    # Verify
    T_check = _build_target_tensor(n, m, p)
    T_recon = np.zeros((n*m, m*p, n*p), dtype=np.float64)
    for r in range(rank):
        for a in range(n*m):
            for b in range(m*p):
                for c in range(n*p):
                    T_recon[a, b, c] += U_r[a, r] * V_r[b, r] * W_r[c, r]
    
    if np.max(np.abs(T_recon - T_check)) < 1e-6:
        return U_r, V_r, W_r
    
    return None

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
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

# EVOLVE-BLOCK-START
def construct():
    global START_TIME
    START_TIME = time.time()
    
    n, m, p = DIMENSIONS
    
    # Try to find lower rank decompositions
    for rank in range(27, n * m * p):  # Start from known lower bounds
        if time.time() - START_TIME > DEADLINE - 10:
            break
        
        result = _decompose_via_optimization(n, m, p, rank, max_time=min(30, DEADLINE - (time.time() - START_TIME) - 10))
        if result is not None:
            U, V, W = result
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fall back to schoolbook
    return _schoolbook(n, m, p)
# EVOLVE-BLOCK-END
