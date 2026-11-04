"""Matrix multiplication 2x3x3 rank search with deadline guard."""
import time
import numpy as np

DIMENSIONS = [2, 3, 3]
DEADLINE_SECONDS = 115.0  # leave 5s margin for the 120s evaluator limit


def _verify(n, m, p, U, V, W, tol=1e-9):
    """Check that U, V, W exactly decompose the (n,m,p) multiplication tensor."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    rank = U.shape[1]
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if k * n + i == j:
                    target = 1.0
                else:
                    target = 0.0
                err = abs(val - target)
                if err > max_err:
                    max_err = err
    return max_err < tol


def _construct_rank18():
    """Schoolbook rank-18 decomposition (always valid fallback)."""
    n, m, p = 2, 3, 3
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def _construct_rank17():
    """
    Construct a rank-17 decomposition using a block-Sassenhouse approach.
    For 2x3x3 we can exploit the structure: split the 2x3 matrix into two 1x3 rows
    and use Strassen-like tricks on sub-problems.
    
    Actually, let me try a more principled approach:
    The multiplication tensor for (2,3,3) has entries T[i,j,k] = delta_{i+j,k} 
    where i in {0,1}, j in {0,1,2}, k in {0,1,2,3,4} -- wait, that's not right.
    
    Let me re-read the convention. The output factor follows W[k*n+i, r].
    So W is indexed by (k, i) with k in {0..p-1}={0,1,2} and i in {0..n-1}={0,1}.
    W row index = k*n + i.
    
    The tensor entry T[i, j, k] (where i=row of A, j=col of A / row of B, k=col of B)
    equals 1 if and only if the (i,k) entry of AB depends on A[i,j]*B[j,k].
    By the trace convention, T[i,j,k] = 1 for all valid (i,j,k) triples, and the
    decomposition must satisfy sum_r U[i,j,r]*V[j,k,r]*W[k*n+i,r] = T[i,j,k].
    
    Wait - actually for matrix multiplication, T[i,j,k] = 1 for ALL i,j,k.
    The "tensor" here is the all-ones tensor of shape (n,m,p) = (2,3,3).
    That's the standard matrix multiplication tensor where every entry is 1.
    
    Hmm, that doesn't sound right either. Let me reconsider.
    
    In the standard matrix multiplication tensor, C[i,k] = sum_j A[i,j] * B[j,k].
    The tensor T has indices (i, j, k) and T[i,j,k] = 1 for all valid indices,
    because C[i,k] = sum_j (1) * A[i,j] * B[j,k].
    
    So yes, T[i,j,k] = 1 for all i in {0,1}, j in {0,1,2}, k in {0,1,2}.
    
    The decomposition is: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k.
    
    This is equivalent to saying that the all-ones tensor of shape (2,3,3) has
    rank <= R. The rank of the all-ones tensor [n,m,p] is known to be related
    to the tensor rank of the multiplication problem.
    
    For the all-ones tensor of shape (n,m,p), a trivial decomposition uses n*m*p
    terms. But we can do better by sharing structure.
    
    Key insight: sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1 for all i,j,k.
    
    If we set U[i,j,r] = a_{i,r}, V[j,k,r] = b_{j,r}, W[k,i,r] = c_{k,i,r} ... no,
    the indices are coupled through j and k.
    
    Let me think about this differently. We need:
    For all (i, j, k): sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1
    
    This means that for each fixed i and k, the matrix sum_r W[k,i,r] * 
    outer(U[i,:,r], V[:,k,r]) must equal the all-ones matrix of size m x m... 
    no wait.
    
    For fixed i, k: sum_j ... no. For fixed (i,k), we need:
    sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 for ALL j.
    
    So for fixed (i,k): let w_r = W[k,i,r]. Then for each j:
    sum_r w_r * U[i,j,r] * V[j,k,r] = 1.
    
    This is a system of m equations (one per j) with R unknowns (the w_r values),
    but U and V are also unknowns.
    
    A simple approach: use rank-18 schoolbook. Can we get rank-17?
    
    Let me try to find a rank-17 decomposition via numerical optimization.
    """
    n, m, p = 2, 3, 3
    target_rank = 17
    
    # Variables: U (n*m x R), V (m*p x R), W (p*n x R)
    # Total variables: (n*m + m*p + p*n) * R = (6 + 9 + 6) * 17 = 21 * 17 = 357
    
    # We'll use scipy.optimize to minimize the squared error
    try:
        from scipy.optimize import minimize
    except ImportError:
        return _construct_rank18()
    
    def make_initial(R, seed=42):
        rng = np.random.RandomState(seed)
        U = rng.randn(n * m, R) * 0.5
        V = rng.randn(m * p, R) * 0.5
        W = rng.randn(p * n, R) * 0.5
        return U, V, W
    
    def compute_error(params, R):
        U = params[:n * m * R].reshape(n * m, R)
        V = params[n * m * R : n * m * R + m * p * R].reshape(m * p, R)
        W = params[n * m * R + m * p * R :].reshape(p * n, R)
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(R):
                        val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    err += (val - 1.0) ** 2
        return err
    
    # Try several random initializations
    best_solution = None
    best_err = float('inf')
    
    start_time = time.time()
    
    for seed in range(50):
        if time.time() - start_time > DEADLINE_SECONDS - 10:
            break
        
        U0, V0, W0 = make_initial(target_rank, seed)
        params0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
        
        result = minimize(
            compute_error, params0, args=(target_rank,),
            method='L-BFGS-B',
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        
        err = result.fun
        if err < best_err:
            best_err = err
            if err < 1e-12:
                best_solution = result.x
                break
    
    if best_solution is not None and best_err < 1e-10:
        U = best_solution[:n * m * R].reshape(n * m, R) if False else \
            best_solution[:n * m * target_rank].reshape(n * m, target_rank)
        V = best_solution[n * m * target_rank : n * m * target_rank + m * p * target_rank].reshape(m * p, target_rank)
        W = best_solution[n * m * target_rank + m * p * target_rank :].reshape(p * n, target_rank)
        
        # Verify
        if _verify(n, m, p, U, V, W, tol=1e-8):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return _construct_rank18()


def _try_rank16_numerical():
    """Attempt to find a rank-16 decomposition via numerical optimization."""
    n, m, p = 2, 3, 3
    target_rank = 16
    
    try:
        from scipy.optimize import minimize
    except ImportError:
        return None
    
    def make_initial(R, seed):
        rng = np.random.RandomState(seed)
        U = rng.randn(n * m, R) * 0.3
        V = rng.randn(m * p, R) * 0.3
        W = rng.randn(p * n, R) * 0.3
        return U, V, W
    
    def compute_error(params, R):
        U = params[:n * m * R].reshape(n * m, R)
        V = params[n * m * R : n * m * R + m * p * R].reshape(m * p, R)
        W = params[n * m * R + m * p * R :].reshape(p * n, R)
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(R):
                        val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    err += (val - 1.0) ** 2
        return err
    
    best_solution = None
    best_err = float('inf')
    start_time = time.time()
    
    for seed in range(80):
        if time.time() - start_time > DEADLINE_SECONDS - 20:
            break
        
        U0, V0, W0 = make_initial(target_rank, seed)
        params0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
        
        result = minimize(
            compute_error, params0, args=(target_rank,),
            method='L-BFGS-B',
            options={'maxiter': 150, 'ftol': 1e-14, 'gtol': 1e-10}
        )
        
        err = result.fun
        if err < best_err:
            best_err = err
            if err < 1e-12:
                best_solution = result.x
                break
    
    if best_solution is not None and best_err < 1e-10:
        U = best_solution[:n * m * target_rank].reshape(n * m, target_rank)
        V = best_solution[n * m * target_rank : n * m * target_rank + m * p * target_rank].reshape(m * p, target_rank)
        W = best_solution[n * m * target_rank + m * p * target_rank :].reshape(p * n, target_rank)
        
        if _verify(n, m, p, U, V, W, tol=1e-8):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return None


def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x3x3 matrix multiplication.
    Returns {'U', 'V', 'W'} with shapes (n*m, rank), (m*p, rank), (p*n, rank).
    """
    start_time = time.time()
    
    # Try rank 16 first
    result16 = _try_rank16_numerical()
    if result16 is not None:
        return result16
    
    if time.time() - start_time > DEADLINE_SECONDS:
        return _construct_rank18()
    
    # Try rank 17
    result17 = _construct_rank17()
    if result17 is not None:
        r = result17["U"][0] if result17["U"] else []
        if len(r) == 17:
            return result17
    
    if time.time() - start_time > DEADLINE_SECONDS:
        return _construct_rank18()
    
    # Fallback to rank 18
    return _construct_rank18()
