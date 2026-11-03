"""Find a rank-minimal exact tensor decomposition for 3x5 by 5x5 matrix multiplication."""
import time
import itertools
import numpy as np

DIMENSIONS = [3, 5, 5]
DEADLINE = 150.0
SAFETY_MARGIN = 5.0  # seconds before deadline to stop searching

def _build_schoolbook(n=3, m=5, p=5):
    """Construct the trivial schoolbook decomposition of rank n*m*p."""
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

def _verify_decomp(U, V, W, n=3, m=5, p=5, tol=1e-10):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor.
    
    The tensor T[i,j,k] should equal sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    for all i in [0,n), j in [0,m), k in [0,p).
    T[i,j,k] = 1 if this is the identity-like tensor for matmul, else 0.
    Actually for matrix multiplication: (A*B)[i,k] = sum_j A[i,j]*B[j,k]
    The tensor has T[i,j,k] = 1 for all valid i,j,k (it's the all-ones tensor in 
    the appropriate sense - actually it's the identity in the middle index).
    
    Wait, let me reconsider. For matrix multiplication of (n x m) by (m x p):
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    The tensor T has indices (i, j, k) where T[i,j,k] = 1 for all i,j,k.
    No wait - the tensor decomposition represents:
    C[i,k] = sum_r (sum_j A[i,j] * U[i*m+j, r]) * (sum_k' ... )
    
    Actually the standard form is:
    T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]  (cyclic)
    
    For matrix multiplication, T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    Because C[i,k] = sum_j A[i,j] B[j,k], so the "tensor" is the all-ones tensor.
    
    Wait no. Let me think again carefully.
    
    The matrix multiplication tensor T ∈ R^{n × m × p} is defined by:
    T[i, j, k] = 1 if the (i,k) entry of A*B depends on A[i,j] and B[j,k].
    
    Since C[i,k] = sum_j A[i,j] * B[j,k], every triple (i,j,k) contributes.
    So T[i,j,k] = 1 for ALL i,j,k.
    
    A rank-r decomposition: T[i,j,k] = sum_{r=0}^{r-1} U[i,j,r] * V[j,k,r] * W[k,i,r]
    
    With the trace convention W[k*n+i, r], the third factor indexes W by (k, i).
    
    So we need: for all i in [0,n), j in [0,m), k in [0,p):
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    """
    rank = len(U[0]) if U else 0
    # Build numpy arrays
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(np.dot(Ua[i*m+j], Va[j*p+k]), Wa[k*n+i])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _try_rank(r, n=3, m=5, p=5, time_limit=None, start_time=None):
    """Try to find a rank-r decomposition using random search + local optimization.
    
    Returns (U, V, W) if found, else None.
    """
    if start_time is None:
        start_time = time.time()
    if time_limit is None:
        time_limit = DEADLINE - SAFETY_MARGIN
    
    nm = n * m  # 15
    mp = m * p  # 25
    pn = p * n  # 15
    
    # The tensor T[i,j,k] = 1 for all i,j,k
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all (i,j,k)
    
    # Strategy: Random initialization + gradient descent on the residual
    # Residual R = T - U*V*W (tensor product)
    # Minimize ||R||^2
    
    # Number of constraints: n*m*p = 75
    # Number of variables: 3 * r (for U, V, W) = 3r
    # For r < 75, we have more variables than constraints potentially
    
    # Let's use a different approach: 
    # Fix U and V randomly, then solve for W (linear system)
    # Or use the fact that this is a bilinear/trilinear problem
    
    # Better approach: Random init, then iterative refinement
    # For each (i,j,k), we need sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
    
    # Let's try: random U, V, then for each k*n+i, solve a linear system for W[k*n+i, :]
    # But W[k*n+i, r] appears multiplied by U[i*m+j, r]*V[j*p+k, r]
    # For fixed (j,k), the equation is: sum_r (U[i*m+j,r]*V[j*p+k,r]) * W[k*n+i,r] = 1 for all i
    # That's n equations in r unknowns (for each fixed j,k)
    
    # Actually for fixed (j,k): we need for all i in [0,n):
    # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
    # This is n linear equations in the r unknowns W[k*n+i,r] for r=0..r-1
    # But W[k*n+i,r] depends on i! So it's not a single linear system.
    
    # Let me re-index. For fixed (j,k), define for each i:
    # a_i[r] = U[i*m+j,r] * V[j*p+k,r]  (length r vector)
    # Then we need: a_i · W[k*n+i, :] = 1 for i = 0..n-1
    
    # So for each i, we have one linear equation: a_i · w_i = 1 where w_i = W[k*n+i,:]
    # These are independent equations (different w_i for different i)
    # So for fixed (j,k), we need to find w_0, ..., w_{n-1} such that a_i · w_i = 1
    # This is always solvable if a_i != 0 (just pick w_i = a_i / ||a_i||^2)
    
    # But the same W[k*n+i,r] is used across different j values!
    # W[k*n+i, r] appears in equations for all j: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
    # So for fixed (k,i), we have m equations (one for each j) in r unknowns W[k*n+i,0..r-1]
    # This is a system of m linear equations in r unknowns.
    # For r >= m, this is generally solvable.
    
    # So the strategy is:
    # 1. Randomly initialize U and V
    # 2. For each (k,i), solve the linear system: for all j, sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
    #    i.e., let A_{k,i} be an m x r matrix with A_{k,i}[j,r] = U[i*m+j,r]*V[j*p+k,r]
    #    Then we need A_{k,i} * W[k*n+i,:] = ones(m)
    # 3. Check if the resulting W satisfies ALL constraints (it should by construction for the (k,i) groups)
    #    Wait, we solved for W given U,V. The constraint is exactly what we solved for.
    #    So if the linear systems are solvable, we get a valid decomposition!
    
    # The key question: when are the linear systems solvable?
    # A_{k,i} is m x r. We need A_{k,i} * w = 1_m to have a solution.
    # This requires 1_m to be in the column space of A_{k,i}, i.e., in the row space... 
    # Actually, A*w = b has a solution iff b is in the column space of A,
    # which is equivalent to b being orthogonal to the left null space of A.
    # For random U,V, A_{k,i} will have full row rank m (if r >= m) with high probability,
    # so any b (including 1_m) is in the column space.
    
    # So for r >= m = 5, random U,V should work!
    # Let's verify: r >= 5 should give a valid decomposition.
    
    # Actually wait - we need r >= m for each (k,i) system to be solvable.
    # m = 5, so r >= 5 should work.
    
    # But can we do better? The lower bound for 3x5x5 matmul...
    # Known results: R(3,5,5) is between some values.
    # The trivial bound is 75. Strassen-like improvements might help.
    # For square n x n x n, best known is ~n^2.5 for large n.
    # For 3x5x5, let's just try to find the minimum r.
    
    # Let's start with r = m = 5 and try random U,V.
    
    best_result = None
    
    for attempt in range(200):
        if time.time() - start_time > time_limit:
            break
        
        # Random initialization of U and V
        Ua = np.random.randn(nm, r)
        Va = np.random.randn(mp, r)
        
        # Solve for W
        Wa = np.zeros((pn, r))
        feasible = True
        for k in range(p):
            for i in range(n):
                # A[j, r_idx] = U[i*m+j, r_idx] * V[j*p+k, r_idx] for j in 0..m-1
                A = np.array([Ua[i*m+j] * Va[j*p+k] for j in range(m)])  # m x r
                b = np.ones(m)
                # Solve A * w = b
                try:
                    w, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
                    # Check if solution is exact
                    if np.linalg.norm(A @ w - b) < 1e-10:
                        Wa[k*n+i] = w
                    else:
                        feasible = False
                        break
                except:
                    feasible = False
                    break
            if not feasible:
                break
        
        if feasible:
            # Convert to lists
            U_list = Ua.tolist()
            V_list = Va.tolist()
            W_list = Wa.tolist()
            
            # Verify
            if _verify_decomp(U_list, V_list, W_list, n, m, p):
                best_result = {"U": U_list, "V": V_list, "W": W_list}
                break
    
    return best_result

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    time_limit = DEADLINE - SAFETY_MARGIN
    
    # Always have the schoolbook as fallback
    schoolbook = _build_schoolbook(n, m, p)
    
    # Try to find lower rank decompositions
    # Lower bound: at least max(n*m, m*p, p*n) = max(15, 25, 15) = 25? No.
    # Actually the rank must be at least the "slice rank" or other lower bounds.
    # For now, try r from small to large.
    
    # The minimum possible rank: we need at least enough to span the constraints.
    # From our analysis, r >= m = 5 should work with random U,V.
    # Let's try r = 5, 6, 7, ...
    
    for r in range(5, 80):
        if time.time() - start_time > time_limit:
            break
        
        result = _try_rank(r, n, m, p, time_limit, start_time)
        if result is not None:
            return result
    
    # Fallback to schoolbook
    return schoolbook

if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
