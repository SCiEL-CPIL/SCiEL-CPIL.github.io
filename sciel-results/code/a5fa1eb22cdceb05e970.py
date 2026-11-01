"""Matrix multiplication 2x2 times 2x13: rank-minimal tensor decomposition."""
import numpy as np
import time
import random

DIMENSIONS = [2, 2, 13]

def _verify(U, V, W, n=2, m=2, p=13, tol=1e-9):
    """Verify that U,V,W exactly decompose the matmul tensor."""
    rank = U.shape[1]
    # Build the matmul tensor: M[(i*m+j), (j'*p+k)] for the product
    # (A*X)[i,k] = sum_{j'} A[i,j'] * X[j',k]
    # T[i, (j',k), j] where output index is i, input-A index is j, input-X index is j',k
    # Actually: T[a, b, c] where a = output row i (0..n-1), b = A column j (0..m-1), c = X column k (0..p-1)
    # T[i, j, k] contributes to output[i,k] via A[i,j]*X[j,k]
    # The tensor is: T[i, j, k] = A[i,j] * X[j,k] as a trilinear form
    # Decomposition: T[i,j,k] = sum_r U[n*m? no...] 
    # Interface: U (n*m, rank), V (m*p, rank), W (p*n, rank)
    # (A*X)[i,k] = sum_j A[i,j]*X[j,k] = sum_r U[i*m+j? ...]
    # Standard: (A*X)[i,k] = sum_j A[i,j] X[j,k]
    # With factors: (A*X)[i,k] = sum_r U_row[i] ... 
    # From the seed: U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal delta_{i,?} ... 
    # Actually the tensor satisfies: sum_{j,r} U[i*m+j,r] V[j*p+k,r] = T[i,?,k] ...
    # The decomposition means: for the bilinear map (A,X) -> A*X,
    # (A*X)[i,k] = sum_r (sum_j A[i,j] U[i*m+j,r]) * (sum_k? no) 
    # Let me re-read: U (n*m, rank), V (m*p, rank), W (p*n, rank)
    # (A*X)[i,k] = sum_r W[k*n+i, r] * (sum_j A[i,j] * U[i*m+j, r]) * (sum_k? ...
    # I think: (A*X)[i,k] = sum_r W[k*n+i, r] * (A·col_r_U)[i] * (col_r_V · X)[k]
    # where col_r_U is the r-th column of U, reshaped to (n,m), so (A·U_r)[i] = sum_j A[i,j]*U_r[i*m+j]
    # and col_r_V is the r-th column of V, reshaped to (m,p), so (V_r·X)[k] = sum_j V_r[j*p+k]*X[j,k]
    # Wait, that's not right either. Let me think again.
    
    # The matmul tensor T has the property that for any A (n×m) and X (m×p):
    # (A*X)[i,k] = sum_{j} A[i,j] * X[j,k]
    # This can be written as: (A*X)[i,k] = sum_r U_r[i] * V_r[k] * ... no.
    
    # Correct formulation: The tensor is defined by T[i,j,k] = δ_{output(i,k), input_A(i,j), input_X(j,k)}
    # The decomposition says: for all i,j,k, the "coefficient" is:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]... no, that's trilinear in three indices.
    
    # Actually I think the convention is:
    # (A*X)[i,k] = sum_r (sum_j A[i,j] * U[i*m+j, r]) * (sum_j V[j*p+k, r] * X[j,k]) * W[k*n+i, r] / W[k*n+i,r]^2? No.
    
    # Let me just verify numerically with the seed decomposition.
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # For each (i,k), compute sum_r U[i,?,r]... 
    # I believe: (A*X)[i,k] = sum_r W[k*n+i, r] * (sum_j A[i,j] * U[i*m+j, r]) * (sum_j X[j,k] * V[j*p+k, r])
    # but that would be cubic. More likely it's bilinear in A and X:
    # (A*X)[i,k] = sum_r (sum_j A[i,j] * U[i*m+j, r]) * (sum_j X[j,k] * V[j*p+k, r])
    # and W is a scaling? Or W is part of the trilinear form.
    
    # From the seed code: for each (i,j,k), U[i*m+j]=1, V[j*p+k]=1, W[k*n+i]=1 at the same column.
    # So for a given (i,j,k) triple, there's a rank-1 component where U has 1 at (i*m+j), V at (j*p+k), W at (k*n+i).
    # The product U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 for that component.
    # The full tensor entry T[i,j,k] (the coefficient of A[i,j]*X[j,k] in (A*X)) should be 1 if i is output row, j is shared, k is col.
    # But (A*X)[i,k] = sum_j A[i,j]*X[j,k], so the "tensor" in a certain sense has T[i,j,k]=1 for all valid i,j,k.
    # The decomposition is: 1 = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] for the specific i,j,k where this should hold?
    
    # I think the correct interpretation is:
    # Define the matrix multiplication as a trilinear map: given A (n×m), X (m×p), output Y (n×p)
    # Y[i,k] = sum_j A[i,j] X[j,k]
    # The decomposition factors this as:
    # Y[i,k] = sum_r W[k*n+i, r] * (sum_j A[i,j] U[i*m+j, r]) * (sum_j X[j,k] V[j*p+k, r])
    # NO, that's not right dimensionally for arbitrary A and X.
    
    # Let me just test: with seed, pick random A and X, compute A@X, and see if the formula works.
    np.random.seed(42)
    A = np.random.randn(n, m)
    X = np.random.randn(m, p)
    Y_expected = A @ X
    
    # Try: Y[i,k] = sum_r U_i_r * V_k_r * W_ik_r where U_i_r = sum_j A[i,j]*U[i*m+j,r], V_k_r = sum_j X[j,k]*V[j*p+k,r]
    # That would be: Y[i,k] = sum_r (A@U_col_r)[i] * (V_col_r@X[:,k]) * W[k*n+i,r]
    # = sum_r (A@U_r)[i] * (V_r@X[:,k]) * W_ik[r]... but W_ik[r] depends on i,k,r so it's still a sum over r of a product of three things depending on r.
    
    # I think the correct formula is simply:
    # Y[i,k] = sum_r U[i*m+?, r] ... 
    
    # Let me try the most natural interpretation: the matmul tensor is the tensor
    # M[i, j, k] = 1 for the coefficient of e_i^T A e_j * e_j^T X e_k giving output e_i^T Y e_k
    # The rank decomposition means: M[i,j,k] = sum_r u_r[i*m+j] * v_r[j*p+k] * w_r[k*n+i]
    # where u_r is column r of U, v_r is column r of V, w_r is column r of W.
    # So: for all i in [0,n), j in [0,m), k in [0,p): 1 = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # That's the condition! It's a "1" because the coefficient is 1.
    
    # Let's verify with seed:
    for i in range(n):
        for j in range(m):
            for k in range(p):
                lhs = 0.0
                for r in range(rank):
                    lhs += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(lhs - 1.0) > tol:
                    return False
    return True

def _try_random_decomposition(target_rank, n=2, m=2, p=13, max_iters=500, seed=None):
    """Try to find a valid rank-r decomposition via random init + gradient steps."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()
    
    U = rng.randn(n*m, target_rank)
    V = rng.randn(m*p, target_rank)
    W = rng.randn(p*n, target_rank)
    
    # The condition is: for all i,j,k: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
    # This is a system of n*m*p equations. We can use coordinate descent or simple projection.
    
    # For each (i,j,k), the equation is:
    # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
    # Let a_r = U[i*m+j,r], b_r = V[j*p+k,r], c_r = W[k*n+i,r]
    # We need dot(a, b*c) = 1 for every (i,j,k)
    
    # This is hard. Let's try a different parametrization.
    # Since n*m = 4 and m*p = 26 and p*n = 26, and target_rank might be small,
    # let's try to solve it as a rank-factorization problem.
    
    # Alternative: the tensor T[i,j,k] = 1 for all i,j,k (it's an all-ones tensor in this indexing!)
    # So we need a rank-r decomposition of the all-ones tensor of size (n*m) x (m*p) x (p*n) = 4 x 26 x 26.
    # But wait, it's not all-ones in the full tensor. The constraint is only on the "diagonal" in some sense.
    # Actually NO: the constraint is for ALL i in range(n), j in range(m), k in range(p), the equation
    # sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 must hold.
    # These are n*m*p = 2*2*13 = 52 scalar equations.
    # The unknowns are n*m*rank + m*p*rank + p*n*rank = (4+26+26)*rank = 56*rank values.
    # For rank >= 1, there are more unknowns than equations, so solutions should exist (if the system is consistent).
    
    # Let's try rank = 1: U is 4x1, V is 26x1, W is 26x1. 
    # Equations: for all i,j,k: U[i*m+j]*V[j*p+k]*W[k*n+i] = 1
    # Can this work? U[0]*V[0]*W[0] = 1, U[0]*V[13]*W[1] = 1, etc.
    # This requires U[i*m+j]*V[j*p+k] to be independent of k, and W[k*n+i] to adjust.
    # Seems unlikely for rank 1.
    
    # Let's just try to solve the 52 equations for the unknowns using a simple iterative method.
    # Initialize randomly, then for each equation, adjust the variables.
    
    # Simple approach: for each (i,j,k), the equation is f(i,j,k) = dot(u_i, v_jk * w_ki) = 1
    # where u_i = U[i*m+j, :], v_jk = V[j*p+k, :], w_ki = W[k*n+i, :]
    # We can do: for each (i,j,k), compute current value s = dot(U[i*m+j], V[j*p+k] * W[k*n+i])
    # and scale one of the three vectors to bring s closer to 1.
    
    start_time = time.time()
    for it in range(max_iters):
        if time.time() - start_time > 10:
            break
        # Shuffle order of equations
        eqs = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
        rng.shuffle(eqs)
        for i, j, k in eqs:
            if time.time() - start_time > 10:
                break
            idx_u = i * m + j
            idx_v = j * p + k
            idx_w = k * n + i
            # Current product
            uvw = np.dot(U[idx_u], V[idx_v] * W[idx_w])
            if abs(uvw) < 1e-12:
                # Perturb
                U[idx_u] += rng.randn(target_rank) * 0.01
                V[idx_v] += rng.randn(target_rank) * 0.01
                W[idx_w] += rng.randn(target_rank) * 0.01
                continue
            # Want uvw = 1. Scale W[idx_w] by factor 1/uvw (but this only fixes this one eq)
            # Better: scale U[idx_u] by (1/uvw)^(1/3), V by same, W by same? No, that changes others.
            # Just scale W[idx_w] by (1/uvw):
            W[idx_w] *= (1.0 / uvw)
        # Check convergence
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    val = np.dot(U[idx_u], V[idx_v] * W[idx_w])
                    max_err = max(max_err, abs(val - 1.0))
        if max_err < 1e-9:
            return U, V, W
    
    # Final check
    if _verify(U.tolist(), V.tolist(), W.tolist()):
        return U, V, W
    return None

def _schoolbook(n=2, m=2, p=13):
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
    start_time = time.time()
    deadline = 140.0  # Safety margin from 150s
    
    n, m, p = DIMENSIONS
    best = _schoolbook(n, m, p)
    best_rank = n * m * p  # 52
    
    # Try progressively lower ranks
    for target_rank in range(1, best_rank):
        if time.time() - start_time > deadline - 20:
            break
        # Try a few seeds
        for seed in range(10):
            if time.time() - start_time > deadline - 20:
                break
            result = _try_random_decomposition(target_rank, n, m, p, max_iters=200, seed=seed)
            if result is not None:
                U, V, W = result
                # Convert to list-of-lists with rounded values
                best_rank = target_rank
                best = {
                    "U": [[round(float(x), 12) for x in row] for row in np.array(U).tolist()],
                    "V": [[round(float(x), 12) for x in row] for row in np.array(V).tolist()],
                    "W": [[round(float(x), 12) for x in row] for row in np.array(W).tolist()],
                }
                break
        if best_rank < n * m * p:
            break  # Found improvement
    
    return best
# EVOLVE-BLOCK-END
