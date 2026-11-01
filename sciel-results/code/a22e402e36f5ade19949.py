"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START

def _build_tensor(n, m, p):
    """Build the 24-dimensional vector representing the matrix multiplication tensor.
    
    The tensor T has entries T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    We flatten it as a vector of length n*m*p with index = i*m*p + j*p + k.
    """
    T = np.ones(n * m * p)
    return T

def _tensor_to_vector(T, n, m, p):
    """Flatten tensor T[i,j,k] to a vector with index i*m*p + j*p + k."""
    return T.flatten()

def _vector_to_tensor(v, n, m, p):
    """Reshape vector back to tensor T[i,j,k]."""
    return v.reshape(n, m, p)

def _generate_candidate_terms(n, m, p, max_val=2):
    """Generate candidate rank-1 terms with integer/half-integer entries.
    
    A rank-1 term is defined by three vectors:
    - u in R^(n*m) with entries in {-max_val, ..., max_val} * 0.5
    - v in R^(m*p) with entries in {-max_val, ..., max_val} * 0.5  
    - w in R^(p*n) with entries in {-max_val, ..., max_val} * 0.5
    
    The contribution to the tensor is: T[i,j,k] += u[i*m+j] * v[j*p+k] * w[k*n+i]
    
    We'll generate a pool of candidates and select a subset that spans the tensor.
    """
    # For efficiency, we'll use a more structured approach.
    # The trivial decomposition uses 24 terms where each term has exactly one non-zero entry
    # in each of u, v, w (all equal to 1).
    
    # Let's generate candidates based on patterns:
    candidates = []
    
    # Pattern 1: Standard basis-like terms (the trivial 24)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = [0] * (n * m)
                v = [0] * (m * p)
                w = [0] * (p * n)
                u[i * m + j] = 1
                v[j * p + k] = 1
                w[k * n + i] = 1
                candidates.append((u, v, w))
    
    # Pattern 2: Terms with two non-zero entries in one vector
    # For example, combine two standard terms
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for i2 in range(n):
                    if i2 == i:
                        continue
                    for j2 in range(m):
                        for k2 in range(p):
                            # Try combining (i,j,k) and (i2,j2,k2)
                            u = [0] * (n * m)
                            v = [0] * (m * p)
                            w = [0] * (p * n)
                            u[i * m + j] = 1
                            u[i2 * m + j2] = 1
                            v[j * p + k] = 1
                            v[j2 * p + k2] = 1
                            w[k * n + i] = 1
                            w[k2 * n + i2] = 1
                            # This gives contributions to 4 tensor entries, not just 1
                            # So this is not a valid rank-1 term for our purpose
                            pass
    
    # Let's take a different approach: use a greedy algorithm to find a small set
    # of rank-1 tensors whose sum equals the all-ones tensor.
    
    return candidates

def _greedy_select_decomposition(n, m, p, target_rank, time_limit=60):
    """Try to find a rank-`target_rank` decomposition using greedy selection.
    
    Strategy: 
    1. Start with the trivial rank-24 decomposition.
    2. Try to merge pairs of terms that can be replaced by a single term.
    3. Use local search to find better combinations.
    """
    start_time = time.time()
    
    # The tensor we want to represent is all ones: T[i,j,k] = 1 for all (i,j,k)
    # A rank-1 term with vectors (u, v, w) contributes:
    #   T[i,j,k] += u[i*m+j] * v[j*p+k] * w[k*n+i]
    
    # We need sum_r u_r[i*m+j] * v_r[j*p+k] * w_r[k*n+i] = 1 for all (i,j,k)
    
    # Let's try a structured approach based on known results.
    # For 2x3 x 3x4, the known lower bound is around 20-22.
    
    # Approach: Use the fact that we can share computation across columns.
    # For each column k, we compute A * B[:,k] which is a 2x3 times 3x1 product.
    # The trivial rank for this is 6.
    # But if we process two columns at a time, we might save.
    
    # Let's try to construct a rank-20 decomposition explicitly.
    
    # One known technique: use the identity
    # (a+b)(c+d) = ac + ad + bc + bd
    # to combine terms.
    
    # For 2x3 x 3x4, let's try the following construction:
    # We'll use 20 terms indexed by r = 0, ..., 19.
    
    # Let me try a different systematic approach:
    # We want to find U (6xR), V (12xR), W (8xR) such that
    # for all i,j,k: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
    
    # This is a system of 24 equations in R*6 + R*12 + R*8 = 26R unknowns.
    # For R=20, we have 520 unknowns and 24 equations, so there's plenty of freedom.
    
    # The constraint is that entries should be integers or half-integers.
    
    # Let's try to solve this using a simple iterative method.
    
    # Initialize with the trivial decomposition and try to reduce rank.
    
    # Actually, let's try a more direct approach: use known patterns for rectangular
    # matrix multiplication.
    
    # For 2x3 x 3x4, one construction:
    # Use the fact that 2x3 x 3x4 can be decomposed as:
    # - 2x3 x 3x2 (rank ~ 12) for first two columns
    # - 2x3 x 3x2 (rank ~ 12) for last two columns
    # But this gives 24, not better.
    
    # Better: share some terms between the two 2x3 x 3x2 blocks.
    
    # Let me try a random search with local optimization.
    
    np.random.seed(42)
    
    best_rank = n * m * p  # Start with trivial
    best_solution = None
    
    # Generate the trivial solution as starting point
    U = np.zeros((n * m, n * m * p))
    V = np.zeros((m * p, n * m * p))
    W = np.zeros((p * n, n * m * p))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    # Now try to reduce rank by finding linear dependencies.
    # If two columns of the "combined" tensor are linearly dependent, we can merge them.
    
    # Actually, let's try a different approach: use numerical optimization.
    
    # We want to minimize the rank subject to the constraint that the decomposition
    # is valid. This is a non-convex problem.
    
    # Let's try a simple approach: randomly initialize U, V, W with small rank
    # and use gradient descent to satisfy the constraints.
    
    for target_rank in [20, 21, 22, 23]:
        if time.time() - start_time > time_limit:
            break
        
        # Try multiple random initializations
        found = False
        for trial in range(100):
            if time.time() - start_time > time_limit:
                break
            
            U = np.random.randn(n * m, target_rank) * 0.5
            V = np.random.randn(m * p, target_rank) * 0.5
            W = np.random.randn(p * n, target_rank) * 0.5
            
            # Simple iterative refinement: adjust entries to satisfy constraints
            # For each (i,j,k), we need sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
            
            # This is a system of 24 equations. Let's try to solve it.
            # We can fix U and V, then solve for W, or vice versa.
            
            # Let's fix U and V, and solve for W.
            # For each (i,k), we have:
            # sum_j sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
            # This is not linear in W because W appears multiplied by other terms.
            
            # Actually, for fixed U and V, the equation is linear in each W[k*n+i,r].
            # Let's write it as:
            # sum_r W[k*n+i,r] * (sum_j U[i*m+j,r] * V[j*p+k,r]) = 1
            
            # For each (i,k), define A_ik,r = sum_j U[i*m+j,r] * V[j*p+k,r]
            # Then we need: sum_r W[k*n+i,r] * A_ik,r = 1
            
            # This is a linear system in W. For each (i,k), we have one equation
            # in target_rank unknowns (the W[k*n+i,r] values).
            
            # There are n*p = 8 such equations, each with target_rank unknowns.
            # So the system is underdetermined for target_rank > 1.
            
            # Let's solve it using least squares.
            
            # Build the system:
            # For each (i,k): sum_r W[k*n+i,r] * A_ik,r = 1
            
            # We can write this as a matrix equation: M @ w = b
            # where w is a vector of length n*p*target_rank = 8*target_rank
            # M is 8 x (8*target_rank)
            # b is 8 x 1
            
            M = np.zeros((n * p, n * p * target_rank))
            b = np.ones(n * p)
            
            for i in range(n):
                for k in range(p):
                    row = k * n + i  # Index in M (0 to n*p-1)
                    for r in range(target_rank):
                        A_ik_r = sum(U[i * m + j, r] * V[j * p + k, r] for j in range(m))
                        M[row, row * target_rank + r] = A_ik_r
            
            # Solve M @ w = b using least squares
            try:
                w, residuals, rank_M, sv = np.linalg.lstsq(M, b, rcond=None)
            except:
                continue
            
            # Reshape w back to W
            W = w.reshape(n * p, target_rank).T  # W is (p*n, target_rank)
            # Wait, W[k*n+i, r], so W has shape (p*n, target_rank)
            # w is indexed by (k*n+i, r), so w.reshape(p*n, target_rank) gives W directly
            W = w.reshape(p * n, target_rank)
            
            # Now verify the full decomposition
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        actual = sum(U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r] 
                                    for r in range(target_rank))
                        if abs(actual - 1.0) > 1e-6:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                # Check if entries are close to integers or half-integers
                all_close = True
                for mat in [U, V, W]:
                    for val in mat.flatten():
                        # Check if val is close to an integer or half-integer
                        nearest_half = round(val * 2) / 2
                        if abs(val - nearest_half) > 1e-4:
                            all_close = False
                            break
                    if not all_close:
                        break
                
                if all_close:
                    # Round to half-integers
                    U_r = np.round(U * 2) / 2
                    V_r = np.round(V * 2) / 2
                    W_r = np.round(W * 2) / 2
                    
                    # Verify after rounding
                    valid_r = True
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                actual = sum(U_r[i * m + j, r] * V_r[j * p + k, r] * W_r[k * n + i, r] 
                                            for r in range(target_rank))
                                if abs(actual - 1.0) > 1e-9:
                                    valid_r = False
                                    break
                            if not valid_r:
                                break
                        if not valid_r:
                            break
                    
                    if valid_r:
                        best_rank = target_rank
                        best_solution = (U_r, V_r, W_r)
                        found = True
                        break
        
        if found:
            break
    
    if best_solution is not None:
        U_r, V_r, W_r = best_solution
        return {
            "U": U_r.tolist(),
            "V": V_r.tolist(),
            "W": W_r.tolist()
        }
    
    return None

def _trivial_decomposition(n, m, p):
    """Return the trivial rank-(n*m*p) decomposition."""
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

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin out of 120s
    
    n, m, p = DIMENSIONS
    
    # Try to find a better decomposition
    result = _greedy_select_decomposition(n, m, p, target_rank=20, time_limit=100)
    
    if result is not None:
        return result
    
    # Fallback: trivial rank-24 decomposition
    return _trivial_decomposition(n, m, p)
# EVOLVE-BLOCK-END
