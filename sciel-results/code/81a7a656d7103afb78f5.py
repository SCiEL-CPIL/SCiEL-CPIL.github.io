import time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
import itertools

_START_TIME = time.time()
_DEADLINE = 140  # 10s safety margin

def _verify(U, V, W, n, m, p, tol=1e-10):
    """Check if U,V,W form a valid decomposition."""
    if not U or not V or not W:
        return False
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Construct target tensor
    T_target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0
                
    # Compute reconstructed tensor
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    
    # Check difference
    if np.max(np.abs(T_recon - T_target)) > tol:
        return False
    return True

def _generate_candidates(n, m, p, max_val=2):
    """
    Generate a pool of candidate rank-1 tensors with small integer entries.
    A candidate is a tuple (u, v, w) where u, v, w are vectors.
    We restrict entries to integers in [-max_val, max_val].
    """
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    
    # Target tensor support
    support = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                support.add((a, b, c))
                
    candidates = []
    
    # To keep the pool manageable, we sample.
    # Full enumeration of {-1,0,1}^dim is too large.
    # We focus on sparse vectors.
    
    # Heuristic: Generate candidates that have non-zeros only where target might be?
    # Actually, a valid rank-1 term u v w^T must have support subset of target support 
    # IF we assume the final sum is exact and no cancellations of non-target terms occur?
    # No, cancellations can occur. But for small integer entries, it's safer to restrict
    # candidates to those that don't "break" the support too much, or just rely on ILP constraints.
    
    # Let's generate random sparse candidates with small integers.
    rng = np.random.RandomState(42)
    num_candidates = 5000
    
    for _ in range(num_candidates):
        # Sparse random vectors
        u = np.zeros(dim_U)
        v = np.zeros(dim_V)
        w = np.zeros(dim_W)
        
        # Randomly select some indices to be non-zero
        # Keep it sparse
        nz_u = rng.randint(1, 4)
        nz_v = rng.randint(1, 4)
        nz_w = rng.randint(1, 4)
        
        idx_u = rng.choice(dim_U, size=nz_u, replace=False)
        idx_v = rng.choice(dim_V, size=nz_v, replace=False)
        idx_w = rng.choice(dim_W, size=nz_w, replace=False)
        
        u[idx_u] = rng.randint(-max_val, max_val + 1, size=nz_u)
        v[idx_v] = rng.randint(-max_val, max_val + 1, size=nz_v)
        w[idx_w] = rng.randint(-max_val, max_val + 1, size=nz_w)
        
        # Skip if all zero
        if np.all(u==0) or np.all(v==0) or np.all(w==0):
            continue
            
        # Check if this candidate "respects" the target support?
        # If u_i != 0 and v_j != 0 and w_k != 0, then T_ijk gets contribution.
        # If T_ijk should be 0, this contribution must be cancelled by others.
        # For ILP, we can just include all and let constraints handle it.
        # However, to speed up ILP, we can filter candidates that have "large" violations
        # of the support structure if we assume no cancellation for now?
        # Let's just add them.
        
        candidates.append((u, v, w))
        
    return candidates

def _solve_ilp(n, m, p, candidates):
    """
    Solve the set cover / exact cover problem using ILP.
    Minimize sum(x_i) subject to sum(x_i * T_i) == T_target.
    """
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    
    T_target = np.zeros((dim_U, dim_V, dim_W))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0
                
    T_flat = T_target.flatten()
    num_constraints = len(T_flat)
    num_vars = len(candidates)
    
    if num_vars == 0:
        return None
        
    # Objective: minimize sum of x_i
    c = np.ones(num_vars)
    
    # Constraints: A @ x == T_flat
    # A is (num_constraints, num_vars)
    # A[i, j] = candidate_j_tensor_flat[i]
    
    A = np.zeros((num_constraints, num_vars))
    
    for j, (u, v, w) in enumerate(candidates):
        t = np.einsum('i,j,k->ijk', u, v, w)
        A[:, j] = t.flatten()
        
    # Variable bounds: x_i >= 0. 
    # For exact cover with integer coefficients, x_i can be integer.
    # scipy.milp requires integer variables.
    integrality = np.ones(num_vars) # All variables are integers
    
    # Equality constraints
    constraints = [LinearConstraint(A, T_flat, T_flat)]
    
    # Bounds: x_i >= 0. Upper bound?
    # Max rank is schoolbook n*m*p = 24. So x_i <= 24 is a safe upper bound?
    # Actually, sum x_i <= 24. So each x_i <= 24.
    bounds = Bounds(lb=0, ub=24)
    
    try:
        res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
        if res.status == 0: # Optimal
            x = res.x
            # Filter out zero variables
            selected = [candidates[j] for j in range(num_vars) if x[j] > 0.5]
            if not selected:
                return None
            return selected
        else:
            return None
    except Exception:
        return None

def _greedy_cover(n, m, p, candidates):
    """
    Greedy set cover heuristic.
    """
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    
    T_target = np.zeros((dim_U, dim_V, dim_W))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0
                
    residual = T_target.copy()
    selected = []
    remaining_candidates = candidates.copy()
    
    # Sort candidates by "coverage" or just iterate?
    # To be efficient, we can check which candidate reduces the L2 norm of residual the most.
    # But that's O(N * M).
    # Simple greedy: Pick candidate that matches most unsatisfied entries?
    
    for _ in range(24): # Max rank bound
        if np.allclose(residual, 0):
            break
        best_idx = -1
        best_score = -1
        best_cand = None
        
        # Sample a subset of remaining candidates to speed up
        # If list is huge, sample
        if len(remaining_candidates) > 100:
            sample_indices = np.random.choice(len(remaining_candidates), 100, replace=False)
            sample_cands = [remaining_candidates[i] for i in sample_indices]
        else:
            sample_cands = remaining_candidates
            
        for idx, (u, v, w) in enumerate(sample_cands):
            t = np.einsum('i,j,k->ijk', u, v, w)
            # We want to minimize ||residual - t||^2
            # Or maximize overlap with residual?
            # Since we need exact match, let's look for exact match on a subset?
            # Let's just compute error
            err = np.sum((residual - t)**2)
            if err < best_score:
                best_score = err
                # Map back to original index
                if len(remaining_candidates) > 100:
                    best_idx = sample_indices[idx]
                else:
                    best_idx = idx
                best_cand = (u, v, w)
                
        if best_idx == -1:
            break
            
        selected.append(best_cand)
        residual -= np.einsum('i,j,k->ijk', best_cand[0], best_cand[1], best_cand[2])
        remaining_candidates.pop(best_idx)
        
    if np.allclose(residual, 0):
        return selected
    return None

def construct():
    n, m, p = 2, 3, 4
    lower_bound = max(n*m, m*p, n*p)
    
    # 1. Try to find a solution using ILP on a generated candidate pool
    # Generate candidates
    candidates = _generate_candidates(n, m, p, max_val=1)
    
    # Also add some "obvious" candidates?
    # E.g., standard basis vectors?
    # The schoolbook solution uses basis vectors.
    # Let's add basis vectors to the pool to ensure we can always find the schoolbook solution.
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    
    for a in range(dim_U):
        u = np.zeros(dim_U); u[a] = 1
        for b in range(dim_V):
            v = np.zeros(dim_V); v[b] = 1
            for c in range(dim_W):
                w = np.zeros(dim_W); w[c] = 1
                # Only add if (a,b,c) is in target support?
                # If not, it's a "bad" candidate that creates error.
                # But ILP can handle it if we allow negative coefficients? 
                # No, we set x_i >= 0. So we should only add candidates that
                # don't create non-zero entries outside the support?
                # Actually, if a candidate creates a non-zero outside support, 
                # it MUST be cancelled by another candidate.
                # With x_i >= 0, this is hard.
                # So, let's filter candidates to those with support subset of target support.
                
                # Check support
                # u has non-zero at a. v at b. w at c.
                # Tensor is non-zero only at (a,b,c).
                # So we just need (a,b,c) in target support.
                i, j = divmod(a, m)
                j2, k = divmod(b, p)
                if i != j2: continue # j must match
                i2, k2 = divmod(c, n)
                if k != i2: continue # k must match i? No, c = k*n + i. So k2=k, i2=i.
                # Wait, c = k*n + i. So divmod(c, n) gives (k, i).
                # So i2=k, k2=i? No.
                # c = k*n + i.
                # divmod(c, n) -> (k, i).
                # So i2 = k, k2 = i.
                # We need i2 == j? No.
                # Target: a=i*m+j, b=j*p+k, c=k*n+i.
                # Check:
                # a -> i, j
                # b -> j, k
                # c -> k, i
                # So we need j from a to equal j from b.
                # And k from b to equal k from c.
                # And i from c to equal i from a.
                
                # My loop above:
                # i, j = divmod(a, m)
                # j2, k = divmod(b, p)
                # if i != j2: continue # j must match j2? Yes.
                # i2, k2 = divmod(c, n) # i2=k, k2=i
                # if k != i2: continue # k must match i2? Yes.
                # if i != k2: continue # i must match k2? Yes.
                
                # So the check is:
                if i != j2 or k != i2 or i != k2:
                    continue
                    
                candidates.append((u, v, w))
                
    # Remove duplicates? Not strictly necessary for ILP, but good for size.
    # Convert to list of tuples for hashing?
    unique_candidates = []
    seen = set()
    for u, v, w in candidates:
        key = (tuple(u), tuple(v), tuple(w))
        if key not in seen:
            seen.add(key)
            unique_candidates.append((u, v, w))
            
    candidates = unique_candidates
    
    # Try ILP
    selected = _solve_ilp(n, m, p, candidates)
    
    if selected:
        # Construct U, V, W
        rank = len(selected)
        U = np.zeros((dim_U, rank))
        V = np.zeros((dim_V, rank))
        W = np.zeros((dim_W, rank))
        
        for r, (u, v, w) in enumerate(selected):
            U[:, r] = u
            V[:, r] = v
            W[:, r] = w
            
        # Verify
        if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
    # Try Greedy
    selected = _greedy_cover(n, m, p, candidates)
    if selected:
        rank = len(selected)
        U = np.zeros((dim_U, rank))
        V = np.zeros((dim_V, rank))
        W = np.zeros((dim_W, rank))
        
        for r, (u, v, w) in enumerate(selected):
            U[:, r] = u
            V[:, r] = v
            W[:, r] = w
            
        if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback: Schoolbook
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
