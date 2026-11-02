import numpy as np
import time
import random

DIMENSIONS = [3, 3, 6]

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W decompose the matrix multiplication tensor."""
    R = U.shape[1]
    # Check non-zero entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                s = 0.0
                for r in range(R):
                    s += U[a, r] * V[b, r] * W[c, r]
                if abs(s - 1.0) > tol:
                    return False
    
    # Check a sample of zero entries
    rng = random.Random(42)
    checked = 0
    max_checks = 2000
    while checked < max_checks:
        i = rng.randint(0, n - 1)
        j = rng.randint(0, m - 1)
        k = rng.randint(0, p - 1)
        a = i * m + j
        j2 = rng.randint(0, m - 1)
        k2 = rng.randint(0, p - 1)
        b = j2 * p + k2
        i2 = rng.randint(0, n - 1)
        k3 = rng.randint(0, p - 1)
        c = k3 * n + i2
        if j2 == j and i2 == i and k3 == k:
            continue
        s = 0.0
        for r in range(R):
            s += U[a, r] * V[b, r] * W[c, r]
        if abs(s) > tol:
            return False
        checked += 1
    return True

def _build_schoolbook(n, m, p):
    R = n * m * p
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, r] = 1.0
                V[j * p + k, r] = 1.0
                W[k * n + i, r] = 1.0
                r += 1
    return U, V, W

def _compute_tensor(U, V, W, n, m, p):
    """Compute the full tensor from the decomposition and return it as a 3D array."""
    R = U.shape[1]
    a_dim = n * m
    b_dim = m * p
    c_dim = p * n
    T = np.zeros((a_dim, b_dim, c_dim))
    for r in range(R):
        T += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
    return T

def _als_step(U, V, W, n, m, p, target_rank, rng, max_iter=500, tol=1e-8):
    """Alternating Least Squares for tensor decomposition.
    
    We fix two factors and solve for the third via linear least squares.
    """
    a_dim = n * m
    b_dim = m * p
    c_dim = p * n
    
    # Precompute the target tensor entries (only non-zero ones matter for fitting)
    # But we need to enforce zero entries too. Let's use a weighted approach.
    # Actually, for exact decomposition, we need all entries to match.
    # Let's use the full tensor constraint.
    
    # For efficiency, we'll use the fact that the tensor is mostly zero.
    # We'll solve the linear system for the factor that we're updating.
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    prev_loss = float('inf')
    
    for it in range(max_iter):
        # Update U: for each row a, solve for U[a, :]
        # The constraint is: sum_r U[a,r] * V[b,r] * W[c,r] = T[a,b,c] for all b,c
        # This is a linear system in U[a, :]
        
        # Build the system for U
        # For a given a, we need: U[a, :] @ (V[b, :] * W[c, :]) = T[a,b,c]
        # This is: sum_r U[a,r] * (V[b,r] * W[c,r]) = T[a,b,c]
        
        # Let's form the matrix M_U where M_U[a, (b,c), r] = V[b,r] * W[c,r]
        # But this is huge. Instead, for each a, we solve a least squares problem.
        
        # Actually, let's use a different formulation.
        # Reshape V and W into a matrix: for fixed a, the equation is
        # U[a, :] @ (V * W)_{b,c} = T[a,b,c]
        # where (V*W)_{b,c,r} = V[b,r]*W[c,r]
        
        # This is equivalent to: for each a, solve min_U sum_{b,c} (U[a,:] @ (V*W)_{b,c} - T[a,b,c])^2
        
        # Let's vectorize: let vwc = V.T @ W (this doesn't work directly)
        # Actually, (V*W) is a 3D array. Let's flatten it.
        
        # For each a, the system is:
        # A_a @ U[a, :]^T = t_a
        # where A_a is (b_dim * c_dim) x target_rank, with A_a[(b,c), r] = V[b,r]*W[c,r]
        # and t_a is (b_dim * c_dim) x 1, with t_a[(b,c)] = T[a,b,c]
        
        # But we don't have T explicitly. We need to construct it or use the known non-zero entries.
        
        # Since the tensor is sparse, let's only use the non-zero entries for the fit,
        # and add a regularization term for the zero entries.
        
        # Actually, for exact decomposition, we must satisfy ALL entries.
        # Let's use a sampled approach for the zero entries.
        
        # For now, let's use the non-zero entries only for the ALS step,
        # and verify at the end.
        
        # Non-zero entries: for each (i,j,k), T[i*m+j, j*p+k, k*n+i] = 1
        # Let's collect these.
        
        nz_indices = []
        nz_values = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    nz_indices.append((a, b, c))
                    nz_values.append(1.0)
        
        nz_indices = np.array(nz_indices)
        nz_values = np.array(nz_values)
        
        # Update U
        for a in range(a_dim):
            # Find all (b, c) such that (a, b, c) is a non-zero entry
            mask = nz_indices[:, 0] == a
            if not np.any(mask):
                continue
            b_idx = nz_indices[mask, 1]
            c_idx = nz_indices[mask, 2]
            vals = nz_values[mask]
            
            # Build the system: sum_r U[a,r] * V[b,r] * W[c,r] = vals
            # A[(b,c), r] = V[b,r] * W[c,r]
            A = np.zeros((len(b_idx), target_rank))
            for idx in range(len(b_idx)):
                A[idx, :] = V[b_idx[idx], :] * W[c_idx[idx], :]
            
            # Solve least squares
            if len(b_idx) >= target_rank:
                sol, residuals, rank, sv = np.linalg.lstsq(A, vals, rcond=None)
                U[a, :] = sol
        
        # Update V
        for b in range(b_dim):
            mask = nz_indices[:, 1] == b
            if not np.any(mask):
                continue
            a_idx = nz_indices[mask, 0]
            c_idx = nz_indices[mask, 2]
            vals = nz_values[mask]
            
            A = np.zeros((len(a_idx), target_rank))
            for idx in range(len(a_idx)):
                A[idx, :] = U[a_idx[idx], :] * W[c_idx[idx], :]
            
            if len(a_idx) >= target_rank:
                sol, residuals, rank, sv = np.linalg.lstsq(A, vals, rcond=None)
                V[b, :] = sol
        
        # Update W
        for c in range(c_dim):
            mask = nz_indices[:, 2] == c
            if not np.any(mask):
                continue
            a_idx = nz_indices[mask, 0]
            b_idx = nz_indices[mask, 1]
            vals = nz_values[mask]
            
            A = np.zeros((len(a_idx), target_rank))
            for idx in range(len(a_idx)):
                A[idx, :] = U[a_idx[idx], :] * V[b_idx[idx], :]
            
            if len(a_idx) >= target_rank:
                sol, residuals, rank, sv = np.linalg.lstsq(A, vals, rcond=None)
                W[c, :] = sol
        
        # Compute loss
        loss = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    s = np.sum(U[a, :] * V[b, :] * W[c, :])
                    loss += (s - 1.0) ** 2
        
        if abs(prev_loss - loss) < tol:
            break
        prev_loss = loss
    
    return U, V, W

def _try_rank_reduction(n, m, p, target_rank, deadline, seed=0):
    """Attempt to find a rank-target_rank decomposition using ALS."""
    rng = np.random.RandomState(seed)
    R = n * m * p
    a_dim = n * m
    b_dim = m * p
    c_dim = p * n
    
    # Initialize with random matrices
    U = rng.randn(a_dim, target_rank) * 0.1
    V = rng.randn(b_dim, target_rank) * 0.1
    W = rng.randn(c_dim, target_rank) * 0.1
    
    # Run ALS
    U, V, W = _als_step(U, V, W, n, m, p, target_rank, rng, max_iter=300)
    
    # Check if we found a valid solution
    if time.time() < deadline:
        if _verify(U, V, W, n, m, p, tol=1e-6):
            return U, V, W
    
    return None

def construct():
    n, m, p = DIMENSIONS
    
    # Deadline guard: return best found before wall-clock limit
    deadline = time.time() + 120 - 10  # 110s safety margin
    
    # Schoolbook baseline: rank 54
    U_school, V_school, W_school = _build_schoolbook(n, m, p)
    
    best_rank = n * m * p
    best_U, best_V, best_W = U_school, V_school, W_school
    
    # Try a few candidate ranks
    for target_rank in [53, 52, 50, 48, 45, 40]:
        if time.time() > deadline - 20:
            break
        
        result = _try_rank_reduction(n, m, p, target_rank, deadline, seed=target_rank)
        if result is not None:
            U_try, V_try, W_try = result
            if _verify(U_try, V_try, W_try, n, m, p, tol=1e-6):
                best_rank = target_rank
                best_U, best_V, best_W = U_try, V_try, W_try
                break
    
    # Return the best result found
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
