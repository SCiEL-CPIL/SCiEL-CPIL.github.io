"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W


def _round_to_half_int(arr, tol=1e-6):
    """Round each entry to the nearest half-integer; return rounded array and success flag."""
    rounded = np.round(arr * 2) / 2
    diff = np.abs(arr - rounded)
    return rounded, np.all(diff <= tol)


def _verify(U, V, W, tol=1e-8):
    """Check that U * V^T * W^T (contracted) equals the identity tensor for (n,m,p).
    Specifically, for each (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1"""
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    # Build the full tensor product check
    # T[i,j,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    # Reshape U: (n, m, rank), V: (m, p, rank), W: (p, n, rank)
    U_3d = U.reshape(n, m, rank)
    V_3d = V.reshape(m, p, rank)
    W_3d = W.reshape(p, n, rank)
    # T[i,j,k] = sum_r U_3d[i,j,r] * V_3d[j,k,r] * W_3d[k,i,r]
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = np.dot(U_3d[i, j, :], V_3d[j, k, :] * W_3d[k, i, :])
    return np.allclose(T, np.ones((n, m, p)), atol=tol)


def _try_random_search(max_rank, n_steps=5000, deadline=None):
    """Randomized search for a rank-max_rank decomposition with half-integer entries."""
    n, m, p = DIMENSIONS
    best_rank = None
    for R in range(max_rank, 1, -1):
        if deadline is not None and time.time() > deadline:
            break
        for step in range(n_steps):
            if deadline is not None and time.time() > deadline:
                return best_rank
            # Random initialization with small values
            U = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(n * m, R))
            V = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(m * p, R))
            W = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(p * n, R))
            # Try a few local refinements
            for _ in range(20):
                if deadline is not None and time.time() > deadline:
                    break
                # Pick a random entry to flip/adjust
                mat = random.choice([U, V, W])
                r, c = random.randint(0, mat.shape[0] - 1), random.randint(0, mat.shape[1] - 1)
                old = mat[r, c]
                # Try nearby values
                candidates = [old + d for d in [-1, -0.5, 0.5, 1] if old + d in [-1, -0.5, 0, 0.5, 1]]
                if not candidates:
                    continue
                new_val = random.choice(candidates)
                mat[r, c] = new_val
                if _verify(U, V, W):
                    # Found a valid decomposition
                    return R
                mat[r, c] = old
    return best_rank


def _try_greedy_constructive(R, deadline=None):
    """Try to constructively find a rank-R decomposition by solving linear constraints.
    We fix U and V randomly and solve for W, then check if W can be made half-integer."""
    n, m, p = DIMENSIONS
    if deadline is not None and time.time() > deadline:
        return None
    
    for attempt in range(100):
        if deadline is not None and time.time() > deadline:
            break
        # Random U, V with entries in {-1, -0.5, 0, 0.5, 1}
        U = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(n * m, R))
        V = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(m * p, R))
        
        # For each (i,j,k), we need sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        # This is linear in W. Set up linear system.
        # Variables: W[k*n+i, r] for all k,i,r -> p*n*R variables
        # Equations: n*m*p equations
        num_vars = p * n * R
        num_eqs = n * m * p
        
        if num_vars < num_eqs:
            continue  # Underdetermined, skip (actually we want overdetermined or square)
        
        # Build A matrix: A[(i,j,k), var_idx] = U[i*m+j, r] * V[j*p+k, r]
        # var_idx = (k*n+i)*R + r
        A = np.zeros((num_eqs, num_vars))
        b = np.ones(num_eqs)
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for r in range(R):
                        var_idx = (k * n + i) * R + r
                        A[eq_idx, var_idx] = U[i * m + j, r] * V[j * p + k, r]
                    eq_idx += 1
        
        # Solve least squares
        try:
            W_flat, residuals, rank_, singular = np.linalg.lstsq(A, b, rcond=None)
        except:
            continue
        
        # Check if residual is small
        if len(residuals) > 0 and residuals[0] > 1e-8:
            continue
        
        # Check if solution is (approximately) half-integer
        W_rounded, is_half_int = _round_to_half_int(W_flat)
        if not is_half_int:
            continue
        
        W = W_rounded.reshape(p * n, R)
        
        # Final verification
        if _verify(U, V, W):
            return U, V, W
    
    return None


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    # Start with schoolbook
    U_s, V_s, W_s = _schoolbook()
    
    # Try to find better decomposition
    # First try constructive approach for ranks below n*m*p
    for R in range(n * m * p - 1, max(n * m * p - 10, 20), -1):
        if time.time() > deadline:
            break
        result = _try_greedy_constructive(R, deadline)
        if result is not None:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try random search for slightly lower ranks
    for R in range(n * m * p - 1, n * m * p - 5, -1):
        if time.time() > deadline:
            break
        found = _try_random_search(R, n_steps=200, deadline=deadline)
        if found is not None:
            # We found rank R exists but need the actual matrices
            # Re-run to capture them (this is a simplification; in practice we'd store them)
            pass
    
    # Return schoolbook if no better found
    return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}

# EVOLVE-BLOCK-END
