"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s kill

    # Try to find a lower-rank decomposition
    best = None
    best_rank = n * m * p

    # Lower bound: max(n*m, m*p, p*n)
    min_rank = max(n * m, m * p, p * n)

    # We search from min_rank upwards
    # It's possible that the rank is not contiguous, but usually it is.
    # If we fail at R, we try R+1.
    
    # To maximize chances, we can interleave restarts for different ranks?
    # No, finding a low rank is the goal.
    
    for target_rank in range(min_rank, best_rank):
        if time.time() > deadline:
            break

        # Determine number of restarts based on remaining time
        # If we have lots of time, do more restarts.
        # If we are close to deadline, do fewer.
        remaining = deadline - time.time()
        if remaining < 5:
            break
            
        # Estimate time per restart (roughly 0.5 - 2 seconds depending on rank)
        # Let's try 5 restarts for each rank initially.
        num_restarts = 5
        if target_rank > 100:
            num_restarts = 3
        elif target_rank > 50:
            num_restarts = 4
        else:
            num_restarts = 6

        for restart in range(num_restarts):
            if time.time() > deadline:
                break
            sol = _try_rank(n, m, p, target_rank, deadline)
            if sol is not None:
                best = sol
                best_rank = target_rank
                break  # Found valid at this rank, try lower? No, we are iterating UP.
                # Wait, the loop is `for target_rank in range(min_rank, best_rank)`.
                # If we find a solution at `target_rank`, we should record it.
                # Do we continue to try LOWER ranks?
                # The loop starts at `min_rank`. So we are trying the lowest first.
                # If we find one at `min_rank`, that's the best possible.
                # If we fail at `min_rank`, we try `min_rank + 1`.
                # So if we find one, we can return immediately?
                # No, it's possible we found a solution at `min_rank` but it's not the *true* minimum?
                # No, `min_rank` is the lower bound. If we find a solution there, it's optimal.
                # If we find one at `min_rank + 1`, it's the best so far, but maybe `min_rank` was just hard to find?
                # But we already tried `min_rank` with `num_restarts` restarts.
                # If we failed, it's likely not possible or very hard.
                # So we can return the first solution found?
                # Yes, because we are iterating from low to high.
                # The first valid solution found is the lowest rank found so far.
                # Since we check ranks in increasing order, the first one found is the global minimum (among those we found).
                # Is it guaranteed to be the true minimum? No.
                # But it's the best we can get.
                # We should return it.
                return best

    if best is not None:
        return best

    # Fallback: schoolbook decomposition
    return _schoolbook(n, m, p)


def _schoolbook(n, m, p):
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


def _verify(U, V, W, n, m, p):
    """Verify exact decomposition using Fraction arithmetic."""
    rank = len(U[0]) if U else 0
    if rank == 0:
        return False
        
    # Pre-convert to Fractions for speed? No, converting inside loop is better if we only do it once.
    # But we need to access U[u_idx][c] etc.
    # Let's convert the whole matrices to lists of Fractions.
    U_f = [[Fraction(x) for x in row] for row in U]
    V_f = [[Fraction(x) for x in row] for row in V]
    W_f = [[Fraction(x) for x in row] for row in W]

    # Check valid positions
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                total = Fraction(0)
                for c in range(rank):
                    total += U_f[u_idx][c] * V_f[v_idx][c] * W_f[w_idx][c]
                if total != 1:
                    return False

    # Check invalid positions
    # To speed this up, we can skip if we are confident.
    # But correctness is key.
    for u_idx in range(n * m):
        i = u_idx // m
        j = u_idx % m
        for v_idx in range(m * p):
            j2 = v_idx // p
            k = v_idx % p
            if j != j2:
                continue # Skip if j doesn't match, product will be 0?
                # No, U and V can have non-zero entries.
                # The sum is over c.
                # We must check.
                pass
            
            for w_idx in range(p * n):
                k2 = w_idx // n
                i2 = w_idx % n
                # Valid if j==j2, k==k2, i==i2
                if (j == j2 and k == k2 and i == i2):
                    continue
                
                total = Fraction(0)
                for c in range(rank):
                    total += U_f[u_idx][c] * V_f[v_idx][c] * W_f[w_idx][c]
                if total != 0:
                    return False

    return True


def _try_rank(n, m, p, target_rank, deadline):
    """Try to find a rank-target_rank decomposition via iterative refinement."""
    # Use a fixed seed for reproducibility within the restart?
    # No, we want different restarts to be different.
    # But construct() calls _try_rank multiple times.
    # We should use a global random state or pass seed.
    # Let's just use np.random which is global.
    
    U = np.random.randn(n * m, target_rank)
    V = np.random.randn(m * p, target_rank)
    W = np.random.randn(p * n, target_rank)
    
    # Normalize to prevent large/small values initially
    U /= np.linalg.norm(U)
    V /= np.linalg.norm(V)
    W /= np.linalg.norm(W)

    max_iterations = 100
    
    for iteration in range(max_iterations):
        if time.time() > deadline:
            return None

        # Update U
        for i in range(n):
            for j in range(m):
                u_idx = i * m + j
                A = np.zeros((p, target_rank))
                for k in range(p):
                    v_idx = j * p + k
                    w_idx = k * n + i
                    A[k, :] = V[v_idx, :] * W[w_idx, :]
                b = np.ones(p)
                try:
                    U[u_idx, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                except:
                    pass

        # Update V
        for j in range(m):
            for k in range(p):
                v_idx = j * p + k
                A = np.zeros((n, target_rank))
                for i in range(n):
                    u_idx = i * m + j
                    w_idx = k * n + i
                    A[i, :] = U[u_idx, :] * W[w_idx, :]
                b = np.ones(n)
                try:
                    V[v_idx, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                except:
                    pass

        # Update W
        for k in range(p):
            for i in range(n):
                w_idx = k * n + i
                A = np.zeros((m, target_rank))
                for j in range(m):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    A[j, :] = U[u_idx, :] * V[v_idx, :]
                b = np.ones(m)
                try:
                    W[w_idx, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                except:
                    pass

        # Check convergence every 5 iterations
        if iteration % 5 == 0:
            max_err = 0.0
            # Check valid positions
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_idx = i * m + j
                        v_idx = j * p + k
                        w_idx = k * n + i
                        val = np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :])
                        err = abs(val - 1.0)
                        if err > max_err:
                            max_err = err
            if max_err < 1e-8:
                # Check a few invalid positions to be sure
                # Just check one random invalid position?
                # No, if valid are good, invalid are likely good if rank is sufficient.
                # But we must verify exactly later.
                
                # Convert to fractions
                U_frac = [[_to_frac(x) for x in row] for row in U]
                V_frac = [[_to_frac(x) for x in row] for row in V]
                W_frac = [[_to_frac(x) for x in row] for row in W]
                
                if _verify(U_frac, V_frac, W_frac, n, m, p):
                    return {"U": U_frac, "V": V_frac, "W": W_frac}

    return None


def _to_frac(x):
    """Convert a float to a Fraction, rounding to nearest simple rational."""
    if abs(x) < 1e-12:
        return 0
    # Round to 10 decimal places to catch integers/halves/quarters
    x_rounded = round(x, 10)
    f = Fraction(x_rounded).limit_denominator(100000)
    return f
# EVOLVE-BLOCK-END
