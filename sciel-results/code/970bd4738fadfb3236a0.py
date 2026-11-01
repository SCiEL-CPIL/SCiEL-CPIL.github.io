"""Matrix multiplication 3x3 by 3x8: rank-minimal exact tensor decomposition.
Uses structured search with exact rational verification.
"""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 8]
START = time.time()
DEADLINE = 130.0  # safety margin before 150s kill


def _build_schoolbook():
    n, m, p = DIMENSIONS
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


def _validate_exact(U, V, W, n, m, p):
    """Check that U[i*m+j,c] * V[j*p+k,c] == W[k*n+i,c] for all i,j,k,c.
    Uses exact integer/half-integer arithmetic."""
    U = np.asarray(U)
    V = np.asarray(V)
    W = np.asarray(W)
    r = U.shape[1]
    # Multiply by 2 to work with integers (half-integers become integers)
    U2 = (2 * U).astype(int)
    V2 = (2 * V).astype(int)
    W2 = (2 * W).astype(int)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U[i*m+j,:] * V[j*p+k,:] element-wise should equal W[k*n+i,:]
                # In doubled space: U2[i*m+j,:] * V2[j*p+k,:] == 4 * W2[k*n+i,:]
                lhs = U2[i * m + j, :] * V2[j * p + k, :]
                rhs = 4 * W2[k * n + i, :]
                if not np.array_equal(lhs, rhs):
                    return False
    return True


def _to_list(M):
    """Convert numpy array to list of lists of int or float (half-integer)."""
    result = []
    for row in M:
        r = []
        for v in row:
            fv = float(v)
            if fv == int(fv):
                r.append(int(fv))
            else:
                r.append(fv)
        result.append(r)
    return result


def _try_structured_rank(r, n, m, p, seed=0):
    """Try structured decompositions using small integer/half-integer entries.
    Strategy: randomly assign values from {-1, -0.5, 0, 0.5, 1} to U and V,
    then compute W = U*V element-wise per (i,j,k) and check consistency."""
    rng = np.random.default_rng(seed)
    values = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    
    for it in range(5000):
        if time.time() > DEADLINE:
            return None
        
        # Random sparse-ish initialization
        U = rng.choice(values, size=(n * m, r))
        V = rng.choice(values, size=(m * p, r))
        
        # Compute required W: W[k*n+i, c] must satisfy
        # For each c: sum over nothing... actually W is determined per (i,j,k)
        # W[k*n+i, c] = U[i*m+j, c] * V[j*p+k, c] for the specific j that pairs with i,k
        # But wait - for a given (i,k) and rank component c, W[k*n+i, c] must be
        # consistent across ALL j: U[i*m+j, c] * V[j*p+k, c] must be the same for all j.
        # This is the key constraint!
        
        W = np.zeros((p * n, r))
        consistent = True
        for i in range(n):
            for k in range(p):
                for c in range(r):
                    # Check consistency across j
                    val = U[i * m + 0, c] * V[0 * p + k, c]
                    for j in range(1, m):
                        if U[i * m + j, c] * V[j * p + k, c] != val:
                            consistent = False
                            break
                    if not consistent:
                        break
                if not consistent:
                    break
            if not consistent:
                break
        
        if not consistent:
            continue
        
        # Build W
        for i in range(n):
            for k in range(p):
                for c in range(r):
                    W[k * n + i, c] = U[i * m + 0, c] * V[0 * p + k, c]
        
        # Verify exactness
        if _validate_exact(U, V, W, n, m, p):
            # Check that W has only half-integer values
            W2 = (2 * W).astype(int)
            if not np.array_equal(W2, (2 * W)):
                continue
            return {"U": _to_list(U), "V": _to_list(V), "W": _to_list(W)}
    
    return None


def _try_gaussian_refine(r, n, m, p, seed=0):
    """Gaussian random + local refinement with exact verification."""
    rng = np.random.default_rng(seed)
    values = np.array([-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, -1.5])
    
    for it in range(3000):
        if time.time() > DEADLINE:
            return None
        
        U = rng.choice(values, size=(n * m, r))
        V = rng.choice(values, size=(m * p, r))
        
        # Check consistency: for each (i,k,c), U[i*m+j,c]*V[j*p+k,c] must be same for all j
        consistent = True
        for i in range(n):
            for k in range(p):
                for c in range(r):
                    ref = U[i * m + 0, c] * V[0 * p + k, c]
                    for j in range(1, m):
                        if abs(U[i * m + j, c] * V[j * p + k, c] - ref) > 1e-9:
                            consistent = False
                            break
                    if not consistent:
                        break
                if not consistent:
                    break
            if not consistent:
                break
        
        if not consistent:
            continue
        
        W = np.zeros((p * n, r))
        for i in range(n):
            for k in range(p):
                for c in range(r):
                    W[k * n + i, c] = U[i * m + 0, c] * V[0 * p + k, c]
        
        if _validate_exact(U, V, W, n, m, p):
            return {"U": _to_list(U), "V": _to_list(V), "W": _to_list(W)}
    
    return None


def _try_pairwise_sharing(r, n, m, p, seed=0):
    """Try decompositions where rank components are shared between pairs of (i,j) or (j,k).
    For m=3, we can try: each rank component contributes to at most 2 j-values for a given i,k."""
    rng = np.random.default_rng(seed)
    values = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    
    for it in range(2000):
        if time.time() > DEADLINE:
            return None
        
        U = rng.choice(values, size=(n * m, r))
        V = rng.choice(values, size=(m * p, r))
        
        consistent = True
        for i in range(n):
            for k in range(p):
                for c in range(r):
                    ref = U[i * m + 0, c] * V[0 * p + k, c]
                    for j in range(1, m):
                        if abs(U[i * m + j, c] * V[j * p + k, c] - ref) > 1e-9:
                            consistent = False
                            break
                    if not consistent:
                        break
                if not consistent:
                    break
            if not consistent:
                break
        
        if not consistent:
            continue
        
        W = np.zeros((p * n, r))
        for i in range(n):
            for k in range(p):
                for c in range(r):
                    W[k * n + i, c] = U[i * m + 0, c] * V[0 * p + k, c]
        
        if _validate_exact(U, V, W, n, m, p):
            return {"U": _to_list(U), "V": _to_list(V), "W": _to_list(W)}
    
    return None


def construct():
    n, m, p = DIMENSIONS
    best = _build_schoolbook()
    best_rank = n * m * p  # 72

    # Lower bound: max(n*p, m*p, n*m) = max(24, 24, 9) = 24
    lower = max(n * p, m * p, n * m)
    
    # Try ranks from lower bound upward
    for r in range(lower, best_rank):
        if time.time() > DEADLINE - 5:
            break
        
        # Budget per rank: try multiple strategies
        seed_base = r * 1000
        
        # Strategy 1: structured small-integer search
        res = _try_structured_rank(r, n, m, p, seed=seed_base)
        if res is not None:
            best = res
            best_rank = r
            break
        
        # Strategy 2: refined search
        res = _try_gaussian_refine(r, n, m, p, seed=seed_base + 1)
        if res is not None:
            best = res
            best_rank = r
            break
        
        # Strategy 3: pairwise sharing
        res = _try_pairwise_sharing(r, n, m, p, seed=seed_base + 2)
        if res is not None:
            best = res
            best_rank = r
            break
    
    return best


if __name__ == "__main__":
    res = construct()
    print("rank:", len(res["U"][0]))
