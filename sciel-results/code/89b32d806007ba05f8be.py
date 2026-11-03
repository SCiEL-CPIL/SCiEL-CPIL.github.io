"""Matrix multiplication 2x4x6: rank-minimal tensor decomposition.

Uses structured slice decomposition exploiting n=2 structure.
For each i in {0,1}, the partial tensor T_i has a specific pattern.
We find minimal rank decompositions for each slice and combine.
"""

import time
import sys
import numpy as np

DIMENSIONS = [2, 4, 6]

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[a,b,c] = 1 iff a=i*m+j, b=j*p+k, c=k*n+i."""
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _verify(U, V, W, T, tol=1e-8):
    """Verify that sum_r U[:,r] V[:,r] W[:,r] == T."""
    S = np.einsum('ar,br,cr->abc', U, V, W)
    return np.allclose(S, T, atol=tol)

def _schoolbook(n, m, p):
    """Trivial rank n*m*p decomposition."""
    r = n * m * p
    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((p * n, r))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                col = i * (m * p) + j * p + k
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U[a, col] = 1.0
                V[b, col] = 1.0
                W[c, col] = 1.0
    return U, V, W

def _find_rank_via_fixed_W(T, r, W_init, max_iter=500, tol=1e-9):
    """
    Given W (C x r), find U (A x r) and V (B x r) such that
    T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r].
    
    Uses alternating least squares with rank-1 enforcement.
    """
    A, B, C = T.shape
    rng = np.random.RandomState(12345)
    
    # Initialize V
    V = rng.randn(B, r) * 0.1
    U = rng.randn(A, r) * 0.1
    
    T_flat_a = T.reshape(A, B * C)
    T_flat_b = np.transpose(T, (1, 0, 2)).reshape(B, A * C)
    
    for it in range(max_iter):
        # Solve for U: T[a,:,:] = sum_r U[a,r] * outer(V[:,r], W[:,r])
        # T_flat_a[a] = sum_r U[a,r] * P[r] where P[r] = outer(V[:,r], W[:,r]).flatten()
        P = np.einsum('br,cr->bcr', V, W).reshape(r, B * C)
        # T_flat_a = U @ P.T  =>  U = lstsq(P.T, T_flat_a.T).T
        U_new, _, _, _ = np.linalg.lstsq(P.T, T_flat_a.T, rcond=None)
        U_new = U_new.T  # (A, r)
        
        # Solve for V: T[:,b,:] = sum_r V[b,r] * outer(U[:,r], W[:,r])
        Q = np.einsum('ar,cr->acr', U_new, W).reshape(r, A * C)
        V_new, _, _, _ = np.linalg.lstsq(Q.T, T_flat_b.T, rcond=None)
        V_new = V_new.T  # (B, r)
        
        # Check convergence
        S = np.einsum('ar,br,cr->abc', U_new, V_new, W)
        err = np.linalg.norm(S - T) / max(1.0, np.linalg.norm(T))
        
        if err < tol:
            return U_new, V_new, W, err
        
        U, V = U_new, V_new
        
        if it % 50 == 49:
            # Periodic check
            pass
    
    S = np.einsum('ar,br,cr->abc', U, V, W)
    err = np.linalg.norm(S - T) / max(1.0, np.linalg.norm(T))
    if err < tol:
        return U, V, W, err
    return None, None, None, err

def _structured_W(C, r, n, p, seed=0):
    """
    Create a structured W matrix based on the c-index pattern.
    c = k*n + i, so c ranges over {0,...,p*n-1} = {0,...,11} for p=6,n=2.
    
    For rank r, we assign each of the r columns to specific (k,i) patterns.
    The idea: W[c, col] should be nonzero only for c values compatible with col.
    """
    rng = np.random.RandomState(seed)
    W = np.zeros((C, r))
    
    # Each column of W corresponds to a "term" in the decomposition.
    # For term t, it involves specific (i,j,k) combinations.
    # c = k*n + i, so for a given term, if it involves i and k, then W[k*n+i, t] != 0.
    
    # Strategy: distribute r terms across the p*n = 12 c-indices.
    # Each c-index k*n+i gets contributions from terms that involve that (k,i) pair.
    
    # For now, use random but structured: each column has nonzeros at a few c-positions.
    for t in range(r):
        # Choose which (k,i) pairs this term involves
        # A term in matrix mult involves one (i,j,k) triple, so it touches one c = k*n+i
        # But in a rank-r decomposition, a single term can combine multiple (i,j,k)
        # if they share the same i (for U), same j (for V), or same k (for W).
        
        # Let's assign this term to a specific i and k
        i_val = t % n
        k_val = (t // n) % p
        c_val = k_val * n + i_val
        W[c_val, t] = 1.0
        
        # Also allow some terms to span multiple k for the same i
        if t >= n * p:
            # Extra terms: allow spanning
            i_val = t % n
            for k_val in range(p):
                c_val = k_val * n + i_val
                W[c_val, t] = rng.randn()
    
    # Normalize columns
    for t in range(r):
        norm = np.linalg.norm(W[:, t])
        if norm > 1e-10:
            W[:, t] /= norm
    
    return W

def _search_rank_structured(T, r, max_restarts=100, deadline=None):
    """Search for rank-r decomposition using structured W initialization."""
    A, B, C = T.shape
    rng = np.random.RandomState(42)
    
    for restart in range(max_restarts):
        if deadline is not None and time.time() > deadline:
            break
        
        # Different initialization strategies
        if restart < 20:
            # Structured W based on (i,k) patterns
            W = _structured_W(C, r, DIMENSIONS[0], DIMENSIONS[2], seed=restart)
        elif restart < 40:
            # Random W with sparsity
            W = rng.randn(C, r)
            # Zero out some entries
            for t in range(r):
                mask = rng.random(C) < 0.5
                W[mask, t] = 0
        elif restart < 60:
            # W based on k-index only (ignore i for W)
            W = np.zeros((C, r))
            for t in range(r):
                k_val = t % DIMENSIONS[2]
                for i_val in range(DIMENSIONS[0]):
                    c_val = k_val * DIMENSIONS[0] + i_val
                    W[c_val, t] = rng.randn()
        else:
            # Fully random
            W = rng.randn(C, r)
        
        U, V, Wsol, err = _find_rank_via_fixed_W(T, r, W, max_iter=300)
        
        if U is not None and err < 1e-9:
            # Final verification
            S = np.einsum('ar,br,cr->abc', U, V, Wsol)
            if np.allclose(S, T, atol=1e-9):
                return U, V, Wsol, r
    
    return None

def _lower_bound(n, m, p):
    """Compute a lower bound on the rank of M_{n,m,p}."""
    # Trivial: max(n*m, m*p, p*n) / min(n,m,p) ... not tight
    # Better: the tensor has n*m*p nonzeros, each rank-1 term covers at most min(n*m, m*p, p*n)
    # but with structure, each term covers at most 1 nonzero in the "diagonal" sense
    # For now, use n*m*p / max(n,m,p) as a weak bound
    return max(1, (n * m * p) // max(n, m, p))

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 2, 4, 6
    T = _build_tensor(n, m, p)
    
    # Start with schoolbook as fallback
    U_sbk, V_sbk, W_sbk = _schoolbook(n, m, p)
    best_rank = n * m * p  # 48
    best_result = (U_sbk, V_sbk, W_sbk)
    
    print(f"Schoolbook rank: {best_rank}", file=sys.stderr)
    
    # Try ranks from lower bound up to best_rank - 1
    lb = _lower_bound(n, m, p)
    # For 2x4x6: lb = 48//6 = 8, but actual rank is likely higher
    # Let's try from a reasonable range
    search_start = max(lb, 15)
    search_end = best_rank - 1
    
    # Try to find the minimal rank
    for r in range(search_start, min(search_end, 40) + 1):
        if time.time() > deadline - 10:
            break
        
        result = _search_rank_structured(T, r, max_restarts=50, deadline=deadline - 5)
        
        if result is not None:
            U, V, W, found_rank = result
            best_rank = found_rank
            best_result = (U, V, W)
            print(f"Found rank {found_rank} decomposition!", file=sys.stderr)
            break  # We want the minimal rank, so stop at the first found
    
    U, V, W = best_result
    print(f"Final rank: {best_rank}", file=sys.stderr)
    
    # Convert to lists
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
