# EVOLVE-BLOCK-START
import numpy as np
import time


def _qap_cost(a, b, p):
    return int(np.sum(a * b[np.ix_(p, p)]))


def _swap_delta(a, b, p, i, j):
    """Compute exact delta for swapping facilities at positions i and j.
    After swap: facility i goes to location p[j], facility j goes to location p[i].
    delta = sum_k (a[i,k] - a[k,j]) * (b[p[j],p[k]] - b[p[i],p[k]])
    """
    pi = p[i]
    pj = p[j]
    dB = b[pj, p] - b[pi, p]
    dA = a[i, :] - a[:, j]
    return int(np.dot(dA, dB))


def _local_search(a, b, p, deadline, best_cost, best_p):
    """First-improvement 2-opt local search until deadline."""
    n = len(p)
    cur = _qap_cost(a, b, p)
    if cur < best_cost:
        best_cost = cur
        best_p = p.copy()

    while time.time() < deadline:
        improved = False
        for i in range(n):
            if time.time() >= deadline:
                break
            for j in range(i + 1, n):
                delta = _swap_delta(a, b, p, i, j)
                if delta < 0:
                    p[i], p[j] = p[j], p[i]
                    cur += delta
                    improved = True
                    if cur < best_cost:
                        best_cost = cur
                        best_p = p.copy()
                    if time.time() >= deadline:
                        return best_cost, best_p
        if not improved:
            break

    if cur < best_cost:
        best_cost = cur
        best_p = p.copy()
    return best_cost, best_p


def _spectral_embedding(a, b):
    """Create initial assignment via spectral embedding of facility and location graphs."""
    n = a.shape[0]
    
    # Build Laplacians
    A = a.astype(np.float64)
    B = b.astype(np.float64)
    
    # Symmetrize
    A_sym = (A + A.T) / 2.0
    B_sym = (B + B.T) / 2.0
    
    # Degree vectors
    dA = A_sym.sum(axis=1)
    dB = B_sym.sum(axis=1)
    
    # Laplacian L = D - A
    LA = np.diag(dA) - A_sym
    LB = np.diag(dB) - B_sym
    
    # Get eigenvectors (skip trivial ones)
    # Use only first few non-trivial eigenvectors for embedding
    k = min(10, n - 1)
    
    try:
        evals_A, evecs_A = np.linalg.eigh(LA)
        evals_B, evecs_B = np.linalg.eigh(LB)
        
        # Take k smallest non-zero eigenvalues' eigenvectors
        # Skip the first (trivial) eigenvector
        idx_A = np.argsort(evals_A)[1:1+k]
        idx_B = np.argsort(evals_B)[1:1+k]
        
        U_A = evecs_A[:, idx_A]
        U_B = evecs_B[:, idx_B]
        
        # Normalize rows to unit length
        normA = np.linalg.norm(U_A, axis=1, keepdims=True)
        normA[normA == 0] = 1
        U_A = U_A / normA
        
        normB = np.linalg.norm(U_B, axis=1, keepdims=True)
        normB[normB == 0] = 1
        U_B = U_B / normB
        
        # Match by minimizing total distance: assign facility i to location j
        # that minimizes ||U_A[i] - U_B[j]||^2 = ||U_A[i]||^2 + ||U_B[j]||^2 - 2*U_A[i]·U_B[j]
        # Since norms are 1, this is equivalent to maximizing U_A[i]·U_B[j]
        
        # Compute similarity matrix
        S = U_A @ U_B.T  # S[i,j] = U_A[i]·U_B[j]
        
        # Greedy assignment: repeatedly pick the most similar unassigned pair
        p = np.full(n, -1, dtype=np.int64)
        used_fac = np.zeros(n, dtype=bool)
        used_loc = np.zeros(n, dtype=bool)
        
        for _ in range(n):
            # Find max similarity among unassigned
            S_masked = S.copy()
            S_masked[used_fac, :] = -np.inf
            S_masked[:, used_loc] = -np.inf
            
            idx = np.unravel_index(np.argmax(S_masked), S_masked.shape)
            f, l = idx
            p[f] = l
            used_fac[f] = True
            used_loc[l] = True
        
    except Exception:
        # Fallback to rank matching
        heavy = np.argsort(-a.sum(axis=1), kind="stable")
        central = np.argsort(b.sum(axis=1), kind="stable")
        p = np.empty(n, dtype=np.int64)
        p[heavy] = central
    
    return p.astype(np.int64)


def _anneal(a, b, p, start_time, deadline, best_cost, best_p, rng):
    """Simulated Annealing refinement."""
    n = len(p)
    cur = _qap_cost(a, b, p)
    if cur < best_cost:
        best_cost = cur
        best_p = p.copy()

    T = 5000.0
    T_end = 1.0
    alpha = 0.995
    
    while T > T_end and (time.time() - start_time) < deadline:
        i = rng.integers(n)
        j = rng.integers(n)
        if i == j:
            continue
            
        delta = _swap_delta(a, b, p, i, j)
        
        if delta < 0 or (rng.random() < np.exp(-delta / T)):
            p[i], p[j] = p[j], p[i]
            cur += delta
            
            if cur < best_cost:
                best_cost = cur
                best_p = p.copy()
                
        T *= alpha
        
        if time.time() - start_time > deadline:
            break
            
    if cur < best_cost:
        best_cost = cur
        best_p = p.copy()
        
    return best_cost, best_p


def solve(a, b):
    a = np.ascontiguousarray(a, dtype=np.int64)
    b = np.ascontiguousarray(b, dtype=np.int64)
    n = len(a)
    t0 = time.time()
    deadline = t0 + 110.0  # safety margin before 120s kill

    # --- Spectral embedding for geometrically coherent initial assignment ---
    p0 = _spectral_embedding(a, b)
    best_cost = _qap_cost(a, b, p0)
    best_p = p0.copy()

    # Local search on spectral start
    p = p0.copy()
    best_cost, best_p = _local_search(a, b, p, deadline, best_cost, best_p)

    # Simulated Annealing for diversification
    rng = np.random.default_rng(12345)
    p = best_p.copy()
    best_cost, best_p = _anneal(a, b, p, t0, deadline, best_cost, best_p, rng)

    # Random restarts with local search if time remains
    while time.time() < deadline - 5.0:
        p = rng.permutation(n).astype(np.int64)
        best_cost, best_p = _local_search(a, b, p, deadline, best_cost, best_p)
        
    return best_p.tolist()
# EVOLVE-BLOCK-END
