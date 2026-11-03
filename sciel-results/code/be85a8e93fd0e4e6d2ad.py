"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np


def _swap_delta(i, j, p, a, b):
    """Delta in cost when swapping the two facilities currently at positions i and j."""
    pi = p[i]
    pj = p[j]
    # term 1: flow between i and j, distance changes sign
    t1 = (a[i, j] + a[j, i]) * (b[pi, pj] - b[pj, pi])
    # term 2: combined flow row against all other locations
    flow = a[i] + a[j]  # a[i,k]+a[j,k] for all k
    # distance row: b[pi, :] - b[pj, :]
    dist = b[pi] - b[pj]
    t2 = flow @ dist
    # subtract the i<->j contribution already counted in t2 (k=i and k=j)
    # For k=i: flow[i]=a[i,i]+a[j,i], dist[i]=b[pi,pi]-b[pj,pi]
    # For k=j: flow[j]=a[i,j]+a[j,j], dist[j]=b[pi,pj]-b[pj,pj]
    t2 -= (a[i, i] + a[j, i]) * (b[pi, pi] - b[pj, pi])
    t2 -= (a[i, j] + a[j, j]) * (b[pi, pj] - b[pj, pj])
    return t1 + t2


def _compute_cost(p, a, b):
    """Compute total cost for permutation p."""
    n = len(p)
    c = 0
    for i in range(n):
        c += a[i] @ b[p[i], p]
    return c


def _local_search(p, a, b, max_rounds, deadline):
    """Steepest-descent 2-opt. Returns improved permutation and its cost."""
    n = len(p)
    p = p.copy()
    c = _compute_cost(p, a, b)
    improved = True
    rounds = 0
    while improved and rounds < max_rounds:
        if time.time() >= deadline:
            break
        improved = False
        rounds += 1
        best_d = 0
        best_i = -1
        best_j = -1
        for i in range(n):
            if time.time() >= deadline:
                break
            for j in range(i + 1, n):
                d = _swap_delta(i, j, p, a, b)
                if d < best_d:
                    best_d = d
                    best_i = i
                    best_j = j
        if best_i >= 0:
            p[best_i], p[best_j] = p[best_j], p[best_i]
            c += best_d
            improved = True
    return p, c


def _perturb(p, k, rng):
    """Apply k random swaps."""
    p = p.copy()
    n = len(p)
    for _ in range(k):
        i, j = rng.integers(0, n, size=2)
        if i != j:
            p[i], p[j] = p[j], p[i]
    return p


def _spectral_init(a, b, n, rng, num_eig=10):
    """
    Spectral initialization: use eigenvectors of transformed matrices
    to generate high-quality initial assignments.
    """
    # Center and scale matrices for better spectral properties
    a_centered = a - a.mean(axis=1, keepdims=True)
    b_centered = b - b.mean(axis=1, keepdims=True)
    
    # Symmetrize for eigenvalue decomposition
    a_sym = (a_centered + a_centered.T) / 2
    b_sym = (b_centered + b_centered.T) / 2
    
    # Get top eigenvectors
    try:
        _, a_eigvecs = np.linalg.eigh(a_sym)
        _, b_eigvecs = np.linalg.eigh(b_sym)
    except np.linalg.LinAlgError:
        # Fallback to random if eigenvalue computation fails
        return rng.permutation(n).astype(np.int64)
    
    # Use top eigenvectors (largest eigenvalues are at the end for eigh)
    a_top = a_eigvecs[:, -num_eig:]
    b_top = b_eigvecs[:, -num_eig:]
    
    # Compute projection scores: for each facility i, project onto eigenspace
    # For each location j, project onto eigenspace
    # Match based on similarity in eigenspace
    
    # Create embedding of facilities and locations in eigenspace
    # Facility i's embedding: [a_top[i,0], a_top[i,1], ..., a_top[i,k-1]]
    # Location j's embedding: [b_top[j,0], b_top[j,1], ..., b_top[j,k-1]]
    
    # We want to match facilities to locations such that the dot product
    # of their embeddings is maximized (or minimized, depending on sign)
    
    # Compute pairwise similarity
    sim = a_top @ b_top.T  # n x n matrix where sim[i,j] = similarity of facility i to location j
    
    # For minimization, we want to assign facilities to locations with low similarity
    # (since high flow should go to low distance)
    
    # Use greedy assignment based on similarity
    p = np.full(n, -1, dtype=np.int64)
    assigned_loc = np.full(n, False)
    
    # Process facilities in order of decreasing "importance" (row sum of a)
    importance = np.argsort(-a.sum(axis=1), kind="stable")
    
    for fac in importance:
        # Find best unassigned location
        best_loc = -1
        best_val = np.inf
        for loc in range(n):
            if assigned_loc[loc]:
                continue
            val = sim[fac, loc]
            if val < best_val:
                best_val = val
                best_loc = loc
        if best_loc < 0:
            # Fallback: pick random unassigned
            unassigned = np.where(~assigned_loc)[0]
            best_loc = unassigned[rng.integers(0, len(unassigned))]
        p[fac] = best_loc
        assigned_loc[best_loc] = True
    
    return p


def _spectral_init_v2(a, b, n, rng, num_eig=8):
    """
    Alternative spectral initialization using different transformation.
    """
    # Use the Hadamard product structure
    # The QAP can be related to matrix factorization
    
    # Center matrices
    a_c = a - a.mean()
    b_c = b - b.mean()
    
    # Symmetrize
    a_s = (a_c + a_c.T) / 2
    b_s = (b_c + b_c.T) / 2
    
    # Get eigendecomposition
    try:
        a_vals, a_vecs = np.linalg.eigh(a_s)
        b_vals, b_vecs = np.linalg.eigh(b_s)
    except np.linalg.LinAlgError:
        return rng.permutation(n).astype(np.int64)
    
    # Use top eigenvectors
    k = min(num_eig, n)
    a_top = a_vecs[:, -k:]
    b_top = b_vecs[:, -k:]
    
    # Sign alignment: ensure consistent orientation
    # Make the first non-zero element of each eigenvector positive
    for i in range(k):
        idx = np.argmax(np.abs(a_top[:, i]) > 1e-10)
        if a_top[idx, i] < 0:
            a_top[:, i] *= -1
        idx = np.argmax(np.abs(b_top[:, i]) > 1e-10)
        if b_top[idx, i] < 0:
            b_top[:, i] *= -1
    
    # Compute assignment matrix
    # M[i,j] = sum_k a_top[i,k] * b_top[j,k]
    M = a_top @ b_top.T
    
    # Greedy assignment: assign facility i to location j that minimizes M[i,j]
    p = np.full(n, -1, dtype=np.int64)
    assigned_loc = np.full(n, False)
    
    # Sort facilities by their "energy" in eigenspace
    energy = np.sum(a_top ** 2, axis=1)
    order = np.argsort(-energy, kind="stable")
    
    for fac in order:
        best_loc = -1
        best_val = np.inf
        for loc in range(n):
            if assigned_loc[loc]:
                continue
            val = M[fac, loc]
            if val < best_val:
                best_val = val
                best_loc = loc
        if best_loc < 0:
            unassigned = np.where(~assigned_loc)[0]
            best_loc = unassigned[rng.integers(0, len(unassigned))]
        p[fac] = best_loc
        assigned_loc[best_loc] = True
    
    return p


def solve(a, b):
    start = time.time()
    deadline = start + 135.0  # 15s safety margin before 150s kill
    n = len(a)
    a = np.ascontiguousarray(a, dtype=np.int64)
    b = np.ascontiguousarray(b, dtype=np.int64)
    rng = np.random.default_rng(12345)

    # --- Build diverse initial solutions ---
    inits = []

    # 1. Rank-matching: heavy flow rows -> central (low distance sum) locations
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p1 = np.empty(n, dtype=np.int64)
    p1[heavy] = central
    inits.append(p1)

    # 2. Spectral initialization v1
    if time.time() < deadline - 60.0:
        p_spec1 = _spectral_init(a, b, n, rng, num_eig=10)
        inits.append(p_spec1)

    # 3. Spectral initialization v2
    if time.time() < deadline - 50.0:
        p_spec2 = _spectral_init_v2(a, b, n, rng, num_eig=8)
        inits.append(p_spec2)

    # 4. Random
    p3 = rng.permutation(n).astype(np.int64)
    inits.append(p3)

    # --- Evaluate each init, keep best after local search ---
    best_p = None
    best_c = np.inf
    for idx, init in enumerate(inits):
        if time.time() >= deadline - 40.0:
            break
        # Give more time to spectral inits
        if idx >= 1:
            dl = deadline - 20.0
        else:
            dl = deadline - 30.0
        p, c = _local_search(init, a, b, max_rounds=25, deadline=dl)
        if c < best_c:
            best_c = c
            best_p = p.copy()

    # --- ILS: perturb best, re-optimize, keep if better ---
    ils_round = 0
    while time.time() < deadline - 5.0:
        ils_round += 1
        # increase perturbation strength slowly
        k = min(2 + ils_round // 4, 12)
        cand = _perturb(best_p, k, rng)
        p, c = _local_search(cand, a, b, max_rounds=25, deadline=deadline - 2.0)
        if c < best_c:
            best_c = c
            best_p = p.copy()
            ils_round = 0

    return best_p.tolist()
# EVOLVE-BLOCK-END
