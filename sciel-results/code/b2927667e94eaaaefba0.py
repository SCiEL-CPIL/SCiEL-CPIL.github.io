"""Multi-start local search for Maximum Cut with ILS and SA polish.

Combines fast steepest-ascent local search from many diverse starts with
Iterated Local Search (ILS) perturbations and simulated annealing to escape
local optima, all under a strict wall-clock deadline.
"""
import time
import numpy as np


def _cut_value(s, A, E):
    # s in {+1,-1};  cut = E/2 - (0.25) * s^T A s
    return 0.5 * E - 0.25 * float(s @ (A @ s))


def _local_search(s, A, E, deadline):
    """Steepest-ascent local search: flip vertex with largest positive gain."""
    n = s.shape[0]
    s = s.astype(np.float64).copy()
    while True:
        if time.time() > deadline:
            break
        Ax = A @ s
        gains = s * Ax
        i = int(np.argmax(gains))
        if gains[i] <= 1e-9:
            break
        s[i] = -s[i]
    return s, _cut_value(s, A, E)


def _pairwise_polish(s, A, E, deadline, max_iters=200):
    """Pairwise (2-opt) flip polish: try flipping pairs of vertices.
    
    For flipping vertices i and j simultaneously:
    gain = 2 * s[i]*s[j] * (A[i,j] - 0.5*A[i,:]@s - 0.5*A[j,:]@s + 0.5*A[i,:]@s_j)
    where s_j is s with j flipped.
    Simpler: gain_2flip = gain_i + gain_j - 2*s[i]*s[j]*A[i,j]
    where gain_i = s[i]*(A@ s)[i] is single-flip gain for i.
    """
    n = s.shape[0]
    s = s.astype(np.float64).copy()
    val = _cut_value(s, A, E)
    
    for _ in range(max_iters):
        if time.time() > deadline:
            break
        Ax = A @ s
        gains = s * Ax  # single-flip gains
        
        # Find best pair (i, j) with i < j
        # gain_2flip(i,j) = gains[i] + gains[j] - 2*s[i]*s[j]*A[i,j]
        # We want to find the pair with largest positive gain_2flip
        # that improves the cut.
        
        best_gain = 0.0
        best_i, best_j = -1, -1
        
        # Use a smarter approach: for each i, find best j
        # gain_2flip(i,j) = gains[i] + gains[j] - 2*s[i]*s[j]*A[i,j]
        # For fixed i: maximize over j of (gains[j] - 2*s[i]*s[j]*A[i,j])
        # = gains[i] + max_j (gains[j] - 2*s[i]*s[j]*A[i,j])
        
        # Efficient: for each i, compute vector of (gains[j] - 2*s[i]*s[j]*A[i,j]) for all j
        # This is gains - 2*s[i]*(s * A[i,:])
        
        improved = False
        for i in range(n):
            if time.time() > deadline:
                break
            # For this i, find best j != i
            # candidate_gain(j) = gains[j] - 2*s[i]*s[j]*A[i,j]
            sA_i = s * A[i, :]  # element-wise: s[j] * A[i,j]
            candidate = gains - 2.0 * s[i] * sA_i
            candidate[i] = -np.inf  # exclude i
            
            j = int(np.argmax(candidate))
            pair_gain = gains[i] + candidate[j]
            
            if pair_gain > best_gain:
                best_gain = pair_gain
                best_i = i
                best_j = j
        
        if best_i == -1 or best_gain <= 1e-9:
            break
        
        # Apply the flip
        s[best_i] = -s[best_i]
        s[best_j] = -s[best_j]
        val = _cut_value(s, A, E)
        improved = True
    
    return s, val


def _perturb(s, rng, k_min=5, k_max=15):
    """Randomly flip k vertices."""
    n = s.shape[0]
    s = s.astype(np.float64).copy()
    k = int(rng.integers(k_min, k_max + 1))
    idx = rng.choice(n, size=k, replace=False)
    s[idx] = -s[idx]
    return s


def _simulated_annealing(s, A, E, deadline, rng, t_start=3.0, t_end=0.05, cooling=0.998):
    """Simulated annealing to escape local optima."""
    n = s.shape[0]
    s = s.astype(np.float64).copy()
    best_s = s.copy()
    best_val = _cut_value(s, A, E)
    cur = best_val
    T = t_start

    while T > t_end:
        if time.time() > deadline:
            break
        # Pick a random vertex to flip
        i = int(rng.integers(0, n))
        Ax = A @ s
        gain = s[i] * Ax[i]

        # Accept if gain > 0, or with probability exp(gain/T) if gain < 0
        if gain > 0 or rng.random() < np.exp(gain / T):
            s[i] = -s[i]
            cur += gain
            if cur > best_val:
                best_val = cur
                best_s = s.copy()

        T *= cooling

    return best_s, best_val


def partition(adjacency):
    A = np.asarray(adjacency, dtype=np.float64)
    A = 0.5 * (A + A.T)
    np.fill_diagonal(A, 0.0)
    n = A.shape[0]
    E = 0.5 * float(A.sum())

    start = time.time()
    deadline = start + 110.0  # safety margin under the 120 s limit
    rng = np.random.default_rng(42)

    # Generate diverse initializations
    inits = []

    # Structured starts
    alt = np.ones(n, dtype=np.float64)
    alt[1::2] = -1.0
    inits.append(alt)
    inits.append(np.ones(n, dtype=np.float64))
    inits.append(-np.ones(n, dtype=np.float64))

    # Random starts
    for _ in range(30):
        inits.append(rng.choice(np.array([-1.0, 1.0]), size=n))

    best_s = None
    best_val = -1.0

    # Phase 1: Local search from all starts
    for s0 in inits:
        if time.time() > deadline - 30.0:
            break
        s, val = _local_search(s0, A, E, deadline - 30.0)
        if val > best_val:
            best_val = val
            best_s = s.copy()

    # Phase 2: ILS on the best solution
    if best_s is not None and time.time() < deadline - 15.0:
        current_s = best_s.copy()
        current_val = best_val
        no_improve_count = 0
        max_iter = 500
        for _ in range(max_iter):
            if time.time() > deadline - 15.0:
                break
            # Perturb
            pert_s = _perturb(current_s, rng)
            # Local search from perturbed
            ls_s, ls_val = _local_search(pert_s, A, E, deadline - 15.0)
            if ls_val > current_val:
                current_s = ls_s
                current_val = ls_val
                no_improve_count = 0
                if current_val > best_val:
                    best_val = current_val
                    best_s = current_s.copy()
            else:
                no_improve_count += 1
                # If stuck for a while, try larger perturbation
                if no_improve_count % 10 == 0:
                    current_s = _perturb(current_s, rng, k_min=10, k_max=25)
                    current_val = _cut_value(current_s, A, E)

    # Phase 3: Simulated annealing polish
    if best_s is not None and time.time() < deadline - 5.0:
        for _ in range(2):
            if time.time() > deadline - 5.0:
                break
            s, val = _simulated_annealing(
                best_s, A, E, deadline - 5.0, rng,
                t_start=3.0, t_end=0.05, cooling=0.998
            )
            # Local search after SA
            s, val = _local_search(s, A, E, deadline - 5.0)
            if val > best_val:
                best_val = val
                best_s = s.copy()

    # Final local search polish
    if best_s is not None and time.time() < deadline:
        s, val = _local_search(best_s, A, E, deadline)
        if val > best_val:
            best_val = val
            best_s = s
    
    # Pairwise polish: try 2-flip moves to escape 1-flip local optima
    if best_s is not None and time.time() < deadline:
        s, val = _pairwise_polish(best_s, A, E, deadline, max_iters=50)
        if val > best_val:
            best_val = val
            best_s = s
        # Run another round of single-flip local search after pairwise
        s, val = _local_search(best_s, A, E, deadline)
        if val > best_val:
            best_val = val
            best_s = s

    return (best_s > 0).astype(int)
