import numpy as np
import time

def priority(adjacency):
    start_time = time.time()
    DEADLINE = 110.0  # Safety margin

    A = np.asarray(adjacency, dtype=float)
    N = A.shape[0]

    if N == 0:
        return np.array([])

    # Ensure symmetric and zero diagonal
    A = (A + A.T) / 2.0
    np.fill_diagonal(A, 0.0)
    A_bool = A > 0.5

    # Precompute features
    D = A.sum(axis=1)
    D_safe = D + 1e-9
    
    # Mean Neighbor Degree
    SND = A @ D
    MeanND = SND / D_safe

    # Triangles (edges among neighbors)
    A_sq = A @ A
    Tri = np.diag(A_sq) / 2.0

    # Clustering Coefficient
    Clus = np.zeros(N)
    mask = D > 1
    Clus[mask] = 2.0 * Tri[mask] / (D[mask] * (D[mask] - 1.0))

    # Open Pairs (neighbor pairs not connected)
    OpenPairs = np.maximum(0, 0.5 * D * (D - 1.0) - Tri)

    # Max Neighbor Degree
    MaxNbrDeg = np.zeros(N)
    for i in range(N):
        nbrs = np.where(A_bool[i])[0]
        if len(nbrs) > 0:
            MaxNbrDeg[i] = D[nbrs].max()

    # Min Neighbor Degree
    MinNbrDeg = np.zeros(N)
    for i in range(N):
        nbrs = np.where(A_bool[i])[0]
        if len(nbrs) > 0:
            MinNbrDeg[i] = D[nbrs].min()

    # Normalize features to [0, 1]
    def norm(x):
        min_val = x.min()
        max_val = x.max()
        if max_val == min_val:
            return np.zeros_like(x)
        return (x - min_val) / (max_val - min_val)

    D_n = norm(D)
    MeanND_n = norm(MeanND)
    Tri_n = norm(Tri)
    Clus_n = norm(Clus)
    OpenPairs_n = norm(OpenPairs)
    MaxNbrDeg_n = norm(MaxNbrDeg)
    MinNbrDeg_n = norm(MinNbrDeg)

    adj_rows = A_bool

    def evaluate_fast(scores):
        indices = np.argsort(-scores, kind='stable')
        is_picked = np.zeros(N, dtype=bool)
        count = 0
        for v in indices:
            if not is_picked[v]:
                count += 1
                is_picked[v] = True
                is_picked[adj_rows[v]] = True
        return count

    # Feature matrix for linear combination
    # Columns: D, MeanND, Tri, Clus, OpenPairs, MaxNbrDeg, MinNbrDeg
    # We will search for weights w such that score = w @ X
    # Note: We want to MINIMIZE degree, so w for D should be negative.
    # To simplify search, we can flip signs in the feature matrix for "bad" features.
    # Let's define features such that HIGHER is BETTER for IS.
    # -D is good. So use -D_n.
    # Clus is good (redundant neighbors). So use Clus_n.
    # OpenPairs is bad (independent neighbors). So use -OpenPairs_n.
    # MeanND is generally bad (heavy neighbors). So use -MeanND_n.
    
    F1 = -D_n
    F2 = Clus_n
    F3 = -OpenPairs_n
    F4 = -MeanND_n
    F5 = -MaxNbrDeg_n
    F6 = MinNbrDeg_n # Low min degree in neighborhood? Maybe neutral.
    
    # Initial guess: Strongly favor low degree, moderately favor high clustering
    best_weights = np.array([1.0, 0.5, 0.5, 0.2, 0.1, 0.0])
    best_scores = F1 * best_weights[0] + F2 * best_weights[1] + F3 * best_weights[2] + \
                  F4 * best_weights[3] + F5 * best_weights[4] + F6 * best_weights[5]
    best_size = evaluate_fast(best_scores)

    # Hill Climbing / Local Search
    rng = np.random.default_rng(42)
    
    # Pre-compute feature columns for speed
    F_cols = np.column_stack((F1, F2, F3, F4, F5, F6))
    
    def get_scores(weights):
        return F_cols @ weights

    def try_update(weights):
        nonlocal best_size, best_scores, best_weights
        scores = get_scores(weights)
        size = evaluate_fast(scores)
        if size > best_size:
            best_size = size
            best_scores = scores.copy()
            best_weights = weights.copy()
            return True
        return False

    # Phase 1: Coarse Grid / Random Start
    for _ in range(20):
        w = rng.uniform(0.0, 1.5, size=6)
        w[0] = rng.uniform(0.5, 2.0) # Degree is most important
        try_update(w)

    # Phase 2: Hill Climbing
    step_size = 0.1
    no_improve = 0
    max_no_improve = 50
    
    while time.time() < DEADLINE:
        if no_improve > max_no_improve:
            break
            
        # Pick a random weight to perturb
        idx = rng.integers(0, 6)
        delta = rng.normal(0, step_size)
        
        new_w = best_weights.copy()
        new_w[idx] += delta
        
        # Keep weights positive (since features are already sign-flipped)
        # Exception: maybe allow some negative if we want to invert logic, 
        # but our features are constructed to be "higher is better".
        new_w = np.maximum(new_w, 0.0)
        
        if try_update(new_w):
            no_improve = 0
            step_size *= 0.9 # Shrink step on success
        else:
            no_improve += 1
            if no_improve > 10:
                step_size *= 0.8 # Shrink step on failure too, slowly

    # Phase 3: Final Polish with very small steps
    if time.time() < DEADLINE - 5.0:
        step_size = 0.01
        for _ in range(500):
            if time.time() > DEADLINE:
                break
            idx = rng.integers(0, 6)
            delta = rng.normal(0, step_size)
            new_w = best_weights.copy()
            new_w[idx] += delta
            new_w = np.maximum(new_w, 0.0)
            try_update(new_w)

    return best_scores
