import numpy as np
import time

def priority(adjacency):
    start_time = time.time()
    DEADLINE = 145.0  # 5s safety margin before 150s kill

    A = np.asarray(adjacency, dtype=float)
    N = A.shape[0]

    if N == 0:
        return np.array([])

    # Ensure symmetric
    A = (A + A.T) / 2.0
    np.fill_diagonal(A, 0.0)
    A_bool = A > 0.5

    # Precompute adjacency rows for fast evaluation
    nbrs = [np.where(A_bool[i])[0] for i in range(N)]
    nbrs_set = [set(nbrs[i]) for i in range(N)] # For potential other uses, but list is faster for blocking

    # Features
    D = A.sum(axis=1)
    D_safe = D + 1e-9

    # Sum of Neighbor Degrees
    SND = A @ D
    
    # Triangles
    A_sq = A @ A
    Tri = np.diag(A_sq) / 2.0

    # Clustering coefficient
    Clus = np.zeros(N)
    mask = D > 1
    Clus[mask] = 2.0 * Tri[mask] / (D[mask] * (D[mask] - 1.0))

    # Core Numbers (k-core decomposition)
    core = np.zeros(N, dtype=int)
    deg = D.copy().astype(int)
    max_deg = int(deg.max()) if N > 0 else 0
    # Simple iterative core number calculation
    # For N=200, a simple O(N^2) or O(NM) approach is fine
    remaining = set(range(N))
    while remaining:
        min_deg = min(deg[i] for i in remaining)
        to_remove = [i for i in remaining if deg[i] == min_deg]
        if not to_remove:
            break
        for i in to_remove:
            core[i] = min_deg
            remaining.discard(i)
        # Update degrees of neighbors
        for i in to_remove:
            for j in nbrs[i]:
                if j in remaining:
                    deg[j] -= 1

    # Distance 2 Non-Neighbors count
    mask_non_nbr = ~A_bool
    np.fill_diagonal(mask_non_nbr, False)
    Dist2NonNbr = (A_sq * mask_non_nbr).sum(axis=1)

    # Laplacian Eigenvectors
    L = np.diag(D) - A
    try:
        eigvals, eigvecs = np.linalg.eigh(L)
        idx = np.argsort(eigvals)
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        
        # Take first few non-trivial eigenvectors
        # eigvecs[:, 0] is constant (all 1s) for connected graph
        # eigvecs[:, 1] is Fiedler
        # We will use eigvecs[:, 1] to eigvecs[:, K]
    except Exception:
        eigvecs = np.zeros((N, 10))

    # PageRank (Power Iteration)
    d = 0.85
    pr = np.ones(N) / N
    for _ in range(50):
        pr_new = (1 - d) / N + d * (A.T @ pr)
        if np.allclose(pr, pr_new):
            break
        pr = pr_new

    # Build Feature Matrix
    # Features:
    # 0: -D
    # 1: -SND
    # 2: -Clus
    # 3: -Core
    # 4: -Dist2NonNbr
    # 5: -Pr
    # 6: eigvecs[:, 1] (Fiedler)
    # 7: -abs(eigvecs[:, 1])
    # 8: eigvecs[:, 2]
    # 9: -abs(eigvecs[:, 2])
    # 10: eigvecs[:, 3]
    # 11: -abs(eigvecs[:, 3])
    
    def norm(x):
        x = np.asarray(x, dtype=float)
        min_val = x.min()
        max_val = x.max()
        if max_val == min_val:
            return np.zeros_like(x)
        return (x - min_val) / (max_val - min_val)

    F = np.zeros((N, 12))
    F[:, 0] = norm(-D)
    F[:, 1] = norm(-SND)
    F[:, 2] = norm(-Clus)
    F[:, 3] = norm(-core.astype(float))
    F[:, 4] = norm(-Dist2NonNbr)
    F[:, 5] = norm(-pr)
    
    # Ensure eigvecs has enough columns
    if eigvecs.shape[1] > 1:
        F[:, 6] = norm(eigvecs[:, 1])
        F[:, 7] = norm(-np.abs(eigvecs[:, 1]))
    if eigvecs.shape[1] > 2:
        F[:, 8] = norm(eigvecs[:, 2])
        F[:, 9] = norm(-np.abs(eigvecs[:, 2]))
    if eigvecs.shape[1] > 3:
        F[:, 10] = norm(eigvecs[:, 3])
        F[:, 11] = norm(-np.abs(eigvecs[:, 3]))

    num_features = F.shape[1]

    # Fast evaluation function
    def evaluate_fast(scores):
        indices = np.argsort(-scores, kind='stable')
        blocked = np.zeros(N, dtype=bool)
        count = 0
        for v in indices:
            if not blocked[v]:
                count += 1
                blocked[v] = True
                if len(nbrs[v]) > 0:
                    blocked[nbrs[v]] = True
        return count

    # Initial best
    # Try a few heuristic weight vectors
    rng = np.random.default_rng(42)
    
    initial_weights = [
        np.array([1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), # Degree
        np.array([0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), # SND
        np.array([1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), # Degree + SND
        np.array([1.0, 0.5, 0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0]), # Mix
        np.array([0.5, 0.5, 0, 0, 0, 0.5, 0, 0, 0, 0, 0, 0]), # + PageRank
    ]
    
    best_size = 0
    best_weights = None
    best_scores = None

    for w in initial_weights:
        scores = F @ w
        sz = evaluate_fast(scores)
        if sz > best_size:
            best_size = sz
            best_weights = w.copy()
            best_scores = scores.copy()

    # Simulated Annealing Search
    # We search in the weight space
    current_w = best_weights.copy()
    current_scores = best_scores.copy()
    current_size = best_size
    
    T = 1.0
    T_min = 0.01
    alpha = 0.999 # Cooling rate
    
    # Estimate steps we can take
    # Evaluation is fast, ~0.1ms?
    # 150s / 0.0001s = 1.5M steps.
    # Let's aim for 500,000 steps to be safe.
    
    while time.time() < DEADLINE:
        # Perturb
        # Pick a random feature to perturb
        i = rng.integers(0, num_features)
        delta = rng.normal(0, 0.1)
        
        new_w = current_w.copy()
        new_w[i] += delta
        
        new_scores = F @ new_w
        new_size = evaluate_fast(new_scores)
        
        # Accept if better, or with probability exp((new_size - current_size)/T)
        if new_size > current_size:
            current_w = new_w
            current_scores = new_scores
            current_size = new_size
            
            if current_size > best_size:
                best_size = current_size
                best_weights = current_w.copy()
                best_scores = current_scores.copy()
        else:
            # Metropolis criterion
            # Since size is integer, delta is often 0 or -1
            # If delta is 0, accept with prob 1? No, that leads to plateaus.
            # If delta < 0, accept with prob exp(delta / T)
            if new_size == current_size:
                # Accept to move around plateau
                if rng.random() < 0.5:
                    current_w = new_w
                    current_scores = new_scores
            else:
                # new_size < current_size
                prob = np.exp((new_size - current_size) / T)
                if rng.random() < prob:
                    current_w = new_w
                    current_scores = new_scores
                    current_size = new_size

        T *= alpha
        if T < T_min:
            T = T_min
            # Restart temperature if stuck? Or just keep going.
            # Let's do a random restart of weights if we haven't improved in a while.
            # Track iterations since last improvement
            # (Simplification: just let it run, the noise will help)

    return best_scores
