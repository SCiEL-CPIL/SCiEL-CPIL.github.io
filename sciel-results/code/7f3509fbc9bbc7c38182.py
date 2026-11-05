import numpy as np
import time

def priority(adjacency):
    start_time = time.time()
    DEADLINE = 145.0  # Safety margin

    A = np.asarray(adjacency, dtype=float)
    N = A.shape[0]

    if N == 0:
        return np.array([])

    # Ensure symmetric and no self-loops
    A = (A + A.T) / 2.0
    np.fill_diagonal(A, 0.0)
    A_bool = A > 0.5

    # Precompute neighbor lists for fast evaluation
    nbrs = [np.where(A_bool[i])[0] for i in range(N)]

    # Features
    D = A.sum(axis=1)
    D_safe = D + 1e-9

    # Sum of Neighbor Degrees
    SND = A @ D
    MeanND = SND / D_safe

    # Triangles
    A_sq = A @ A
    Tri = np.diag(A_sq) / 2.0

    # Clustering coefficient
    Clus = np.zeros(N)
    mask = D > 1
    Clus[mask] = 2.0 * Tri[mask] / (D[mask] * (D[mask] - 1.0))

    # Fiedler Vector (Second smallest eigenvector of Laplacian)
    L = D - A
    # For N=200, eigh is fast and accurate
    try:
        eigvals, eigvecs = np.linalg.eigh(L)
        # eigvals are sorted ascending. 0 is smallest (connected component).
        # Second smallest is index 1.
        fiedler = eigvecs[:, 1]
        Fiedler_mag = np.abs(fiedler)
    except Exception:
        # Fallback if eigendecomposition fails or graph is disconnected in a way that causes issues
        Fiedler_mag = np.zeros(N)

    # Normalize features
    def norm(x):
        min_val = x.min()
        max_val = x.max()
        if max_val == min_val:
            return np.zeros_like(x)
        return (x - min_val) / (max_val - min_val)

    D_n = norm(D)
    SND_n = norm(SND)
    MeanND_n = norm(MeanND)
    Tri_n = norm(Tri)
    Clus_n = norm(Clus)
    Fiedler_n = norm(Fiedler_mag)

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

    # Baseline: Low Degree
    scores_base = -D_n
    best_size = evaluate_fast(scores_base)
    best_scores = scores_base.copy()

    # Search Strategy:
    # We search over a linear combination of features.
    # Score = w0 - w1*D_n - w2*Fiedler_n - w3*Clus_n - w4*SND_n
    # We use a local search / hill climbing on the weights.
    
    rng = np.random.default_rng(42)
    
    # Initial candidate weights
    # (w1, w2, w3, w4)
    candidates = [
        (1.0, 0.0, 0.0, 0.0),   # -D
        (0.8, 0.2, 0.0, 0.0),   # -0.8D - 0.2Fiedler
        (0.8, 0.0, 0.2, 0.0),   # -0.8D - 0.2Clus
        (0.7, 0.1, 0.1, 0.1),   # Mixed
        (0.6, 0.2, 0.1, 0.1),   # More Fiedler
        (0.5, 0.3, 0.1, 0.1),   # Even more Fiedler
        (0.9, 0.0, 0.1, 0.0),   # -0.9D - 0.1Clus
        (1.0, 0.1, 0.0, 0.0),   # -D - 0.1Fiedler
    ]
    
    for w in candidates:
        s = - (w[0]*D_n + w[1]*Fiedler_n + w[2]*Clus_n + w[3]*SND_n)
        sz = evaluate_fast(s)
        if sz > best_size:
            best_size = sz
            best_scores = s.copy()

    # Simulated Annealing on the score vector directly
    # This allows for non-linear interactions that a linear combination of features might miss.
    current_scores = best_scores.copy()
    current_size = best_size
    T = 0.1  # Initial temperature
    cooling_rate = 0.999
    
    # Max iterations limited by time
    max_iterations = 100000
    for it in range(max_iterations):
        if time.time() > DEADLINE - 10.0:
            break
        
        # Perturb scores
        # Smaller perturbations for fine-tuning
        perturbation_scale = T
        perturbed = current_scores + rng.normal(0, perturbation_scale, size=N)
        
        sz = evaluate_fast(perturbed)
        
        if sz > current_size:
            current_scores = perturbed.copy()
            current_size = sz
            if sz > best_size:
                best_size = sz
                best_scores = perturbed.copy()
        else:
            delta = sz - current_size
            if delta < 0:
                prob = np.exp(delta / max(T, 1e-6))
                if rng.random() < prob:
                    current_scores = perturbed.copy()
                    current_size = sz
        
        T *= cooling_rate

    # Final Local Polish: Single vertex adjustments
    if time.time() < DEADLINE - 2.0:
        for _ in range(2000):
            if time.time() > DEADLINE - 1.0:
                break
            v = rng.integers(0, N)
            # Try increasing and decreasing the score of a single vertex
            # Increasing score makes it more likely to be picked
            # Decreasing score makes it less likely
            for delta in [0.01, -0.01, 0.05, -0.05]:
                perturbed = best_scores.copy()
                perturbed[v] += delta
                sz = evaluate_fast(perturbed)
                if sz > best_size:
                    best_size = sz
                    best_scores = perturbed.copy()
                    break # Move to next vertex if improved

    return best_scores
