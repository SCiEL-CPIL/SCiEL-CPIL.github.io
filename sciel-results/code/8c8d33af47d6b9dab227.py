import numpy as np
import time

def priority(adjacency):
    start_time = time.time()
    DEADLINE = 145.0  # 5s safety margin

    A = np.asarray(adjacency, dtype=float)
    N = A.shape[0]

    if N == 0:
        return np.array([])

    # Ensure symmetric
    A = (A + A.T) / 2.0
    np.fill_diagonal(A, 0.0)
    A_bool = A > 0.5
    
    # Precompute neighbors
    nbrs = [np.where(A_bool[i])[0] for i in range(N)]
    D = np.array([len(nbrs[i]) for i in range(N)], dtype=float)

    # Features
    # 1. Degree
    F_deg = D.copy()
    
    # 2. Sum of Neighbor Degrees
    F_snd = np.array([np.sum(D[nbrs[i]]) if len(nbrs[i]) > 0 else 0 for i in range(N)])
    
    # 3. Average Neighbor Degree
    F_avg_nbr_deg = np.array([np.mean(D[nbrs[i]]) if len(nbrs[i]) > 0 else 0 for i in range(N)])
    
    # 4. Core Number
    core = np.zeros(N, dtype=int)
    deg = D.copy().astype(int)
    max_deg = int(deg.max()) if N > 0 else 0
    if max_deg > 0:
        buckets = [[] for _ in range(max_deg + 1)]
        for i in range(N):
            buckets[min(deg[i], max_deg)].append(i)
        remaining = np.ones(N, dtype=bool)
        current_max = max_deg
        while current_max >= 0:
            while buckets[current_max]:
                i = buckets[current_max].pop()
                if not remaining[i]:
                    continue
                core[i] = current_max
                remaining[i] = False
                for j in nbrs[i]:
                    if remaining[j]:
                        old_deg = deg[j]
                        deg[j] -= 1
                        try:
                            buckets[old_deg].remove(j)
                        except ValueError:
                            pass
                        new_deg = max(0, min(deg[j], max_deg))
                        buckets[new_deg].append(j)
            current_max -= 1
    F_core = core.astype(float)
    
    # 5. Eigenvector Centrality (Top eigenvector of A)
    try:
        eigvals, eigvecs = np.linalg.eigh(A)
        # Largest eigenvalue is last
        top_vec = eigvecs[:, -1]
        # Make it positive for consistency
        if np.sum(top_vec) < 0:
            top_vec = -top_vec
        F_eig = top_vec
    except Exception:
        F_eig = np.zeros(N)
        
    # 6. Cluster Coefficient (approximate: triangles / possible triangles)
    # Triangles
    A_sq = A @ A
    F_tri = np.diag(A_sq) / 2.0
    # Possible triangles: d*(d-1)/2
    poss_tri = D * (D - 1) / 2.0
    F_clust = np.zeros(N)
    mask = poss_tri > 0
    F_clust[mask] = F_tri[mask] / poss_tri[mask]
    
    # Normalize features
    def norm(x):
        x = np.asarray(x, dtype=float)
        min_val = x.min()
        max_val = x.max()
        if max_val == min_val:
            return np.zeros_like(x)
        return (x - min_val) / (max_val - min_val)
        
    F_norm = [
        norm(F_deg),
        norm(F_snd),
        norm(F_avg_nbr_deg),
        norm(F_core),
        norm(F_eig),
        norm(F_clust)
    ]
    
    F_mat = np.column_stack(F_norm)
    
    # Fast evaluation
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

    # Initialize scores with a strong heuristic
    # We want low degree, low neighbor degree, low core, high eigenvector centrality (peripheral nodes?), low cluster coefficient
    # Let's try a weighted sum.
    # Initial guess: Low Degree is usually best for MIS.
    w_init = np.array([-1.0, -0.5, -0.5, -0.5, 0.5, -0.5])
    scores = F_mat @ w_init
    
    # Add small noise to break ties
    rng = np.random.default_rng(42)
    scores += rng.normal(0, 0.01, N)
    
    current_size = evaluate_fast(scores)
    best_scores = scores.copy()
    best_size = current_size
    
    # Simulated Annealing
    T = 1.0
    T_min = 0.01
    alpha = 0.99995 # Cooling rate, needs to be tuned for ~100k-1M iterations
    
    # Estimate iterations based on time
    # Each iteration: generate move, evaluate (O(N^2) ~ 40k ops).
    # 40k ops is fast. Let's aim for as many as possible.
    
    iter_count = 0
    while time.time() < DEADLINE - 1.0:
        # Generate move
        move_type = rng.integers(0, 3)
        
        trial_scores = scores.copy()
        
        if move_type == 0:
            # Swap two vertices
            i, j = rng.integers(0, N, 2)
            if i != j:
                trial_scores[i], trial_scores[j] = trial_scores[j], trial_scores[i]
        elif move_type == 1:
            # Perturb one vertex
            i = rng.integers(0, N)
            trial_scores[i] += rng.normal(0, 0.5)
        else:
            # Perturb two vertices
            i, j = rng.integers(0, N, 2)
            if i != j:
                trial_scores[i] += rng.normal(0, 0.5)
                trial_scores[j] -= rng.normal(0, 0.5)
                
        trial_size = evaluate_fast(trial_scores)
        
        delta = trial_size - current_size
        
        if delta > 0 or rng.random() < np.exp(delta / T):
            scores = trial_scores
            current_size = trial_size
            if current_size > best_size:
                best_size = current_size
                best_scores = scores.copy()
                
        # Cool down
        T *= alpha
        if T < T_min:
            T = T_min
            # Reheat occasionally? Or just continue at T_min.
            # Let's do a quick reheat if stuck for a while
            # Simple reheat: if no improvement in last 1000 steps, T = 0.5
            # To keep it simple, we just let T go to T_min.
            
        iter_count += 1
        
        # Check time frequently to avoid overshoot
        if iter_count % 100 == 0 and time.time() > DEADLINE - 0.5:
            break

    return best_scores
