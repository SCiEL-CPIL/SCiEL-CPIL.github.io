import numpy as np
import time

def priority(adjacency):
    start_time = time.time()
    DEADLINE = 135.0  # Safety margin (evaluator kills at 150s)

    A = np.asarray(adjacency, dtype=float)
    N = A.shape[0]

    if N == 0:
        return np.array([])

    # Ensure symmetric and zero diagonal
    A = (A + A.T) / 2.0
    np.fill_diagonal(A, 0.0)
    A_bool = A > 0.5

    # Precompute basic features
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

    # Feature matrix: columns are features where HIGHER is BETTER for IS
    # -D is good (low degree)
    # Clus is good (redundant neighbors, less conflict)
    # -OpenPairs is good (fewer independent neighbors)
    # -MeanND is good (lighter neighbors)
    # -MaxNbrDeg is good
    # MinNbrDeg is neutral
    
    F1 = -D_n
    F2 = Clus_n
    F3 = -OpenPairs_n
    F4 = -MeanND_n
    F5 = -MaxNbrDeg_n
    F6 = MinNbrDeg_n
    
    F_cols = np.column_stack((F1, F2, F3, F4, F5, F6))

    # Initial guess: Strongly favor low degree, moderately favor high clustering
    best_weights = np.array([1.0, 0.5, 0.5, 0.2, 0.1, 0.0])
    best_scores = F_cols @ best_weights
    best_size = evaluate_fast(best_scores)

    # Local Search with Tabu and Simulated Annealing hybrid
    rng = np.random.default_rng(42)
    
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

    # Phase 1: Coarse Random Starts
    for _ in range(20):
        w = rng.uniform(0.0, 1.5, size=6)
        w[0] = rng.uniform(0.5, 2.0) # Degree is most important
        try_update(w)

    # Phase 2: Simulated Annealing
    current_w = best_weights.copy()
    current_size = best_size
    T = 1.0  # Temperature
    cooling = 0.995
    step_size = 0.1
    no_improve = 0
    max_no_improve = 200
    tabu_list = []
    tabu_tenure = 20
    
    while time.time() < DEADLINE and no_improve < max_no_improve:
        # Check for timeout
        if time.time() > DEADLINE - 5.0:
            break
            
        # Pick a random weight to perturb
        idx = rng.integers(0, 6)
        delta = rng.normal(0, step_size)
        
        new_w = current_w.copy()
        new_w[idx] += delta
        new_w = np.maximum(new_w, 0.0)
        
        # Tabu check: avoid recent moves
        is_tabu = False
        for tw in tabu_list:
            if np.allclose(tw, new_w, atol=1e-6):
                is_tabu = True
                break
        
        if is_tabu:
            no_improve += 1
            continue
        
        new_scores = get_scores(new_w)
        new_size = evaluate_fast(new_scores)
        
        delta_obj = new_size - current_size
        
        # Accept if better, or with probability exp(delta_obj / T)
        if delta_obj > 0 or (T > 1e-6 and rng.random() < np.exp(delta_obj / T)):
            current_w = new_w
            current_size = new_size
            
            # Update best
            if current_size > best_size:
                best_size = current_size
                best_scores = new_scores.copy()
                best_weights = current_w.copy()
                no_improve = 0
                tabu_list = []  # Clear tabu on new best
            else:
                no_improve += 1
                
            # Add to tabu
            tabu_list.append(current_w.copy())
            if len(tabu_list) > tabu_tenure:
                tabu_list.pop(0)
            
            # Cool down
            T *= cooling
        else:
            no_improve += 1
            # Slightly increase step size if stuck
            if no_improve > 50:
                step_size *= 1.05

    # Phase 3: Final Polish with very small steps (Hill Climbing)
    if time.time() < DEADLINE - 2.0:
        step_size = 0.01
        no_improve = 0
        max_no_improve = 100
        current_w = best_weights.copy()
        current_size = best_size
        
        while time.time() < DEADLINE - 1.0 and no_improve < max_no_improve:
            idx = rng.integers(0, 6)
            delta = rng.normal(0, step_size)
            new_w = current_w.copy()
            new_w[idx] += delta
            new_w = np.maximum(new_w, 0.0)
            
            new_scores = get_scores(new_w)
            new_size = evaluate_fast(new_scores)
            
            if new_size > current_size:
                current_w = new_w
                current_size = new_size
                no_improve = 0
                if current_size > best_size:
                    best_size = current_size
                    best_scores = new_scores.copy()
                    best_weights = current_w.copy()
            else:
                no_improve += 1

    return best_scores
