import numpy as np
import time

def priority(adjacency):
    start_time = time.time()
    DEADLINE = 135.0  # Safety margin before 150s kill

    A = np.asarray(adjacency, dtype=float)
    N = A.shape[0]

    if N == 0:
        return np.array([])

    # Ensure symmetric and zero diagonal
    A = (A + A.T) / 2.0
    np.fill_diagonal(A, 0.0)
    A_bool = A > 0.5

    # Precompute adjacency rows and degrees
    nbrs = [np.where(A_bool[i])[0] for i in range(N)]
    D = np.array([len(nbrs[i]) for i in range(N)], dtype=float)

    # Features
    # 1. Degree
    F_deg = D.copy()

    # 2. Sum of Neighbor Degrees
    F_snd = np.array([np.sum(D[nbrs[i]]) if len(nbrs[i]) > 0 else 0 for i in range(N)])

    # 3. Block Cost: Sum of 1/(1+deg[neighbor])
    F_block_cost = np.array([np.sum(1.0 / (1.0 + D[nbrs[i]])) if len(nbrs[i]) > 0 else 0 for i in range(N)])

    # 4. Triangles
    A_sq = A @ A
    F_tri = np.diag(A_sq) / 2.0

    # 5. Core Number
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

    # 6. Fiedler Vector (2nd smallest eigenvalue of Laplacian)
    L = np.diag(D) - A
    try:
        eigvals, eigvecs = np.linalg.eigh(L)
        idx = np.argsort(eigvals)
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        fiedler = eigvecs[:, 1]
        F_fiedler_abs = np.abs(fiedler)
    except Exception:
        F_fiedler_abs = np.zeros(N)

    # 7. Average neighbor degree
    F_avg_nbr_deg = np.array([np.mean(D[nbrs[i]]) if len(nbrs[i]) > 0 else 0 for i in range(N)])

    # 8. Principal Eigenvector (Perron Vector)
    try:
        eigvals_A, eigvecs_A = np.linalg.eigh(A)
        idx_A = np.argsort(eigvals_A)
        eigvals_A = eigvals_A[idx_A]
        eigvecs_A = eigvecs_A[:, idx_A]
        v1 = eigvecs_A[:, -1]
        max_val = np.max(np.abs(v1))
        if max_val > 0:
            v1 = v1 / max_val
        else:
            v1 = np.zeros(N)
    except Exception:
        v1 = np.zeros(N)
    F_perron = v1

    # 9. Complement Clique Density: For each vertex, count edges among its non-neighbors
    #    This measures how "clustered" the non-neighbors are (dense in complement = isolated in original)
    #    Use efficient computation: for vertex i, non-nbrs are all except i and nbrs[i]
    #    Count = number of pairs (u,v) in non-nbrs[i] such that A[u,v] == 0
    F_clique_density = np.zeros(N)
    for i in range(N):
        if len(nbrs[i]) == N - 1:
            F_clique_density[i] = 0
            continue
        # Get non-neighbors (excluding self)
        non_nbr_mask = np.ones(N, dtype=bool)
        non_nbr_mask[i] = False
        non_nbr_mask[nbrs[i]] = False
        non_nbr_idx = np.where(non_nbr_mask)[0]
        if len(non_nbr_idx) < 2:
            F_clique_density[i] = 0
            continue
        # Count edges among non-neighbors in original graph
        # If many edges among non-neighbors in original, they are NOT a clique in complement
        # We want: few edges among non-neighbors in original = dense clique in complement
        sub_A = A[np.ix_(non_nbr_idx, non_nbr_idx)]
        edge_count = np.sum(sub_A > 0.5) / 2.0
        possible_edges = len(non_nbr_idx) * (len(non_nbr_idx) - 1) / 2.0
        if possible_edges > 0:
            # Score: 1 - (edges / possible) = fraction of non-edges among non-neighbors
            F_clique_density[i] = 1.0 - (edge_count / possible_edges)
        else:
            F_clique_density[i] = 0

    # 10. Weighted independent set heuristic: sum of 1/(1+deg[j]) for non-neighbors j
    #     This approximates the expected IS size if we pick i and then greedily pick from non-neighbors
    F_wis_heuristic = np.zeros(N)
    for i in range(N):
        if len(nbrs[i]) == N - 1:
            F_wis_heuristic[i] = 0
            continue
        non_nbr_mask = np.ones(N, dtype=bool)
        non_nbr_mask[i] = False
        non_nbr_mask[nbrs[i]] = False
        non_nbr_idx = np.where(non_nbr_mask)[0]
        if len(non_nbr_idx) == 0:
            F_wis_heuristic[i] = 0
        else:
            F_wis_heuristic[i] = np.sum(1.0 / (1.0 + D[non_nbr_idx]))

    # Normalize features to [0, 1]
    def norm(x):
        x = np.asarray(x, dtype=float)
        min_val = x.min()
        max_val = x.max()
        if max_val == min_val:
            return np.zeros_like(x)
        return (x - min_val) / (max_val - min_val)

    def norm_signed(x):
        x = np.asarray(x, dtype=float)
        x = x - x.mean()
        max_abs = np.max(np.abs(x))
        if max_abs == 0:
            return np.zeros_like(x)
        return x / max_abs

    F_norm = [
        norm(F_deg),
        norm(F_snd),
        norm(F_block_cost),
        norm(F_tri),
        norm(F_core),
        norm(F_fiedler_abs),
        norm(F_avg_nbr_deg),
        norm(F_clique_density),
        norm(F_wis_heuristic)
    ]
    F_perron_norm = norm_signed(F_perron)

    # Feature Matrix: N x 10
    F_mat = np.column_stack(F_norm + [F_perron_norm])
    n_feat = F_mat.shape[1]

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

    # Multiple initial weight strategies - include weights for new features
    rng = np.random.default_rng(42)

    strategies = [
        np.array([-1.0, 1.0, -1.0, 0.0, -1.0, 1.0, 1.0, 0.5, 0.5, 0.5]),
        np.array([-1.0, 0.5, -1.0, 0.0, -1.0, 0.5, 1.0, 0.5, 0.5, 0.5]),
        np.array([-0.5, 1.0, -0.5, 0.0, -0.5, 1.0, 0.5, 0.5, 0.5, 0.5]),
        np.array([-1.0, 1.0, -1.0, 0.5, -1.0, 1.0, 0.5, 0.5, 0.5, 0.5]),
        np.array([-1.0, 1.0, -1.0, -0.5, -1.0, 1.0, 1.0, 0.5, 0.5, 0.5]),
        np.array([-1.5, 1.0, -1.0, 0.0, -1.0, 0.5, 1.5, 0.5, 0.5, 0.5]),
        np.array([-1.0, 1.0, -1.0, 0.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
        np.array([-1.0, 1.0, -1.0, 0.0, -1.0, 1.0, 1.0, -1.0, -1.0, -1.0]),
        # New strategies emphasizing clique density
        np.array([-1.0, 1.0, -1.0, 0.0, -1.0, 1.0, 1.0, 1.0, 1.0, 0.5]),
        np.array([-0.5, 0.5, -0.5, 0.0, -0.5, 0.5, 0.5, 1.0, 1.0, 0.5]),
        np.array([-1.0, 1.0, -1.0, 0.0, -1.0, 1.0, 1.0, 0.8, 0.8, 0.8]),
        rng.normal(0, 1, n_feat),
        rng.normal(0, 1, n_feat),
    ]

    best_size = 0
    best_scores = None

    for w in strategies:
        s = F_mat @ w
        sz = evaluate_fast(s)
        if sz > best_size:
            best_size = sz
            best_scores = s.copy()

    # Coordinate descent on weights
    def coordinate_descent(w, max_iter=500):
        current_w = w.copy()
        current_s = F_mat @ current_w
        current_sz = evaluate_fast(current_s)
        step = 0.1
        for _ in range(max_iter):
            if time.time() > DEADLINE - 5:
                break
            improved = False
            for d in range(n_feat):
                for delta in [step, -step]:
                    trial_w = current_w.copy()
                    trial_w[d] += delta
                    trial_s = F_mat @ trial_w
                    trial_sz = evaluate_fast(trial_s)
                    if trial_sz > current_sz:
                        current_w = trial_w
                        current_s = trial_s
                        current_sz = trial_sz
                        improved = True
            if not improved:
                step *= 0.5
                if step < 0.001:
                    break
        return current_w, current_s, current_sz

    # Run coordinate descent from best strategy
    if best_scores is not None:
        w_best = np.linalg.lstsq(F_mat, best_scores, rcond=None)[0]
        w_cd, s_cd, sz_cd = coordinate_descent(w_best, max_iter=300)
        if sz_cd > best_size:
            best_size = sz_cd
            best_scores = s_cd.copy()

    # Random restarts with coordinate descent
    for _ in range(30):
        if time.time() > DEADLINE - 10:
            break
        w_rand = rng.normal(0, 1, n_feat)
        w_cd, s_cd, sz_cd = coordinate_descent(w_rand, max_iter=100)
        if sz_cd > best_size:
            best_size = sz_cd
            best_scores = s_cd.copy()

    # Local polish: direct score perturbations (Gaussian noise)
    if best_scores is not None and time.time() < DEADLINE - 3:
        for _ in range(5000):
            if time.time() > DEADLINE - 1:
                break
            s_pert = best_scores + rng.normal(0, 0.02, N)
            sz = evaluate_fast(s_pert)
            if sz > best_size:
                best_size = sz
                best_scores = s_pert.copy()

    # Simulated Annealing on permutations (structural change from current local search)
    if best_scores is not None and time.time() < DEADLINE - 5:
        # Convert best_scores to a permutation
        order = np.argsort(-best_scores, kind='stable')
        
        def perm_to_scores(perm):
            scores = np.zeros(N)
            for i, v in enumerate(perm):
                scores[v] = N - i
            return scores
        
        current_perm = order.copy()
        current_scores = perm_to_scores(current_perm)
        current_size = evaluate_fast(current_scores)
        
        T = 1.0
        T_min = 0.01
        cooling_rate = 0.999
        
        sa_improved = False
        max_sa_iter = 10000
        
        for it in range(max_sa_iter):
            if time.time() > DEADLINE - 2:
                break
            if T < T_min:
                break
                
            # Swap two random positions
            i, j = rng.integers(0, N, size=2)
            if i == j:
                continue
                
            new_perm = current_perm.copy()
            new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
            new_scores = perm_to_scores(new_perm)
            new_size = evaluate_fast(new_scores)
            
            delta = new_size - current_size
            if delta > 0 or rng.random() < np.exp(delta / T):
                current_perm = new_perm
                current_scores = new_scores
                current_size = new_size
                if current_size > best_size:
                    best_size = current_size
                    best_scores = current_scores.copy()
                    sa_improved = True
            
            T *= cooling_rate
        
        # If SA improved, do a final local polish
        if sa_improved and time.time() < DEADLINE - 1:
            for _ in range(1000):
                if time.time() > DEADLINE - 0.5:
                    break
                s_pert = best_scores + rng.normal(0, 0.01, N)
                sz = evaluate_fast(s_pert)
                if sz > best_size:
                    best_size = sz
                    best_scores = s_pert.copy()

    # Final high-precision polish: very small perturbations with many iterations
    if best_scores is not None and time.time() < DEADLINE - 0.5:
        remaining_time = DEADLINE - 0.5 - time.time()
        # Estimate iterations per second from previous polish
        for _ in range(20000):
            if time.time() > DEADLINE - 0.1:
                break
            s_pert = best_scores + rng.normal(0, 0.005, N)
            sz = evaluate_fast(s_pert)
            if sz > best_size:
                best_size = sz
                best_scores = s_pert.copy()

    if best_scores is None:
        return -D.astype(float)

    return best_scores.astype(float)
