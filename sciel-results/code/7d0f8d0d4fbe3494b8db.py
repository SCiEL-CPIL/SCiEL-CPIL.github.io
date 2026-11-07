import time
import random
import itertools
from collections import Counter

def construct():
    start_time = time.time()
    deadline = start_time + 82  # safety margin before 90s kill

    n = 14
    m = 35
    max_val = 10000

    all_pairs = list(itertools.combinations(range(n), 2))
    num_pairs = len(all_pairs)  # 91

    def calc_score(vals, edge_indices):
        """edge_indices is a list of indices into all_pairs"""
        sums = set()
        prods = set()
        for ei in edge_indices:
            i, j = all_pairs[ei]
            sums.add(vals[i] + vals[j])
            prods.add(vals[i] * vals[j])
        return max(len(sums), len(prods))

    def edges_from_indices(edge_indices):
        return [[all_pairs[ei][0], all_pairs[ei][1]] for ei in edge_indices]

    def initial_edges(vals, seed=42):
        """Greedy initialization: pick edges that reuse sums/prods most."""
        random.seed(seed)
        
        pair_data = []
        for idx, (i, j) in enumerate(all_pairs):
            s = vals[i] + vals[j]
            p = vals[i] * vals[j]
            pair_data.append((idx, i, j, s, p))
        
        sum_freq = Counter(s for _, _, _, s, _ in pair_data)
        prod_freq = Counter(p for _, _, _, _, p in pair_data)
        
        # Score: prefer high frequency (reuses sum/prod)
        def edge_score(idx, i, j, s, p):
            return -(sum_freq[s] + prod_freq[p])
        
        pair_data.sort(key=lambda x: edge_score(*x))
        
        selected = []
        seen_sums = set()
        seen_prods = set()
        
        for idx, i, j, s, p in pair_data:
            if len(selected) >= m:
                break
            selected.append(idx)
            seen_sums.add(s)
            seen_prods.add(p)
        
        return selected

    def simulated_annealing(vals, edge_indices, max_iters=5000, t0=1.0, t_end=0.01):
        """SA on edge selection: swap one edge at a time."""
        current = list(edge_indices)
        current_score = calc_score(vals, current)
        best = list(current)
        best_score = current_score
        
        selected_set = set(current)
        
        # Precompute for speed
        val_sums = {}
        val_prods = {}
        for idx, (i, j) in enumerate(all_pairs):
            val_sums[idx] = vals[i] + vals[j]
            val_prods[idx] = vals[i] * vals[j]
        
        # Build frequency maps for quick delta computation
        def compute_delta(remove_idx, add_idx):
            """Compute change in score when removing remove_idx and adding add_idx."""
            # This is approximate; full recompute is safer but slower
            # For now, just return 0 and let caller recompute
            return 0
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Temperature
            temp = t0 * (t_end / t0) ** (it / max_iters)
            
            # Pick a random edge to remove
            pos = random.randint(0, m - 1)
            remove_idx = current[pos]
            
            # Pick a random edge to add (not currently selected)
            add_idx = random.randint(0, num_pairs - 1)
            if add_idx in selected_set:
                continue
            
            # Try the swap
            new_current = list(current)
            new_current[pos] = add_idx
            
            new_score = calc_score(vals, new_current)
            
            delta = new_score - current_score
            
            if delta <= 0 or random.random() < pow(2.718281828, -delta / max(temp, 1e-10)):
                # Accept
                current = new_current
                current_score = new_score
                selected_set = set(current)
                
                if current_score < best_score:
                    best_score = current_score
                    best = list(current)
        
        return best, best_score

    def joint_sa(vals, edge_indices, max_iters=3000, t0=10.0, t_end=0.01):
        """SA that can also perturb values (change one value)."""
        current_vals = list(vals)
        current_edges = list(edge_indices)
        current_score = calc_score(current_vals, current_edges)
        best_vals = list(current_vals)
        best_edges = list(current_edges)
        best_score = current_score
        
        selected_set = set(current_edges)
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            temp = t0 * (t_end / t0) ** (it / max_iters)
            
            # Choose move type: 70% edge swap, 30% value perturbation
            if random.random() < 0.7:
                # Edge swap
                pos = random.randint(0, m - 1)
                remove_idx = current_edges[pos]
                add_idx = random.randint(0, num_pairs - 1)
                if add_idx in selected_set:
                    continue
                
                new_edges = list(current_edges)
                new_edges[pos] = add_idx
                
                new_score = calc_score(current_vals, new_edges)
                delta = new_score - current_score
                
                if delta <= 0 or random.random() < pow(2.718281828, -delta / max(temp, 1e-10)):
                    current_edges = new_edges
                    current_score = new_score
                    selected_set = set(current_edges)
                    
                    if current_score < best_score:
                        best_score = current_score
                        best_vals = list(current_vals)
                        best_edges = list(current_edges)
            else:
                # Value perturbation: change one value
                vpos = random.randint(0, n - 1)
                old_val = current_vals[vpos]
                # Random new value
                if random.random() < 0.5:
                    new_val = max(1, min(max_val, old_val + random.randint(-50, 50)))
                else:
                    new_val = random.randint(1, max_val)
                
                if new_val == old_val:
                    continue
                
                new_vals = list(current_vals)
                new_vals[vpos] = new_val
                
                # Check distinctness
                if len(set(new_vals)) != n:
                    continue
                
                new_score = calc_score(new_vals, current_edges)
                delta = new_score - current_score
                
                if delta <= 0 or random.random() < pow(2.718281828, -delta / max(temp, 1e-10)):
                    current_vals = new_vals
                    current_score = new_score
                    
                    if current_score < best_score:
                        best_score = current_score
                        best_vals = list(current_vals)
                        best_edges = list(current_edges)
        
        return best_vals, best_edges, best_score

    best_score = float('inf')
    best_values = None
    best_edges = None

    # Generate diverse initial value sets
    value_sets = []
    
    random.seed(123)
    
    # Arithmetic progressions
    for start in range(1, 30):
        for step in range(1, 15):
            vals = [start + i * step for i in range(n)]
            if max(vals) <= max_val and min(vals) >= 1:
                value_sets.append(vals)
    
    # Geometric progressions
    for start in range(2, 10):
        for ratio in [2, 3]:
            vals = [start * (ratio ** i) for i in range(n)]
            if max(vals) <= max_val:
                value_sets.append(vals)
    
    # Random small sets
    for _ in range(50):
        vals = sorted(random.sample(range(1, 200), n))
        value_sets.append(vals)
    
    # Mixed: some close together, some spread
    for _ in range(30):
        base = random.randint(1, 50)
        vals = []
        for i in range(n):
            if i < 7:
                v = base + random.randint(0, 20)
            else:
                v = base + random.randint(50, 500)
            vals.append(v)
        vals = sorted(set(vals))
        while len(vals) < n:
            vals.append(random.randint(1, max_val))
        vals = sorted(vals[:n])
        if len(set(vals)) == n:
            value_sets.append(vals)
    
    # Process value sets with joint SA
    for vals in value_sets:
        if time.time() > deadline - 10:
            break
        
        # Initial edge selection
        init_edges = initial_edges(vals)
        init_score = calc_score(vals, init_edges)
        
        # Run joint SA
        if time.time() < deadline - 15:
            res_vals, res_edges, res_score = joint_sa(vals, init_edges, max_iters=2000, t0=10.0, t_end=0.01)
            
            if res_score < best_score:
                best_score = res_score
                best_values = res_vals[:]
                best_edges = edges_from_indices(res_edges)
        
        # Also try pure edge SA
        if time.time() < deadline - 10:
            res_edges2, res_score2 = simulated_annealing(vals, init_edges, max_iters=3000, t0=1.0, t_end=0.01)
            
            if res_score2 < best_score:
                best_score = res_score2
                best_values = vals[:]
                best_edges = edges_from_indices(res_edges2)
        
        # Try multiple random initializations for this value set
        for seed in [1, 7, 42, 99]:
            if time.time() > deadline - 10:
                break
            
            init_edges_r = initial_edges(vals, seed=seed)
            
            if time.time() < deadline - 15:
                res_vals_r, res_edges_r, res_score_r = joint_sa(vals, init_edges_r, max_iters=1000, t0=5.0, t_end=0.01)
                
                if res_score_r < best_score:
                    best_score = res_score_r
                    best_values = res_vals_r[:]
                    best_edges = edges_from_indices(res_edges_r)

    # Fallback
    if best_values is None:
        best_values = list(range(1, n + 1))
        init_edges = initial_edges(best_values)
        best_edges = edges_from_indices(init_edges)
        best_score = calc_score(best_values, best_edges)

    # Final type conversion
    best_values = [int(v) for v in best_values]
    best_edges = [[int(i), int(j)] for i, j in best_edges]

    return {'values': best_values, 'edges': best_edges}
