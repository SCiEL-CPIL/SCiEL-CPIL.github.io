import random
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s margin
    
    points = list(range(16))
    all_triples = list(itertools.combinations(points, 3))
    n_triples = len(all_triples)
    
    # Precompute masks for all triples
    triple_masks = []
    for t in all_triples:
        mask = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
        triple_masks.append(mask)
    
    # Precompute popcount for all 16-bit masks
    popcount = [0] * (1 << 16)
    for i in range(1, 1 << 16):
        popcount[i] = popcount[i >> 1] + (i & 1)
    
    # Build mask to triple mapping
    mask_to_triple = {}
    for t, m in zip(all_triples, triple_masks):
        mask_to_triple[m] = list(t)
    
    def get_alpha_fast(edge_masks):
        """Find independence number by checking subsets from largest to smallest."""
        if not edge_masks:
            return 16
        
        # Check if all points are covered
        all_points = (1 << 16) - 1
        
        # For each size from 16 down to 1, check if there's an independent set
        for size in range(16, 0, -1):
            # Generate all subsets of this size and check
            # Use bit manipulation to generate combinations
            found = False
            # Iterate through all masks with popcount == size
            for m in range(1 << 16):
                if popcount[m] != size:
                    continue
                is_indep = True
                for e in edge_masks:
                    if (m & e) == e:
                        is_indep = False
                        break
                if is_indep:
                    return size
            # If we get here, no independent set of this size exists
            # Continue to smaller sizes
        return 0
    
    def get_alpha_with_bitset(edge_masks):
        """Faster independence check using precomputed coverage."""
        if not edge_masks:
            return 16
        
        n_edges = len(edge_masks)
        
        # For each point, track which edges contain it
        point_edges = [set() for _ in range(16)]
        for i, e in enumerate(edge_masks):
            for p in range(16):
                if e & (1 << p):
                    point_edges[p].add(i)
        
        # Check sizes from 16 down
        for size in range(16, 0, -1):
            # Try all subsets of this size
            # Use recursive generation with pruning
            def check_subset(subset, start, remaining):
                if remaining == 0:
                    # Check if this subset is independent
                    for e in edge_masks:
                        if (e & subset) == e:
                            return False
                    return True
                for p in range(start, 16 - remaining + 1):
                    if subset & (1 << p):
                        continue
                    # Prune: if adding p creates a complete edge
                    new_subset = subset | (1 << p)
                    for e in edge_masks:
                        if (e & new_subset) == e:
                            break
                    else:
                        if check_subset(new_subset, p + 1, remaining - 1):
                            return True
                return False
            
            if check_subset(0, 0, size):
                return size
        return 0
    
    def is_linear(new_mask, edge_masks):
        for e in edge_masks:
            inter = new_mask & e
            if inter and (inter & (inter - 1)):
                return False
        return True
    
    # Greedy initialization with multiple strategies
    def greedy_init(seed):
        rng = random.Random(seed)
        edges = []
        shuffled_indices = list(range(n_triples))
        rng.shuffle(shuffled_indices)
        for idx in shuffled_indices:
            m = triple_masks[idx]
            if is_linear(m, edges):
                edges.append(m)
        return edges
    
    def greedy_init_by_degree(seed):
        """Greedy that prioritizes triples covering uncovered points."""
        rng = random.Random(seed)
        edges = []
        covered = 0  # bitmask of covered points
        
        # Sort triples by how many new points they cover (recompute each iteration)
        available = list(range(n_triples))
        rng.shuffle(available)
        
        for _ in range(len(available)):
            best_idx = -1
            best_new = -1
            for idx in available:
                m = triple_masks[idx]
                new_points = m & ~covered
                new_count = popcount[new_points]
                if new_count > best_new:
                    best_new = new_count
                    best_idx = idx
            
            if best_idx == -1:
                break
            
            m = triple_masks[best_idx]
            if is_linear(m, edges):
                edges.append(m)
                covered |= m
                available.remove(best_idx)
        
        return edges
    
    # Local search
    def local_search(edges, time_limit, seed):
        rng = random.Random(seed)
        current_edges = edges[:]
        current_alpha = get_alpha_fast(current_edges)
        best_edges = current_edges[:]
        best_alpha = current_alpha
        
        T = 1.0
        T_min = 0.01
        local_start = time.time()
        
        while time.time() < local_start + time_limit:
            # Generate neighbor
            r = rng.random()
            new_edges = current_edges[:]
            
            if r < 0.6:
                # Swap: remove one, add one
                if current_edges:
                    idx_rm = rng.randint(0, len(current_edges) - 1)
                    removed = current_edges[idx_rm]
                    new_edges.pop(idx_rm)
                    
                    # Try to find a good replacement
                    added = False
                    for _ in range(100):
                        cand_idx = rng.randint(0, n_triples - 1)
                        cand_mask = triple_masks[cand_idx]
                        if cand_mask != removed and is_linear(cand_mask, new_edges):
                            new_edges.append(cand_mask)
                            added = True
                            break
                    if not added:
                        new_edges.insert(idx_rm, removed)
            elif r < 0.8:
                # Add
                added = False
                for _ in range(30):
                    cand_idx = rng.randint(0, n_triples - 1)
                    cand_mask = triple_masks[cand_idx]
                    if is_linear(cand_mask, new_edges):
                        new_edges.append(cand_mask)
                        added = True
                        break
                if not added:
                    continue
            else:
                # Remove
                if current_edges:
                    idx_rm = rng.randint(0, len(current_edges) - 1)
                    new_edges.pop(idx_rm)
                else:
                    continue
            
            new_alpha = get_alpha_fast(new_edges)
            
            if new_alpha < current_alpha:
                current_edges = new_edges
                current_alpha = new_alpha
                if current_alpha < best_alpha:
                    best_alpha = current_alpha
                    best_edges = current_edges[:]
                    if best_alpha <= 5:
                        break
            else:
                # Accept with probability based on temperature
                delta = new_alpha - current_alpha
                if delta > 0:
                    accept_prob = 1.0 / (1.0 + delta * delta)
                    if rng.random() < accept_prob * T:
                        current_edges = new_edges
                        current_alpha = new_alpha
                T *= 0.9995
                if T < T_min:
                    T = T_min
        
        return best_edges, best_alpha
    
    # Try multiple strategies and restarts
    best_alpha = 16
    best_edges = []
    
    # Initial greedy attempts with different seeds
    for seed in range(10):
        if time.time() > deadline - 10:
            break
        edges = greedy_init(seed)
        alpha = get_alpha_fast(edges)
        if alpha < best_alpha:
            best_alpha = alpha
            best_edges = edges[:]
    
    # Try degree-based greedy
    for seed in range(5):
        if time.time() > deadline - 10:
            break
        edges = greedy_init_by_degree(seed)
        alpha = get_alpha_fast(edges)
        if alpha < best_alpha:
            best_alpha = alpha
            best_edges = edges[:]
    
    if best_alpha <= 5:
        return [mask_to_triple[m] for m in best_edges]
    
    # Local search from best solution
    remaining_time = deadline - time.time()
    if remaining_time > 5:
        # Do local search with time budget
        search_time = min(remaining_time - 5, 120)
        best_edges, best_alpha = local_search(best_edges, search_time, seed=42)
    
    if best_alpha <= 5:
        return [mask_to_triple[m] for m in best_edges]
    
    # If still not good, try more restarts
    if time.time() < deadline - 10:
        for attempt in range(20):
            if time.time() > deadline - 10:
                break
            seed = 100 + attempt
            edges = greedy_init(seed)
            alpha = get_alpha_fast(edges)
            if alpha < best_alpha:
                best_alpha = alpha
                best_edges = edges[:]
                if best_alpha <= 5:
                    break
            
            # Quick local search
            if time.time() < deadline - 10:
                quick_time = min(5, deadline - time.time() - 10)
                if quick_time > 1:
                    edges, alpha = local_search(edges, quick_time, seed=seed)
                    if alpha < best_alpha:
                        best_alpha = alpha
                        best_edges = edges[:]
                        if best_alpha <= 5:
                            break
    
    return [mask_to_triple[m] for m in best_edges]
