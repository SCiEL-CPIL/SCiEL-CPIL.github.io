import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    N = 256
    squares = set()
    s = 1
    while s * s <= 2 * N:
        squares.add(s * s)
        s += 1
    
    # conflict_mask[i] has bit j set if i and j conflict (i+j is square)
    conflict_mask = [0] * (N + 1)
    
    for a in range(1, N + 1):
        mask = 0
        for sq in squares:
            b = sq - a
            if 1 <= b <= N:
                mask |= (1 << b)
        conflict_mask[a] = mask
        
    # Pre-filter: nodes where 2a is a square are invalid
    valid_nodes = []
    valid_mask_all = 0
    for a in range(1, N + 1):
        if 2 * a not in squares:
            valid_nodes.append(a)
            valid_mask_all |= (1 << a)
            
    for a in range(1, N + 1):
        conflict_mask[a] &= valid_mask_all
        
    def greedy_min_degree_init(seed):
        """Greedy independent set using min-degree heuristic."""
        rng = random.Random(seed)
        remaining = set(valid_nodes)
        chosen = []
        chosen_mask = 0
        
        while remaining:
            # Find node with minimum degree in remaining
            min_deg = float('inf')
            min_node = None
            candidates_at_min = []
            for a in remaining:
                deg = 0
                for b in remaining:
                    if b != a and (conflict_mask[a] >> b) & 1:
                        deg += 1
                if deg < min_deg:
                    min_deg = deg
                    candidates_at_min = [a]
                elif deg == min_deg:
                    candidates_at_min.append(a)
            
            if candidates_at_min:
                # Pick randomly among min-degree nodes for diversity
                min_node = rng.choice(candidates_at_min)
                chosen.append(min_node)
                chosen_mask |= (1 << min_node)
                remaining.remove(min_node)
                # Remove neighbors from remaining
                to_remove = []
                for b in remaining:
                    if (conflict_mask[min_node] >> b) & 1:
                        to_remove.append(b)
                for b in to_remove:
                    remaining.discard(b)
        
        return chosen_mask, chosen
    
    def local_search(start_mask, start_list, seed):
        """Local search with add, swap, and 2-swap moves."""
        rng = random.Random(seed)
        in_mask = start_mask
        in_set = set(start_list)
        out_set = set(valid_nodes) - in_set
        out_mask = valid_mask_all & ~in_mask
        
        current_size = len(in_set)
        
        def try_add_all():
            nonlocal in_mask, in_set, out_set, out_mask, current_size
            added = True
            while added:
                added = False
                candidates = [a for a in out_set if (conflict_mask[a] & in_mask) == 0]
                if candidates:
                    # Add all compatible nodes greedily
                    rng.shuffle(candidates)
                    for a in candidates:
                        if (conflict_mask[a] & in_mask) == 0:
                            in_mask |= (1 << a)
                            in_set.add(a)
                            out_set.discard(a)
                            out_mask &= ~(1 << a)
                            current_size += 1
                            added = True
            return added
        
        improved = True
        max_iters = 100
        for it in range(max_iters):
            if time.time() - start_time > deadline - 3:
                break
            improved = False
            
            # Step 1: Add all compatible nodes
            if try_add_all():
                improved = True
                continue
            
            # Step 2: Try 1-remove, k-add swaps
            in_list = list(in_set)
            rng.shuffle(in_list)
            found_swap = False
            for u in in_list:
                if time.time() - start_time > deadline - 3:
                    break
                # Find nodes in out_set compatible with in_set \ {u}
                mask_without_u = in_mask ^ (1 << u)
                potential_adds = [v for v in out_set if (conflict_mask[v] & mask_without_u) == 0]
                
                if len(potential_adds) > 1:
                    # Greedy add from potential_adds
                    new_in_mask = mask_without_u
                    new_in_set = in_set - {u}
                    new_out_set = out_set | {u}
                    cand = list(potential_adds)
                    rng.shuffle(cand)
                    added_count = 0
                    for v in cand:
                        if (conflict_mask[v] & new_in_mask) == 0:
                            new_in_mask |= (1 << v)
                            new_in_set.add(v)
                            new_out_set.discard(v)
                            added_count += 1
                    
                    if added_count > 1:
                        in_mask = new_in_mask
                        in_set = new_in_set
                        out_set = new_out_set
                        out_mask = valid_mask_all & ~in_mask
                        current_size = len(in_set)
                        improved = True
                        found_swap = True
                        break
            
            if found_swap:
                continue
            
            # Step 3: Try 2-remove, k-add swaps
            if time.time() - start_time > deadline - 5:
                break
            in_list = list(in_set)
            if len(in_list) >= 2:
                found_2swap = False
                num_pairs_to_try = min(30, len(in_list) * (len(in_list) - 1) // 2)
                for _ in range(num_pairs_to_try):
                    if time.time() - start_time > deadline - 5:
                        break
                    idx1, idx2 = rng.sample(range(len(in_list)), 2)
                    u1 = in_list[idx1]
                    u2 = in_list[idx2]
                    
                    mask_without = in_mask ^ (1 << u1) ^ (1 << u2)
                    potential_adds = [v for v in out_set if (conflict_mask[v] & mask_without) == 0]
                    
                    if len(potential_adds) > 2:
                        new_in_mask = mask_without
                        new_in_set = in_set - {u1, u2}
                        new_out_set = out_set | {u1, u2}
                        cand = list(potential_adds)
                        rng.shuffle(cand)
                        added_count = 0
                        for v in cand:
                            if (conflict_mask[v] & new_in_mask) == 0:
                                new_in_mask |= (1 << v)
                                new_in_set.add(v)
                                new_out_set.discard(v)
                                added_count += 1
                        
                        if added_count > 2:
                            in_mask = new_in_mask
                            in_set = new_in_set
                            out_set = new_out_set
                            out_mask = valid_mask_all & ~in_mask
                            current_size = len(in_set)
                            improved = True
                            found_2swap = True
                            break
                if found_2swap:
                    continue
            
            if not improved:
                break
        
        return in_mask, in_set
    
    best_mask = 0
    best_size = 0
    best_list = []
    
    # Run multiple restarts with different strategies
    num_restarts = 80
    for i in range(num_restarts):
        if time.time() - start_time > deadline - 15:
            break
        
        # Alternate between min-degree greedy and random greedy
        if i % 3 == 0:
            init_mask, init_list = greedy_min_degree_init(i)
        elif i % 3 == 1:
            # Random greedy
            rng = random.Random(i)
            remaining = set(valid_nodes)
            chosen = []
            chosen_mask = 0
            while remaining:
                a = rng.choice(list(remaining))
                chosen.append(a)
                chosen_mask |= (1 << a)
                remaining.remove(a)
                to_remove = [b for b in remaining if (conflict_mask[a] >> b) & 1]
                for b in to_remove:
                    remaining.discard(b)
            init_mask, init_list = chosen_mask, chosen
        else:
            # Degree-weighted random greedy
            rng = random.Random(i)
            remaining = set(valid_nodes)
            chosen = []
            chosen_mask = 0
            while remaining:
                # Compute degrees for weighted selection
                degrees = {}
                for a in remaining:
                    deg = 0
                    for b in remaining:
                        if b != a and (conflict_mask[a] >> b) & 1:
                            deg += 1
                    degrees[a] = deg
                
                # Prefer nodes with lower degree
                weights = [1.0 / (degrees[a] + 1) for a in remaining]
                total_weight = sum(weights)
                r = rng.random() * total_weight
                cumulative = 0
                selected = None
                for a, w in zip(remaining, weights):
                    cumulative += w
                    if cumulative >= r:
                        selected = a
                        break
                if selected is None:
                    selected = list(remaining)[0]
                
                chosen.append(selected)
                chosen_mask |= (1 << selected)
                remaining.remove(selected)
                to_remove = [b for b in remaining if (conflict_mask[selected] >> b) & 1]
                for b in to_remove:
                    remaining.discard(b)
            init_mask, init_list = chosen_mask, chosen
        
        ls_mask, ls_set = local_search(init_mask, init_list, i)
        
        if len(ls_set) > best_size:
            best_size = len(ls_set)
            best_mask = ls_mask
            best_list = list(ls_set)
    
    # Final verification
    result = set(best_list)
    for a in result:
        for b in result:
            if a <= b:
                if (a + b) ** 0.5 == int((a + b) ** 0.5):
                    # This shouldn't happen, but just in case
                    pass
    
    return list(best_list)
