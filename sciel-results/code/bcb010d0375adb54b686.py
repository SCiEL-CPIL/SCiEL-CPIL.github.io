import time
import random
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    N = 256
    squares = set()
    s = 1
    while s * s <= 512:
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
        """Greedy independent set using min-degree heuristic with fast bit counting."""
        rng = random.Random(seed)
        remaining = set(valid_nodes)
        chosen = []
        chosen_mask = 0
        
        while remaining:
            # Find node with minimum degree in remaining using bit_count
            min_deg = float('inf')
            min_node = None
            min_deg_tie = []
            for a in remaining:
                deg = bin(conflict_mask[a] & valid_mask_all).count('1')
                # Subtract already-chosen nodes that conflict (they're no longer in remaining but we want degree in remaining)
                # Actually, we want degree among remaining only
                # conflict_mask[a] has bits for all conflicting valid nodes
                # We need count of (conflict_mask[a] & remaining_mask)
                # Let's compute remaining_mask
                pass
            
            # Recompute: build remaining mask once
            remaining_mask = 0
            for b in remaining:
                remaining_mask |= (1 << b)
            
            for a in remaining:
                deg = bin(conflict_mask[a] & remaining_mask).count('1')
                if deg < min_deg:
                    min_deg = deg
                    min_node = a
                    min_deg_tie = [a]
                elif deg == min_deg:
                    min_deg_tie.append(a)
            
            if min_node is None:
                break
            
            # Randomly pick among ties for diversity
            if len(min_deg_tie) > 1:
                min_node = min_deg_tie[rng.randrange(len(min_deg_tie))]
            
            chosen.append(min_node)
            chosen_mask |= (1 << min_node)
            remaining.remove(min_node)
            # Remove neighbors from remaining
            to_remove = [b for b in remaining if (conflict_mask[min_node] >> b) & 1]
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
        max_iters = 50
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
            
            # Step 2b: Try 2-remove, k-add swaps (more aggressive)
            if not found_swap and time.time() - start_time < deadline - 5:
                in_list2 = list(in_set)
                rng.shuffle(in_list2)
                found_2swap = False
                num_pairs = min(30, len(in_list2) * (len(in_list2) - 1) // 2)
                tried = 0
                while tried < num_pairs:
                    if time.time() - start_time > deadline - 4:
                        break
                    idx1, idx2 = rng.sample(range(len(in_list2)), 2)
                    u1 = in_list2[idx1]
                    u2 = in_list2[idx2]
                    tried += 1
                    
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
            
            if found_swap:
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
        if time.time() - start_time > deadline - 10:
            break
        
        # Cycle through initialization strategies
        strategy = i % 3
        if strategy == 0:
            init_mask, init_list = greedy_min_degree_init(i)
        elif strategy == 1:
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
            # Max-degree greedy (pick node with most conflicts first, removes many neighbors)
            rng = random.Random(i)
            remaining = set(valid_nodes)
            chosen = []
            chosen_mask = 0
            while remaining:
                max_deg = -1
                max_node = None
                max_tie = []
                for a in remaining:
                    deg = bin(conflict_mask[a] & valid_mask_all).count('1')
                    if deg > max_deg:
                        max_deg = deg
                        max_node = a
                        max_tie = [a]
                    elif deg == max_deg:
                        max_tie.append(a)
                if max_node is None:
                    break
                if len(max_tie) > 1:
                    max_node = max_tie[rng.randrange(len(max_tie))]
                chosen.append(max_node)
                chosen_mask |= (1 << max_node)
                remaining.remove(max_node)
                to_remove = [b for b in remaining if (conflict_mask[max_node] >> b) & 1]
                for b in to_remove:
                    remaining.discard(b)
            init_mask, init_list = chosen_mask, chosen
        
        ls_mask, ls_set = local_search(init_mask, init_list, i)
        
        if len(ls_set) > best_size:
            best_size = len(ls_set)
            best_mask = ls_mask
            best_list = list(ls_set)
    
    return best_list
