import time
import random
import math
import numpy as np
from collections import deque

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
    
    # Map valid nodes to compact indices 0..n-1 for B&B
    node_to_idx = {}
    idx_to_node = []
    for idx, node in enumerate(valid_nodes):
        node_to_idx[node] = idx
        idx_to_node.append(node)
    n_valid = len(valid_nodes)
    
    # Build compact adjacency bitmask (index-based)
    adj = [0] * n_valid
    for i, node_i in enumerate(valid_nodes):
        mask = 0
        for j, node_j in enumerate(valid_nodes):
            if i != j and (conflict_mask[node_i] >> node_j) & 1:
                mask |= (1 << j)
        adj[i] = mask
        
    def greedy_independent_set(remaining_mask):
        """Greedy independent set on remaining vertices (index-based mask)."""
        chosen_mask = 0
        remaining = remaining_mask
        while remaining:
            # Pick vertex with minimum degree in remaining
            min_deg = n_valid + 1
            min_idx = -1
            for i in range(n_valid):
                if (remaining >> i) & 1:
                    deg = bin(adj[i] & remaining).count('1')
                    if deg < min_deg:
                        min_deg = deg
                        min_idx = i
            if min_idx < 0:
                break
            chosen_mask |= (1 << min_idx)
            # Remove chosen and its neighbors
            remaining &= ~(1 << min_idx)
            remaining &= ~adj[min_idx]
        return chosen_mask
    
    def branch_and_bound(remaining_mask, current_size, best_size_ref):
        """Branch and bound for maximum independent set.
        Returns (best_size, best_mask) for this subtree."""
        if time.time() > deadline - 1:
            return current_size, 0
        
        # Upper bound: current_size + greedy upper bound on remaining
        if remaining_mask == 0:
            return current_size, 0
        
        # Quick upper bound: count bits in remaining
        ub = current_size + bin(remaining_mask).count('1')
        if ub <= best_size_ref[0]:
            return best_size_ref[0], best_size_ref[1]
        
        # Greedy upper bound (tighter)
        greedy_bound = bin(greedy_independent_set(remaining_mask)).count('1')
        if current_size + greedy_bound <= best_size_ref[0]:
            return best_size_ref[0], best_size_ref[1]
        
        # Pick a vertex to branch on (highest degree for better pruning)
        max_deg = -1
        branch_idx = -1
        for i in range(n_valid):
            if (remaining_mask >> i) & 1:
                deg = bin(adj[i] & remaining_mask).count('1')
                if deg > max_deg:
                    max_deg = deg
                    branch_idx = i
        
        if branch_idx < 0:
            return current_size, 0
        
        # Branch 1: Include branch_idx
        new_remaining = remaining_mask & ~(1 << branch_idx) & ~adj[branch_idx]
        size1, mask1 = branch_and_bound(new_remaining, current_size + 1, best_size_ref)
        
        # Branch 2: Exclude branch_idx
        new_remaining2 = remaining_mask & ~(1 << branch_idx)
        size2, mask2 = branch_and_bound(new_remaining2, current_size, best_size_ref)
        
        if size1 >= size2:
            return size1, mask1
        else:
            return size2, mask2
    
    def greedy_min_degree_init(seed):
        """Greedy independent set using min-degree heuristic with fast bit counting."""
        rng = random.Random(seed)
        remaining = set(valid_nodes)
        chosen = []
        chosen_mask = 0
        
        while remaining:
            remaining_mask = 0
            for b in remaining:
                remaining_mask |= (1 << b)
            
            min_deg = float('inf')
            min_node = None
            min_deg_tie = []
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
            
            if len(min_deg_tie) > 1:
                min_node = min_deg_tie[rng.randrange(len(min_deg_tie))]
            
            chosen.append(min_node)
            chosen_mask |= (1 << min_node)
            remaining.remove(min_node)
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
            
            if try_add_all():
                improved = True
                continue
            
            in_list = list(in_set)
            rng.shuffle(in_list)
            found_swap = False
            for u in in_list:
                if time.time() - start_time > deadline - 3:
                    break
                mask_without_u = in_mask ^ (1 << u)
                potential_adds = [v for v in out_set if (conflict_mask[v] & mask_without_u) == 0]
                
                if len(potential_adds) > 1:
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
    
    # Phase 1: Get a good initial solution via local search
    init_mask, init_list = greedy_min_degree_init(0)
    ls_mask, ls_set = local_search(init_mask, init_list, 0)
    best_size = len(ls_set)
    best_mask = ls_mask
    best_list = list(ls_set)
    
    # Phase 2: Try B&B with time limit
    if time.time() < deadline - 15:
        # Convert best_list to compact index mask
        best_idx_mask = 0
        for node in best_list:
            if node in node_to_idx:
                best_idx_mask |= (1 << node_to_idx[node])
        
        full_remaining = (1 << n_valid) - 1
        best_size_ref = [best_size, best_idx_mask]
        
        bb_size, bb_mask = branch_and_bound(full_remaining, 0, best_size_ref)
        
        if bb_size > best_size:
            best_size = bb_size
            # Convert back to node values
            bb_list = [idx_to_node[i] for i in range(n_valid) if (bb_mask >> i) & 1]
            best_list = bb_list
            best_mask = 0
            for node in best_list:
                best_mask |= (1 << node)
    
    # Phase 3: Run more local search restarts with remaining time
    num_restarts = 40
    for i in range(num_restarts):
        if time.time() - start_time > deadline - 5:
            break
        
        strategy = i % 3
        if strategy == 0:
            init_mask, init_list = greedy_min_degree_init(i + 100)
        elif strategy == 1:
            rng = random.Random(i + 200)
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
            rng = random.Random(i + 300)
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
        
        ls_mask, ls_set = local_search(init_mask, init_list, i + 400)
        
        if len(ls_set) > best_size:
            best_size = len(ls_set)
            best_mask = ls_mask
            best_list = list(ls_set)
    
    return best_list
