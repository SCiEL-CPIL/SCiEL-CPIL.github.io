import numpy as np
import itertools
import time
from collections import defaultdict

def kissing_number11() -> np.ndarray:
    d = 11
    start_time = time.time()
    deadline = start_time + 580  # safety margin before 630s limit
    
    # Base points: 2*e_i and -2*e_i
    base_points = []
    for i in range(d):
        p1 = np.zeros(d, dtype=np.int64)
        p1[i] = 2
        base_points.append(p1)
        
        p2 = np.zeros(d, dtype=np.int64)
        p2[i] = -2
        base_points.append(p2)
    
    # Generate all 4-combinations
    combos = list(itertools.combinations(range(d), 4))
    n_combos = len(combos)
    
    # Precompute intersection sizes
    combo_masks = []
    for c in combos:
        mask = 0
        for idx in c:
            mask |= (1 << idx)
        combo_masks.append(mask)
    
    inter_sizes = np.zeros((n_combos, n_combos), dtype=np.int8)
    for i in range(n_combos):
        for j in range(i + 1, n_combos):
            inter = combo_masks[i] & combo_masks[j]
            sz = bin(inter).count('1')
            inter_sizes[i, j] = sz
            inter_sizes[j, i] = sz
    
    # Conflict: intersection > 2
    conflict = inter_sizes > 2
    
    # Greedy with multiple strategies
    best_selected = None
    best_count = 0
    
    # Strategy 1: Greedy by degree (ascending)
    degrees = np.sum(conflict, axis=1).astype(int)
    
    # Precompute conflict matrix as list of lists for faster access in Python loops
    # conflict is a boolean numpy array. We will use it directly but maybe convert to list of sets for neighbor lookup?
    # For now, keep numpy array.
    
    def greedy_select(order):
        selected = []
        forbidden = np.zeros(n_combos, dtype=bool)
        for ci in order:
            if not forbidden[ci]:
                selected.append(ci)
                forbidden[conflict[ci]] = True
        return selected

    def local_search_1swap(selected, deadline):
        # Best improvement 1-swap
        improved = True
        while improved and time.time() < deadline:
            improved = False
            best_gain = 0
            best_move = None
            current_len = len(selected)
            selected_set = set(selected)
            
            # Try all removals
            for si in range(current_len):
                removed = selected[si]
                # Candidates to add: all nodes not in selected_set
                # We want to find a node cj that is compatible with all in selected \ {removed}
                # and cj != removed
                # Actually, cj can be any node not in selected.
                # If we add cj, size remains same.
                # So gain is 0?
                # Wait, 1-swap maintains size. It helps to reach a better local optimum for 2-swap or just to escape?
                # No, 1-swap alone doesn't increase size.
                # To increase size, we need to add a node without removing one.
                # If we can add a node, we should do it.
                
                # First, try to ADD a node (if possible)
                # This is a "growth" step.
                # If we can add a node, we do it and restart the loop (or continue).
                # If we can't add, we try swaps to enable future additions.
                
                pass # We will handle growth separately or integrate it.
            
            # Let's do a simpler loop:
            # 1. Try to add any node. If found, add it and continue.
            # 2. If no node can be added, try 1-swap to change the set (to potentially allow a different node to be added later).
            # This is standard for Max Independent Set.
            
            # Step 1: Growth
            added = False
            for cj in range(n_combos):
                if cj in selected_set:
                    continue
                # Check if cj is compatible with all in selected
                ok = True
                for sj in selected:
                    if conflict[cj, sj]:
                        ok = False
                        break
                if ok:
                    selected.append(cj)
                    selected_set.add(cj)
                    added = True
                    break
            
            if added:
                continue # Restart growth/swap loop
            
            # Step 2: Swap (if no growth)
            # Try all pairs (remove i, add j)
            # We want to find a swap that doesn't decrease size (it doesn't) but might lead to growth.
            # Since size is constant, we just pick the first valid swap or a random one?
            # Best improvement: pick the swap that maximizes "potential"? Hard to define.
            # Let's just do First Improvement Swap.
            
            for si in range(len(selected)):
                if time.time() > deadline:
                    return selected
                removed = selected[si]
                remaining_set = selected_set - {removed}
                
                for cj in range(n_combos):
                    if cj in remaining_set:
                        continue
                    # Check if cj is compatible with all in remaining_set
                    ok = True
                    for sj in remaining_set:
                        if conflict[cj, sj]:
                            ok = False
                            break
                    if ok:
                        # Perform swap
                        selected[si] = cj
                        selected_set.discard(removed)
                        selected_set.add(cj)
                        improved = True
                        break
                if improved:
                    break
        
        return selected

    # Run ILS
    rng = np.random.default_rng(42)
    
    # Initial Greedy
    orders = [np.argsort(degrees), np.argsort(-degrees)]
    for seed in range(50):
        if time.time() > deadline - 60:
            break
        orders.append(rng.permutation(n_combos))
    
    for order in orders:
        if time.time() > deadline - 60:
            break
        
        selected = greedy_select(order)
        selected = local_search_1swap(selected, deadline - 30)
        
        if len(selected) > best_count:
            best_count = len(selected)
            best_selected = selected[:]
            
        # If we have time, do more ILS iterations on the best found
        if best_selected is not None and time.time() < deadline - 30:
            # Perturb best_selected
            current_best = best_selected[:]
            for _ in range(20): # 20 perturbations
                if time.time() > deadline - 30:
                    break
                # Remove 5 random nodes
                if len(current_best) > 10:
                    remove_idx = rng.choice(len(current_best), size=5, replace=False)
                    for idx in sorted(remove_idx, reverse=True):
                        current_best.pop(idx)
                
                # Local search
                current_best = local_search_1swap(current_best, deadline - 30)
                
                if len(current_best) > best_count:
                    best_count = len(current_best)
                    best_selected = current_best[:]
    
    # Verify and count
    selected_set = set(best_selected) if best_selected else set()
    
    # Build points array
    points = list(base_points)
    
    if best_selected:
        for ci in best_selected:
            c = combos[ci]
            indices = list(c)
            for mask in range(16):
                p = np.zeros(d, dtype=np.int64)
                for k, idx in enumerate(indices):
                    if (mask >> k) & 1:
                        p[idx] = -1
                    else:
                        p[idx] = 1
                points.append(p)
    
    points_array = np.array(points, dtype=np.int64)
    
    return points_array
