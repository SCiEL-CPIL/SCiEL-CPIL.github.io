import numpy as np
import itertools
import time
import random
from collections import defaultdict

def kissing_number11() -> np.ndarray:
    d = 11
    start_time = time.time()
    deadline = start_time + 600  # 600 second deadline with safety margin
    
    # Generate all 4-combinations of indices
    all_combos = list(itertools.combinations(range(d), 4))
    n_combos = len(all_combos)
    
    # Build conflict graph: two combos conflict if they share 3 or more indices
    # For efficiency, precompute conflicts using bitmasks
    combo_to_mask = {}
    for i, c in enumerate(all_combos):
        mask = 0
        for idx in c:
            mask |= (1 << idx)
        combo_to_mask[i] = mask
    
    # Build adjacency: which combos conflict with which
    # Two combos conflict if their intersection has size >= 3
    # Since each combo has size 4, intersection >= 3 means they share 3 or 4 indices
    # We'll use a different approach: for each combo, find all combos that share >= 3 indices
    
    # Precompute: for each triple of indices, which 4-combinations contain that triple
    # If two 4-combinations share a triple, they conflict
    triple_to_combos = defaultdict(list)
    for i, c in enumerate(all_combos):
        # Get all 3-subsets of this 4-combination
        for j in range(4):
            for k in range(j+1, 4):
                for l in range(k+1, 4):
                    triple = (c[j], c[k], c[l])
                    triple_to_combos[triple].append(i)
    
    # Build conflict list for each combo
    conflicts = [set() for _ in range(n_combos)]
    for triple, combo_list in triple_to_combos.items():
        for i in range(len(combo_list)):
            for j in range(i+1, len(combo_list)):
                ci = combo_list[i]
                cj = combo_list[j]
                conflicts[ci].add(cj)
                conflicts[cj].add(ci)
    
    # Now do greedy independent set selection with multiple strategies
    best_selected = []
    best_size = 0
    
    # Strategy 1: Original greedy with random ordering
    for seed in range(20):
        if time.time() > deadline:
            break
        rng = np.random.default_rng(seed)
        order = list(range(n_combos))
        rng.shuffle(order)
        
        selected = []
        selected_set = set()
        
        for i in order:
            if i in selected_set:
                continue
            # Check if i conflicts with any selected
            conflict_found = False
            for s in selected:
                if s in conflicts[i]:
                    conflict_found = True
                    break
            if not conflict_found:
                selected.append(i)
                selected_set.add(i)
        
        if len(selected) > best_size:
            best_size = len(selected)
            best_selected = selected[:]
    
    # Strategy 2: Greedy by minimum degree (pick combo with fewest conflicts first)
    for seed in range(20):
        if time.time() > deadline:
            break
        rng = np.random.default_rng(1000 + seed)
        
        remaining = set(range(n_combos))
        selected = []
        
        while remaining and time.time() < deadline:
            # Find combo with minimum number of conflicts among remaining
            min_conflicts = float('inf')
            best_combo = None
            for i in remaining:
                conf_count = len(conflicts[i] & remaining)
                if conf_count < min_conflicts:
                    min_conflicts = conf_count
                    best_combo = i
            
            if best_combo is None:
                break
            
            selected.append(best_combo)
            remaining.remove(best_combo)
            # Remove all conflicts of best_combo from remaining
            for c in conflicts[best_combo]:
                remaining.discard(c)
        
        if len(selected) > best_size:
            best_size = len(selected)
            best_selected = selected[:]
    
    # Strategy 3: Local search - start from best found, try to improve
    if best_selected and time.time() < deadline - 60:
        current = list(best_selected)
        current_set = set(current)
        current_size = len(current)
        
        # Try to add more combos
        for i in range(n_combos):
            if time.time() > deadline - 30:
                break
            if i in current_set:
                continue
            # Check if i can be added
            can_add = True
            for s in current:
                if s in conflicts[i]:
                    can_add = False
                    break
            if can_add:
                current.append(i)
                current_set.add(i)
                current_size += 1
        
        if current_size > best_size:
            best_size = current_size
            best_selected = current[:]
    
    # Build points array
    points = []
    
    # Add Type A points: 2*e_i and -2*e_i
    for i in range(d):
        p1 = np.zeros(d, dtype=np.int64)
        p1[i] = 2
        points.append(p1)
        
        p2 = np.zeros(d, dtype=np.int64)
        p2[i] = -2
        points.append(p2)
        
    # Add Type B points for selected combinations
    for idx in best_selected:
        c = all_combos[idx]
        indices = list(c)
        # Generate all 16 sign patterns
        for mask in range(16):
            p = np.zeros(d, dtype=np.int64)
            for k, idx2 in enumerate(indices):
                # Bit k of mask determines sign
                if (mask >> k) & 1:
                    p[idx2] = -1
                else:
                    p[idx2] = 1
            points.append(p)
            
    points_array = np.array(points, dtype=np.int64)
    
    return points_array
