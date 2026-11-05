import time
import random
from itertools import combinations
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill
    
    points = list(range(16))
    all_triples = list(combinations(points, 3))
    n_triples = len(all_triples)
    
    # Precompute bitmask for each triple
    triple_masks = [0] * n_triples
    for i, t in enumerate(all_triples):
        triple_masks[i] = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
    
    # Precompute conflict pairs: two triples conflict if they share 2+ points
    # For MILP, we need to build constraints. But first, let's use a faster approach.
    
    # Strategy: Use MILP to maximize covered pairs subject to linearity constraints
    # Universe: all C(16,2) = 120 pairs
    # Each triple covers exactly 3 pairs
    # Constraint: no two selected triples share an edge (pair)
    
    # Build pair-to-triple mapping
    pair_to_triples = {}
    for i, t in enumerate(all_triples):
        for j in range(3):
            for k in range(j+1, 3):
                pair = (t[j], t[k])
                if pair not in pair_to_triples:
                    pair_to_triples[pair] = []
                pair_to_triples[pair].append(i)
    
    n_pairs = len(pair_to_triples)
    pairs_list = list(pair_to_triples.keys())
    
    # MILP formulation:
    # Variables: x_i in {0,1} for each triple i
    # Maximize: sum over pairs of (1 - y_p) where y_p = 1 if pair p is covered
    # Actually, maximize number of covered pairs = sum_p z_p where z_p = 1 if any triple covering p is selected
    # Constraint: for each pair p, at most one triple covering p is selected (linearity)
    # i.e., for each pair p, sum_{i in pair_to_triples[p]} x_i <= 1
    
    # Let's reformulate:
    # Maximize: sum_p z_p
    # Subject to:
    #   z_p <= sum_{i in pair_to_triples[p]} x_i  for each pair p
    #   sum_{i in pair_to_triples[p]} x_i <= 1    for each pair p (linearity)
    #   x_i in {0,1}, z_p in {0,1}
    
    # This is a set packing problem with an additional objective.
    # Actually, the linearity constraint is exactly that no two triples share a pair.
    # So we want to maximize the number of distinct pairs covered by selected triples,
    # subject to the selected triples being pairwise disjoint in pairs.
    
    # Since each triple covers 3 pairs, and no two triples can share a pair,
    # the number of covered pairs = 3 * (number of selected triples).
    # So maximizing covered pairs = maximizing number of triples = finding a maximum packing.
    
    # But we want to MINIMIZE the independence number, not maximize the number of triples.
    # A larger packing may or may not have a smaller independence number.
    
    # Let me think differently. The key insight from the knowledge:
    # "Switching to a local search that specifically targets the boundary between alpha=8 and alpha=7"
    
    # Let me implement a targeted approach:
    # 1. Build a good initial system using greedy + local search
    # 2. Use a fast check for whether alpha <= 7 (i.e., no independent set of size 8)
    # 3. If alpha <= 7, we're done with a good score
    # 4. Otherwise, try to reduce alpha further
    
    # Fast check: does there exist an independent set of size 8?
    # An independent set of size 8 is a subset of 8 points containing no complete triple.
    # We can check this by: for each subset of 8 points, check if any triple is fully contained.
    
    # Precompute all 8-subsets for fast checking
    all_8_subsets = list(combinations(points, 8))
    n_8_subsets = len(all_8_subsets)
    
    # Precompute mask for each 8-subset
    subset8_masks = [0] * n_8_subsets
    for i, s in enumerate(all_8_subsets):
        m = 0
        for p in s:
            m |= (1 << p)
        subset8_masks[i] = m
    
    def has_independent_set_of_size_8(triple_set):
        """Check if there's an independent set of size 8 (no triple fully contained)."""
        ts = set(triple_set)
        for i in range(n_8_subsets):
            sm = subset8_masks[i]
            # Check if any triple is fully contained in this 8-set
            found = False
            for ti in ts:
                if (triple_masks[ti] & sm) == triple_masks[ti]:
                    found = True
                    break
            if not found:
                return True
        return False
    
    def compute_alpha(triple_set):
        """Compute exact independence number."""
        ts = set(triple_set)
        for k in range(16, 0, -1):
            for subset in combinations(points, k):
                sm = 0
                for p in subset:
                    sm |= (1 << p)
                found = False
                for ti in ts:
                    if (triple_masks[ti] & sm) == triple_masks[ti]:
                        found = True
                        break
                if not found:
                    return k
        return 0
    
    # Greedy construction with balance heuristic
    def greedy_construct(seed=None):
        if seed is not None:
            random.seed(seed)
        selected = []
        selected_set = set()
        point_usage = [0] * 16
        
        # Build conflict check: two triples conflict if they share >= 2 points
        # We'll check on the fly using masks
        
        available = list(range(n_triples))
        random.shuffle(available)
        
        for idx in available:
            mi = triple_masks[idx]
            can_add = True
            for sel in selected_set:
                common = mi & triple_masks[sel]
                if bin(common).count('1') >= 2:
                    can_add = False
                    break
            if can_add:
                selected.append(idx)
                selected_set.add(idx)
                for p in all_triples[idx]:
                    point_usage[p] += 1
        return selected
    
    # Local search targeting alpha reduction
    def local_search_targeted(triple_list, max_iters=500):
        """Local search that targets reducing alpha below 8."""
        current = list(triple_list)
        current_set = set(current)
        
        # Check current alpha
        if not has_independent_set_of_size_8(current_set):
            alpha = compute_alpha(current)
            return current, alpha
        
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            
            if time.time() > deadline - 5:
                break
            
            # Try adding a triple that hits the most 8-sets
            # For each candidate triple, count how many 8-subsets it destroys
            # (i.e., 8-subsets that currently have no triple but would have this one)
            
            # First, find which 8-subsets are currently independent
            independent_8sets = []
            for i in range(n_8_subsets):
                sm = subset8_masks[i]
                found = False
                for ti in current_set:
                    if (triple_masks[ti] & sm) == triple_masks[ti]:
                        found = True
                        break
                if not found:
                    independent_8sets.append(i)
            
            if not independent_8sets:
                # No independent 8-set, so alpha <= 7
                alpha = compute_alpha(current)
                return current, alpha
            
            # For each candidate triple, count how many independent 8-sets it hits
            candidates = []
            for idx in range(n_triples):
                if idx in current_set:
                    continue
                mi = triple_masks[idx]
                
                # Check linearity
                can_add = True
                for sel in current_set:
                    common = mi & triple_masks[sel]
                    if bin(common).count('1') >= 2:
                        can_add = False
                        break
                if not can_add:
                    continue
                
                # Count how many independent 8-sets this triple hits
                hits = 0
                for i in independent_8sets:
                    if (mi & subset8_masks[i]) == mi:
                        hits += 1
                
                if hits > 0:
                    candidates.append((hits, idx))
            
            if not candidates:
                break
            
            # Try the top candidates
            candidates.sort(key=lambda x: -x[0])
            random.shuffle(candidates[:min(50, len(candidates))])
            
            for hits, idx in candidates[:min(20, len(candidates))]:
                new_set = current_set | {idx}
                if not has_independent_set_of_size_8(new_set):
                    # Great, we reduced alpha below 8
                    current = list(new_set)
                    current_set = new_set
                    alpha = compute_alpha(current)
                    return current, alpha
                else:
                    # Check if it reduced the number of independent 8-sets
                    old_count = len(independent_8sets)
                    new_indep = []
                    for i in range(n_8_subsets):
                        sm = subset8_masks[i]
                        found = False
                        for ti in new_set:
                            if (triple_masks[ti] & sm) == triple_masks[ti]:
                                found = True
                                break
                        if not found:
                            new_indep.append(i)
                    if len(new_indep) < old_count:
                        current = list(new_set)
                        current_set = new_set
                        improved = True
                        break
            
            # Try removing a triple that's not helping
            if not improved and len(current) > 3:
                for i in range(len(current)):
                    new_set = current_set - {current[i]}
                    # Check if removal helps (reduces independent 8-sets is bad, so we want to keep)
                    # Actually, removal might allow adding better triples later
                    # Just try it
                    if not has_independent_set_of_size_8(new_set):
                        current = list(new_set)
                        current_set = new_set
                        alpha = compute_alpha(current)
                        return current, alpha
                    improved = True
                    break
        
        # Final evaluation
        alpha = compute_alpha(current)
        return current, alpha
    
    best_alpha = 16
    best_triples = []
    
    # Phase 1: Multiple greedy + targeted local search
    num_restarts = 50
    for restart in range(num_restarts):
        if time.time() > deadline - 10:
            break
        
        system = greedy_construct(seed=restart * 137 + 42)
        if not system:
            continue
        
        optimized, alpha = local_search_targeted(system, max_iters=100)
        
        if alpha < best_alpha:
            best_alpha = alpha
            best_triples = optimized[:]
            
            if best_alpha <= 6:
                break
    
    # Phase 2: If we have alpha <= 7, try to improve further
    if best_alpha <= 7 and time.time() < deadline - 10:
        # Try to get alpha <= 6
        for attempt in range(20):
            if time.time() > deadline - 5:
                break
            
            # Start from best and try random perturbations
            current = best_triples[:]
            current_set = set(current)
            
            # Randomly remove 1-2 triples
            for _ in range(random.randint(1, 2)):
                if current:
                    idx_to_remove = random.choice(current)
                    current.remove(idx_to_remove)
                    current_set.discard(idx_to_remove)
            
            # Re-run local search
            optimized, alpha = local_search_targeted(current, max_iters=80)
            if alpha < best_alpha:
                best_alpha = alpha
                best_triples = optimized[:]
    
    # Phase 3: Final exact evaluation and polish
    if best_triples and time.time() < deadline - 3:
        final_alpha = compute_alpha(best_triples)
        if final_alpha < best_alpha:
            best_alpha = final_alpha
    
    # Convert indices back to triples
    result = [all_triples[i] for i in best_triples]
    
    return result
