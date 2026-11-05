"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from fractions import Fraction
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    N = 256
    # Weights: reciprocal of each number (use Fraction for exactness)
    weights = [Fraction(1, i) for i in range(1, N + 1)]
    float_weights = [float(w) for w in weights]
    
    # Precompute conflicts (neighbors in conflict graph)
    # neighbors[i] is a set of indices j such that i and j form a 3-AP with some z
    neighbors = [set() for _ in range(N)]
    
    for i in range(N):
        x = i + 1
        for j in range(i + 1, N):
            y = j + 1
            conflict = False
            # Check if x, y can be part of a 3-AP
            # 1. x, y are 1st, 2nd -> z = 2y - x. Need z > y and z <= N
            z = 2 * y - x
            if y < z <= N:
                conflict = True
            # 2. y, x are 1st, 2nd -> z = 2x - y. Need z > x and z <= N
            z = 2 * x - y
            if x < z <= N:
                conflict = True
            # 3. x, y are 1st, 3rd -> z = (x+y)/2. Need z integer, 1 <= z <= N, z != x, z != y
            s = x + y
            if s % 2 == 0:
                z = s // 2
                if 1 <= z <= N and z != x and z != y:
                    conflict = True
            
            if conflict:
                neighbors[i].add(j)
                neighbors[j].add(i)
                
    # Helper to check if a vertex can be added to a set
    # selected is a set of indices
    def can_add(vertex, selected):
        # Check if vertex conflicts with any in selected
        for neighbor in neighbors[vertex]:
            if neighbor in selected:
                return False
        return True
        
    # Helper to calculate exact weight of a set
    def calc_weight(selected):
        total = Fraction(0, 1)
        for v in selected:
            total += weights[v]
        return total

    best_set = set()
    best_weight = Fraction(0, 1)
    
    # Multi-start greedy with local search
    # We will run several iterations. Each iteration starts with a randomized greedy fill.
    # Then we do local search: try to add any vertex that fits.
    # If stuck, we might do a "kick" (remove some, refill).
    
    # Strategy:
    # 1. Generate a random permutation of vertices.
    # 2. Greedily add vertices if they don't conflict.
    # 3. Local Search: Iterate through all vertices. If a vertex not in set can be added, add it. Repeat until no more can be added.
    # 4. To improve, try "Swaps": 
    #    For each selected vertex v, temporarily remove it.
    #    Then greedily add any vertices that can now be added (including v? No, v is removed).
    #    If the new set is better, keep it.
    #    This is O(N^2) per swap attempt, which is fine for N=256.
    
    def local_search(selected):
        # Phase 1: Add any possible vertex
        changed = True
        while changed:
            changed = False
            for v in range(N):
                if v not in selected and can_add(v, selected):
                    selected.add(v)
                    changed = True
        # Phase 2: Try swaps
        improved = True
        while improved:
            improved = False
            for v in list(selected):
                # Try removing v
                selected.remove(v)
                # Try adding any vertex that can now be added
                added_any = False
                for u in range(N):
                    if u not in selected and can_add(u, selected):
                        selected.add(u)
                        added_any = True
                # Check if the set is now better? 
                # Actually, simply adding everything possible might lead to a worse weight 
                # if we removed a high weight vertex and added low weight ones.
                # We need to compare weights.
                # Let's refine: 
                # 1. Remove v.
                # 2. Find all vertices that can be added to (selected without v).
                # 3. If sum(weights of added) > weight(v), then do the swap.
                # 4. If not, revert.
                
                # Let's re-implement the swap logic more carefully
                # We already modified 'selected' in place above. We need to be able to revert.
                # Let's do this properly.
                
                # Actually, the loop above is dangerous. Let's use a candidate set approach.
                pass
        
        # Let's implement a safer local search
        return selected

    def safe_local_search(initial_selected):
        selected = set(initial_selected)
        
        # 1. Add any vertex that fits
        changed = True
        while changed:
            changed = False
            for v in range(N):
                if v not in selected and can_add(v, selected):
                    selected.add(v)
                    changed = True
        
        # 2. Try to improve by swapping
        # For each v in selected, check if removing v allows adding a set of vertices 
        # with total weight > weight(v)
        improved = True
        while improved:
            improved = False
            for v in list(selected):
                # Temporarily remove v
                selected.remove(v)
                
                # Find all vertices that can be added to current 'selected'
                # Note: 'selected' no longer contains v
                candidates_to_add = []
                for u in range(N):
                    if u not in selected and can_add(u, selected):
                        candidates_to_add.append(u)
                        
                # We want to add the best subset of candidates_to_add.
                # Since candidates_to_add are mutually non-conflicting with 'selected',
                # they might conflict with each other.
                # Greedily add them by weight?
                # Let's just try adding ALL of them? No, they might conflict.
                # Let's greedily add them by weight descending.
                
                gain = Fraction(0, 1)
                new_selected = set(selected)
                # Sort candidates by weight descending
                candidates_to_add.sort(key=lambda x: -float_weights[x])
                for u in candidates_to_add:
                    if can_add(u, new_selected):
                        new_selected.add(u)
                        gain += weights[u]
                        
                # Compare gain with weight of v
                if gain > weights[v]:
                    # Accept the swap
                    selected = new_selected
                    improved = True
                    break # Restart the loop as the set changed significantly
                else:
                    # Revert
                    selected.add(v)
                    
        return selected

    # Run multiple iterations
    num_iterations = 50 # Adjust based on time
    
    for i in range(num_iterations):
        if time.time() > deadline:
            break
            
        # Randomized Greedy Initialization
        # Order vertices by weight, but add random noise to break ties or shuffle slightly
        # Or just shuffle the order completely?
        # High weight vertices (small numbers) are more valuable. 
        # Let's try a mix: 50% natural order (by weight), 50% random order.
        
        if random.random() < 0.5:
            order = sorted(range(N), key=lambda x: -float_weights[x])
            # Shuffle the last half to introduce diversity
            mid = len(order) // 2
            order[mid:] = random.sample(order[mid:], len(order[mid:]))
        else:
            order = list(range(N))
            random.shuffle(order)
            
        current_set = set()
        for v in order:
            if can_add(v, current_set):
                current_set.add(v)
                
        # Local Search
        current_set = safe_local_search(current_set)
        
        # Evaluate
        current_weight = calc_weight(current_set)
        if current_weight > best_weight:
            best_weight = current_weight
            best_set = set(current_set)
            
        # Optional: Perturbation for next iteration
        # Remove 1-3 random vertices from best_set to start next iteration?
        # The loop above starts fresh each time. 
        # To explore better, we can start next iteration from a perturbed best_set.
        if i < num_iterations - 1:
             # 50% chance to restart from scratch, 50% to perturb best
             if random.random() < 0.5:
                 pass # Next iteration will be fresh
             else:
                 # Perturb best_set for next iteration's initialization?
                 # The current loop structure doesn't easily allow passing state.
                 # Let's just rely on the randomization in the initialization.
                 pass

    # Final check: try to add any remaining vertices to best_set
    changed = True
    while changed:
        changed = False
        for v in range(N):
            if v not in best_set and can_add(v, best_set):
                best_set.add(v)
                changed = True
                best_weight = calc_weight(best_set)

    # Extract result
    result = sorted([i + 1 for i in best_set])
    
    if not result:
        # Fallback: just {1}
        result = [1]
        
    return result
# EVOLVE-BLOCK-END
