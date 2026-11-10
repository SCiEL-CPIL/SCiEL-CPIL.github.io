"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin
    
    # Build conflict graph: edge between a,b if (a*b)%(a+b)==0
    n = 256
    # conflicts[i] is a bitmask of vertices that conflict with vertex i (1-indexed)
    conflicts = [0] * (n + 1)
    
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            if (a * b) % (a + b) == 0:
                conflicts[a] |= (1 << b)
                conflicts[b] |= (1 << a)
    
    # Precompute degree for each vertex
    degree = [bin(conflicts[i]).count('1') for i in range(1, n + 1)]
    
    def is_valid_set(vertices_set):
        """Check if a set of vertices forms an independent set."""
        for a in vertices_set:
            for b in vertices_set:
                if a != b and (conflicts[a] >> b) & 1:
                    return False
        return True
    
    def greedy_max_independent_set(order):
        """Greedy MIS: process vertices in given order, add if no conflict with already added."""
        selected = []
        selected_mask = 0
        for v in order:
            # Check if v conflicts with any already selected
            if (conflicts[v] & selected_mask) == 0:
                selected.append(v)
                selected_mask |= (1 << v)
        return selected
    
    def local_search_improve(initial_set, max_iters=5000):
        """Try to improve the set by swapping or adding/removing vertices."""
        current = set(initial_set)
        current_size = len(current)
        current_mask = 0
        for v in current:
            current_mask |= (1 << v)
        
        best_set = set(current)
        best_size = current_size
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Try to add a vertex not in current
            added = False
            removed = False
            
            # Find candidates to add
            candidates = []
            for v in range(1, n + 1):
                if (current_mask >> v) & 1:
                    continue
                # Check if v can be added
                if (conflicts[v] & current_mask) == 0:
                    candidates.append(v)
            
            if candidates:
                # Pick the one that allows most future additions (heuristic)
                # Simple: pick random candidate
                v_add = random.choice(candidates)
                current.add(v_add)
                current_mask |= (1 << v_add)
                current_size += 1
                added = True
            
            # Try to remove a vertex and add multiple others
            if not added and current_size > 0:
                # Try removing a high-degree vertex
                removable = list(current)
                random.shuffle(removable)
                for v_remove in removable[:min(5, len(removable))]:  # try up to 5 removals
                    trial_mask = current_mask & ~(1 << v_remove)
                    trial_set = current - {v_remove}
                    
                    # Try to add vertices that were previously blocked
                    added_count = 0
                    for v_add in range(1, n + 1):
                        if (trial_mask >> v_add) & 1:
                            continue
                        if (conflicts[v_add] & trial_mask) == 0:
                            trial_mask |= (1 << v_add)
                            trial_set.add(v_add)
                            added_count += 1
                    
                    if len(trial_set) > current_size:
                        current = trial_set
                        current_mask = trial_mask
                        current_size = len(current)
                        removed = True
                        break
            
            if current_size > best_size:
                best_set = set(current)
                best_size = current_size
            
            # Occasionally restart from best
            if it % 200 == 199 and current_size < best_size:
                current = set(best_set)
                current_mask = 0
                for v in current:
                    current_mask |= (1 << v)
                current_size = len(current)
        
        return list(best_set)
    
    # Strategy 1: Greedy with different orderings
    best_result = []
    best_size = 0
    
    # Sort by degree (ascending) - prefer low-degree vertices first
    order_by_degree = sorted(range(1, n + 1), key=lambda x: degree[x])
    result = greedy_max_independent_set(order_by_degree)
    best_result = result
    best_size = len(result)
    
    # Sort by value (ascending)
    result = greedy_max_independent_set(range(1, n + 1))
    if len(result) > best_size:
        best_result = result
        best_size = len(result)
    
    # Sort by value (descending)
    result = greedy_max_independent_set(range(n, 0, -1))
    if len(result) > best_size:
        best_result = result
        best_size = len(result)
    
    # Multiple random orderings
    for _ in range(20):
        if time.time() > deadline:
            break
        random_order = list(range(1, n + 1))
        random.shuffle(random_order)
        result = greedy_max_independent_set(random_order)
        if len(result) > best_size:
            best_result = result
            best_size = len(result)
    
    # Local search improvement
    if time.time() < deadline:
        improved = local_search_improve(best_result)
        if len(improved) > best_size:
            best_result = improved
            best_size = len(improved)
    
    # Additional local search rounds from different starting points
    for _ in range(5):
        if time.time() > deadline:
            break
        random_order = list(range(1, n + 1))
        random.shuffle(random_order)
        greedy_result = greedy_max_independent_set(random_order)
        improved = local_search_improve(greedy_result, max_iters=2000)
        if len(improved) > best_size:
            best_result = improved
            best_size = len(improved)
    
    return best_result
# EVOLVE-BLOCK-END
