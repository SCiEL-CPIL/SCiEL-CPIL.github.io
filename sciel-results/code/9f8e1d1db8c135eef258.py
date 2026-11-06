import math
import time
import random

def construct():
    start_time = time.time()
    deadline = 85  # safety margin before 90s kill
    
    n = 256
    max_sum = 2 * n
    
    # Perfect squares up to max_sum
    squares = set()
    i = 1
    while i * i <= max_sum:
        squares.add(i * i)
        i += 1
    
    # Filter candidates: 2*a must not be a perfect square
    candidates = []
    for a in range(1, n + 1):
        if 2 * a not in squares:
            candidates.append(a)
    
    m = len(candidates)
    cand_set = set(candidates)
    
    # Build conflict graph: conflict[a][b] = True if a+b is a perfect square
    # Use adjacency list for efficiency
    conflicts = {a: set() for a in candidates}
    for idx_a, a in enumerate(candidates):
        for b in candidates[idx_a+1:]:
            if a + b in squares:
                conflicts[a].add(b)
                conflicts[b].add(a)
    
    # Greedy initialization: pick vertices in order of lowest degree first
    def greedy_init():
        remaining = list(candidates)
        result = []
        while remaining:
            # Find vertex with minimum conflicts among remaining
            min_conf = min(remaining, key=lambda x: len(conflicts[x] & set(remaining)))
            result.append(min_conf)
            remaining.remove(min_conf)
            # Remove all conflicts of min_conf from remaining
            rem_set = set(remaining)
            remaining = [x for x in remaining if x not in conflicts[min_conf]]
        return set(result)
    
    best_set = greedy_init()
    best_size = len(best_set)
    
    # Local search: try to add vertices, remove conflicts if needed
    def try_improve(current_set, max_iters=100000):
        current = set(current_set)
        current_size = len(current)
        no_improve_count = 0
        iters = 0
        
        while iters < max_iters and time.time() - start_time < deadline - 2:
            iters += 1
            no_improve = True
            
            # Try adding each non-member vertex
            non_members = [x for x in candidates if x not in current]
            random.shuffle(non_members)
            
            for v in non_members:
                # Check if v conflicts with current set
                conflicting = conflicts[v] & current
                if not conflicting:
                    # Can add v directly
                    current.add(v)
                    current_size += 1
                    no_improve = False
                    if current_size > best_size:
                        best_size = current_size
                        best_set = set(current)
                    break
                else:
                    # Try removing conflicting vertices and adding v
                    # Only worthwhile if we remove fewer than we gain (always gain 1, remove len(conflicting))
                    # So only try if len(conflicting) <= 1... actually we gain 1 and remove len(conflicting)
                    # Net change = 1 - len(conflicting). Only positive if len(conflicting) == 0 (already handled)
                    # So skip this branch - it never helps for pure size maximization
                    # But wait - removing one and adding one keeps size same but might enable future additions
                    # Let's do a 1-swap: remove one conflicting, add v
                    if len(conflicting) == 1:
                        to_remove = next(iter(conflicting))
                        new_set = current - {to_remove} | {v}
                        # Check if this is valid (it should be by construction)
                        # This doesn't increase size, but might help future additions
                        # Skip for now - focus on pure additions
                        pass
            
            if no_improve:
                # Try 2-swap: remove two, add one (net -1, bad) or other moves
                # Try removing a vertex that has many conflicts and see if we can add more
                # Actually, let's try: for each vertex in current, temporarily remove it
                # and see if we can add back more vertices
                improved = False
                for v in list(current):
                    trial = current - {v}
                    # Try to greedily add back vertices
                    trial_size = len(trial)
                    trial_set = set(trial)
                    non_members2 = [x for x in candidates if x not in trial_set]
                    random.shuffle(non_members2)
                    for u in non_members2:
                        if not (conflicts[u] & trial_set):
                            trial_set.add(u)
                            trial_size += 1
                    if trial_size > current_size:
                        current = trial_set
                        current_size = trial_size
                        improved = True
                        if current_size > best_size:
                            best_size = current_size
                            best_set = set(current)
                        break
                if not improved:
                    no_improve_count += 1
                    if no_improve_count > 50:
                        break
            else:
                no_improve_count = 0
        
        return current, current_size
    
    # Run local search with random restarts
    num_restarts = 0
    while time.time() - start_time < deadline - 3:
        num_restarts += 1
        
        if num_restarts == 1:
            current_set = best_set
        else:
            # Random restart with perturbation
            if random.random() < 0.3:
                current_set = greedy_init()
            else:
                # Perturb best set: remove some random vertices
                current_set = set(best_set)
                to_remove = random.sample(list(current_set), min(len(current_set)//4, 10))
                current_set -= set(to_remove)
        
        current_set, current_size = try_improve(current_set, max_iters=50000)
        
        if current_size > best_size:
            best_size = current_size
            best_set = set(current_set)
        
        # If we found a perfect or near-perfect solution, stop
        if best_size >= 100:
            break
    
    return sorted(best_set)

