import time
import itertools
import numpy as np
import random

N = 16
DEADLINE = 300.0  # seconds, leave margin before 330s kill

def _dist2(a, b):
    return (a[0] - b[0])**2 + (a[1] - b[1])**2

def _is_isosceles(a, b, c):
    d1 = _dist2(a, b)
    d2 = _dist2(a, c)
    d3 = _dist2(b, c)
    return d1 == d2 or d1 == d3 or d2 == d3

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE - 15.0  # 15s safety margin
    
    points = [(x, y) for x in range(N) for y in range(N)]
    npts = len(points)
    
    # Precompute isosceles conflicts using bitmasks for speed
    # conflict[i] = bitmask of j such that i and j are in some isosceles triple with a third point
    # Actually, we need: for each pair (i,j), the set of k that form isosceles with them
    # But for independent set formulation: a set S is valid if no three points in S form isosceles
    # So conflict[i] should be: which j cannot coexist with i in S? 
    # Wait - any two points can coexist. Three points cannot if they form isosceles.
    # This is a 3-uniform hypergraph independent set problem.
    
    # For branch-and-bound, we need efficient checks.
    # Let's precompute: for each pair (i,j), the bitmask of k that form isosceles with i,j
    pair_conflict = np.zeros((npts, npts), dtype=object)
    for i in range(npts):
        for j in range(i+1, npts):
            mask = 0
            for k in range(npts):
                if k != i and k != j and _is_isosceles(points[i], points[j], points[k]):
                    mask |= (1 << k)
            pair_conflict[i][j] = mask
            pair_conflict[j][i] = mask
    
    # Greedy construction with randomization for diversity
    def greedy_construct(seed):
        rng = random.Random(seed)
        S = set()
        S_list = []
        
        # Track which points are "blocked" (would create isosceles with some pair in S)
        blocked = 0  # bitmask
        
        while True:
            if time.time() > deadline:
                break
            
            # Find all compatible points (not in S, not blocked)
            available = 0
            for p in range(npts):
                if p not in S and not (blocked & (1 << p)):
                    available |= (1 << p)
            
            if available == 0:
                break
            
            # Score each available point by how many new points it would block
            best_p = None
            best_score = -1
            avail_list = [p for p in range(npts) if (available & (1 << p))]
            rng.shuffle(avail_list)
            
            for p in avail_list:
                # Count how many new points would be blocked
                new_blocked = 0
                for a in S:
                    new_blocked |= pair_conflict[p][a]
                new_blocked &= ~blocked
                new_blocked &= ~(1 << p)
                score = bin(new_blocked).count('1')
                # Add small random tiebreaker
                score += rng.random() * 0.001
                if score > best_score:
                    best_score = score
                    best_p = p
            
            if best_p is None:
                break
            
            S.add(best_p)
            S_list.append(best_p)
            # Update blocked
            for a in S:
                if a != best_p:
                    blocked |= pair_conflict[best_p][a]
        
        # Check if maximal: can we add any more points?
        # If all non-S points are blocked, it's maximal
        return S_list, blocked
    
    # Local search: try to remove points and re-add to improve
    def local_search(initial_S, seed):
        rng = random.Random(seed)
        S = set(initial_S)
        best_S = list(S)
        best_size = len(S)
        
        # Compute blocked for current S
        def compute_blocked(S_set):
            blocked = 0
            S_list = list(S_set)
            for i in range(len(S_list)):
                for j in range(i+1, len(S_list)):
                    blocked |= pair_conflict[S_list[i]][S_list[j]]
            return blocked
        
        max_iterations = 500
        for iteration in range(max_iterations):
            if time.time() > deadline:
                break
            
            if len(S) >= best_size:
                continue
            
            blocked = compute_blocked(S)
            
            # Try removing one point and see if we can do better
            S_list = list(S)
            rng.shuffle(S_list)
            
            improved = False
            for remove_p in S_list:
                if len(S) - 1 >= best_size:
                    continue
                
                S.remove(remove_p)
                new_blocked = compute_blocked(S)
                
                # Try to add back different points
                added = False
                for add_p in range(npts):
                    if add_p in S or (new_blocked & (1 << add_p)):
                        continue
                    # Check if adding add_p keeps S valid (no isosceles)
                    valid = True
                    for a in S:
                        for b in S:
                            if a != b and (pair_conflict[a][b] & (1 << add_p)):
                                valid = False
                                break
                        if not valid:
                            break
                    if valid:
                        S.add(add_p)
                        added = True
                        break
                
                if not added:
                    S.add(remove_p)
                    continue
                
                # Check if this new S is maximal
                final_blocked = compute_blocked(S)
                is_maximal = True
                for p in range(npts):
                    if p not in S and not (final_blocked & (1 << p)):
                        is_maximal = False
                        break
                
                if is_maximal and len(S) < best_size:
                    best_S = list(S)
                    best_size = len(S)
                    improved = True
                else:
                    S.remove(add_p)
                    S.add(remove_p)
            
            if not improved:
                break
        
        return best_S
    
    # Branch and bound for finding smaller maximal independent sets
    def branch_and_bound(current_S, blocked_mask, depth, best_size, seed):
        """
        Try to extend current_S to a maximal set with size < best_size.
        Returns list of points if found, else None.
        """
        if time.time() > deadline:
            return None
        
        if len(current_S) >= best_size:
            return None
        
        # Check if current_S is already maximal
        is_maximal = True
        for p in range(npts):
            if p not in current_S and not (blocked_mask & (1 << p)):
                is_maximal = False
                break
        
        if is_maximal:
            if len(current_S) < best_size:
                return list(current_S)
            return None
        
        # Find available points
        available = []
        for p in range(npts):
            if p not in current_S and not (blocked_mask & (1 << p)):
                available.append(p)
        
        if not available:
            return None
        
        # Sort by coverage (points that would be blocked)
        def coverage_score(p):
            new_blocked = 0
            for a in current_S:
                new_blocked |= pair_conflict[p][a]
            new_blocked &= ~blocked_mask
            return bin(new_blocked).count('1')
        
        available.sort(key=coverage_score, reverse=True)
        
        # Try each available point
        for p in available[:min(20, len(available))]:  # Limit branching
            if time.time() > deadline:
                break
            
            new_blocked = blocked_mask
            for a in current_S:
                new_blocked |= pair_conflict[p][a]
            
            result = branch_and_bound(
                current_S | {p}, 
                new_blocked, 
                depth + 1, 
                best_size, 
                seed
            )
            if result is not None:
                return result
        
        return None
    
    # Main: run multiple greedy + local search iterations
    best_result = None
    best_size = npts
    
    # First, get a baseline greedy solution
    baseline_S, _ = greedy_construct(42)
    if baseline_S:
        best_result = baseline_S
        best_size = len(baseline_S)
    
    # Run multiple randomized greedy + local search
    num_trials = 50
    for trial in range(num_trials):
        if time.time() > deadline:
            break
        
        seed = 1000 + trial
        S, _ = greedy_construct(seed)
        if not S:
            continue
        
        # Local search
        improved_S = local_search(S, seed)
        
        if len(improved_S) < best_size:
            best_size = len(improved_S)
            best_result = improved_S
    
    # Try branch and bound from best solution
    if best_result is not None and time.time() < deadline:
        # Try to improve by removing points and using B&B
        for remove_attempt in range(min(5, len(best_result))):
            if time.time() > deadline:
                break
            
            S_copy = set(best_result)
            # Remove a random point
            to_remove = random.Random(2000 + remove_attempt).choice(list(S_copy))
            S_copy.remove(to_remove)
            
            # Compute blocked for remaining
            blocked = 0
            S_list = list(S_copy)
            for i in range(len(S_list)):
                for j in range(i+1, len(S_list)):
                    blocked |= pair_conflict[S_list[i]][S_list[j]]
            
            # Try to extend to maximal with size < best_size
            result = branch_and_bound(S_copy, blocked, 0, best_size, 3000)
            if result is not None and len(result) < best_size:
                best_size = len(result)
                best_result = result
    
    # Final validation: ensure best_result is valid and maximal
    if best_result is not None:
        S_set = set(best_result)
        # Check no isosceles
        valid = True
        for i in range(len(best_result)):
            for j in range(i+1, len(best_result)):
                for k in range(j+1, len(best_result)):
                    if _is_isosceles(points[best_result[i]], points[best_result[j]], points[best_result[k]]):
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if not valid:
            # Fallback to baseline
            if baseline_S:
                best_result = baseline_S
            else:
                # Return empty set (shouldn't happen)
                best_result = []
    
    return [points[i] for i in best_result] if best_result else [points[0]]
