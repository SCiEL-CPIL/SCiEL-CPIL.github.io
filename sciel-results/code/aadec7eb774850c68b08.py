import time
import random
import math

N = 64

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin for 330s limit
    
    # Precompute all possible points
    all_points = [(x, y) for x in range(N) for y in range(N)]
    num_points = len(all_points)
    
    # Precompute squared distances between all pairs
    # dist[i][j] = squared distance between point i and point j
    # Storing as a list of lists of ints
    # 4096 * 4096 is ~16M entries. In Python this is heavy but manageable if we are smart.
    # Actually, let's just compute on the fly or use a lookup table if possible.
    # 64x64 grid. Max dist2 = 63^2 + 63^2 = 3969 + 3969 = 7938.
    # We can map point index to (x,y).
    
    # To speed up distance checks, we can precompute a distance matrix?
    # 4096x4096 matrix of ints is 16M ints. In Python, an int object is ~28 bytes. 
    # 16M * 28 bytes = ~450 MB. This might be too much memory or slow to initialize.
    # Let's try to avoid the full matrix and compute dist2 quickly.
    
    # Alternative: Represent points as indices 0..4095.
    # x = i // N, y = i % N.
    # dist2(i, j) = (xi-xj)^2 + (yi-yj)^2.
    
    # State representation:
    # selected: list of point indices currently in the set.
    # selected_set: set of point indices for O(1) lookup.
    # dist_counts: For each point q in selected, we need to know the set of distances to other selected points.
    # Actually, the constraint is:
    # 1. For any p in selected, all dist2(p, q) for q in selected, q!=p must be distinct.
    # 2. For any q in selected, dist2(p, q) must not be in {dist2(q, r) for r in selected, r!=q, r!=p}.
    #    Wait, condition 2 is symmetric. If dist(p,q) == dist(q,r), then p,q,r form an isosceles triangle with vertex q.
    #    So the condition is simply: No three points p, q, r in the set form an isosceles triangle.
    #    This is equivalent to: For every q in the set, the multiset of distances from q to all other points in the set contains no duplicates.
    
    # So, for each q in selected, let D_q = { dist2(q, p) | p in selected, p != q }.
    # The set is valid iff for all q, len(D_q) == len(selected) - 1.
    
    # Efficient check for adding p:
    # 1. Check if for the new point p, all distances to existing points are distinct.
    # 2. Check if for every existing point q, dist(p, q) is not already in D_q.
    
    # Data structures for fast updates:
    # selected: list of indices
    # point_set: set of indices
    # dist_sets: dict mapping point index q -> set of distances from q to other selected points.
    
    # Initialization: Greedy to get a good starting point
    selected = []
    point_set = set()
    dist_sets = {}
    
    def _dist2_idx(i, j):
        xi = i // N
        yi = i % N
        xj = j // N
        yj = j % N
        dx = xi - xj
        dy = yi - yj
        return dx*dx + dy*dy

    def _can_add(p_idx):
        if p_idx in point_set:
            return False
        k = len(selected)
        if k == 0:
            return True
        
        # Check distances from p to all selected points
        seen_dists = set()
        for q_idx in selected:
            d = _dist2_idx(p_idx, q_idx)
            # Condition 1: Distances from p must be unique
            if d in seen_dists:
                return False
            seen_dists.add(d)
            
            # Condition 2: d must not be in dist_sets[q_idx]
            if d in dist_sets[q_idx]:
                return False
        
        return True

    def _add(p_idx):
        point_set.add(p_idx)
        selected.append(p_idx)
        dist_sets[p_idx] = set()
        
        for q_idx in selected[:-1]: # All except p_idx
            d = _dist2_idx(p_idx, q_idx)
            dist_sets[p_idx].add(d)
            dist_sets[q_idx].add(d)

    def _remove(p_idx):
        point_set.remove(p_idx)
        selected.remove(p_idx)
        
        # Update dist_sets for all remaining points
        for q_idx in selected:
            d = _dist2_idx(p_idx, q_idx)
            dist_sets[q_idx].remove(d)
        
        del dist_sets[p_idx]

    # 1. Greedy Initialization
    rng = random.Random(42)
    candidates = list(range(num_points))
    rng.shuffle(candidates)
    
    for p_idx in candidates:
        if _can_add(p_idx):
            _add(p_idx)
            
    best_size = len(selected)
    best_set_copy = selected[:]
    
    # 2. Simulated Annealing
    # Energy function: We want to maximize size.
    # We can define a "temperature" that controls the probability of removing a point (worsening the size)
    # to allow future additions.
    
    current_temp = 1.0
    cooling_rate = 0.999
    min_temp = 1e-5
    
    # We need to track "stuck" state to restart or adjust strategy if needed, 
    # but standard SA should handle it.
    
    # To make SA effective, we need to propose moves.
    # Move types:
    # A: Add a random unselected point (if valid). Size +1. Always accept.
    # B: Remove a random selected point. Size -1. Accept with prob exp(-delta/temp) where delta=1?
    #    Actually, since we maximize size, removing a point is "bad" by 1 unit.
    #    But it might allow adding 2 points later.
    #    We accept removal with probability exp(-1/temp).
    
    # Optimization: Instead of iterating all unselected points for "Add", 
    # we can keep a list of "blocked" points and update it? 
    # That's complex. 
    # Simple approach:
    # 1. Try to add random unselected points.
    # 2. If no additions possible (stuck), start removing points with SA probability.
    
    # Let's refine the loop:
    # While time < deadline:
    #   1. Try to add random unselected points. If any added, reset "stuck" counter.
    #   2. If no additions in a batch of tries, perform a "removal" move with probability based on temp.
    #   3. Cool down.
    
    # To speed up "Try to add", we can maintain a list of unselected points.
    unselected = [i for i in range(num_points) if i not in point_set]
    
    # We need to update unselected when we add/remove.
    # Removing from a list is O(N). 
    # Since N=4096, and we do this many times, it might be slow.
    # Alternative: Just iterate through a random sample of unselected points.
    
    # Let's use a simpler heuristic loop:
    # 1. Shuffle unselected. Try to add first few.
    # 2. If fail, remove a random point (with prob).
    
    # Actually, let's just use the standard SA structure with explicit state checks.
    
    # Pre-compute neighbors? No, grid is dense.
    
    # Let's optimize the "Add" phase.
    # Instead of checking random unselected, we can keep track of which points are "potentially addable"?
    # A point p is addable if it doesn't conflict with current set.
    # This is hard to maintain incrementally without heavy bookkeeping.
    
    # Let's stick to random sampling but make it smart.
    
    stuck_count = 0
    max_stuck_before_remove = 50 # Try 50 random adds before considering removal
    
    while time.time() < deadline:
        # 1. Attempt additions
        added_any = False
        
        # Sample a few random unselected points
        # To avoid O(N) list maintenance, we just pick random indices and check if they are in point_set.
        # If they are, skip. If not, try to add.
        
        for _ in range(100): # Try 100 random candidates per step
            if time.time() > deadline:
                break
                
            p_idx = rng.randint(0, num_points - 1)
            if p_idx in point_set:
                continue
                
            if _can_add(p_idx):
                _add(p_idx)
                added_any = True
                if len(selected) > best_size:
                    best_size = len(selected)
                    best_set_copy = selected[:]
        
        if added_any:
            stuck_count = 0
        else:
            stuck_count += 1
            
            # 2. Attempt removal (Simulated Annealing step)
            # If we are stuck, remove a point to escape local optimum.
            # Probability of removal increases as we get more stuck or temp is high.
            # Standard SA: accept worse move with prob exp(-delta/E * 1/T).
            # Here delta = 1 (size reduction).
            # prob = exp(-1/current_temp)
            
            if selected:
                # We only remove if we are sufficiently stuck or if temp is high enough
                # To avoid oscillating at low temp, we can scale the probability.
                # Let's use a dynamic probability: 
                # If stuck_count > threshold, force a removal or increase prob.
                
                # Simple rule: 
                # If stuck_count > max_stuck_before_remove, remove a random point.
                # Otherwise, remove with prob exp(-1/current_temp).
                
                do_remove = False
                if stuck_count > max_stuck_before_remove:
                    do_remove = True
                else:
                    prob = math.exp(-1.0 / current_temp)
                    if rng.random() < prob:
                        do_remove = True
                
                if do_remove:
                    # Remove a random point
                    # Prefer removing points that are "central" or have high connectivity?
                    # Random is fine for now.
                    idx = rng.randint(0, len(selected) - 1)
                    p_idx = selected[idx]
                    _remove(p_idx)
                    stuck_count = 0 # Reset stuck counter after a move
                    
        # Cooling
        current_temp *= cooling_rate
        if current_temp < min_temp:
            # If temp is too low, we might get stuck in local optima permanently.
            # Reheat? Or just stop?
            # Reheating can be effective.
            # Let's reheat to 0.5 * initial_temp occasionally if we are stuck for a long time.
            if stuck_count > 1000:
                current_temp = 0.5
                stuck_count = 0
            else:
                current_temp = min_temp

    return [(i // N, i % N) for i in best_set_copy]

# EVOLVE-BLOCK-END
