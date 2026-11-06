import numpy as np
import time
import random

N = 64

def construct():
    start_time = time.time()
    deadline = start_time + 295  # 295s deadline, 35s margin for evaluator overhead
    
    # Precompute all squared distances between pairs of grid points?
    # Grid is 64x64 -> 4096 points.
    # Distance squared ranges from 1 to 63^2 + 63^2 = 7938.
    # We can represent the set of used distances from a point as a bitset (int).
    # Max distance index is 7938. Python ints are arbitrary precision, so this is fine.
    # Actually, using a boolean array of size 7939 might be faster for membership checks?
    # Or just a set? Sets are O(1) average. Bitsets are very fast for "is this bit set".
    # Let's try sets first for simplicity, then optimize if needed.
    
    # A more powerful structural change: 
    # Instead of greedy add + local search for adding, use a **simulated annealing** or **hill climbing** 
    # that allows **swaps** and **moves**. 
    # State: set of points.
    # Move: pick a random point in S, remove it, pick a random point NOT in S, add it.
    # Score: number of points - penalty * (number of isosceles triangles).
    # But we want EXACTLY 0 isosceles triangles. 
    # So we can define the objective as: maximize |S| subject to 0 violations.
    # Or, minimize violations, then maximize |S|.
    # A better objective for local search: 
    # f(S) = |S| - M * V(S), where V(S) is the number of isosceles triangles, and M is large.
    # If we find a state with V(S)=0, we can then try to add points greedily.
    
    # Let's implement a local search that minimizes violations.
    # Initial state: random set of K points.
    # Move: swap a point in S with a point not in S.
    # Accept if it reduces violations, or with some probability if it increases (annealing).
    
    # To compute violations efficiently, we need an incremental update.
    # When we remove point p from S:
    #   For each pair (q, r) in S\{p}, check if (q,r,p) was an isosceles triangle.
    #   This is O(|S|^2) which is too slow if |S| is large.
    
    # Alternative: maintain for each point p in S, the set of distances to other points.
    # V(S) = sum over all pairs (p, q) of [number of other points r such that dist(p,r) == dist(q,r)] ... no, that's not right.
    # An isosceles triangle is defined by 3 points. 
    # Let's define V(S) as the number of triples (p, q, r) such that they form an isosceles triangle.
    # This is hard to update incrementally.
    
    # Let's stick to the constraint satisfaction approach:
    # We want to find a large set S with 0 violations.
    # The previous program did greedy add. The bottleneck is that greedy add gets stuck in local optima.
    # A swap-based local search can escape these.
    
    # Let's implement a local search that tries to **add** points, but if it fails, it tries to **swap** points.
    # Actually, a simpler and effective method is **iterated local search** on the "add" problem, but with a twist:
    # Instead of just adding, we allow **replacing** a point.
    
    # Let's try a different structural change: **Genetic Algorithm** or **Memetic Algorithm**?
    # No, let's try **Simulated Annealing** on the set of points, where the state is a subset of the grid.
    # Objective: maximize |S| - penalty * V(S).
    # If we can maintain V(S) incrementally, this is feasible.
    
    # Incremental update for V(S):
    # When adding point p to S:
    #   New violations = number of pairs (q, r) in S such that (p, q, r) is isosceles.
    #   This is O(|S|^2).
    # When removing point p from S:
    #   Removed violations = number of pairs (q, r) in S\{p} such that (p, q, r) is isosceles.
    #   This is O(|S|^2).
    # If |S| is ~50, |S|^2 = 2500. This is fast enough for many steps.
    
    # Let's implement Simulated Annealing.
    
    def is_isosceles(p, q, r):
        d_pq = (p[0]-q[0])**2 + (p[1]-q[1])**2
        d_pr = (p[0]-r[0])**2 + (p[1]-r[1])**2
        d_qr = (q[0]-r[0])**2 + (q[1]-r[1])**2
        return d_pq == d_pr or d_pq == d_qr or d_pr == d_qr

    # Precompute all points
    all_points = [(x, y) for x in range(N) for y in range(N)]
    
    best_points = []
    best_count = 0
    
    def count_violations(S):
        """Count the number of isosceles triangles in S."""
        S = list(S)
        n = len(S)
        v = 0
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if is_isosceles(S[i], S[j], S[k]):
                        v += 1
        return v

    # This O(n^3) count is too slow for large n. We need incremental.
    
    def add_point_violations(S, p):
        """Return the number of new isosceles triangles formed by adding p to S."""
        count = 0
        S_list = list(S)
        for i in range(len(S_list)):
            for j in range(i+1, len(S_list)):
                if is_isosceles(p, S_list[i], S_list[j]):
                    count += 1
        return count

    def remove_point_violations(S, p):
        """Return the number of isosceles triangles removed by removing p from S."""
        count = 0
        S_list = [q for q in S if q != p]
        for i in range(len(S_list)):
            for j in range(i+1, len(S_list)):
                if is_isosceles(p, S_list[i], S_list[j]):
                    count += 1
        return count

    # Let's implement a local search that starts with a random set and tries to reduce violations to 0,
    # then maximize size.
    
    def simulated_annealing(seed_val):
        random.seed(seed_val)
        
        # Start with a small random set
        initial_size = 10
        S = set(random.sample(all_points, initial_size))
        current_violations = count_violations(S)
        
        # We want to minimize violations, then maximize size.
        # Let's define a combined objective: 
        # Objective = -|S| * 1000 + violations
        # We want to minimize this.
        
        def objective(S, v):
            return -len(S) * 1000 + v
        
        current_obj = objective(S, current_violations)
        best_S = set(S)
        best_v = current_violations
        best_obj = current_obj
        
        T = 100.0
        cooling_rate = 0.999
        min_T = 0.1
        max_steps = 50000
        
        for step in range(max_steps):
            if time.time() > deadline - 5:
                break
            if T < min_T:
                break
                
            # Generate a neighbor: swap a point in S with a point not in S
            if len(S) == 0:
                new_p = random.choice(all_points)
                S.add(new_p)
                delta_v = add_point_violations(S - {new_p}, new_p)
                new_v = current_violations + delta_v
                new_obj = objective(S, new_v)
            else:
                # Pick a point to remove
                p_remove = random.choice(list(S))
                # Pick a point to add
                p_add = random.choice(all_points)
                if p_add in S:
                    continue
                    
                # Calculate delta violations
                v_remove = remove_point_violations(S, p_remove)
                v_add = add_point_violations(S - {p_remove}, p_add)
                delta_v = v_add - v_remove
                new_v = current_violations + delta_v
                new_obj = objective(S - {p_remove} | {p_add}, new_v)
                
                delta_obj = new_obj - current_obj
                if delta_obj < 0 or random.random() < np.exp(-delta_obj / T):
                    S.remove(p_remove)
                    S.add(p_add)
                    current_violations = new_v
                    current_obj = new_obj
                    
                    if current_obj < best_obj:
                        best_obj = current_obj
                        best_S = set(S)
                        best_v = current_violations
                        
            T *= cooling_rate
            
        # After SA, try to greedily add points to best_S
        S = set(best_S)
        v = best_v
        # Greedy add
        candidates = [p for p in all_points if p not in S]
        random.shuffle(candidates)
        for p in candidates:
            if time.time() > deadline - 5:
                break
            delta_v = add_point_violations(S, p)
            if delta_v == 0:
                S.add(p)
                v += delta_v # v remains 0
                
        return list(S), v

    # Run multiple SA trials
    trial_count = 0
    while time.time() < deadline - 10:
        trial_count += 1
        seed_val = trial_count * 7919 + 1
        S, v = simulated_annealing(seed_val)
        
        if v == 0:
            if len(S) > best_count:
                best_points = S
                best_count = len(S)
        else:
            # Even if v > 0, maybe we can fix it?
            # For now, ignore.
            pass
            
    return best_points
