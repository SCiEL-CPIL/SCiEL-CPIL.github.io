"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd, exp
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # 1. Generate Candidate Pool
    # We want a diverse set of integer points in [-100, 100]
    # A grid is good for creating lines.
    # Let's use a grid in [-40, 40] with step 1.
    # 81 x 81 = 6561 points.
    # Pairs: ~21 million. Too slow for precomputation in Python.
    # Let's reduce pool size.
    # Grid [-20, 20] step 1 -> 41x41 = 1681 points.
    # Pairs: ~1.4 million. Feasible.
    
    pool = []
    for x in range(-20, 21):
        for y in range(-20, 21):
            pool.append((x, y))
    
    # Add some random points to break grid symmetry and add variety
    for _ in range(500):
        x = random.randint(-100, 100)
        y = random.randint(-100, 100)
        pool.append((x, y))
        
    # Remove duplicates
    pool = list(set(pool))
    random.shuffle(pool)
    
    # Limit pool size to ensure precomputation is fast
    # 2000 points -> 2M pairs. Should be okay in < 5-10s.
    pool = pool[:2000]
    n_pool = len(pool)
    
    if n_pool < 20:
        return [[i, 0] for i in range(20)] # Fallback

    # 2. Precompute Line Structures
    def get_line_key(p1, p2):
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        if dx == 0 and dy == 0:
            return None
        g = gcd(abs(dx), abs(dy))
        if g == 0:
            g = 1
        dx_n = dx // g
        dy_n = dy // g
        if dx_n < 0 or (dx_n == 0 and dy_n < 0):
            dx_n = -dx_n
            dy_n = -dy_n
        c = dy_n * p1[0] - dx_n * p1[1]
        return (dx_n, dy_n, c)

    line_map = defaultdict(set)
    point_lines = defaultdict(list)
    
    for i in range(n_pool):
        for j in range(i + 1, n_pool):
            lk = get_line_key(pool[i], pool[j])
            if lk:
                line_map[lk].add(i)
                line_map[lk].add(j)
                point_lines[i].append(lk)
                point_lines[j].append(lk)
                
    # Filter lines to only those with >= 3 points in the pool?
    # For cost calculation, we only care about lines that *could* have 3 or 4 selected points.
    # A line with only 2 points in the pool can never have 3 selected points.
    # So we can ignore lines with len < 3.
    active_lines = {lk: idxs for lk, idxs in line_map.items() if len(idxs) >= 3}
    
    # Update point_lines to only include active lines?
    # Actually, for delta calculation, we need to know which lines are affected.
    # If a line is not in active_lines, it can't contribute to cost.
    # So we can just check if the line key is in active_lines during delta calc.
    
    # 3. Initialize State
    # Random selection of 20 points
    current_indices = random.sample(range(n_pool), 20)
    current_set = set(current_indices)
    
    # line_counts: dict line_key -> count of selected points on this line
    line_counts = defaultdict(int)
    for idx in current_indices:
        for lk in point_lines[idx]:
            if lk in active_lines:
                line_counts[lk] += 1
                
    def calc_cost(line_counts):
        penalty = 0
        reward = 0
        for lk, count in line_counts.items():
            if count >= 4:
                penalty += 1
            elif count == 3:
                reward += 1
        return penalty * 10000 - reward
        
    current_cost = calc_cost(line_counts)
    
    best_cost = current_cost
    best_indices = current_indices[:]
    
    # 4. Simulated Annealing
    T = 50.0
    cooling = 0.999
    min_T = 0.01
    
    # Precompute list of active line keys for faster iteration if needed?
    # We use point_lines for delta.
    
    while time.time() < deadline and T > min_T:
        # Generate Neighbor
        # Swap one selected with one unselected
        i_out = random.choice(current_indices)
        # Pick random unselected
        # To avoid building list of unselected every time, we can just pick random and check
        # But n_pool is 2000, so list is small.
        unselected = [i for i in range(n_pool) if i not in current_set]
        if not unselected:
            break
        i_in = random.choice(unselected)
        
        # Calculate Delta Cost
        # Lines affected are those containing i_out or i_in
        # We need to update line_counts for these lines
        affected_lines = set()
        for lk in point_lines[i_out]:
            if lk in active_lines:
                affected_lines.add(lk)
        for lk in point_lines[i_in]:
            if lk in active_lines:
                affected_lines.add(lk)
                
        old_costs = {}
        new_costs = {}
        
        for lk in affected_lines:
            old_count = line_counts.get(lk, 0)
            old_costs[lk] = old_count
            
            # Simulate change
            new_count = old_count
            if i_out in active_lines[lk]:
                new_count -= 1
            if i_in in active_lines[lk]:
                new_count += 1
            new_costs[lk] = new_count
            
        delta_cost = 0
        for lk in affected_lines:
            oc = old_costs[lk]
            nc = new_costs[lk]
            
            # Cost contribution of a line:
            # If count >= 4: 10000
            # If count == 3: -1
            # Else: 0
            
            def line_cost(c):
                if c >= 4:
                    return 10000
                elif c == 3:
                    return -1
                else:
                    return 0
                    
            delta_cost += line_cost(nc) - line_cost(oc)
            
        new_cost = current_cost + delta_cost
        
        # Acceptance
        if new_cost < current_cost or (T > 0 and random.random() < exp(-delta_cost / T)):
            # Accept
            # Update state
            current_indices[current_indices.index(i_out)] = i_in
            current_set.remove(i_out)
            current_set.add(i_in)
            
            for lk in affected_lines:
                line_counts[lk] = new_costs[lk]
                
            current_cost = new_cost
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_indices = current_indices[:]
                
        T *= cooling
        
    # 5. Final Verification and Repair
    # The best solution might still have 4-collinear points if the penalty wasn't high enough 
    # or if we accepted a bad move during high T.
    # We must ensure the returned set is valid (no 4 collinear).
    
    final_indices = best_indices[:]
    final_set = set(final_indices)
    
    # Check validity
    def is_valid(indices):
        idx_set = set(indices)
        for lk, idxs in active_lines.items():
            count = 0
            for i in idxs:
                if i in idx_set:
                    count += 1
            if count >= 4:
                return False
        return True
        
    if not is_valid(final_indices):
        # Repair: Move points that are part of 4-collinear violations
        # Simple heuristic: Find a bad line, move one of its points to a random new spot
        for _ in range(100): # Max repair attempts
            if is_valid(final_indices):
                break
                
            # Find a bad line
            bad_line = None
            bad_idx = None
            idx_set = set(final_indices)
            for lk, idxs in active_lines.items():
                count = 0
                on_line = []
                for i in idxs:
                    if i in idx_set:
                        count += 1
                        on_line.append(i)
                if count >= 4:
                    bad_line = lk
                    bad_idx = random.choice(on_line)
                    break
                    
            if bad_idx is None:
                break
                
            # Move bad_idx to a random unselected point
            unselected = [i for i in range(n_pool) if i not in idx_set]
            if not unselected:
                break
            new_idx = random.choice(unselected)
            
            final_indices[final_indices.index(bad_idx)] = new_idx
            final_set.remove(bad_idx)
            final_set.add(new_idx)
            
        if not is_valid(final_indices):
            # Fallback to a known valid structure: Parabola
            # y = x^2 for x in -10..9
            return [[x, x*x] for x in range(-10, 10)]
            
    return [[pool[i][0], pool[i][1]] for i in final_indices]
# EVOLVE-BLOCK-END
