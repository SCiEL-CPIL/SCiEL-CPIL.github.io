"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import lcm, gcd
from fractions import Fraction
import random

def construct():
    start_time = time.time()
    deadline = start_time + 88  # Safety margin for 90s limit
    
    N = 256
    candidates = list(range(2, N + 1))
    
    # Precompute incompatibility using bitmasks for speed
    # compat_mask[i] is a bitmask where bit j is set if candidates[i] is compatible with candidates[j]
    idx_map = {val: i for i, val in enumerate(candidates)}
    n_cand = len(candidates)
    compat_mask = [0] * n_cand
    
    for i in range(n_cand):
        for j in range(i + 1, n_cand):
            a, b = candidates[i], candidates[j]
            if lcm(a, b) > 256:
                compat_mask[i] |= (1 << j)
                compat_mask[j] |= (1 << i)
    
    def is_compat_idx(i, j):
        if i == j:
            return True
        return (compat_mask[i] >> j) & 1
    
    def is_compat_val(a, b):
        if a == b:
            return True
        ia, ib = idx_map[a], idx_map[b]
        return is_compat_idx(ia, ib)
    
    def can_add(val, current_vals):
        """Check if val can be added to current_vals."""
        iv = idx_map[val]
        for cv in current_vals:
            if not is_compat_idx(iv, idx_map[cv]):
                return False
        return True
    
    def reciprocal_sum(subset):
        total = Fraction(0, 1)
        for x in subset:
            total += Fraction(1, x)
        return total
    
    def is_valid(subset):
        s = set(subset)
        for i, a in enumerate(subset):
            for b in subset[i+1:]:
                if not is_compat_val(a, b):
                    return False
        return True
    
    # Greedy construction with different orderings
    def greedy(order):
        selected = []
        selected_set = set()
        for x in order:
            if can_add(x, selected):
                selected.append(x)
                selected_set.add(x)
        return selected
    
    # Generate multiple initial solutions
    best_solution = None
    best_sum = Fraction(0, 1)
    
    # Strategy 1: Ascending order (small first - high reciprocal)
    sol1 = greedy(candidates)
    s1 = reciprocal_sum(sol1)
    if s1 > best_sum:
        best_sum = s1
        best_solution = sol1[:]
    
    # Strategy 2: Descending order
    sol2 = greedy(sorted(candidates, reverse=True))
    s2 = reciprocal_sum(sol2)
    if s2 > best_sum:
        best_sum = s2
        best_solution = sol2[:]
    
    # Strategy 3: By number of incompatibilities (fewest first)
    inc_count = {}
    for x in candidates:
        ix = idx_map[x]
        count = 0
        for y in candidates:
            if x != y and not is_compat_idx(ix, idx_map[y]):
                count += 1
        inc_count[x] = count
    order3 = sorted(candidates, key=lambda x: (inc_count[x], x))
    sol3 = greedy(order3)
    s3 = reciprocal_sum(sol3)
    if s3 > best_sum:
        best_sum = s3
        best_solution = sol3[:]
    
    # Strategy 4: Randomized greedy - multiple restarts
    for r in range(50):
        if time.time() > deadline - 20:
            break
        random.shuffle(candidates)
        sol_r = greedy(candidates)
        s_r = reciprocal_sum(sol_r)
        if s_r > best_sum:
            best_sum = s_r
            best_solution = sol_r[:]
    
    # Local search with bitmask-based checks
    current = best_solution[:]
    current_set = set(current)
    current_sum = best_sum
    
    def try_improve(current, current_set, current_sum):
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            
            # Try adding any compatible element not in current (prefer smallest for max reciprocal gain)
            for x in sorted(candidates, key=lambda v: 1.0/v):
                if x in current_set:
                    continue
                if can_add(x, current):
                    new_set = current + [x]
                    new_set.sort()
                    new_set_set = set(new_set)
                    new_sum = current_sum + Fraction(1, x)
                    if new_sum > current_sum:
                        current = new_set
                        current_set = new_set_set
                        current_sum = new_sum
                        improved = True
                        break
            
            # Try 1-swap: replace one element with another that gives net gain
            if not improved:
                current.sort()
                for i in range(len(current)):
                    if time.time() > deadline - 5:
                        break
                    old_elem = current[i]
                    others = [current[j] for j in range(len(current)) if j != i]
                    # Try all candidates not in current
                    for x in sorted(candidates, key=lambda v: 1.0/v):
                        if x in current_set:
                            continue
                        if can_add(x, others):
                            new_sum = current_sum - Fraction(1, old_elem) + Fraction(1, x)
                            if new_sum > current_sum:
                                new_set = current[:]
                                new_set[i] = x
                                new_set.sort()
                                new_set_set = set(new_set)
                                current = new_set
                                current_set = new_set_set
                                current_sum = new_sum
                                improved = True
                                break
                    if improved:
                        break
            
            # Try 2-swap: remove two elements, add two others
            if not improved and len(current) >= 2:
                current.sort()
                for i in range(len(current)):
                    if time.time() > deadline - 5:
                        break
                    for j in range(i + 1, len(current)):
                        if time.time() > deadline - 5:
                            break
                        old1, old2 = current[i], current[j]
                        others = [current[k] for k in range(len(current)) if k != i and k != j]
                        # Find best pair to add
                        best_pair_gain = Fraction(0, 1)
                        best_pair = None
                        for x in sorted(candidates, key=lambda v: 1.0/v):
                            if x in current_set:
                                continue
                            if not can_add(x, others):
                                continue
                            for y in sorted(candidates, key=lambda v: 1.0/v):
                                if y in current_set or y == x:
                                    continue
                                if not can_add(y, others) or not is_compat_val(x, y):
                                    continue
                                gain = Fraction(1, x) + Fraction(1, y) - Fraction(1, old1) - Fraction(1, old2)
                                if gain > best_pair_gain:
                                    best_pair_gain = gain
                                    best_pair = (x, y)
                        if best_pair and best_pair_gain > 0:
                            new_set = others + [best_pair[0], best_pair[1]]
                            new_set.sort()
                            new_set_set = set(new_set)
                            new_sum = current_sum + best_pair_gain
                            current = new_set
                            current_set = new_set_set
                            current_sum = new_sum
                            improved = True
                            break
                    if improved:
                        break
        
        return current, current_set, current_sum
    
    current, current_set, current_sum = try_improve(current, current_set, current_sum)
    if current_sum > best_sum:
        best_sum = current_sum
        best_solution = current[:]
    
    # Simulated annealing with better neighborhood
    if time.time() < deadline - 8:
        sa_start = time.time()
        sa_time_budget = deadline - 8 - sa_start
        current = best_solution[:]
        current_set = set(current)
        current_sum = best_sum
        
        def sa_neighbor(current, current_set, current_sum):
            """Generate a neighbor by adding, removing, or swapping one element."""
            r = random.random()
            if r < 0.35:
                # Add a random compatible element
                candidates_to_add = [x for x in candidates if x not in current_set and can_add(x, current)]
                if candidates_to_add:
                    x = random.choice(candidates_to_add)
                    new_set = current + [x]
                    new_set.sort()
                    return new_set, set(new_set), current_sum + Fraction(1, x)
            elif r < 0.6:
                # Remove a random element
                if len(current) > 1:
                    idx = random.randint(0, len(current) - 1)
                    removed = current[idx]
                    new_set = current[:idx] + current[idx+1:]
                    return new_set, set(new_set), current_sum - Fraction(1, removed)
            else:
                # Swap: remove one, add one compatible
                if len(current) > 0:
                    idx = random.randint(0, len(current) - 1)
                    old_elem = current[idx]
                    others = [current[j] for j in range(len(current)) if j != idx]
                    candidates_to_add = [x for x in candidates if x not in current_set and x != old_elem and can_add(x, others)]
                    if candidates_to_add:
                        x = random.choice(candidates_to_add)
                        new_set = current[:]
                        new_set[idx] = x
                        new_set.sort()
                        return new_set, set(new_set), current_sum - Fraction(1, old_elem) + Fraction(1, x)
            return current, current_set, current_sum
        
        import math
        temp = 0.05
        cooling = 0.997
        min_temp = 0.0005
        while time.time() < sa_start + sa_time_budget:
            new_set, new_set_set, new_sum = sa_neighbor(current, current_set, current_sum)
            delta = float(new_sum - current_sum)
            if delta > 0 or (temp > min_temp and random.random() < math.exp(delta / temp)):
                current = new_set
                current_set = new_set_set
                current_sum = new_sum
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_solution = current[:]
            temp *= cooling
            if temp < min_temp:
                break
    
    # Final validation
    if not is_valid(best_solution):
        # Fallback to simple greedy
        selected = []
        for x in range(2, N + 1):
            if can_add(x, selected):
                selected.append(x)
        return selected
    
    return best_solution
# EVOLVE-BLOCK-END
