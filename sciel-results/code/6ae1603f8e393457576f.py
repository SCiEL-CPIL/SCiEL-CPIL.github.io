"""Optimized program: backtracking with pruning + timeout guard to maximize sequence length."""
import time
import sys
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s kill

    # Greedy baseline (fast, gives a valid starting point)
    def greedy():
        values, sums = [], set()
        for x in range(1, 257):
            tails, total = [x], x
            for a in reversed(values):
                total += a
                tails.append(total)
            if not sums.intersection(tails):
                values.append(x)
                sums.update(tails)
        return values

    best = greedy()
    best_len = len(best)

    # Backtracking search to try to beat the greedy result
    def check_add(values, sums, x):
        """Check if x can be appended; return new sums set or None."""
        total = x
        new_sums = []
        new_sums.append(total)
        for a in reversed(values):
            total += a
            new_sums.append(total)
        for s in new_sums:
            if s in sums:
                return None
        return new_sums

    # Backtracking: try to build a sequence of target length
    def backtrack(target_len, values, sums, start_idx, depth):
        if time.time() > deadline:
            return None
        if len(values) == target_len:
            return list(values)
        lower = values[-1] + 1 if values else 1
        for x in range(lower, 257):
            if time.time() > deadline:
                return None
            new_sums = check_add(values, sums, x)
            if new_sums is None:
                continue
            values.append(x)
            sums_copy = sums | set(new_sums)
            result = backtrack(target_len, values, sums_copy, x + 1, depth + 1)
            if result is not None:
                return result
            values.pop()
        return None

    # Try to find longer sequences via backtracking
    for target_len in range(best_len + 1, best_len + 15):
        if time.time() > deadline:
            break
        sub_deadline = min(deadline, time.time() + 5)

        def bt(target, vals, sm, lo):
            if time.time() > sub_deadline:
                return None
            if len(vals) == target:
                return list(vals)
            for x in range(lo, 257):
                if time.time() > sub_deadline:
                    return None
                ns = check_add(vals, sm, x)
                if ns is None:
                    continue
                vals.append(x)
                sm2 = sm | set(ns)
                res = bt(target, vals, sm2, x + 1)
                if res is not None:
                    return res
                vals.pop()
            return None

        result = bt(target_len, [], set(), 1)
        if result is not None and len(result) > best_len:
            best = result
            best_len = len(result)

    # Simulated annealing on difference vectors to find longer valid sequences
    def count_collisions(values):
        """Count the number of duplicate block sums. 0 means valid."""
        if not values:
            return 0
        n = len(values)
        sums = []
        for i in range(n):
            s = 0
            for j in range(i, n):
                s += values[j]
                sums.append(s)
        # Count duplicates
        seen = set()
        collisions = 0
        for s in sums:
            if s in seen:
                collisions += 1
            else:
                seen.add(s)
        return collisions

    def is_valid(values):
        return count_collisions(values) == 0

    def sa_search(initial_seq, max_iters=50000, t_start=10.0, t_end=0.01):
        """Simulated annealing on difference vectors.
        
        State: list of values (strictly increasing, in [1,256])
        Objective: minimize collision count, then maximize length
        Moves: 
          1. Perturb a difference (change one gap)
          2. Try to extend by adding a new element at the end
          3. Try to remove an element
        """
        random.seed(42)
        current = list(initial_seq)
        current_cost = count_collisions(current)
        best_found = list(current) if current_cost == 0 else []
        best_found_len = len(best_found)
        
        t = t_start
        cooling = (t_end / t_start) ** (1.0 / max(1, max_iters - 1))
        
        for it in range(max_iters):
            if time.time() > deadline - 2:
                break
            
            t *= cooling
            
            # Choose a move type
            move_type = random.random()
            
            new_seq = None
            
            if move_type < 0.5:
                # Perturb a difference: change one gap
                if len(current) >= 2:
                    idx = random.randint(0, len(current) - 2)
                    old_gap = current[idx + 1] - current[idx]
                    # Try to change the gap by a small amount
                    delta = random.randint(-3, 3)
                    if delta == 0:
                        continue
                    new_gap = old_gap + delta
                    if new_gap < 1:
                        continue
                    # Rebuild from idx+1 onward
                    new_val = current[idx] + new_gap
                    if new_val > 256:
                        continue
                    new_seq = list(current[:idx + 1])
                    for k in range(idx + 1, len(current)):
                        old_gap_k = current[k + 1] - current[k] if k + 1 < len(current) else 0
                        if k == idx:
                            nv = new_val
                        else:
                            nv = new_seq[-1] + (current[k + 1] - current[k])
                        if nv > 256:
                            new_seq = None
                            break
                        new_seq.append(nv)
                    if new_seq is None or len(new_seq) != len(current):
                        continue
                    # Check strict increasing
                    for k in range(len(new_seq) - 1):
                        if new_seq[k + 1] <= new_seq[k]:
                            new_seq = None
                            break
                    if new_seq is None:
                        continue
                    
            elif move_type < 0.8:
                # Try to extend by adding a new element
                if len(current) < 20:
                    # Try adding a random value greater than last
                    if current:
                        lo = current[-1] + 1
                    else:
                        lo = 1
                    if lo <= 256:
                        new_val = random.randint(lo, min(256, lo + 20))
                        new_seq = list(current) + [new_val]
                        
            else:
                # Try to remove a random element (to allow re-optimization)
                if len(current) > 3:
                    idx = random.randint(0, len(current) - 1)
                    new_seq = list(current[:idx]) + list(current[idx + 1:])
                    # Ensure still strictly increasing
                    for k in range(len(new_seq) - 1):
                        if new_seq[k + 1] <= new_seq[k]:
                            new_seq = None
                            break
                    if new_seq is None:
                        continue
            
            if new_seq is None:
                continue
            
            new_cost = count_collisions(new_seq)
            
            # Accept if better or with probability exp(-delta/t)
            if new_cost < current_cost:
                current = new_seq
                current_cost = new_cost
                if current_cost == 0 and len(current) > best_found_len:
                    best_found = list(current)
                    best_found_len = len(current)
            else:
                # Accept worse with some probability
                delta = new_cost - current_cost
                if delta == 0:
                    # Same cost, accept if longer
                    if len(new_seq) > len(current):
                        current = new_seq
                        current_cost = new_cost
                        if current_cost == 0 and len(current) > best_found_len:
                            best_found = list(current)
                            best_found_len = len(current)
                else:
                    import math
                    if t > 0:
                        prob = math.exp(-delta / t)
                        if random.random() < prob:
                            current = new_seq
                            current_cost = new_cost
        
        return best_found

    # Run SA from the greedy solution and from modified versions
    if time.time() < deadline - 5:
        # SA from greedy
        sa_result = sa_search(best, max_iters=30000, t_start=5.0, t_end=0.001)
        if len(sa_result) > best_len:
            best = sa_result
            best_len = len(sa_result)
    
    # SA from a shorter prefix of greedy (to explore different regions)
    if time.time() < deadline - 5:
        for prefix_len in [best_len - 3, best_len - 5]:
            if prefix_len < 3:
                continue
            prefix = best[:prefix_len]
            sa_result = sa_search(prefix, max_iters=20000, t_start=8.0, t_end=0.005)
            if len(sa_result) > best_len:
                best = sa_result
                best_len = len(sa_result)
            if time.time() > deadline - 5:
                break
    
    # Final backtracking attempt with more time if we have room
    if time.time() < deadline - 3:
        for target_len in range(best_len + 1, best_len + 5):
            if time.time() > deadline - 1:
                break
            sub_deadline = min(deadline - 1, time.time() + 3)

            def bt2(target, vals, sm, lo):
                if time.time() > sub_deadline:
                    return None
                if len(vals) == target:
                    return list(vals)
                for x in range(lo, 257):
                    if time.time() > sub_deadline:
                        return None
                    ns = check_add(vals, sm, x)
                    if ns is None:
                        continue
                    vals.append(x)
                    sm2 = sm | set(ns)
                    res = bt2(target, vals, sm2, x + 1)
                    if res is not None:
                        return res
                    vals.pop()
                return None

            result = bt2(target_len, [], set(), 1)
            if result is not None and len(result) > best_len:
                best = result
                best_len = len(result)
                break

    return best
# EVOLVE-BLOCK-END
