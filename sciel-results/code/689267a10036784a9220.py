"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import random

def construct():
    t_start = time.time()
    DEADLINE = t_start + 85.0
    
    N = 4096
    
    def try_add(x, selected, triple_sums):
        """Check if x can be added. Returns (ok, new_sums_set)."""
        n = len(selected)
        new_sums = set()
        
        # Triples with one x: x + a_i + a_j for i <= j
        for i in range(n):
            ai = selected[i]
            for j in range(i, n):
                s = x + ai + selected[j]
                if s in new_sums or s in triple_sums:
                    return False, set()
                new_sums.add(s)
        
        # Triples with two x's: 2*x + a_i
        for i in range(n):
            s = 2 * x + selected[i]
            if s in new_sums or s in triple_sums:
                return False, set()
            new_sums.add(s)
        
        # Triple with three x's
        s = 3 * x
        if s in new_sums or s in triple_sums:
            return False, set()
        new_sums.add(s)
        
        return True, new_sums
    
    def build_greedy(start=1, step=1):
        """Greedy: add numbers in order with given start and step."""
        selected = []
        triple_sums = set()
        
        for x in range(start, N + 1, step):
            if time.time() > DEADLINE - 15:
                break
            ok, new_sums = try_add(x, selected, triple_sums)
            if ok:
                selected.append(x)
                triple_sums.update(new_sums)
        
        return selected
    
    def compute_triple_sums(arr):
        """Compute all unordered triple sums with repetition."""
        m = len(arr)
        sums = set()
        for i in range(m):
            for j in range(i, m):
                for k in range(j, m):
                    sums.add(arr[i] + arr[j] + arr[k])
        return sums
    
    def estimate_collision_density(x, selected, triple_sums):
        """Estimate how many triple sums x would collide with. Lower is better."""
        n = len(selected)
        collisions = 0
        # Sample a subset for speed
        if n > 30:
            indices = random.sample(range(n), min(n, 30))
        else:
            indices = list(range(n))
        
        for i in indices:
            ai = selected[i]
            for j in range(i, n):
                s = x + ai + selected[j]
                if s in triple_sums:
                    collisions += 1
                if collisions > 50:
                    return collisions
            s = 2 * x + ai
            if s in triple_sums:
                collisions += 1
        s = 3 * x
        if s in triple_sums:
            collisions += 1
        return collisions
    
    def simulated_annealing(initial_set, triple_sums, time_budget):
        """Simulated annealing to maximize cardinality."""
        current = list(initial_set)
        current_sums = set(triple_sums)
        current_size = len(current)
        best = list(current)
        best_size = current_size
        best_sums = set(current_sums)
        
        T = 50.0
        cooling = 0.995
        T_min = 0.5
        
        end_time = time.time() + time_budget
        stuck_count = 0
        
        while time.time() < end_time and T > T_min:
            T *= cooling
            move_type = random.random()
            
            if move_type < 0.4:
                # Try adding a random element
                x = random.randint(1, N)
                if x not in current:
                    ok, new_sums = try_add(x, current, current_sums)
                    if ok:
                        new_current = current + [x]
                        new_size = len(new_current)
                        if new_size > best_size:
                            best = list(new_current)
                            best_size = new_size
                            best_sums = set(current_sums) | new_sums
                            current = new_current
                            current_sums = best_sums
                            current_size = new_size
                            stuck_count = 0
                        else:
                            # Accept with probability exp(-delta/T) where delta is negative (improvement)
                            # Since we only accept if ok, this is always an improvement in validity
                            # But size doesn't change, so no gain. Skip.
                            pass
                    else:
                        stuck_count += 1
                else:
                    stuck_count += 1
                    
            elif move_type < 0.7:
                # Try swapping: remove one, add another
                if len(current) < 2:
                    continue
                i = random.randint(0, len(current) - 1)
                removed = current[i]
                temp = current[:i] + current[i+1:]
                
                # Compute triple sums after removal
                # We need to recompute or track. For simplicity, recompute.
                if len(temp) > 40:
                    continue  # Too expensive
                temp_sums = compute_triple_sums(temp)
                
                # Try to add a better element
                candidates = []
                for x in random.sample(range(1, N), min(100, N)):
                    if x not in temp:
                        coll = estimate_collision_density(x, temp, temp_sums)
                        candidates.append((coll, x))
                candidates.sort()
                
                for coll, x in candidates[:5]:
                    ok, new_sums = try_add(x, temp, temp_sums)
                    if ok:
                        new_current = temp + [x]
                        new_size = len(new_current)
                        if new_size > best_size:
                            best = list(new_current)
                            best_size = new_size
                            best_sums = set(temp_sums) | new_sums
                            current = new_current
                            current_sums = best_sums
                            current_size = new_size
                            stuck_count = 0
                        elif new_size == current_size and random.random() < np.exp(-0.5 / T):
                            # Accept non-improving move with probability
                            current = new_current
                            current_sums = set(temp_sums) | new_sums
                            current_size = new_size
                        break
                else:
                    stuck_count += 1
                    
            else:
                # Try removing an element that causes many collisions, then refill
                if len(current) > 10:
                    # Find the element with highest "cost" (most collisions it causes)
                    # Approximate: remove random element, see if we can add two back
                    i = random.randint(0, len(current) - 1)
                    removed = current[i]
                    temp = current[:i] + current[i+1:]
                    
                    if len(temp) > 40:
                        continue
                    temp_sums = compute_triple_sums(temp)
                    
                    # Try to add two elements
                    added = 0
                    for x in random.sample(range(1, N), min(200, N)):
                        if x not in temp and x != removed:
                            ok, new_sums = try_add(x, temp, temp_sums)
                            if ok:
                                temp.append(x)
                                temp_sums.update(new_sums)
                                added += 1
                                if added >= 2:
                                    break
                        if time.time() > end_time:
                            break
                    
                    new_size = len(temp)
                    if new_size > best_size:
                        best = list(temp)
                        best_size = new_size
                        best_sums = set(temp_sums)
                        current = temp
                        current_sums = best_sums
                        current_size = new_size
                        stuck_count = 0
                    elif new_size == current_size and added >= 1 and random.random() < np.exp(-1.0 / T):
                        current = temp
                        current_sums = set(temp_sums)
                        current_size = new_size
                    else:
                        stuck_count += 1
            
            if stuck_count > 200:
                T = max(T * 0.9, T_min)
                stuck_count = 0
        
        return best, best_sums, best_size
    
    # Phase 1: Basic greedy from 1
    result = build_greedy(1, 1)
    best_size = len(result)
    best_result = list(result)
    
    # Phase 2: Try various greedy strategies
    if time.time() < DEADLINE - 20:
        strategies = [
            (1, 2), (1, 3), (2, 1), (2, 2), (3, 1), (3, 3),
            (1, 4), (5, 1), (7, 1), (1, 5), (1, 6), (1, 7), (1, 8)
        ]
        for start, step in strategies:
            if time.time() > DEADLINE - 20:
                break
            candidate = build_greedy(start, step)
            if len(candidate) > best_size:
                best_result = candidate
                best_size = len(candidate)
    
    # Phase 3: Local search - remove one element, greedily refill
    if time.time() < DEADLINE - 25:
        for _ in range(5):
            if time.time() > DEADLINE - 25:
                break
            improved = False
            for i in range(len(best_result)):
                if time.time() > DEADLINE - 25:
                    break
                temp = best_result[:i] + best_result[i+1:]
                triple_sums = compute_triple_sums(temp)
                
                new_list = list(temp)
                for x in range(1, N + 1):
                    if x in new_list:
                        continue
                    if time.time() > DEADLINE - 25:
                        break
                    ok, new_sums = try_add(x, new_list, triple_sums)
                    if ok:
                        new_list.append(x)
                        triple_sums.update(new_sums)
                
                if len(new_list) > best_size:
                    best_result = new_list
                    best_size = len(best_result)
                    improved = True
                    break
    
    # Phase 4: Simulated annealing to escape local optima
    if time.time() < DEADLINE - 5:
        remaining_time = min(30.0, DEADLINE - 5 - time.time())
        if remaining_time > 3:
            # Get triple sums for current best
            current_sums = compute_triple_sums(best_result)
            sa_result, sa_sums, sa_size = simulated_annealing(best_result, current_sums, remaining_time)
            if sa_size > best_size:
                best_result = sa_result
                best_size = sa_size
    
    return best_result
# EVOLVE-BLOCK-END
