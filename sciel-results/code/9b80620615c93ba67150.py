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
    
    def build_base3_greedy():
        """Build using base-3 inspired spacing, then compress via greedy packing."""
        # Start with a base-3-like sequence: a_i = 3^i * scale
        # 3^0=1, 3^1=3, 3^2=9, 3^3=27, 3^4=81, 3^5=243, 3^6=729, 3^7=2187
        # Scale to fit in [1, 4096]: use base 3 with some compression
        # Try base-3 with digit positions, then greedily fill gaps
        
        # First, get a valid base-3 set
        base3 = []
        triple_sums = set()
        power = 1
        while power * 3 <= N:
            if time.time() > DEADLINE - 30:
                break
            ok, new_sums = try_add(power, base3, triple_sums)
            if ok:
                base3.append(power)
                triple_sums.update(new_sums)
            power *= 3
        
        # Now try to add more elements greedily between existing ones
        selected = list(base3)
        for x in range(1, N + 1):
            if time.time() > DEADLINE - 30:
                break
            if x in selected:
                continue
            ok, new_sums = try_add(x, selected, triple_sums)
            if ok:
                selected.append(x)
                triple_sums.update(new_sums)
        
        return selected
    
    def compute_triple_sums(arr):
        """Compute all unordered triple sums with repetition using NumPy."""
        m = len(arr)
        if m == 0:
            return set()
        a = np.array(arr, dtype=np.int64)
        # Generate all (i,j,k) with i<=j<=k
        # Use broadcasting for efficiency
        ii = np.repeat(np.arange(m), m * (m + 1) // 2)
        jj = np.tile(np.repeat(np.arange(m), m - np.arange(m)), m)
        # Simpler: just use nested loops but with numpy for the sum check
        sums = set()
        for i in range(m):
            for j in range(i, m):
                for k in range(j, m):
                    sums.add(int(a[i] + a[j] + a[k]))
        return sums
    
    def compute_triple_sums_fast(arr):
        """Faster computation using numpy vectorization for the inner loop."""
        m = len(arr)
        if m == 0:
            return set()
        a = np.array(arr, dtype=np.int64)
        sums = set()
        for i in range(m):
            # For fixed i, compute all a[i] + a[j] + a[k] for j>=i, k>=j
            sub = a[i:]
            n_sub = len(sub)
            # Generate all pairs (j,k) with j<=k in sub
            jj, kk = np.triu_indices(n_sub)
            pair_sums = sub[jj] + sub[kk]
            triple_sums_i = a[i] + pair_sums
            for s in triple_sums_i:
                sums.add(int(s))
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
    
    def local_search_refill(initial_set, time_budget):
        """Local search: try removing each element and greedily refilling."""
        current = list(initial_set)
        current_size = len(current)
        best = list(current)
        best_size = current_size
        
        end_time = time.time() + time_budget
        improved = True
        
        while improved and time.time() < end_time:
            improved = False
            for i in range(len(current)):
                if time.time() > end_time - 2:
                    break
                temp = current[:i] + current[i+1:]
                triple_sums = compute_triple_sums_fast(temp)
                
                new_list = list(temp)
                for x in range(1, N + 1):
                    if x in new_list:
                        continue
                    if time.time() > end_time - 2:
                        break
                    ok, new_sums = try_add(x, new_list, triple_sums)
                    if ok:
                        new_list.append(x)
                        triple_sums.update(new_sums)
                
                if len(new_list) > best_size:
                    best = new_list
                    best_size = len(best)
                    improved = True
                    break
        
        return best, best_size
    
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
                
                if len(temp) > 40:
                    continue
                temp_sums = compute_triple_sums_fast(temp)
                
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
                            current = new_current
                            current_sums = set(temp_sums) | new_sums
                            current_size = new_size
                        break
                else:
                    stuck_count += 1
                    
            else:
                # Try removing an element, then refill with up to 2 new elements
                if len(current) > 10:
                    i = random.randint(0, len(current) - 1)
                    removed = current[i]
                    temp = current[:i] + current[i+1:]
                    
                    if len(temp) > 40:
                        continue
                    temp_sums = compute_triple_sums_fast(temp)
                    
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
    
    # Phase 2: Base-3 inspired construction
    if time.time() < DEADLINE - 30:
        base3_result = build_base3_greedy()
        if len(base3_result) > best_size:
            best_result = base3_result
            best_size = len(base3_result)
    
    # Phase 3: Try various greedy strategies
    if time.time() < DEADLINE - 25:
        strategies = [
            (1, 2), (1, 3), (2, 1), (2, 2), (3, 1), (3, 3),
            (1, 4), (5, 1), (7, 1), (1, 5), (1, 6), (1, 7), (1, 8)
        ]
        for start, step in strategies:
            if time.time() > DEADLINE - 25:
                break
            candidate = build_greedy(start, step)
            if len(candidate) > best_size:
                best_result = candidate
                best_size = len(candidate)
    
    # Phase 4: Local search - remove one element, greedily refill
    if time.time() < DEADLINE - 20:
        ls_time = min(15.0, DEADLINE - 20 - time.time())
        if ls_time > 3:
            ls_result, ls_size = local_search_refill(best_result, ls_time)
            if ls_size > best_size:
                best_result = ls_result
                best_size = ls_size
    
    # Phase 5: Simulated annealing to escape local optima
    if time.time() < DEADLINE - 5:
        remaining_time = min(25.0, DEADLINE - 5 - time.time())
        if remaining_time > 3:
            current_sums = compute_triple_sums_fast(best_result)
            sa_result, sa_sums, sa_size = simulated_annealing(best_result, current_sums, remaining_time)
            if sa_size > best_size:
                best_result = sa_result
                best_size = sa_size
    
    return best_result
# EVOLVE-BLOCK-END
