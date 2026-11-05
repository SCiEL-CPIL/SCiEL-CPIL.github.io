"""Combined program: greedy + integer verification + randomized restarts + local search."""
# EVOLVE-BLOCK-START
import time
import random
from math import lcm

def construct():
    start_time = time.time()
    timeout = 58.0  # Safety margin for 60s limit
    
    universe = list(range(1, 23))
    
    # Precompute scaled reciprocal values: LCM(1..22) / x
    L = 1
    for i in range(1, 23):
        L = lcm(L, i)
    
    scaled = {}
    for x in universe:
        scaled[x] = L // x
    
    def is_dissociated(S):
        """Check if all subset sums of reciprocals of S are distinct using integer arithmetic."""
        if not S:
            return True
        sums = {0}
        for x in S:
            val = scaled[x]
            # Check collision before updating
            for s in sums:
                if (s + val) in sums:
                    return False
            # Update sums
            new_sums = [s + val for s in sums]
            sums.update(new_sums)
        return True
    
    def compute_sums(S):
        """Compute all subset sums for a set S."""
        sums = {0}
        for x in S:
            val = scaled[x]
            new_sums = [s + val for s in sums]
            sums.update(new_sums)
        return sums
    
    def try_add(S, x, sums):
        """Check if adding x to S (with current sums) maintains dissociation."""
        val = scaled[x]
        for s in sums:
            if (s + val) in sums:
                return False, sums
        new_sums = set(sums)
        for s in sums:
            new_sums.add(s + val)
        return True, new_sums
    
    best_set = []
    best_size = 0
    
    def greedy_construct(order, current_set=None, sums=None):
        """Greedy construction maintaining subset sums incrementally."""
        if current_set is None:
            current_set = []
            sums = {0}
        for x in order:
            if x in current_set:
                continue
            val = scaled[x]
            # Check collision
            collision = False
            for s in sums:
                if (s + val) in sums:
                    collision = True
                    break
            if not collision:
                current_set.append(x)
                new_sums = set(sums)
                for s in sums:
                    new_sums.add(s + val)
                sums = new_sums
        return current_set, sums
    
    def local_search(S, sums, max_iterations=100):
        """Try to improve the set by removing one element and adding two."""
        S = S[:]
        sums = set(sums)
        
        for _ in range(max_iterations):
            if time.time() - start_time > timeout:
                break
            improved = False
            # Try removing one element and adding two
            for i in range(len(S)):
                if time.time() - start_time > timeout:
                    break
                # Try removing S[i]
                x_remove = S[i]
                new_S = S[:i] + S[i+1:]
                # Recompute sums without x_remove
                new_sums = {0}
                for x in new_S:
                    val = scaled[x]
                    ns = [s + val for s in new_sums]
                    new_sums.update(ns)
                
                # Try adding two elements not in new_S
                remaining = [x for x in universe if x not in new_S]
                for j in range(len(remaining)):
                    if time.time() - start_time > timeout:
                        break
                    x1 = remaining[j]
                    ok1, sums1 = try_add(new_S, x1, new_sums)
                    if not ok1:
                        continue
                    for k in range(len(remaining)):
                        if k == j:
                            continue
                        x2 = remaining[k]
                        ok2, sums2 = try_add(new_S + [x1], x2, sums1)
                        if ok2:
                            # Found improvement: removed 1, added 2
                            S = new_S + [x1, x2]
                            sums = sums2
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                break
        return S, sums
    
    # Strategy 1: Greedy with different deterministic orders
    orders = [
        sorted(universe, reverse=True),
        sorted(universe),
        [n for n in universe if n >= 12],
        [n for n in universe if n % 2 == 0],
        [n for n in universe if n % 2 == 1],
        sorted(universe, key=lambda x: -scaled[x]),  # Largest reciprocals first
        sorted(universe, key=lambda x: scaled[x]),   # Smallest reciprocals first
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22],
        [22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
    ]
    
    for order in orders:
        if time.time() - start_time > timeout:
            break
        current_set, sums = greedy_construct(order)
        
        # Try local search
        current_set, sums = local_search(current_set, sums, max_iterations=20)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
    
    # Strategy 2: Randomized restarts with incremental sums
    random.seed(42)
    restart_count = 0
    while time.time() - start_time < timeout:
        restart_count += 1
        order = universe[:]
        random.shuffle(order)
        
        current_set, sums = greedy_construct(order)
        
        # Try local search
        current_set, sums = local_search(current_set, sums, max_iterations=10)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
            # If we found a very large set, we can stop early
            if best_size >= 18:
                break
    
    # Strategy 3: Targeted expansion from best set
    if best_set and time.time() - start_time < timeout:
        sums = compute_sums(best_set)
        remaining = [x for x in universe if x not in best_set]
        for x in remaining:
            if time.time() - start_time > timeout:
                break
            ok, new_sums = try_add(best_set, x, sums)
            if ok:
                best_set.append(x)
                sums = new_sums
    
    # Fallback to a known valid set if search fails
    if not best_set:
        best_set = [1, 2, 3, 5, 7, 11, 13, 17, 19]
    
    return best_set
# EVOLVE-BLOCK-END
