import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    def validate(sums, max_k):
        """Check if all cardinality sums are pairwise disjoint."""
        for k in range(max_k + 1):
            for j in range(k + 1, max_k + 1):
                if sums[k] & sums[j]:
                    return False
        return True
    
    def try_add(sums, x, max_k):
        """
        Try to add x. sums is a list of Python ints (bitsets).
        Returns new sums list or None.
        """
        # A[k] = sums[k-1] << x for k=1..max_k+1
        # Check: A[k] & sums[j] == 0 for k != j
        # Check: A[k] & A[j] == 0 for k != j
        
        new_sums = [0] * (max_k + 2)
        new_sums[0] = sums[0]
        
        # Compute A
        A = [0] * (max_k + 2)
        for k in range(1, max_k + 2):
            A[k] = sums[k - 1] << x
        
        # Check A[k] against old sums[j] for j != k
        for k in range(1, max_k + 2):
            if A[k] == 0:
                continue
            for j in range(0, max_k + 1):
                if j == k:
                    continue
                if sums[j] == 0:
                    continue
                if (A[k] & sums[j]) != 0:
                    return None
        
        # Check A[k] against A[j] for k < j
        for k in range(1, max_k + 2):
            if A[k] == 0:
                continue
            for j in range(k + 1, max_k + 2):
                if A[j] == 0:
                    continue
                if (A[k] & A[j]) != 0:
                    return None
        
        # Build new sums
        for k in range(1, max_k + 2):
            if k <= max_k:
                new_sums[k] = sums[k] | A[k]
            else:
                new_sums[k] = A[k]
        
        return new_sums
    
    def greedy_with_order(order, deadline):
        """Greedy: try each x in order, add if valid."""
        sums = [0] * 61
        sums[0] = 1  # empty subset sum = 0
        chosen = []
        max_k = 0
        used = set()
        
        for x in order:
            if time.time() > deadline:
                break
            if x in used:
                continue
            new_sums = try_add(sums, x, max_k)
            if new_sums is not None:
                sums = new_sums
                max_k += 1
                chosen.append(x)
                used.add(x)
        
        return chosen
    
    def local_search(initial_set, deadline):
        """
        Start from initial_set, try to improve by:
        1. Adding new elements
        2. Swapping elements
        3. Removing and re-adding
        """
        if not initial_set:
            return []
        
        sums = [0] * 61
        sums[0] = 1
        max_k = 0
        used = set()
        
        for x in initial_set:
            new_sums = try_add(sums, x, max_k)
            if new_sums is not None:
                sums = new_sums
                max_k += 1
                used.add(x)
        
        best_set = list(initial_set)
        best_sums = sums
        best_max_k = max_k
        best_used = used
        
        # Try adding elements
        for x in range(1, 257):
            if time.time() > deadline:
                break
            if x in used:
                continue
            new_sums = try_add(sums, x, max_k)
            if new_sums is not None:
                sums = new_sums
                max_k += 1
                used.add(x)
                best_set = list(used)
                best_sums = sums
                best_max_k = max_k
                best_used = used
        
        # Try swapping: remove one, add another
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(best_set)):
                if time.time() > deadline:
                    break
                # Try removing best_set[i] and adding a new element
                removed = best_set[i]
                remaining = [x for x in best_set if x != removed]
                
                # Rebuild sums without removed
                sums2 = [0] * 61
                sums2[0] = 1
                max_k2 = 0
                used2 = set()
                for x in remaining:
                    new_sums2 = try_add(sums2, x, max_k2)
                    if new_sums2 is not None:
                        sums2 = new_sums2
                        max_k2 += 1
                        used2.add(x)
                
                # Try adding each unused element
                for x in range(1, 257):
                    if time.time() > deadline:
                        break
                    if x in used2:
                        continue
                    new_sums3 = try_add(sums2, x, max_k2)
                    if new_sums3 is not None:
                        # We have a valid set of same size with different composition
                        # Check if it allows further additions
                        sums4 = new_sums3
                        max_k4 = max_k2 + 1
                        used4 = used2 | {x}
                        
                        # Try to add more
                        for y in range(1, 257):
                            if time.time() > deadline:
                                break
                            if y in used4:
                                continue
                            new_sums5 = try_add(sums4, y, max_k4)
                            if new_sums5 is not None:
                                # Improved!
                                best_set = list(used4 | {y})
                                best_sums = new_sums5
                                best_max_k = max_k4 + 1
                                best_used = used4 | {y}
                                improved = True
                                break
                        
                        if improved:
                            break
                
                if improved:
                    break
        
        return best_set
    
    # Strategy:
    # 1. Start with known good set (powers of 2, size 9)
    # 2. Use greedy with various orders
    # 3. Use local search to improve
    
    # Known good: powers of 2
    pow2 = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    best_set = pow2[:]
    
    # Verify pow2
    s = [0] * 10
    s[0] = 1
    for v in pow2:
        ns = [0] * 10
        for k in range(9, 0, -1):
            ns[k] = s[k] | (s[k - 1] << v)
        ns[0] = s[0]
        s = ns
    valid = True
    for k in range(10):
        for j in range(k + 1, 10):
            if s[k] & s[j]:
                valid = False
                break
        if not valid:
            break
    if not valid:
        # Shouldn't happen, but fallback
        best_set = [1, 2, 4, 8, 16, 32, 64, 128]
    
    # Try greedy with various orders
    rng = random.Random(12345)
    
    # Order 1: ascending
    orders = [
        list(range(1, 257)),
        list(range(256, 0, -1)),
    ]
    
    # Order 2: start with small numbers, then fill in
    small_first = list(range(1, 30)) + [x for x in range(30, 257) if x not in set(range(1, 30))]
    orders.append(small_first)
    
    # Random orders
    for _ in range(200):
        o = list(range(1, 257))
        rng.shuffle(o)
        orders.append(o)
    
    for o in orders:
        if time.time() > deadline - 5:
            break
        res = greedy_with_order(o, deadline - 5)
        if len(res) > len(best_set):
            best_set = res
    
    # Local search from best set
    if time.time() < deadline - 10:
        improved = local_search(best_set, deadline - 10)
        if len(improved) > len(best_set):
            best_set = improved
    
    # More local search iterations
    while time.time() < deadline - 5:
        improved = local_search(best_set, deadline - 5)
        if len(improved) > len(best_set):
            best_set = improved
        else:
            break
    
    # Final verification
    final_sums = [0] * (len(best_set) + 1)
    final_sums[0] = 1
    for x in best_set:
        ns = [0] * (len(best_set) + 1)
        for k in range(len(best_set), 0, -1):
            ns[k] = final_sums[k] | (final_sums[k - 1] << x)
        ns[0] = final_sums[0]
        final_sums = ns
    
    valid_final = True
    for k in range(len(best_set) + 1):
        for j in range(k + 1, len(best_set) + 1):
            if final_sums[k] & final_sums[j]:
                valid_final = False
                break
        if not valid_final:
            break
    
    if not valid_final:
        # Fallback to powers of 2
        best_set = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    
    return list(best_set)

construct()
