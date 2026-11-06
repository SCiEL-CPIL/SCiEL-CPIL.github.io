import random
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin
    best_set = []
    best_size = 0
    
    max_n = 1000
    sum_size = 2 * max_n + 1
    
    def run_greedy(order):
        chosen = []
        counts = np.zeros(sum_size, dtype=np.int8)
        
        for c in order:
            valid = True
            
            # Check 2*c
            if counts[2 * c] >= 2:
                valid = False
            
            if valid:
                # Check c + x for x in chosen
                if chosen:
                    sums = c + np.array(chosen)
                    if np.any(counts[sums] >= 2):
                        valid = False
            
            if valid:
                # Update counts
                counts[2 * c] += 1
                chosen.append(c)
                if len(chosen) > 1:
                    sums = c + np.array(chosen[:-1])
                    counts[sums] += 1
                
        return chosen

    # 1. Deterministic Ascending
    asc_order = list(range(1, max_n + 1))
    current_set = run_greedy(asc_order)
    best_set = current_set
    best_size = len(current_set)

    # 2. Random Restarts
    while time.time() - start_time < deadline - 10.0:  # Leave 10s for local search
        order = list(range(1, max_n + 1))
        random.shuffle(order)
        current_set = run_greedy(order)
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set

    # 3. Local Search on Best Set
    if time.time() - start_time < deadline - 5.0:
        best_set = local_search(best_set, start_time, deadline - 5.0)
        best_size = len(best_set)

    return best_set

def local_search(A, start_time, deadline):
    A = set(A)
    max_n = 1000
    sum_size = 2 * max_n + 1
    counts = np.zeros(sum_size, dtype=np.int8)
    
    # Build initial counts correctly
    A_list = sorted(A)
    for i in range(len(A_list)):
        a = A_list[i]
        counts[2 * a] += 1
        for j in range(i + 1, len(A_list)):
            b = A_list[j]
            counts[a + b] += 1
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        A_list = sorted(A)
        
        # Try Insert
        candidates = [x for x in range(1, max_n + 1) if x not in A]
        random.shuffle(candidates)
        
        for b in candidates:
            if time.time() > deadline: break
            # Check if b is valid
            if counts[2 * b] >= 2:
                continue
            if A:
                sums = b + np.array(A_list)
                if np.any(counts[sums] >= 2):
                    continue
            
            # Valid, insert
            A.add(b)
            A_list = sorted(A)
            counts[2 * b] += 1
            other_A = [x for x in A_list if x != b]
            if other_A:
                sums = b + np.array(other_A)
                counts[sums] += 1
            improved = True
            break
        
        if improved:
            continue
            
        # Try Swap: remove a, add b (net size same, but may enable future inserts)
        # More useful: try to find a swap that allows an immediate insert after
        # But simpler: just try swaps to see if they help
        A_list = sorted(A)
        random.shuffle(A_list)
        
        for a in A_list:
            if time.time() > deadline: break
            
            # Temporarily remove a
            A.remove(a)
            A_list_rem = [x for x in A_list if x != a]
            
            # Subtract a's contribution from counts
            counts[2 * a] -= 1
            for x in A_list_rem:
                counts[a + x] -= 1
            
            # Try to find a b to insert (this is the swap: we removed a, now try to add b)
            # If we can add b, then we've effectively swapped a for b (same size)
            # But more importantly, if removing a allows an insert, that's a net gain
            # Actually, if we just remove a and don't add anything, we lose size.
            # So we need to: remove a, add b (swap), OR remove a, add b1, add b2 (net gain of 1)
            
            # Let's try: after removing a, can we insert any element?
            # If yes, we've achieved net gain of 0 (removed 1, added 1) but the new element
            # might be better positioned for future inserts.
            # Better: try to find if removing a allows TWO inserts (net gain +1)
            
            # For simplicity, let's just try swaps: remove a, add b where b was previously blocked
            candidates = [x for x in range(1, max_n + 1) if x not in A and x != a]
            random.shuffle(candidates)
            
            found_swap = False
            for b in candidates[:50]:
                if time.time() > deadline: break
                # Check if b is valid in A (which now doesn't have a)
                if counts[2 * b] >= 2:
                    continue
                if A:
                    sums = b + np.array(list(A))
                    if np.any(counts[sums] >= 2):
                        continue
                
                # b is valid in A without a. Add it back.
                A.add(b)
                A_list = sorted(A)
                counts[2 * b] += 1
                other_A = [x for x in A_list if x != b]
                if other_A:
                    sums = b + np.array(other_A)
                    counts[sums] += 1
                found_swap = True
                break
            
            if not found_swap:
                # Restore a
                A.add(a)
                A_list = sorted(A)
                counts[2 * a] += 1
                for x in A_list:
                    if x != a:
                        counts[a + x] += 1
            else:
                improved = True
                break
        
        if time.time() > deadline:
            break

    return list(A)
