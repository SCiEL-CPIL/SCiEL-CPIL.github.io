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
    
    def can_add(c, counts, A_np):
        """Check if c can be added to the set with given sum counts and array of elements."""
        if counts[2 * c] >= 2:
            return False
        if len(A_np) > 0:
            sums = c + A_np
            if np.any(counts[sums] >= 2):
                return False
        return True
    
    def add_element(c, counts, A_np):
        """Add element c, updating counts and returning new A_np."""
        counts[2 * c] += 1
        if len(A_np) > 0:
            sums = c + A_np
            counts[sums] += 1
        return np.append(A_np, c)
    
    def remove_element(c, counts, A_np):
        """Remove element c, updating counts and returning new A_np."""
        idx = np.where(A_np == c)[0][0]
        A_new = np.delete(A_np, idx)
        counts[2 * c] -= 1
        if len(A_new) > 0:
            sums = c + A_new
            counts[sums] -= 1
        return A_new
    
    def saturate(A_np, counts, deadline):
        """Try to add any valid element repeatedly until no more can be added."""
        A_set = set(A_np.tolist())
        all_candidates = np.array([x for x in range(1, max_n + 1) if x not in A_set], dtype=np.int32)
        
        while time.time() < deadline:
            if len(all_candidates) == 0:
                break
            
            # Try to find any valid candidate
            found = False
            for i in range(len(all_candidates)):
                c = all_candidates[i]
                if can_add(c, counts, A_np):
                    # Add it
                    A_np = add_element(c, counts, A_np)
                    A_set.add(int(c))
                    all_candidates = np.delete(all_candidates, i)
                    found = True
                    break
            
            if not found:
                break
        
        return A_np, counts
    
    def run_greedy(order):
        chosen = []
        counts = np.zeros(sum_size, dtype=np.int8)
        A_np = np.array([], dtype=np.int32)
        
        for c in order:
            if can_add(c, counts, A_np):
                A_np = add_element(c, counts, A_np)
        
        return A_np, counts

    # 1. Deterministic Ascending
    asc_order = list(range(1, max_n + 1))
    A_np, counts = run_greedy(asc_order)
    # Saturate
    A_np, counts = saturate(A_np, counts, deadline - 20.0)
    best_set = A_np.tolist()
    best_size = len(best_set)

    # 2. Random Restarts
    while time.time() - start_time < deadline - 20.0:  # Leave 20s for local search
        order = list(range(1, max_n + 1))
        random.shuffle(order)
        A_np, counts = run_greedy(order)
        # Saturate
        A_np, counts = saturate(A_np, counts, deadline - 20.0)
        if len(A_np) > best_size:
            best_size = len(A_np)
            best_set = A_np.tolist()

    # 3. Local Search on Best Set
    if time.time() - start_time < deadline - 5.0:
        best_set = local_search(best_set, start_time, deadline - 5.0)
        best_size = len(best_set)

    return best_set

def local_search(A, start_time, deadline):
    A_np = np.array(sorted(A), dtype=np.int32)
    A_set = set(A_np.tolist())
    max_n = 1000
    sum_size = 2 * max_n + 1
    counts = np.zeros(sum_size, dtype=np.int8)
    
    # Build initial counts
    for i in range(len(A_np)):
        a = A_np[i]
        counts[2 * a] += 1
        for j in range(i + 1, len(A_np)):
            b = A_np[j]
            counts[a + b] += 1
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        A_list = A_np.tolist()
        
        # Try Insert
        candidates = [x for x in range(1, max_n + 1) if x not in A_set]
        random.shuffle(candidates)
        
        for b in candidates:
            if time.time() > deadline: break
            # Check if b is valid
            if counts[2 * b] >= 2:
                continue
            if len(A_np) > 0:
                sums = b + A_np
                if np.any(counts[sums] >= 2):
                    continue
            
            # Valid, insert
            A_np = add_element(b, counts, A_np)
            A_set.add(b)
            improved = True
            break
        
        if improved:
            continue
            
        # Try Remove-One-Add-Two: remove a, add b1, add b2 (net +1)
        A_list = A_np.tolist()
        random.shuffle(A_list)
        
        for a in A_list:
            if time.time() > deadline: break
            
            # Temporarily remove a
            A_np_rem = remove_element(a, counts, A_np)
            A_set_rem = A_set - {a}
            
            # Find two elements to add
            candidates = [x for x in range(1, max_n + 1) if x not in A_set_rem and x != a]
            random.shuffle(candidates)
            
            found_two = False
            for b1 in candidates:
                if time.time() > deadline: break
                if counts[2 * b1] >= 2:
                    continue
                if len(A_np_rem) > 0:
                    sums = b1 + A_np_rem
                    if np.any(counts[sums] >= 2):
                        continue
                
                # b1 is valid, try to find b2
                # Temporarily add b1 to check
                A_np_temp = add_element(b1, counts, A_np_rem)
                A_set_temp = A_set_rem | {b1}
                
                for b2 in candidates:
                    if b2 == b1:
                        continue
                    if time.time() > deadline: break
                    if counts[2 * b2] >= 2:
                        continue
                    if len(A_np_temp) > 0:
                        sums = b2 + A_np_temp
                        if np.any(counts[sums] >= 2):
                            continue
                    
                    # Both valid! Accept the move
                    A_np = add_element(b2, counts, A_np_temp)
                    A_set = A_set_temp | {b2}
                    found_two = True
                    break
                
                # Restore counts for b1
                A_np_rem = remove_element(b1, counts, A_np_temp)
                
                if found_two:
                    break
            
            if not found_two:
                # Restore a
                A_np = add_element(a, counts, A_np_rem)
                A_set.add(a)
            else:
                improved = True
                break
        
        if time.time() > deadline:
            break

    return A_np.tolist()
