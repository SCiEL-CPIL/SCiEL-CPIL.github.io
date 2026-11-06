import time
import random
import math

def construct():
    start = time.time()
    deadline = start + 50.0  # Safety margin
    
    TARGET = 200
    MAX_ELE = 100
    MASK = (1 << (TARGET + 1)) - 1
    TARGET_BIT = 1 << TARGET
    
    def can_add(bitset, x):
        """Check if adding element x would create a subset sum of 200."""
        return not ((bitset << x) & TARGET_BIT)
    
    def add_element(bitset, x):
        """Add element x to the reachable sums bitset."""
        return bitset | ((bitset << x) & MASK)
    
    def compute_bitset(elems):
        """Compute the reachable sums bitset for a list of elements."""
        bs = 1
        for x in elems:
            bs = add_element(bs, x)
        return bs
    
    def greedy_from(start_list, candidates):
        """
        Start with start_list, try to add elements from candidates if possible.
        Returns (list, bitset, size).
        """
        cur = list(start_list)
        cur_set = set(cur)
        bitset = 1
        for x in cur:
            bitset = add_element(bitset, x)
            
        for x in candidates:
            if x in cur_set:
                continue
            if can_add(bitset, x):
                cur.append(x)
                cur_set.add(x)
                bitset = add_element(bitset, x)
                
        return cur, bitset, len(cur)

    best_set = []
    best_len = 0
    best_bitset = 1

    # Strategy 1: Greedy Ascending (1 to 100)
    s1, b1, l1 = greedy_from([], range(1, MAX_ELE + 1))
    if l1 > best_len:
        best_set, best_len, best_bitset = s1, l1, b1
        
    # Strategy 2: Greedy Descending (100 to 1)
    s2, b2, l2 = greedy_from([], range(MAX_ELE, 0, -1))
    if l2 > best_len:
        best_set, best_len, best_bitset = s2, l2, b2
        
    # Strategy 3: Start with known good interval [66, 100], add smaller
    base_interval = list(range(66, MAX_ELE + 1))
    s3, b3, l3 = greedy_from(base_interval, range(65, 0, -1))
    if l3 > best_len:
        best_set, best_len, best_bitset = s3, l3, b3
        
    # Strategy 4: Start with known good interval [66, 100], add smaller (ascending)
    s4, b4, l4 = greedy_from(base_interval, range(1, 66))
    if l4 > best_len:
        best_set, best_len, best_bitset = s4, l4, b4
        
    # Strategy 5: Randomized Greedy
    rng = random.Random(42)
    for _ in range(200):
        if time.time() > deadline - 5.0:
            break
        candidates = list(range(1, MAX_ELE + 1))
        rng.shuffle(candidates)
        s5, b5, l5 = greedy_from([], candidates)
        if l5 > best_len:
            best_set, best_len, best_bitset = s5, l5, b5

    # Strategy 6: Local Search on Best Set (Simple Swap)
    cur = list(best_set)
    cur_set = set(cur)
    bitset = best_bitset
    
    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        # Rebuild bitset to be safe
        bitset = 1
        for x in cur:
            bitset = add_element(bitset, x)
            
        n = len(cur)
        for i in range(n):
            if time.time() > deadline - 5.0:
                break
            removed = cur[i]
            
            # Compute bitset without cur[i]
            temp_cur = cur[:i] + cur[i+1:]
            temp_bitset = 1
            for x in temp_cur:
                temp_bitset = add_element(temp_bitset, x)
                
            # Try to add any x not in cur
            for x in range(1, MAX_ELE + 1):
                if x in cur_set:
                    continue
                if can_add(temp_bitset, x):
                    # Swap cur[i] with x
                    cur[i] = x
                    cur_set.discard(removed)
                    cur_set.add(x)
                    improved = True
                    break
            if improved:
                break
                
    if len(cur) > best_len:
        best_set = cur
        best_len = len(cur)
        best_bitset = compute_bitset(cur)

    # Strategy 7: Simulated Annealing to increase cardinality
    # Start from best_set, try to remove one element and add two to increase size by 1
    cur = list(best_set)
    cur_set = set(cur)
    cur_bitset = best_bitset
    cur_size = len(cur)
    
    # Temperature schedule
    T0 = 1.0
    T_min = 0.01
    cooling_rate = 0.95
    
    best_sa_size = cur_size
    best_sa_set = list(cur)
    
    T = T0
    max_iterations = 50000
    
    for iteration in range(max_iterations):
        if time.time() > deadline - 2.0:
            break
        
        # Try to remove one element and add two new elements
        # This would increase cardinality by 1
        if cur_size < 100:
            # Pick a random element to remove
            idx_remove = rng.randint(0, cur_size - 1)
            removed_elem = cur[idx_remove]
            
            # Compute bitset without removed_elem
            temp_list = cur[:idx_remove] + cur[idx_remove+1:]
            temp_set = cur_set - {removed_elem}
            temp_bitset = compute_bitset(temp_list)
            
            # Try to find two elements that can be added
            added = []
            for x1 in range(1, MAX_ELE + 1):
                if x1 in temp_set or x1 == removed_elem:
                    continue
                if can_add(temp_bitset, x1):
                    # Try to add x1, then look for x2
                    new_bitset_after_x1 = add_element(temp_bitset, x1)
                    for x2 in range(1, MAX_ELE + 1):
                        if x2 in temp_set or x2 == removed_elem or x2 == x1:
                            continue
                        if can_add(new_bitset_after_x1, x2):
                            # Both can be added
                            added = [x1, x2]
                            break
                    if added:
                        break
            
            if len(added) == 2:
                # Accept the move: remove one, add two
                cur.remove(removed_elem)
                cur_set.discard(removed_elem)
                for x in added:
                    cur.append(x)
                    cur_set.add(x)
                cur_size += 1
                cur_bitset = compute_bitset(cur)
                
                if cur_size > best_sa_size:
                    best_sa_size = cur_size
                    best_sa_set = list(cur)
                
                # Reset temperature on improvement
                T = T0
            else:
                # No valid move found; try a worse move with probability
                # This allows escaping local optima
                if rng.random() < math.exp(-1.0 / T):
                    # Accept a random removal without replacement (worsening)
                    # But this decreases size, so we only do this rarely
                    pass
        
        # Cool down
        T = max(T * cooling_rate, T_min)
        
        # If we've reached a good size, try to polish
        if best_sa_size > best_len:
            best_set = best_sa_set
            best_len = best_sa_size
            best_bitset = compute_bitset(best_set)
            # Reset temperature for more exploration
            T = T0 * 0.5
    
    # Final polish: try to add any remaining elements
    final_bitset = compute_bitset(best_set)
    final_set = set(best_set)
    for x in range(1, MAX_ELE + 1):
        if x not in final_set and can_add(final_bitset, x):
            best_set.append(x)
            final_set.add(x)
            final_bitset = add_element(final_bitset, x)
            best_len += 1
    
    return best_set if best_set else [MAX_ELE]
