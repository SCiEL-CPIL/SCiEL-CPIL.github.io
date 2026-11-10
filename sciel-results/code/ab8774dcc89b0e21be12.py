import time
import random
import numpy as np

def construct():
    N = 512
    start_time = time.time()
    deadline = start_time + 58.0  # Safety margin before 60s limit

    # Precompute all 4-term arithmetic progressions
    all_aps = []
    max_d = (N - 1) // 3
    for d in range(1, max_d + 1):
        max_a = N - 3 * d
        for a in range(1, max_a + 1):
            all_aps.append((a, a + d, a + 2 * d, a + 3 * d))
    
    num_aps = len(all_aps)
    
    # Map each number to the indices of APs it belongs to
    ap_indices_by_num = {i: [] for i in range(1, N + 1)}
    for idx, ap in enumerate(all_aps):
        for x in ap:
            ap_indices_by_num[x].append(idx)
    
    # Precompute for each number, the list of "other three" tuples for each AP it's in
    ap_other_elements = {i: [] for i in range(1, N + 1)}
    for idx, ap in enumerate(all_aps):
        for pos, x in enumerate(ap):
            others = [ap[j] for j in range(4) if j != pos]
            ap_other_elements[x].append(others)
    
    # Use a bitset for the current set to speed up membership checks
    # We'll use a Python list of 0/1 for simplicity, or a set for O(1) but maybe slower due to hashing
    # Actually, for N=512, a simple boolean list is fastest.
    # Let's use a list `in_set` where in_set[x] is True/False.
    
    def make_set_from_list(lst):
        s = [False] * (N + 1)
        for x in lst:
            s[x] = True
        return s

    def is_valid_set(S):
        """Check if set S (list of bools or set) contains any 4-AP"""
        if isinstance(S, list):
            in_set = S
        else:
            in_set = make_set_from_list(S)
        for ap in all_aps:
            if all(in_set[x] for x in ap):
                return False
        return True

    def can_add_element(x, in_set):
        """Check if x can be added to set (in_set list) without creating a 4-AP"""
        for others in ap_other_elements[x]:
            if in_set[others[0]] and in_set[others[1]] and in_set[others[2]]:
                return False
        return True

    def greedy_with_order(order, S_init_list):
        """Greedy add elements in the given order to S_init (list of bools)"""
        in_set = S_init_list[:]
        for x in order:
            if not in_set[x] and can_add_element(x, in_set):
                in_set[x] = True
        return in_set

    def list_to_set(lst):
        return [i for i in range(1, N+1) if lst[i]]

    # Base-4 {0,1} construction
    base4_list = [False] * (N + 1)
    for i in range(1, N + 1):
        n = i
        valid = True
        while n > 0:
            if n % 4 > 1:
                valid = False
                break
            n //= 4
        if valid:
            base4_list[i] = True
    
    if not is_valid_set(base4_list):
        base4_list = [False] * (N + 1)
    
    # Initial best from various heuristics
    candidates_for_best = []
    
    # 1. Base-4 set
    candidates_for_best.append(base4_list)
    
    # 2. Greedy from low to high
    S_greedy_low = greedy_with_order(range(1, N + 1), [False] * (N + 1))
    candidates_for_best.append(S_greedy_low)
    
    # 3. Greedy from high to low
    S_greedy_high = greedy_with_order(range(N, 0, -1), [False] * (N + 1))
    candidates_for_best.append(S_greedy_high)
    
    # 4. Greedy starting from base-4 set, try to add more
    S_greedy_from_base4 = greedy_with_order(range(1, N + 1), base4_list)
    candidates_for_best.append(S_greedy_from_base4)
    
    # 5. Randomized greedy with different seeds
    for seed in range(20):
        if time.time() > deadline - 10:
            break
        rng = random.Random(seed)
        order = list(range(1, N + 1))
        rng.shuffle(order)
        S_rand = greedy_with_order(order, [False] * (N + 1))
        candidates_for_best.append(S_rand)
    
    best_list = max(candidates_for_best, key=lambda x: sum(x))
    best_size = sum(best_list)
    
    # Local Search: 1-move (remove one, add one) and 2-move (remove one, add multiple)
    # We will focus on 2-exchange: remove one, try to add as many as possible.
    # This is effectively a "kick" move.
    
    # To make it faster, we can precompute for each element x, the list of elements y that "conflict" with x?
    # No, the constraint is global (4-way).
    # However, we can optimize the "can_add" check.
    
    # Let's implement the local search with the bitset representation.
    
    for restart in range(20000):
        if time.time() > deadline - 2:
            break
        
        current = best_list[:]
        elements = list_to_set(current)
        if not elements:
            continue
        
        random.shuffle(elements)
        
        # Try 1-step kick: remove one, add greedily
        remove_elem = elements[0]
        current[remove_elem] = False
        
        # Try to add candidates in random order
        candidates = [j for j in range(1, N + 1) if not current[j]]
        random.shuffle(candidates)
        
        added = []
        for c in candidates:
            if can_add_element(c, current):
                current[c] = True
                added.append(c)
        
        new_size = sum(current)
        
        if new_size > best_size:
            best_size = new_size
            best_list = current
            # Accept and continue from here? Or restart?
            # Usually, accepting and continuing is good, but we might get stuck.
            # Let's accept and continue the inner loop.
        else:
            # Revert
            current[remove_elem] = True
            for a in added:
                current[a] = False

    # Phase 2: 2-exchange (remove 2, add greedily)
    for restart in range(10000):
        if time.time() > deadline - 1:
            break
        
        current = best_list[:]
        elements = list_to_set(current)
        if len(elements) < 2:
            continue
        
        # Remove 2 random elements
        idx1 = random.randint(0, len(elements) - 1)
        idx2 = random.randint(0, len(elements) - 1)
        if idx1 == idx2:
            idx2 = (idx2 + 1) % len(elements)
        
        rem1 = elements[idx1]
        rem2 = elements[idx2]
        
        current[rem1] = False
        current[rem2] = False
        
        candidates = [j for j in range(1, N + 1) if not current[j]]
        random.shuffle(candidates)
        
        added = []
        for c in candidates:
            if can_add_element(c, current):
                current[c] = True
                added.append(c)
        
        new_size = sum(current)
        
        if new_size > best_size:
            best_size = new_size
            best_list = current
        else:
            # Revert
            current[rem1] = True
            current[rem2] = True
            for a in added:
                current[a] = False

    # Phase 3: Simulated Annealing style with 1-removal kicks
    # Temperature decay
    current_best = best_list[:]
    current_size = best_size
    T = 1.0
    T_min = 0.01
    
    for sa_iter in range(5000):
        if time.time() > deadline - 0.5:
            break
        
        current = current_best[:]
        elements = list_to_set(current)
        if not elements:
            continue
            
        remove_elem = random.choice(elements)
        current[remove_elem] = False
        
        candidates = [j for j in range(1, N + 1) if not current[j]]
        random.shuffle(candidates)
        
        added = []
        for c in candidates:
            if can_add_element(c, current):
                current[c] = True
                added.append(c)
        
        new_size = sum(current)
        delta = new_size - current_size
        
        if delta > 0:
            current_best = current
            current_size = new_size
            if current_size > best_size:
                best_size = current_size
                best_list = current
        else:
            # Accept with probability exp(delta / T)
            # Since delta is negative, this is < 1
            if random.random() < np.exp(delta / T):
                current_best = current
                current_size = new_size
            else:
                # Revert
                current[remove_elem] = True
                for a in added:
                    current[a] = False
        
        T *= 0.999 # Slow decay

    # Final polish: try to add any remaining elements
    final_set = best_list[:]
    for i in range(1, N + 1):
        if not final_set[i]:
            if can_add_element(i, final_set):
                final_set[i] = True
    
    # Verify validity
    if not is_valid_set(final_set):
        # Fallback to a known valid set if something went wrong
        return sorted(list_to_set(base4_list))
    
    return sorted(list_to_set(final_set))
