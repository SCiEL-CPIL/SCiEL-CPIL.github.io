import time
import random

def construct():
    N = 512
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill

    # Precompute all 4-term arithmetic progressions
    # A 4-AP is (a, a+d, a+2d, a+3d) for d >= 1, a >= 1, a+3d <= N
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

    # Precompute for each AP the other 3 elements (as a tuple)
    # ap_others[ap_idx] = (elem1, elem2, elem3) where these are the other 3 in the AP
    # For can_add_element(x), we need to check: if the other 3 are all in S, then x cannot be added
    # Actually, for x in AP, the condition is: if all other 3 elements are in S, then adding x creates the AP.
    # So for each x and each AP containing x, we store the other 3 elements.
    
    # Better: for each element x, store a list of "conflict tuples" - for each AP containing x,
    # the set of the other 3 elements. If all 3 are in S, x can't be added.
    
    # Even better for speed: use a bitset representation
    # S as a set of ints, but for fast membership, use a boolean array
    
    S_array = [False] * (N + 1)  # S_array[x] = True if x is in set
    S_set = set()  # for iteration
    
    def S_add(x):
        S_array[x] = True
        S_set.add(x)
    
    def S_remove(x):
        S_array[x] = False
        S_set.discard(x)
    
    def S_clear():
        for i in range(1, N + 1):
            S_array[i] = False
        S_set.clear()
    
    def can_add_element(x):
        """Check if x can be added to current set without creating a 4-AP"""
        for ap_idx in ap_indices_by_num[x]:
            ap = all_aps[ap_idx]
            count = 0
            for elem in ap:
                if elem == x:
                    continue
                if S_array[elem]:
                    count += 1
                if count >= 3:
                    return False
        return True

    def greedy_with_order(order):
        """Greedy add elements in the given order to empty set"""
        S_clear()
        result = set()
        for x in order:
            if can_add_element(x):
                S_add(x)
                result.add(x)
        return result

    def greedy_from_init(S_init, order):
        """Greedy add elements in order, starting from S_init"""
        S_clear()
        for x in S_init:
            S_add(x)
        result = set(S_init)
        for x in order:
            if x not in result and can_add_element(x):
                S_add(x)
                result.add(x)
        return result

    # Base-4 {0,1} construction (Salem-Spencer style)
    base4_set = set()
    for i in range(1, N + 1):
        n = i
        valid = True
        while n > 0:
            if n % 4 > 1:
                valid = False
                break
            n //= 4
        if valid:
            base4_set.add(i)

    # Initial candidates
    candidates = []
    
    # 1. Base-4 set
    candidates.append(base4_set)
    
    # 2. Greedy from low to high
    candidates.append(greedy_with_order(range(1, N + 1)))
    
    # 3. Greedy from high to low
    candidates.append(greedy_with_order(range(N, 0, -1)))
    
    # 4. Greedy starting from base-4 set
    candidates.append(greedy_from_init(base4_set, range(1, N + 1)))
    
    # 5. Greedy with step 2 (evens first, then odds)
    evens = list(range(2, N + 1, 2))
    odds = list(range(1, N + 1, 2))
    candidates.append(greedy_with_order(evens + odds))
    candidates.append(greedy_with_order(odds + evens))
    
    # 6. Random order greedy (several tries)
    for _ in range(5):
        random_order = list(range(1, N + 1))
        random.shuffle(random_order)
        candidates.append(greedy_with_order(random_order))

    best_set = max(candidates, key=len)
    best_size = len(best_set)

    # Local search phase 1: remove one, add many
    for restart in range(5000):
        if time.time() - start_time > DEADLINE:
            break

        S_clear()
        for x in best_set:
            S_add(x)
        
        elements = list(best_set)
        random.shuffle(elements)
        
        for i in range(min(len(elements), 20)):
            if time.time() - start_time > DEADLINE:
                break
            
            remove_elem = elements[i]
            S_remove(remove_elem)
            
            # Try to add candidates
            candidates_to_add = [j for j in range(1, N + 1) if not S_array[j]]
            random.shuffle(candidates_to_add)
            
            added = []
            for c in candidates_to_add:
                if can_add_element(c):
                    S_add(c)
                    added.append(c)
            
            if len(added) > 1:
                if len(S_set) > best_size:
                    best_size = len(S_set)
                    best_set = set(S_set)
                break
            else:
                S_add(remove_elem)
                for a in added:
                    S_remove(a)

    # Local search phase 2: swap-based
    for restart in range(5000):
        if time.time() - start_time > DEADLINE:
            break

        S_clear()
        for x in best_set:
            S_add(x)
        
        # Try random swaps: remove x, try to add y
        x = random.choice(list(best_set))
        S_remove(x)
        
        # Try to add any available element
        improvement = False
        for y in range(1, N + 1):
            if not S_array[y] and can_add_element(y):
                S_add(y)
                improvement = True
                break
        
        if improvement:
            # After swap, try to add more
            for y in range(1, N + 1):
                if not S_array[y]:
                    if can_add_element(y):
                        S_add(y)
        
        if len(S_set) > best_size:
            best_size = len(S_set)
            best_set = set(S_set)
        else:
            # Revert
            S_clear()
            for x in best_set:
                S_add(x)

    # Local search phase 3: multi-remove and re-greedy
    for restart in range(3000):
        if time.time() - start_time > DEADLINE:
            break

        S_clear()
        for x in best_set:
            S_add(x)
        
        # Remove 2-4 random elements
        elements = list(S_set)
        random.shuffle(elements)
        num_remove = random.randint(2, 4)
        to_remove = elements[:num_remove]
        
        for r in to_remove:
            S_remove(r)
        
        # Re-greedy: try to add all available in random order
        available = [j for j in range(1, N + 1) if not S_array[j]]
        random.shuffle(available)
        
        for c in available:
            if can_add_element(c):
                S_add(c)
        
        if len(S_set) > best_size:
            best_size = len(S_set)
            best_set = set(S_set)

    # Final polish: try single-element improvements
    S_clear()
    for x in best_set:
        S_add(x)
    
    changed = True
    while changed and (time.time() - start_time) < DEADLINE:
        changed = False
        for x in list(S_set):
            if time.time() - start_time > DEADLINE:
                break
            S_remove(x)
            for y in range(1, N + 1):
                if not S_array[y] and can_add_element(y):
                    S_add(y)
                    if len(S_set) > best_size:
                        best_size = len(S_set)
                        best_set = set(S_set)
                    changed = True
                    break
            if not S_array[y] and not changed:
                S_add(x)  # revert
            elif S_array[y] and y != x:
                S_remove(x)
                S_add(y)
                if len(S_set) > best_size:
                    best_size = len(S_set)
                    best_set = set(S_set)
                    changed = True
            else:
                S_add(x)

    return sorted(best_set)
