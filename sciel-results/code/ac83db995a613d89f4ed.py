import time

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin (90s limit)
    
    n = 256
    
    def can_add(x, S_set, S_list):
        """Check if x can be added to S. x must be smaller than all elements in S."""
        if not S_list:
            return True
        # Check if any b,c in S_list have (b+c) % x == 0
        # Use remainder grouping for speed
        max_val = max(S_list)
        if x > 2 * max_val:
            return True
        # Group by remainder mod x
        rem_groups = {}
        for s in S_list:
            r = s % x
            if r in rem_groups:
                rem_groups[r].append(s)
            else:
                rem_groups[r] = [s]
        # Check: for any pair (b,c), b+c ≡ 0 (mod x)
        # i.e., (b % x) + (c % x) ≡ 0 (mod x)
        # So we need r1 + r2 ≡ 0 (mod x) where r1, r2 are remainders
        # Case 1: r1 == r2 and 2*r1 % x == 0
        # Case 2: r1 != r2 and r1 + r2 == x (or r1+r2 == 0, but remainders are in [0,x-1])
        for r, elems in rem_groups.items():
            if len(elems) >= 2:
                # Two elements with same remainder r: sum ≡ 2r (mod x)
                if (2 * r) % x == 0:
                    return False
            # Check complement remainder
            comp = (x - r) % x
            if comp == r:
                # Already handled above (need at least 2 elements)
                pass
            elif comp in rem_groups:
                return False
        return True
    
    # Phase 1: Greedy construction from top down
    S = []
    S_set = set()
    for x in range(n, 0, -1):
        if time.time() > deadline:
            break
        if can_add(x, S_set, S):
            S.append(x)
            S_set.add(x)
    
    best_size = len(S)
    
    # Phase 2: Local search - try to add skipped elements by removing others
    # Track which elements were skipped
    skipped = [x for x in range(n, 0, -1) if x not in S_set]
    
    # Try adding each skipped element by removing one or two existing elements
    improved = True
    while improved and time.time() < deadline:
        improved = False
        S_sorted = sorted(S)
        S_set = set(S)
        
        for skip_x in skipped:
            if time.time() > deadline:
                break
            if skip_x in S_set:
                continue
            
            # Try removing one element to allow skip_x
            for i, rem_elem in enumerate(S_sorted):
                if time.time() > deadline:
                    break
                new_S = [s for j, s in enumerate(S_sorted) if j != i]
                new_S_set = set(new_S)
                if can_add(skip_x, new_S_set, new_S):
                    # Also verify the whole set is valid (skip_x as 'a' is checked, but need to check other constraints)
                    # Since skip_x < all elements in new_S, only need to check skip_x as 'a'
                    # But we also need to verify new_S itself is still valid (it was valid before)
                    # Actually new_S is a subset of old valid S, so it's valid
                    new_S.append(skip_x)
                    new_S_set.add(skip_x)
                    if len(new_S) > best_size:
                        S = new_S
                        S_set = new_S_set
                        best_size = len(S)
                        improved = True
                        # Update skipped list
                        skipped = [x for x in range(n, 0, -1) if x not in S_set]
                        break
            
            # Try removing two elements
            if not improved and len(S_sorted) >= 2:
                for i in range(len(S_sorted)):
                    if time.time() > deadline:
                        break
                    for j in range(i + 1, len(S_sorted)):
                        if time.time() > deadline:
                            break
                        new_S = [s for k, s in enumerate(S_sorted) if k != i and k != j]
                        new_S_set = set(new_S)
                        if can_add(skip_x, new_S_set, new_S):
                            new_S.append(skip_x)
                            new_S_set.add(skip_x)
                            if len(new_S) > best_size:
                                S = new_S
                                S_set = new_S_set
                                best_size = len(S)
                                improved = True
                                skipped = [x for x in range(n, 0, -1) if x not in S_set]
                                break
                    if improved:
                        break
    
    # Phase 3: Try to add more elements from below (smaller numbers)
    # After local search, try to add any remaining small numbers
    S_set = set(S)
    for x in range(1, n + 1):
        if time.time() > deadline:
            break
        if x in S_set:
            continue
        if can_add(x, S_set, S):
            S.append(x)
            S_set.add(x)
            best_size = len(S)
    
    return S
