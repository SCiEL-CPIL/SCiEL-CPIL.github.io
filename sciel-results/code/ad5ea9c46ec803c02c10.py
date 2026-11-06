"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill

    N = 128
    elements = list(range(1, N + 1))

    # Precompute GCD table
    gcd_table = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            gcd_table[i][j] = gcd(i, j)
            gcd_table[j][i] = gcd_table[i][j]

    # For each element x, for each possible gcd value d,
    # precompute the bitmask of elements y where gcd(x,y) = d
    gcd_mask = [[0] * (N + 1) for _ in range(N + 1)]
    for x in range(1, N + 1):
        for y in range(1, N + 1):
            d = gcd_table[x][y]
            gcd_mask[x][d] |= (1 << y)

    # Precompute pairs_by_gcd
    pairs_by_gcd = defaultdict(list)
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            d = gcd_table[a][b]
            pairs_by_gcd[d].append((a, b))

    # For each x, precompute bad_pairs_for_x: list of (a,b) such that 
    # if both a and b are in S, then x cannot be added
    bad_pairs_for_x = [[] for _ in range(N + 1)]
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            d = gcd_table[a][b]
            # x is in bad_pairs_for_x if gcd(x,a) == d and gcd(x,b) == d
            # i.e., x is in gcd_mask[a][d] AND x is in gcd_mask[b][d]
            mask_a = gcd_mask[a][d]
            mask_b = gcd_mask[b][d]
            common = mask_a & mask_b
            # For each x in common (excluding a and b), add (a,b) to bad_pairs_for_x[x]
            x = 1
            while x <= N:
                if common & (1 << x) and x != a and x != b:
                    bad_pairs_for_x[x].append((a, b))
                x += 1

    def can_add_fast(x, selected_mask):
        """Check if x can be added to set represented by selected_mask."""
        for (a, b) in bad_pairs_for_x[x]:
            if (selected_mask & (1 << a)) and (selected_mask & (1 << b)):
                return False
        return True

    def mask_to_list(mask):
        return [i for i in range(1, N + 1) if mask & (1 << i)]

    def mask_size(mask):
        return bin(mask).count('1')

    def greedy_add(order):
        selected_mask = 0
        for x in order:
            if can_add_fast(x, selected_mask):
                selected_mask |= (1 << x)
        return selected_mask

    def greedy_remove():
        """Start with all elements, greedily remove elements that cause most conflicts."""
        S = (1 << (N + 1)) - 2  # bits 1..N set
        remaining = list(range(1, N + 1))
        
        while remaining:
            worst = -1
            worst_count = 0
            for x in remaining:
                count = 0
                for (a, b) in bad_pairs_for_x[x]:
                    if (S & (1 << a)) and (S & (1 << b)):
                        count += 1
                if count > worst_count:
                    worst_count = count
                    worst = x
            
            if worst == -1 or worst_count == 0:
                break
            
            S &= ~(1 << worst)
            remaining.remove(worst)
        
        return S

    def local_search(mask, time_limit):
        best_mask = mask
        best_size = mask_size(best_mask)
        
        while time.time() < time_limit:
            improved = False
            
            # Try adding
            for x in range(1, N + 1):
                if best_mask & (1 << x):
                    continue
                if can_add_fast(x, best_mask):
                    best_mask |= (1 << x)
                    best_size += 1
                    improved = True
                    break
            if improved:
                continue
            
            # Try swap: remove one, add multiple
            for i in range(1, N + 1):
                if not (best_mask & (1 << i)):
                    continue
                temp = best_mask & ~(1 << i)
                added = 0
                for x in range(1, N + 1):
                    if x == i or (temp & (1 << x)):
                        continue
                    if can_add_fast(x, temp):
                        temp |= (1 << x)
                        added += 1
                        if added >= 2:
                            break
                if added > 1:
                    best_mask = temp
                    best_size += added - 1
                    improved = True
                    break
            
            if not improved:
                break
        
        return best_mask

    # Branch and bound with upper bound pruning
    def branch_and_bound(available_mask, selected_mask, best_size_ref, time_limit):
        """
        Recursive branch and bound.
        available_mask: elements not yet decided
        selected_mask: elements already selected
        best_size_ref: [current best size] (mutable list)
        Returns improved selected_mask
        """
        if time.time() > time_limit:
            return selected_mask
        
        if available_mask == 0:
            if mask_size(selected_mask) > best_size_ref[0]:
                best_size_ref[0] = mask_size(selected_mask)
                return selected_mask
            return selected_mask
        
        # Upper bound: current selected + all available
        ub = mask_size(selected_mask) + mask_size(available_mask)
        if ub <= best_size_ref[0]:
            return selected_mask
        
        # Pick an element to branch on
        # Choose the element with the most conflicts in available set
        best_x = -1
        best_conflicts = -1
        temp = available_mask
        x = 1
        while x <= N and temp:
            if temp & (1 << x):
                # Count how many elements in available would conflict with x
                conflicts = 0
                temp2 = available_mask & ~(1 << x)
                y = 1
                while y <= N and temp2:
                    if temp2 & (1 << y):
                        if not can_add_fast(x, selected_mask | (1 << y)):
                            conflicts += 1
                    y += 1
                if conflicts > best_conflicts:
                    best_conflicts = conflicts
                    best_x = x
            x += 1
        
        if best_x == -1:
            # No elements left to branch on
            return selected_mask
        
        # Branch: include best_x
        if can_add_fast(best_x, selected_mask):
            new_selected = selected_mask | (1 << best_x)
            new_available = available_mask & ~(1 << best_x)
            result1 = branch_and_bound(new_available, new_selected, best_size_ref, time_limit)
            if mask_size(result1) > best_size_ref[0]:
                best_size_ref[0] = mask_size(result1)
                selected_mask = result1
        
        # Branch: exclude best_x
        new_available = available_mask & ~(1 << best_x)
        result2 = branch_and_bound(new_available, selected_mask, best_size_ref, time_limit)
        if mask_size(result2) > best_size_ref[0]:
            best_size_ref[0] = mask_size(result2)
            selected_mask = result2
        
        return selected_mask

    # Multiple greedy with different orderings
    best_mask = 0

    # Ordering 1: ascending
    m = greedy_add(range(1, N + 1))
    if mask_size(m) > mask_size(best_mask):
        best_mask = m

    if time.time() > deadline - 20:
        return mask_to_list(best_mask) if best_mask else [1]

    # Ordering 2: descending
    m = greedy_add(range(N, 0, -1))
    if mask_size(m) > mask_size(best_mask):
        best_mask = m

    if time.time() > deadline - 15:
        return mask_to_list(best_mask) if best_mask else [1]

    # Ordering 3: by number of prime factors
    def num_prime_factors(n):
        count = 0
        d = 2
        while d * d <= n:
            while n % d == 0:
                count += 1
                n //= d
            d += 1
        if n > 1:
            count += 1
        return count

    order3 = sorted(range(1, N + 1), key=lambda x: (num_prime_factors(x), x))
    m = greedy_add(order3)
    if mask_size(m) > mask_size(best_mask):
        best_mask = m

    if time.time() > deadline - 10:
        return mask_to_list(best_mask) if best_mask else [1]

    # Greedy remove
    m = greedy_remove()
    if mask_size(m) > mask_size(best_mask):
        best_mask = m

    if time.time() > deadline - 8:
        return mask_to_list(best_mask) if best_mask else [1]

    # Random orderings
    rng = random.Random(42)
    for trial in range(50):
        if time.time() > deadline - 5:
            break
        order = list(range(1, N + 1))
        rng.shuffle(order)
        m = greedy_add(order)
        if mask_size(m) > mask_size(best_mask):
            best_mask = m

    if time.time() > deadline - 3:
        return mask_to_list(best_mask) if best_mask else [1]

    # Local search
    best_mask = local_search(best_mask, deadline - 2)

    # Try branch and bound on a smaller subset if time permits
    if time.time() < deadline - 10:
        # Take the best mask and try to improve with B&B on remaining elements
        remaining_mask = ((1 << (N + 1)) - 2) & ~best_mask
        # Only consider elements that can potentially be added
        bbb_available = 0
        for x in range(1, N + 1):
            if remaining_mask & (1 << x):
                if can_add_fast(x, best_mask):
                    bbb_available |= (1 << x)
        
        if bbb_available:
            best_size_ref = [mask_size(best_mask)]
            # Limit B&B to a reasonable number of elements
            # Take the top 20 most promising elements
            candidates = [x for x in range(1, N + 1) if bbb_available & (1 << x)]
            candidates.sort(key=lambda x: len(bad_pairs_for_x[x]))  # Fewer conflicts first
            limited_candidates = candidates[:20]
            limited_mask = 0
            for x in limited_candidates:
                limited_mask |= (1 << x)
            
            result = branch_and_bound(limited_mask, best_mask, best_size_ref, deadline - 1)
            if mask_size(result) > mask_size(best_mask):
                best_mask = result

    return mask_to_list(best_mask) if best_mask else [1]
# EVOLVE-BLOCK-END
