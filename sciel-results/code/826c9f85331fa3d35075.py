import time

def construct():
    start = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill

    n = 256

    # Precompute conflict graph: conflict[a][b] = True if a and b cannot coexist in a 3-AP-free set
    # Two elements a,b conflict if there exists a third element c in [1,256] such that
    # {a,b,c} forms a nonconstant 3-term AP.
    # For a pair (a,b) with a < b:
    #   - If b - a is even: midpoint c = (a+b)/2 exists. If c in [1,256], conflict.
    #   - If a - (b-a) >= 1: predecessor c = 2a - b exists. If c in [1,256], conflict.
    #   - If b + (b-a) <= 256: successor c = 2b - a exists. If c in [1,256], conflict.
    # Actually, any pair (a,b) with a != b always has at least one of these in range
    # unless the AP would go out of bounds on both sides. Let me just check all three.

    # conflict_mask[x] is a bitmask of all y that conflict with x
    conflict_mask = [0] * (n + 1)
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            d = b - a
            # Check midpoint
            if d % 2 == 0:
                c = (a + b) // 2
                if 1 <= c <= n:
                    conflict_mask[a] |= (1 << b)
                    conflict_mask[b] |= (1 << a)
                    continue
            # Check predecessor: 2a - b
            c = 2 * a - b
            if 1 <= c <= n:
                conflict_mask[a] |= (1 << b)
                conflict_mask[b] |= (1 << a)
                continue
            # Check successor: 2b - a
            c = 2 * b - a
            if 1 <= c <= n:
                conflict_mask[a] |= (1 << b)
                conflict_mask[b] |= (1 << a)

    # Now conflict_mask[x] contains all y that form a 3-AP with x (as the two endpoints or with x as middle)
    # Wait, I need to be more careful. If a and b are in the set, they conflict if adding both creates a 3-AP.
    # The 3-AP could be (c, a, b) where c = 2a-b, or (a, c, b) where c=(a+b)/2, or (a, b, c) where c=2b-a.
    # So a and b conflict if any of these c values is in [1,256].
    # My loop above handles this correctly.

    # Weight: 1/x
    weight = [0.0] * (n + 1)
    for i in range(1, n + 1):
        weight[i] = 1.0 / i

    # Greedy initialization: add elements in order of decreasing weight (smallest index first)
    S_bits = 0
    for x in range(1, n + 1):
        # Check if x conflicts with any element already in S
        # x conflicts with y in S if (S_bits & conflict_mask[x]) != 0
        if (S_bits & conflict_mask[x]) == 0:
            S_bits |= (1 << x)

    def get_set(S_bits):
        """Return list of elements in the set."""
        result = []
        temp = S_bits
        while temp:
            low = temp & -temp
            idx = low.bit_length() - 1
            result.append(idx)
            temp &= temp - 1
        return result

    def value_of(S_bits):
        """Compute sum of 1/x for x in set."""
        total = 0.0
        temp = S_bits
        while temp:
            low = temp & -temp
            idx = low.bit_length() - 1
            total += weight[idx]
            temp &= temp - 1
        return total

    def can_add(x, S_bits):
        """Check if x can be added to S without creating a 3-AP."""
        return (S_bits & conflict_mask[x]) == 0

    def can_remove_and_add(rem, add, S_bits):
        """Check if removing 'rem' and adding 'add' keeps the set 3-AP-free."""
        # After removing rem, check if add conflicts with remaining elements
        S_without_rem = S_bits & ~(1 << rem)
        return (S_without_rem & conflict_mask[add]) == 0

    def can_swap_two(rem1, rem2, add1, add2, S_bits):
        """Check if removing rem1,rem2 and adding add1,add2 keeps set 3-AP-free."""
        S_new = S_bits & ~(1 << rem1) & ~(1 << rem2) | (1 << add1) | (1 << add2)
        # Check add1 and add2 don't conflict with each other
        if add1 == add2:
            return False
        if conflict_mask[add1] & (1 << add2):
            return False
        # Check add1 doesn't conflict with remaining
        S_without_adds = S_new & ~(1 << add1) & ~(1 << add2)
        if S_without_adds & conflict_mask[add1]:
            return False
        if S_without_adds & conflict_mask[add2]:
            return False
        return True

    best_value = value_of(S_bits)
    best_S = S_bits

    def update_best(S_bits, val):
        nonlocal best_value, best_S
        if val > best_value:
            best_value = val
            best_S = S_bits

    # Local search loop
    improved = True
    while improved and time.time() < DEADLINE:
        improved = False
        S = S_bits
        val = best_value

        # Phase 1: Try adding any element
        for x in range(1, n + 1):
            if not (S & (1 << x)):
                if can_add(x, S):
                    S |= (1 << x)
                    val += weight[x]
                    improved = True

        # Phase 2: Try 1-for-1 swaps (remove u, add v where v < u for weight gain)
        changed = True
        while changed and time.time() < DEADLINE:
            changed = False
            S_list = get_set(S)
            for u in S_list:
                if not (S & (1 << u)):
                    continue
                S_without_u = S & ~(1 << u)
                for v in range(1, u):
                    if S_without_u & (1 << v):
                        continue
                    if can_add(v, S_without_u):
                        # Swap u -> v
                        S = S_without_u | (1 << v)
                        val += weight[v] - weight[u]
                        improved = True
                        changed = True
                        break
                if changed:
                    break

        # Phase 3: Try 2-for-1 swaps (remove u, add v1 and v2 where weight gain is positive)
        # This is expensive, so do it with time check
        if time.time() < DEADLINE - 5:
            S_list = get_set(S)
            for u in S_list:
                if not (S & (1 << u)):
                    continue
                S_without_u = S & ~(1 << u)
                # Try pairs of additions
                for v1 in range(1, u):
                    if S_without_u & (1 << v1):
                        continue
                    if not can_add(v1, S_without_u):
                        continue
                    S_with_v1 = S_without_u | (1 << v1)
                    for v2 in range(v1 + 1, u):
                        if S_with_v1 & (1 << v2):
                            continue
                        if not can_add(v2, S_with_v1):
                            continue
                        # Check weight improvement
                        if weight[v1] + weight[v2] > weight[u]:
                            S = S_with_v1 | (1 << v2)
                            val += weight[v1] + weight[v2] - weight[u]
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break

        # Phase 4: Try 1-for-2 swaps (remove u and w, add v where weight gain is positive)
        # Remove two large elements, add one small element
        if time.time() < DEADLINE - 5:
            S_list = get_set(S)
            for i, u in enumerate(S_list):
                if not (S & (1 << u)):
                    continue
                for j in range(i + 1, len(S_list)):
                    w = S_list[j]
                    if not (S & (1 << w)):
                        continue
                    S_without_uw = S & ~(1 << u) & ~(1 << w)
                    for v in range(1, min(u, w)):
                        if S_without_uw & (1 << v):
                            continue
                        if not can_add(v, S_without_uw):
                            continue
                        if weight[v] > weight[u] + weight[w]:
                            S = S_without_uw | (1 << v)
                            val += weight[v] - weight[u] - weight[w]
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break

        # Phase 5: Try 2-for-2 swaps
        if time.time() < DEADLINE - 10:
            S_list = get_set(S)
            found_swap = False
            for i, u in enumerate(S_list):
                if not (S & (1 << u)):
                    continue
                for j in range(i + 1, len(S_list)):
                    w = S_list[j]
                    if not (S & (1 << w)):
                        continue
                    S_without_uw = S & ~(1 << u) & ~(1 << w)
                    # Try adding v1, v2
                    for v1 in range(1, min(u, w)):
                        if S_without_uw & (1 << v1):
                            continue
                        if not can_add(v1, S_without_uw):
                            continue
                        S_with_v1 = S_without_uw | (1 << v1)
                        for v2 in range(v1 + 1, min(u, w)):
                            if S_with_v1 & (1 << v2):
                                continue
                            if not can_add(v2, S_with_v1):
                                continue
                            if weight[v1] + weight[v2] > weight[u] + weight[w]:
                                S = S_with_v1 | (1 << v2)
                                val += weight[v1] + weight[v2] - weight[u] - weight[w]
                                improved = True
                                found_swap = True
                                break
                        if found_swap:
                            break
                    if found_swap:
                        break
                if found_swap:
                    break

        # Update best
        if val > best_value:
            best_value = val
            best_S = S

    # Final: ensure we return the best found
    S_bits = best_S
    return get_set(S_bits)
