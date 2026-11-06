import time
import bisect

def construct():
    start = time.time()
    deadline = start + 85.0
    n = 256

    def full_validate(S_list):
        n2 = len(S_list)
        for i in range(n2):
            a = S_list[i]
            for j in range(i+1, n2):
                b = S_list[j]
                for k in range(j, n2):
                    c = S_list[k]
                    if (b + c) % a == 0:
                        return False
        return True

    def build_sumset(S_list):
        """Return a set of all pair sums b+c where b<=c and b,c in S_list."""
        sums = set()
        n2 = len(S_list)
        for i in range(n2):
            for j in range(i, n2):
                sums.add(S_list[i] + S_list[j])
        return sums

    def can_add_fast(a, S_set, S_list, sumset):
        """Check if a can be added. For each multiple m of a in [2(a+1), 2n],
        check if m is in the sumset."""
        for m in range(2 * (a + 1), 2 * n + 1, a):
            if m in sumset:
                return False
        return True

    def interval_valid(m_start):
        """Check if [m_start, n] is valid using the 3*min > 2*max condition
        plus direct check for edge cases."""
        # For [K, n], need 3K > 2n for guaranteed validity
        # 3*m_start > 2*256 = 512, so m_start >= 171
        if 3 * m_start > 2 * n:
            return True
        # Otherwise check directly but efficiently
        base = list(range(m_start, n + 1))
        for i in range(len(base)):
            a = base[i]
            # b+c ranges from 2*(a+1) to 2*n
            # Need to check if any b+c (with b,c in base, b<=c, b>a) is divisible by a
            for j in range(i + 1, len(base)):
                b = base[j]
                for k in range(j, len(base)):
                    c = base[k]
                    if (b + c) % a == 0:
                        return False
        return True

    # Start with [171, 256]
    best_S = set(range(171, n + 1))
    best_size = len(best_S)

    # Find the smallest valid starting threshold
    for m_start in range(171, 100, -1):
        if time.time() > deadline - 10:
            break
        if interval_valid(m_start):
            base = list(range(m_start, n + 1))
            if len(base) > best_size:
                best_S = set(base)
                best_size = len(base)
        else:
            # If m_start is invalid, no larger m_start will help (we want smaller)
            # Actually we're going downward, so if m_start is invalid, try m_start+1
            # But we already checked 171 which is valid, so we just continue
            pass

    # Use the best base found
    # Determine the actual best m_start
    best_m_start = 171
    for m_start in range(171, 100, -1):
        if time.time() > deadline - 10:
            break
        if interval_valid(m_start):
            best_m_start = m_start
        else:
            break  # Going lower won't help if this one fails (monotonicity)

    S_list = list(range(best_m_start, n + 1))
    S_set = set(S_list)
    sumset = build_sumset(S_list)

    # Add elements below best_m_start, trying from largest to smallest
    for a in range(best_m_start - 1, 0, -1):
        if time.time() > deadline - 5:
            break
        if a in S_set:
            continue
        if can_add_fast(a, S_set, S_list, sumset):
            pos = bisect.bisect_left(S_list, a)
            S_list.insert(pos, a)
            S_set.add(a)
            # Update sumset: add sums of a with all existing elements
            for x in S_list:
                if x != a:
                    sumset.add(a + x)
            sumset.add(2 * a)
            if len(S_list) > best_size:
                best_S = set(S_list)
                best_size = len(S_list)

    # Final polish: try to add any remaining elements to best_S
    if time.time() < deadline - 15:
        S_list = sorted(best_S)
        S_set = set(S_list)
        sumset = build_sumset(S_list)
        changed = True
        while changed and time.time() < deadline - 5:
            changed = False
            for a in range(n, 0, -1):
                if a in S_set:
                    continue
                if can_add_fast(a, S_set, S_list, sumset):
                    pos = bisect.bisect_left(S_list, a)
                    S_list.insert(pos, a)
                    S_set.add(a)
                    for x in S_list:
                        if x != a:
                            sumset.add(a + x)
                    sumset.add(2 * a)
                    changed = True
                    if len(S_list) > best_size:
                        best_S = set(S_list)
                        best_size = len(S_list)

        # Second pass: try ascending order
        if time.time() < deadline - 5:
            S_list = sorted(best_S)
            S_set = set(S_list)
            sumset = build_sumset(S_list)
            for a in range(1, n + 1):
                if a in S_set:
                    continue
                if can_add_fast(a, S_set, S_list, sumset):
                    pos = bisect.bisect_left(S_list, a)
                    S_list.insert(pos, a)
                    S_set.add(a)
                    for x in S_list:
                        if x != a:
                            sumset.add(a + x)
                    sumset.add(2 * a)
                    if len(S_list) > best_size:
                        best_S = set(S_list)
                        best_size = len(S_list)

    # Local search: try swapping elements to improve
    if time.time() < deadline - 10:
        S_list = sorted(best_S)
        S_set = set(S_list)
        sumset = build_sumset(S_list)
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            n2 = len(S_list)
            for i in range(n2):
                if time.time() > deadline - 5:
                    break
                a = S_list[i]
                # Try removing a and adding a different element
                for b in range(n, a + 1, -1):
                    if b in S_set:
                        continue
                    # Check if S - {a} + {b} is valid
                    # Quick check: can b be added to S - {a}?
                    # Build sumset for S - {a} efficiently
                    # S - {a} has all pairs except those involving a
                    # For can_add_fast, we need: for each multiple m of b in [2(b+1), 2n],
                    # m should NOT be a sum of two elements in S-{a}
                    # m = x+y where x,y in S-{a}, x<=y, x>b (since b is new min potentially)
                    # Actually b could be anywhere. Let's just check directly.
                    
                    # Efficient check: for each multiple m of b, check if m-x is in S-{a} for some x in S-{a}
                    valid_swap = True
                    S_without_a = S_set - {a}
                    for m in range(2 * (b + 1), 2 * n + 1, b):
                        # Check if m = x + y for some x, y in S_without_a with x <= y
                        # x must be > b (since b is the new element, and we need a<b<=c)
                        # Wait, the condition is: for every a'<b'<=c' in new set, a' does not divide b'+c'
                        # When we add b, b is the new element. The constraints involving b are:
                        #   - b as 'a': for all b'<c' in S-{a} with b'<b... no wait
                        #   Actually b replaces a. New set = (S - {a}) ∪ {b}
                        #   New constraints:
                        #   1. For each a'' in S-{a} with a'' < b: a'' must not divide b + c'' for any c'' in S-{a} with c'' >= b
                        #   2. For b as 'a': for all b' < c' in S-{a} with b' > b... no, b' <= c' and both in S-{a}
                        #      Actually: for a'=b, we need b' > b and b' <= c' in S-{a}, b must not divide b'+c'
                        #   3. Existing constraints among S-{a} elements remain valid (they were valid in S)
                        #      Wait, they were valid in S which included a. Removing a doesn't create new violations.
                        #      So only new constraints are those involving b.
                        
                        # Constraint type 1: a'' in S-{a}, a'' < b, check a'' | (b + c'') for c'' in S-{a}, c'' >= b
                        # Constraint type 2: b as divisor, check b | (b' + c') for b', c' in S-{a}, b < b' <= c'
                        
                        # For type 2: b divides m means m = k*b. Need b' + c' = m with b < b' <= c' in S-{a}
                        for m in range(2 * (b + 1), 2 * n + 1, b):
                            # Check if m = b' + c' for b', c' in S-{a}, b < b' <= c'
                            b_min = b + 1
                            b_max = m // 2
                            # Find elements in S-{a} in [b_min, b_max]
                            # Use S_list for binary search
                            left = bisect.bisect_left(S_list, b_min)
                            right = bisect.bisect_right(S_list, b_max)
                            found = False
                            for idx in range(left, right):
                                bp = S_list[idx]
                                if bp == a:
                                    continue
                                cp = m - bp
                                if cp >= bp and cp in S_without_a:
                                    found = True
                                    break
                            if found:
                                valid_swap = False
                                break
                        if not valid_swap:
                            break
                        
                        # For type 1: for each a'' in S-{a} with a'' < b, check a'' | (b + c'')
                        if valid_swap:
                            for idx2 in range(n2):
                                a2 = S_list[idx2]
                                if a2 == a or a2 >= b:
                                    continue
                                # Check if a2 divides b + c'' for any c'' in S-{a} with c'' >= b
                                # b + c'' must be a multiple of a2
                                # c'' >= b, so b + c'' >= 2b
                                # b + c'' = k * a2 for some k
                                # c'' = k*a2 - b, need c'' in S-{a} and c'' >= b
                                # k*a2 - b >= b => k*a2 >= 2b => k >= ceil(2b/a2)
                                # k*a2 - b <= n => k <= (n+b)/a2
                                k_min = (2 * b + a2 - 1) // a2
                                k_max = (n + b) // a2
                                for k in range(k_min, k_max + 1):
                                    c_val = k * a2 - b
                                    if c_val >= b and c_val in S_without_a:
                                        valid_swap = False
                                        break
                                if not valid_swap:
                                    break
                        
                        if not valid_swap:
                            break
                    
                    if valid_swap:
                        # Perform the swap
                        S_list = S_list[:i] + S_list[i+1:] + [b]
                        S_list.sort()
                        S_set = set(S_list)
                        sumset = build_sumset(S_list)
                        improved = True
                        if len(S_list) > best_size:
                            best_S = set(S_list)
                            best_size = len(S_list)
                        break

    # Final validation
    if time.time() < deadline:
        S_final = sorted(best_S)
        if len(S_final) > 5 and not full_validate(S_final):
            S_final = list(range(171, n + 1))
        return S_final

    return sorted(best_S)
