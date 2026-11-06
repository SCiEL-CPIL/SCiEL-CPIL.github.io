import time
import bisect

def construct():
    start = time.time()
    deadline = start + 82.0
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

    def can_add_slow(a, S_set, S_list):
        """Slower but correct check without sumset."""
        for m in range(2 * (a + 1), 2 * n + 1, a):
            b_min = a + 1
            b_max = m // 2
            left = bisect.bisect_left(S_list, b_min)
            right = bisect.bisect_right(S_list, b_max)
            for idx in range(left, right):
                b = S_list[idx]
                c = m - b
                if c >= b and c in S_set:
                    return False
        return True

    # Start with [171, 256]
    best_S = set(range(171, n + 1))
    best_size = len(best_S)

    # Try various starting thresholds
    for m_start in [171, 170, 169, 168, 167, 166, 165, 164, 163, 162, 161, 160,
                    159, 158, 157, 156, 155, 154, 153, 152, 151, 150, 149, 148, 147,
                    146, 145, 144, 143, 142, 141, 140, 139, 138, 137, 136, 135, 134,
                    133, 132, 131, 130, 129, 128, 127, 126, 125, 124, 123, 122, 121,
                    120, 119, 118, 117, 116, 115, 114, 113, 112, 111, 110, 109, 108,
                    107, 106, 105, 104, 103, 102, 101, 100]:
        if time.time() > deadline - 10:
            break

        # Check if [m_start, n] is valid
        base = list(range(m_start, n + 1))
        valid = True
        for i in range(len(base)):
            a = base[i]
            for j in range(i + 1, len(base)):
                b = base[j]
                for k in range(j, len(base)):
                    c = base[k]
                    if (b + c) % a == 0:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break

        if not valid:
            continue

        S_list = base[:]
        S_set = set(S_list)
        sumset = build_sumset(S_list)

        if len(S_list) > best_size:
            best_S = set(S_list)
            best_size = len(S_list)

        # Add elements below m_start, trying from largest to smallest
        for a in range(m_start - 1, 0, -1):
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
    if time.time() < deadline - 8:
        S_list = sorted(best_S)
        S_set = set(S_list)
        sumset = build_sumset(S_list)
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for i, a in enumerate(S_list):
                if time.time() > deadline - 5:
                    break
                # Try removing a and adding a different element
                for b in range(n, 0, -1):
                    if b in S_set or b <= a:
                        continue
                    # Check if S - {a} + {b} is valid
                    # First check if b can be added to S - {a}
                    S_list_no_a = S_list[:i] + S_list[i+1:]
                    S_set_no_a = S_set - {a}
                    # Rebuild sumset for S - {a}
                    sumset_no_a = build_sumset(S_list_no_a)
                    if can_add_fast(b, S_set_no_a, S_list_no_a, sumset_no_a):
                        # Check validity of S - {a} + {b}
                        new_S = S_list_no_a + [b]
                        new_S.sort()
                        if full_validate(new_S):
                            S_list = new_S
                            S_set = set(new_S)
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
