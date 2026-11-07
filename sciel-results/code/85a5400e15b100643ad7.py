import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 5s safety margin for 60s limit

    # Precompute for each x: list of (a, b) with a < b, a + b = x, 1 <= a < b <= 128
    sum_pairs = [[] for _ in range(129)]
    for a in range(1, 129):
        for b in range(a + 1, 129):
            s = a + b
            if s <= 128:
                sum_pairs[s].append((a, b))

    # Precompute for each x: list of y in 1..128 with y != x and x + y <= 128
    partners = [[] for _ in range(129)]
    for x in range(1, 129):
        for y in range(1, 129):
            if y != x and x + y <= 128:
                partners[x].append(y)

    def can_add(S_mask, x):
        """Check if S_mask U {x} is a valid set (no forbidden triple)."""
        # Case 1: x is a member of the triple (a = x).
        # Need distinct b, c in S such that x+b, x+c, b+c are all in S U {x}.
        # Since x+b, x+c > x, they must be in S. b+c in S or b+c == x.
        # V = { y in S | x+y in S } = S & (S >> x)
        V_mask = S_mask & (S_mask >> x)
        if V_mask.bit_count() >= 2:
            v_list = []
            temp = V_mask
            while temp:
                lsb = temp & -temp
                v_list.append(lsb.bit_length() - 1)
                temp -= lsb
            n_v = len(v_list)
            for i in range(n_v):
                b = v_list[i]
                for j in range(i + 1, n_v):
                    c = v_list[j]
                    s = b + c
                    if s == x or ((S_mask >> s) & 1):
                        return False

        # Case 2: x is a sum of two members (a+b = x).
        # Need distinct a, b, c in S with a+b=x, a+c in S, b+c in S.
        for a, b in sum_pairs[x]:
            if ((S_mask >> a) & 1) == 0:
                continue
            if ((S_mask >> b) & 1) == 0:
                continue
            # c in S \ {a,b} such that a+c in S and b+c in S
            U_mask = (S_mask & (S_mask >> a)) & (S_mask & (S_mask >> b))
            U_mask &= ~(1 << a)
            U_mask &= ~(1 << b)
            if U_mask != 0:
                return False

        return True

    def greedy_construct(order, S_mask):
        """Greedy construction given a fixed order of candidates."""
        for x in order:
            if time.time() > deadline:
                break
            if can_add(S_mask, x):
                S_mask |= (1 << x)
        return S_mask

    def local_search(S_mask, max_passes=30, swap_limit=15):
        """Improve S via additions and systematic swaps."""
        S_set = set()
        temp = S_mask
        while temp:
            lsb = temp & -temp
            idx = lsb.bit_length() - 1
            S_set.add(idx)
            temp -= lsb

        not_in_S = [x for x in range(1, 129) if x not in S_set]

        for _ in range(max_passes):
            if time.time() > deadline:
                break

            improved = False
            random.shuffle(not_in_S)

            for x in not_in_S[:]:
                if time.time() > deadline:
                    break

                # Try to add x directly
                if can_add(S_mask, x):
                    S_mask |= (1 << x)
                    S_set.add(x)
                    not_in_S.remove(x)
                    improved = True
                    continue

                # Try swaps: remove y from S, add x
                S_list = list(S_set)
                random.shuffle(S_list)

                found_swap = False
                for y in S_list[:swap_limit]:
                    if time.time() > deadline:
                        break
                    S_temp_mask = S_mask ^ (1 << y)
                    if can_add(S_temp_mask, x):
                        S_mask = S_temp_mask | (1 << x)
                        S_set.remove(y)
                        S_set.add(x)
                        not_in_S.append(y)
                        if x in not_in_S:
                            not_in_S.remove(x)
                        found_swap = True
                        improved = True
                        break

                if found_swap:
                    break

            if not improved:
                break

        return S_mask

    # Precompute a static "conflict degree" for ordering
    degree = [0] * 129
    for x in range(1, 129):
        degree[x] = len(partners[x])

    degree_order = sorted(range(1, 129), key=lambda x: degree[x])

    random.seed(42)

    best_mask = 0

    # Multiple greedy constructions with diverse orderings
    S_mask = greedy_construct(list(range(1, 129)), 0)
    if S_mask.bit_count() > best_mask.bit_count():
        best_mask = S_mask

    S_mask = greedy_construct(list(range(128, 0, -1)), 0)
    if S_mask.bit_count() > best_mask.bit_count():
        best_mask = S_mask

    S_mask = greedy_construct(list(degree_order), 0)
    if S_mask.bit_count() > best_mask.bit_count():
        best_mask = S_mask

    S_mask = greedy_construct(list(reversed(degree_order)), 0)
    if S_mask.bit_count() > best_mask.bit_count():
        best_mask = S_mask

    for i in range(50):
        if time.time() > deadline:
            break
        candidates = list(range(1, 129))
        random.shuffle(candidates)
        S_mask = greedy_construct(candidates, 0)
        if S_mask.bit_count() > best_mask.bit_count():
            best_mask = S_mask

    # Local search from best set
    if time.time() < deadline:
        best_mask = local_search(best_mask, max_passes=30)

    # More random restarts with local search
    for i in range(50):
        if time.time() > deadline:
            break
        candidates = list(range(1, 129))
        random.shuffle(candidates)
        S_mask = greedy_construct(candidates, 0)
        S_mask = local_search(S_mask, max_passes=20)
        if S_mask.bit_count() > best_mask.bit_count():
            best_mask = S_mask

    # Final: try to add any remaining elements
    if time.time() < deadline:
        for x in range(1, 129):
            if (best_mask >> x) & 1:
                continue
            if can_add(best_mask, x):
                best_mask |= (1 << x)

    # One more local search pass
    if time.time() < deadline and best_mask.bit_count() < 128:
        best_mask = local_search(best_mask, max_passes=15)

    # Convert mask to list of ints
    result = []
    temp = best_mask
    while temp:
        lsb = temp & -temp
        idx = lsb.bit_length() - 1
        result.append(int(idx))
        temp -= lsb

    return result
