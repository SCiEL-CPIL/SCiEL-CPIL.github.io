import time
import random
from fractions import Fraction

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 82.0  # Safety margin before 90s kill

    # Precompute reciprocals
    recip = [0.0] * (n + 1)
    for i in range(1, n + 1):
        recip[i] = 1.0 / i

    # Precompute suffix sums of reciprocals for upper bound
    suffix_sum = [0.0] * (n + 2)
    for i in range(n, 0, -1):
        suffix_sum[i] = suffix_sum[i + 1] + recip[i]

    # Precompute conflict structure
    # For each x, conflict_pairs[x] = list of (a, b) where a < b < x and a + x = 2*b
    # When selecting x, if both a and b are selected, conflict
    conflict_pairs = [[] for _ in _ in range(n + 1)]
    conflict_pairs = [[] for _ in range(n + 1)]
    for x in range(1, n + 1):
        for b in range(1, x):
            a = 2 * b - x
            if 1 <= a < b:
                conflict_pairs[x].append((a, b))

    # For each x, precompute which a < x would forbid c = 2*x - a > x
    # When selecting x, for each a in selected with a < x, c = 2*x - a is forbidden if c > x and c <= n
    forbid_map = [[] for _ in range(n + 1)]
    for x in range(1, n + 1):
        for a in range(1, x):
            c = 2 * x - a
            if c > x and c <= n:
                forbid_map[x].append((a, c))

    # Greedy construction: add elements in order of decreasing weight (smallest first)
    def greedy_construct():
        in_set = [False] * (n + 1)
        for x in range(1, n + 1):
            # Check if x can be added
            can_add = True
            for a, b in conflict_pairs[x]:
                if in_set[a] and in_set[b]:
                    can_add = False
                    break
            if can_add:
                in_set[x] = True
        return in_set

    def get_sum(in_set):
        total = 0.0
        for i in range(1, n + 1):
            if in_set[i]:
                total += recip[i]
        return total

    def can_add_check(x, in_set):
        """Check if x can be added to the set."""
        for a, b in conflict_pairs[x]:
            if in_set[a] and in_set[b]:
                return False
        return True

    # Local search: try single swaps and double swaps
    def local_search(in_set, max_iters=500):
        current_sum = get_sum(in_set)
        elements = [i for i in range(1, n + 1) if in_set[i]]
        non_elements = [i for i in range(1, n + 1) if not in_set[i]]

        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False

            # Try single swaps: remove one, add one
            for rem in elements[:]:
                if not in_set[rem]:
                    continue
                in_set[rem] = False
                best_add = None
                best_gain = 0.0
                for add in non_elements:
                    if in_set[add]:
                        continue
                    if can_add_check(add, in_set):
                        gain = recip[add] - recip[rem]
                        if gain > best_gain + 1e-15:
                            best_gain = gain
                            best_add = add
                if best_add is not None:
                    in_set[best_add] = True
                    elements.remove(rem)
                    non_elements.remove(best_add)
                    elements.append(best_add)
                    non_elements.append(rem)
                    current_sum += best_gain
                    improved = True
                else:
                    in_set[rem] = True

        return in_set

    # Branch and bound with forward checking
    # Process variables in order 1 to n
    # State: selected (bitset as int), forbidden (bitset as int), current_sum
    # Bit i (0-indexed) represents number i+1

    def branch_and_bound():
        best_sum = 0.0
        best_set = [False] * (n + 1)

        # We'll use a recursive approach with pruning
        # For efficiency, use bitsets (Python ints)
        # selected_bits: bit i set means (i+1) is selected
        # forbidden_bits: bit i set means (i+1) is forbidden

        def bb_recursive(x, selected_bits, forbidden_bits, current_sum):
            nonlocal best_sum, best_set

            if time.time() > deadline - 5.0:
                return

            if x > n:
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_set = [False] * (n + 1)
                    for i in range(n):
                        if (selected_bits >> i) & 1:
                            best_set[i + 1] = True
                return

            # Upper bound: current_sum + sum of reciprocals of all y >= x not forbidden
            # Quick bound using suffix_sum
            ub = current_sum
            for y in range(x, n + 1):
                if not ((forbidden_bits >> (y - 1)) & 1):
                    ub += recip[y]
            if ub <= best_sum + 1e-12:
                return

            # If x is forbidden, skip it
            if (forbidden_bits >> (x - 1)) & 1:
                bb_recursive(x + 1, selected_bits, forbidden_bits, current_sum)
                return

            # Option 1: Don't select x
            bb_recursive(x + 1, selected_bits, forbidden_bits, current_sum)

            # Option 2: Select x (if valid)
            # Check conflicts: for each (a, b) in conflict_pairs[x], if both a and b in selected, invalid
            valid = True
            for a, b in conflict_pairs[x]:
                if ((selected_bits >> (a - 1)) & 1) and ((selected_bits >> (b - 1)) & 1):
                    valid = False
                    break

            if not valid:
                return

            # New selected bits
            new_selected = selected_bits | (1 << (x - 1))

            # New forbidden bits: for each (a, c) in forbid_map[x], if a in selected, forbid c
            new_forbidden = forbidden_bits
            for a, c in forbid_map[x]:
                if (selected_bits >> (a - 1)) & 1:
                    new_forbidden |= (1 << (c - 1))

            # Also, if x itself would be forbidden by future selections, we don't need to check now
            # But we should ensure x is not already forbidden (already checked above)

            bb_recursive(x + 1, new_selected, new_forbidden, current_sum + recip[x])

        bb_recursive(1, 0, 0, 0.0)
        return best_set, best_sum

    # Simulated annealing for polishing
    def simulated_annealing(in_set, max_iters=2000):
        current_sum = get_sum(in_set)
        best_in_set = in_set[:]
        best_sum = current_sum

        elements = [i for i in range(1, n + 1) if in_set[i]]
        non_elements = [i for i in range(1, n + 1) if not in_set[i]]

        T_start = 0.1
        T_end = 0.0001
        cooling = (T_end / T_start) ** (1.0 / max_iters)
        T = T_start

        for it in range(max_iters):
            if time.time() > deadline - 5.0:
                break

            T *= cooling
            move_type = random.random()

            if move_type < 0.5 and elements and non_elements:
                rem = random.choice(elements)
                add = random.choice(non_elements)

                in_set[rem] = False
                if can_add_check(add, in_set):
                    in_set[add] = True
                    delta = recip[add] - recip[rem]
                    new_sum = current_sum + delta

                    if new_sum > current_sum or random.random() < pow(2.718281828, delta / max(T, 1e-10)):
                        current_sum = new_sum
                        elements.remove(rem)
                        non_elements.remove(add)
                        elements.append(add)
                        non_elements.append(rem)

                        if current_sum > best_sum + 1e-15:
                            best_sum = current_sum
                            best_in_set = in_set[:]
                    else:
                        in_set[add] = False
                in_set[rem] = True

            elif move_type < 0.8 and len(elements) >= 2 and len(non_elements) >= 2:
                rem1, rem2 = random.sample(elements, 2)
                add1, add2 = random.sample(non_elements, 2)

                in_set[rem1] = False
                in_set[rem2] = False

                valid = True
                if not can_add_check(add1, in_set):
                    valid = False
                else:
                    in_set[add1] = True
                    if not can_add_check(add2, in_set):
                        valid = False
                        in_set[add1] = False
                    else:
                        in_set[add2] = True

                if valid:
                    delta = recip[add1] + recip[add2] - recip[rem1] - recip[rem2]
                    new_sum = current_sum + delta

                    if new_sum > current_sum or random.random() < pow(2.718281828, delta / max(T, 1e-10)):
                        current_sum = new_sum
                        elements.remove(rem1)
                        elements.remove(rem2)
                        non_elements.remove(add1)
                        non_elements.remove(add2)
                        elements.extend([add1, add2])
                        non_elements.extend([rem1, rem2])

                        if current_sum > best_sum + 1e-15:
                            best_sum = current_sum
                            best_in_set = in_set[:]
                    else:
                        in_set[add1] = False
                        in_set[add2] = False

                in_set[rem1] = True
                in_set[rem2] = True

            elif move_type < 0.9 and elements:
                rem = random.choice(elements)
                in_set[rem] = False
                delta = -recip[rem]
                new_sum = current_sum + delta

                if new_sum > current_sum or random.random() < pow(2.718281828, delta / max(T, 1e-10)):
                    current_sum = new_sum
                    elements.remove(rem)
                    non_elements.append(rem)

                    if current_sum > best_sum + 1e-15:
                        best_sum = current_sum
                        best_in_set = in_set[:]
                else:
                    in_set[rem] = True

            else:
                if not non_elements:
                    continue
                add = random.choice(non_elements)
                if can_add_check(add, in_set):
                    in_set[add] = True
                    delta = recip[add]
                    new_sum = current_sum + delta

                    if new_sum > current_sum or random.random() < pow(2.718281828, delta / max(T, 1e-10)):
                        current_sum = new_sum
                        non_elements.remove(add)
                        elements.append(add)

                        if current_sum > best_sum + 1e-15:
                            best_sum = current_sum
                            best_in_set = in_set[:]
                    else:
                        in_set[add] = False

        return best_in_set

    # Main algorithm
    random.seed(42)

    # Phase 1: Greedy construction
    best_in_set = greedy_construct()
    best_sum = get_sum(best_in_set)

    # Phase 2: Branch and bound (try to find better solution)
    if time.time() < deadline - 20.0:
        bb_set, bb_sum = branch_and_bound()
        if bb_sum > best_sum + 1e-15:
            best_in_set = bb_set
            best_sum = bb_sum

    # Phase 3: Local search polish
    if time.time() < deadline - 10.0:
        polished = local_search(best_in_set[:])
        p_sum = get_sum(polished)
        if p_sum > best_sum + 1e-15:
            best_in_set = polished
            best_sum = p_sum

    # Phase 4: Simulated annealing with restarts
    num_restarts = 3
    for r in range(num_restarts):
        if time.time() > deadline - 8.0:
            break

        perturbed = best_in_set[:]
        num_removals = random.randint(2, 6)
        elements = [i for i in range(1, n + 1) if perturbed[i]]

        for _ in range(min(num_removals, len(elements))):
            if not elements:
                break
            idx = random.randint(0, len(elements) - 1)
            rem = elements[idx]
            perturbed[rem] = False
            elements.pop(idx)

        sa_result = simulated_annealing(perturbed, max_iters=1500)
        sa_sum = get_sum(sa_result)

        if sa_sum > best_sum + 1e-15:
            best_in_set = sa_result
            best_sum = sa_sum

    # Phase 5: Final local search
    if time.time() < deadline - 3.0:
        polished = local_search(best_in_set[:])
        p_sum = get_sum(polished)
        if p_sum > best_sum + 1e-15:
            best_in_set = polished
            best_sum = p_sum

    # Extract result
    result = [i for i in range(1, n + 1) if best_in_set[i]]

    # Validation
    s_set = set(result)
    for x in range(1, n + 1):
        for a, b in conflict_pairs[x]:
            if a in s_set and b in s_set and x in s_set:
                raise AssertionError(f"3-AP found: {a}, {b}, {x}")

    return result
