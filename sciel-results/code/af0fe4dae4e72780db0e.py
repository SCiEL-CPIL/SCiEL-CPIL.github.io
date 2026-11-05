import time
import random

def construct():
    start_time = time.time()
    DEADLINE = start_time + 85.0  # Safety margin for 90s limit
    n = 256

    def can_add(S_sorted, x):
        """Check if x can be added to S_sorted without violating constraints.
        Uses residue counting: for x to be 'a', we need no pair (b,c) in S with
        b <= c, b > x, and x | (b+c).
        Also need to check x as 'b' or 'c' in triples where a < x.
        """
        S_set = set(S_sorted)
        max_s = S_sorted[-1] if S_sorted else 0

        # Case 1: x is the smallest (a = x), need b, c in S with x < b <= c and x | (b+c)
        # b+c must be a multiple of x: k*x for k >= 2 (since b+c > 2x when b > x)
        # b+c ranges from 2*(x+1) to 2*max_s
        # For each k*x in that range, check if there exist b, c in S with b <= c, b > x, b+c = k*x
        # Equivalently: for each b in S with b > x and b <= k*x/2, check if k*x - b is in S and >= b
        k_min = (2 * (x + 1) + x - 1) // x  # smallest k such that k*x >= 2*(x+1)
        k_max = (2 * max_s) // x
        for k in range(k_min, k_max + 1):
            target = k * x
            # b in S, x < b <= target//2, and target - b in S, target - b >= b
            # b ranges from x+1 to target//2
            # Iterate through S_sorted elements in range (x, target//2]
            # Since S_sorted is sorted, we can find the range
            # For efficiency, just check all b in S with x < b <= target//2
            for b in S_sorted:
                if b <= x:
                    continue
                if b > target // 2:
                    break
                c = target - b
                if c < b:
                    continue
                if c in S_set:
                    return False

        # Case 2: x is the middle element (a < x <= c), a in S, c in S, a | (x+c)
        # a < x, so a in S with a < x
        # c >= x, c in S
        # x + c = k*a for some k >= 2 (since x+c > 2a when c >= x > a)
        # Actually k*a = x+c, with a < x <= c
        # c = k*a - x, need c >= x, c in S
        # k*a - x >= x => k*a >= 2x => k >= ceil(2x/a)
        # k*a - x <= max_s => k <= (max_s + x) / a
        for a in S_sorted:
            if a >= x:
                break
            k_min = (2 * x + a - 1) // a
            k_max = (max_s + x) // a
            for k in range(k_min, k_max + 1):
                c = k * a - x
                if c < x:
                    continue
                if c in S_set:
                    return False

        # Case 3: x is the largest element (a < b <= x), a, b in S, a | (b+x)
        # a < b <= x, a, b in S
        # b + x = k*a, k >= 2
        # b = k*a - x, need a < b <= x, b in S
        # k*a - x > a => k*a > a + x => k > (a+x)/a = 1 + x/a
        # k*a - x <= x => k*a <= 2x => k <= 2x/a
        for a in S_sorted:
            if a >= x:
                break
            k_min = (x + a + 1) // a  # k*a > a + x
            k_max = (2 * x) // a
            for k in range(k_min, k_max + 1):
                b = k * a - x
                if b <= a or b > x:
                    continue
                if b in S_set:
                    return False

        return True

    def is_valid(S_list):
        """Verify the entire set is valid."""
        S_list = sorted(S_list)
        S_set = set(S_list)
        for i, a in enumerate(S_list):
            for j in range(i + 1, len(S_list)):
                b = S_list[j]
                for c_val in S_list[j:]:
                    c = c_val
                    if (b + c) % a == 0:
                        return False
        return True

    # Strategy 1: Start from [171, 256] and greedily add smaller elements
    best_solution = list(range(171, 257))
    best_size = len(best_solution)

    # Verify base
    if not is_valid(best_solution):
        # Fallback to a safe interval
        best_solution = list(range(192, 257))
        best_size = len(best_solution)

    S1 = list(best_solution)
    S1_sorted = sorted(S1)
    # Try adding elements from 170 down to 1
    for x in range(170, 0, -1):
        if time.time() > DEADLINE:
            break
        if can_add(S1_sorted, x):
            S1.append(x)
            S1_sorted = sorted(S1)

    if len(S1) > best_size:
        best_size = len(S1)
        best_solution = list(S1)

    # Strategy 2: Try different base intervals
    for k in range(155, 178):
        if time.time() > DEADLINE:
            break
        S2 = list(range(k, n + 1))
        S2_sorted = sorted(S2)
        # Verify base
        if not is_valid(S2):
            continue
        # Greedily add smaller elements
        for x in range(k - 1, 0, -1):
            if time.time() > DEADLINE:
                break
            if can_add(S2_sorted, x):
                S2.append(x)
                S2_sorted = sorted(S2)
        if len(S2) > best_size:
            best_size = len(S2)
            best_solution = list(S2)

    # Strategy 3: Local search with random restarts and swaps
    random.seed(42)
    for restart in range(50):
        if time.time() > DEADLINE:
            break

        # Start from best solution with perturbation
        S_cur = list(best_solution)
        if restart > 0 and len(S_cur) > 15:
            # Remove 1-2 random elements
            for _ in range(random.randint(1, 2)):
                if S_cur:
                    idx = random.randint(0, len(S_cur) - 1)
                    S_cur.pop(idx)

        S_cur_sorted = sorted(S_cur)
        S_cur_set = set(S_cur)

        # Phase 1: Try to add any missing element (greedy fill)
        changed = True
        while changed and time.time() < DEADLINE:
            changed = False
            for x in range(1, n + 1):
                if time.time() > DEADLINE:
                    break
                if x not in S_cur_set:
                    if can_add(S_cur_sorted, x):
                        S_cur.append(x)
                        S_cur_set.add(x)
                        S_cur_sorted = sorted(S_cur)
                        changed = True
                        break

        # Phase 2: Try swaps to improve
        for _ in range(30):
            if time.time() > DEADLINE:
                break
            if len(S_cur) <= 10:
                break

            S_cur_set = set(S_cur)
            S_cur_sorted = sorted(S_cur)

            # Pick a random element to remove
            y = random.choice(S_cur)
            S_temp = [v for v in S_cur if v != y]
            S_temp_set = set(S_temp)
            S_temp_sorted = sorted(S_temp)

            # Try to add an element that wasn't in S_cur
            best_add = None
            for x in range(1, n + 1):
                if time.time() > DEADLINE:
                    break
                if x not in S_temp_set and x != y:
                    if can_add(S_temp_sorted, x):
                        best_add = x
                        break

            if best_add is not None and len(S_temp) + 1 >= len(S_cur):
                S_cur = S_temp + [best_add]

        if len(S_cur) > best_size:
            best_size = len(S_cur)
            best_solution = list(S_cur)

    # Final verification
    S_final = sorted(best_solution)
    if not is_valid(S_final):
        # Fallback to verified base
        S_final = list(range(171, 257))
        if not is_valid(S_final):
            S_final = list(range(192, 257))

    return S_final
