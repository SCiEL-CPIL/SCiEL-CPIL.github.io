import time
import random
import math

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # Safety margin for 90s limit

    n = 256

    # Precompute divisibility: divisors[x] = set of a in [1,x-1] such that a | x
    # This is used for fast conflict checking.
    divisors = [set() for _ in range(2 * n + 1)]
    for a in range(1, n + 1):
        for multiple in range(2 * a, 2 * n + 1, a):
            divisors[multiple].add(a)

    def is_valid_addition(x, S):
        """Check if adding x to S keeps the set valid."""
        if not S:
            return True

        less = [y for y in S if y < x]
        greater = [y for y in S if y > x]

        # Case 1: x is 'a' (smallest in triple)
        # b, c in greater, b <= c, check x | (b+c)
        if len(greater) >= 2:
            g_sorted = sorted(greater)
            for i in range(len(g_sorted)):
                b = g_sorted[i]
                for j in range(i, len(g_sorted)):
                    c = g_sorted[j]
                    if x in divisors[b + c]:
                        return False

        # Case 2: x is 'b' (middle in triple)
        # a in less, c in S with c >= x
        for a in less:
            # c = x (b=c=x)
            if a in divisors[2 * x]:
                return False
            # c in greater
            for c in greater:
                if a in divisors[x + c]:
                    return False

        # Case 3: x is 'c' (largest in triple)
        # a, b in less, a < b, check a | (b+x)
        if len(less) >= 2:
            l_sorted = sorted(less)
            for i in range(len(l_sorted)):
                a = l_sorted[i]
                for j in range(i + 1, len(l_sorted)):
                    b = l_sorted[j]
                    if a in divisors[b + x]:
                        return False

        return True

    def is_valid_set(S):
        """Check if the entire set S is valid."""
        S_sorted = sorted(S)
        m = len(S_sorted)
        for i in range(m):
            a = S_sorted[i]
            for j in range(i + 1, m):
                b = S_sorted[j]
                for k in range(j, m):
                    c = S_sorted[k]
                    if a in divisors[b + c]:
                        return False
        return True

    # Strategy:
    # 1. Greedy from large to small
    # 2. Upper tail pruning + fill
    # 3. Random restarts with local improvement
    # 4. Conflict-graph-based greedy (new)
    # 5. Final polish

    best_set = []
    best_size = 0

    # --- Phase 1: Greedy from large to small ---
    S_greedy = []
    for x in range(n, 0, -1):
        if time.time() > deadline:
            break
        if is_valid_addition(x, S_greedy):
            S_greedy.append(x)

    if len(S_greedy) > best_size:
        best_size = len(S_greedy)
        best_set = S_greedy[:]

    # --- Phase 2: Upper tail pruning + fill ---
    for start_val in [128, 140, 150, 160, 170, 180, 200]:
        if time.time() > deadline:
            break

        S_pruned = list(range(start_val, n + 1))

        # Prune until valid
        changed = True
        while changed:
            changed = False
            if is_valid_set(S_pruned):
                break

            S_sorted = sorted(S_pruned)
            conflict_found = False
            for i in range(len(S_sorted)):
                a = S_sorted[i]
                for j in range(i + 1, len(S_sorted)):
                    b = S_sorted[j]
                    for k in range(j, len(S_sorted)):
                        c = S_sorted[k]
                        if a in divisors[b + c]:
                            # Remove the element that causes the most conflicts
                            # Heuristic: remove 'a' (smallest, divides many sums)
                            S_pruned.remove(a)
                            changed = True
                            conflict_found = True
                            break
                    if conflict_found:
                        break
                if conflict_found:
                    break

            if time.time() > deadline:
                break

        # Fill with smaller elements
        S_targeted = S_pruned[:]
        for x in range(1, start_val):
            if time.time() > deadline:
                break
            if x in S_targeted:
                continue
            if is_valid_addition(x, S_targeted):
                S_targeted.append(x)

        if len(S_targeted) > best_size:
            best_size = len(S_targeted)
            best_set = S_targeted[:]

    # --- Phase 3: Random restarts with local improvement ---
    for attempt in range(25):
        if time.time() > deadline:
            break

        candidates = list(range(1, n + 1))
        random.shuffle(candidates)

        S_try = []
        for x in candidates:
            if time.time() > deadline:
                break
            if is_valid_addition(x, S_try):
                S_try.append(x)

        # Local improvement: try to add any remaining numbers
        S_improved = S_try[:]
        for x in range(1, n + 1):
            if time.time() > deadline:
                break
            if x in S_improved:
                continue
            if is_valid_addition(x, S_improved):
                S_improved.append(x)

        if len(S_improved) > best_size:
            best_size = len(S_improved)
            best_set = S_improved[:]

    # --- Phase 4: Conflict-graph-based greedy ---
    # Build conflict graph: vertices are [1..n], edge between b,c if there exists a in set
    # such that a | (b+c) and a < b <= c.
    # Actually, the conflict is triple-based. Let's think differently.
    # For a given set S, a new element x conflicts if adding it creates a bad triple.
    # The conflict graph approach: for each pair (b,c) with b<=c, the "forbidden a's" are
    # divisors of (b+c) that are < b. If any of those a's are in S, then (a,b,c) is a conflict.
    # So for greedy selection, when considering adding x, we check all pairs involving x.
    # This is what is_valid_addition does.
    # Instead, let's try a different greedy: sort by "conflict degree" and pick low-degree first.

    # Compute for each x, how many conflicts it would have with a dense upper set.
    # Actually, let's try: start with empty set, repeatedly add the element with fewest
    # conflicts with the current set, breaking ties by larger value.

    for attempt in range(10):
        if time.time() > deadline:
            break

        S_cg = []
        remaining = set(range(1, n + 1))

        while remaining:
            if time.time() > deadline:
                break

            best_x = None
            best_conflicts = float('inf')
            best_val = -1

            for x in remaining:
                # Count conflicts if we add x to S_cg
                # This is the same as checking is_valid_addition, but we want to count
                # rather than just check.
                # For efficiency, let's just check validity.
                if is_valid_addition(x, S_cg):
                    # Valid: prefer larger x
                    if x > best_val:
                        best_x = x
                        best_val = x
                        best_conflicts = 0

            if best_x is None:
                break

            S_cg.append(best_x)
            remaining.remove(best_x)

        if len(S_cg) > best_size:
            best_size = len(S_cg)
            best_set = S_cg[:]

    # --- Phase 5: Swap-based local search on best_set ---
    # Try to swap elements in best_set with elements not in best_set to improve size.
    # Actually, since we're maximizing cardinality, we can't swap (same size).
    # Instead, try to find a set of k elements to remove and k+1 to add.
    # This is complex. Let's try a simpler approach: try to add elements one by one,
    # and if that fails, try removing one element and adding two.

    S_final = best_set[:]
    S_set = set(S_final)

    # Try single additions
    for x in range(1, n + 1):
        if time.time() > deadline:
            break
        if x in S_set:
            continue
        if is_valid_addition(x, S_final):
            S_final.append(x)
            S_set.add(x)

    # Try double additions (remove 1, add 2)
    for x in range(1, n + 1):
        if time.time() > deadline:
            break
        if x in S_set:
            continue
        # Try to find a y in S_final such that removing y and adding x is valid,
        # and then also add another z.
        # This is expensive. Let's just try removing each y and adding x.
        for y in list(S_final):
            if time.time() > deadline:
                break
            S_temp = [v for v in S_final if v != y]
            if is_valid_addition(x, S_temp):
                S_temp.append(x)
                # Now try to add another element
                for z in range(1, n + 1):
                    if time.time() > deadline:
                        break
                    if z in S_set or z == x:
                        continue
                    if is_valid_addition(z, S_temp):
                        S_temp.append(z)
                        if len(S_temp) > len(S_final):
                            S_final = S_temp
                            S_set = set(S_final)
                        break

    if len(S_final) > best_size:
        best_size = len(S_final)
        best_set = S_final[:]

    # --- Final polish: Try to add any remaining numbers to best_set ---
    S_polish = best_set[:]
    S_polish_set = set(S_polish)
    for x in range(1, n + 1):
        if time.time() > deadline:
            break
        if x in S_polish_set:
            continue
        if is_valid_addition(x, S_polish):
            S_polish.append(x)
            S_polish_set.add(x)

    if len(S_polish) > best_size:
        best_size = len(S_polish)
        best_set = S_polish[:]

    return best_set
# EVOLVE-BLOCK-END
