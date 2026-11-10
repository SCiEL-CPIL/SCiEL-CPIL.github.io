import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    n = 192
    max_k = 48

    # Precompute for each x in [1..n]:
    # For condition 1: x | (a+b+c) for distinct a,b,c in S
    #   For each x, precompute a set of "forbidden pair sums" mod x
    #   i.e., for each pair (a,b) with a<b, a!=x, b!=x, compute (a+b) % x
    #   Then when checking if x can be added to S, for each c in S\{x},
    #   check if (a+b) % x == (-c) % x for any pair (a,b) in S\{x,c}
    #   Actually simpler: precompute all pair sums mod x for pairs from [1..n]\{x}
    #   Then for a set S, check: does there exist c in S such that
    #   (-c) % x is in the set of pair_sums from S\{c}?
    #   Even simpler: for each x, precompute forbidden_pair_sums_mod_x = set of (a+b)%x for all a<b, a,b!=x
    #   Then x can be added to S iff for all c in S: (-c)%x not in { (a+b)%x : a<b, a,b in S\{c} }
    #   But this requires knowing which pairs are in S, not just global pairs.
    #
    # Better approach: For each x, precompute a 2D boolean table:
    #   conflict[x][a][b] = True if there exists some c (c != x, c != a, c != b)
    #   in [1..n] such that x | (a+b+c)
    #   But this is O(n^4) which is 192^4 ≈ 1.3 billion - too much.
    #
    # Let me think differently. The constraint is:
    # For x in S: no three distinct a,b,c in S\{x} have x | (a+b+c)
    # For a in S: no two distinct u,v in S\{a} have a | (x+u+v) [where x is the new element]
    #
    # Fast check strategy:
    # For adding x to S:
    #   Part 1: Check if any triple in S sums to a multiple of x.
    #     Precompute for each x: residues of all pair sums (a+b) mod x for a<b from [1..n]\{x}
    #     Store as a set. Then for each c in S, check if (-c)%x is achievable as a pair sum
    #     from elements in S\{c}. But we need pairs from S specifically, not global.
    #     Alternative: for each x, maintain a count array count_x[r] = number of pairs (a,b) in S
    #     with a<b, a,b != x, and (a+b)%x == r. Then x is valid iff for all c in S\{x},
    #     count_x[(-c)%x] == 0 OR the only pairs achieving (-c)%x involve c itself.
    #     This is getting complex. Let's use a simpler O(|S|^2) approach with early termination.
    #
    # Actually, let me just optimize the O(|S|^3) check with numpy vectorization.
    #
    # New structural approach:
    # 1. Use a different formulation: work with residue classes.
    # 2. Key observation: if all elements are > n/2, then no element can divide a sum of
    #    three others (since min sum > n/2*3 > n/2*1, and max element <= n, so x | sum
    #    requires sum >= x, and if x > n/2, sum of three elements each > n/2 is > 3n/2,
    #    so sum could be x, 2x, or 3x... wait that doesn't help directly).
    #
    # Let me try a completely different structural approach:
    # Use a SAT-like constraint satisfaction with iterative deepening.
    # Or better: use a different search strategy based on "density" of residues.

    # Precompute pair sums for fast triple-sum check
    # For each x, we want to quickly check: does S contain three elements a,b,c
    # such that (a+b+c) % x == 0?
    # Equivalent: for each c in S, does S\{c} contain a pair (a,b) with (a+b) % x == (-c) % x?
    # Precompute for each x: a set of all possible pair-sum residues from [1..n]\{x}
    # Then maintain for current S: for each x, a count of pair-sum residues in S.

    # Precompute: for each x in 1..n, all pair sums mod x from elements in [1..n]\{x}
    # pair_sums_mod[x] = set of (a+b) % x for all a < b, a != x, b != x, 1 <= a,b <= n
    pair_sums_mod = [None] * (n + 1)
    for x in range(1, n + 1):
        residues = set()
        for a in range(1, n + 1):
            if a == x:
                continue
            for b in range(a + 1, n + 1):
                if b == x:
                    continue
                residues.add((a + b) % x)
        pair_sums_mod[x] = residues

    # For the second condition: for each a in S, check if there exist u,v in S\{a}
    # such that a | (x + u + v), i.e., (x + u + v) % a == 0
    # Equivalent: (u + v) % a == (-x) % a
    # Precompute for each a: pair_sums_mod[a] already exists.
    # So for adding x to S: for each a in S, check if (-x) % a is in the pair-sum residues
    # achievable from S\{a}. Again need S-specific pairs.

    # Let me just do a fast O(|S|^2) check with early termination:
    def can_add_fast(x, S_list):
        """Check if x can be added to set given as list. Returns True if valid."""
        n_s = len(S_list)
        S_set = set(S_list)

        # Part 1: x does not divide sum of any 3 distinct elements in S
        # For each c in S, check if there's a pair (a,b) in S\{c} with (a+b+c) % x == 0
        # i.e., (a+b) % x == (-c) % x
        if n_s >= 3:
            # Build a frequency map of pair sums mod x for pairs in S
            # But we need to exclude pairs involving c
            # Simpler: for each c, scan all pairs in S\{c}
            for c in S_list:
                target = (-c) % x
                found = False
                for i in range(n_s):
                    a = S_list[i]
                    if a == c:
                        continue
                    for j in range(i + 1, n_s):
                        b = S_list[j]
                        if b == c:
                            continue
                        if (a + b) % x == target:
                            found = True
                            break
                    if found:
                        break
                if found:
                    return False

        # Part 2: For each a in S, a does not divide (x + u + v) for any distinct u,v in S\{a}
        # i.e., (x + u + v) % a != 0 for all distinct u,v in S\{a}
        # Equivalent: (u + v) % a != (-x) % a for all distinct u,v in S\{a}
        if n_s >= 2:
            for a in S_list:
                target = (-x) % a
                found = False
                for i in range(n_s):
                    u = S_list[i]
                    if u == a:
                        continue
                    for j in range(i + 1, n_s):
                        v = S_list[j]
                        if v == a:
                            continue
                        if (u + v) % a == target:
                            found = True
                            break
                    if found:
                        break
                if found:
                    return False

        return True

    # Even faster: precompute for each x, a lookup table
    # For a given set S, maintain:
    #   pair_residues[x] = dict mapping residue r -> count of pairs (a,b) in S with a<b, a,b!=x, (a+b)%x==r
    # This allows O(|S|) check for part 1.
    # But maintaining this dynamically is complex. Let's use a hybrid approach.

    # New approach: use a different construction strategy.
    # Key structural insight: if we pick all numbers in a range (n/2, n], 
    # then any sum of 3 such numbers is in (3n/2, 3n], and any element x in (n/2, n]
    # can only divide the sum if sum = x, 2x, or 3x (since sum < 4x for x > n/2... 
    # actually sum can be up to 3n and x can be as small as n/2+1, so sum/x can be up to ~6).
    # This doesn't immediately simplify things.

    # Let me try a completely different structural approach:
    # Formulate as a maximum independent set in a conflict graph.
    # Two elements x and y are "compatible" if they can coexist in a valid set.
    # But compatibility is not pairwise - it depends on the whole set.
    # So this doesn't directly work.

    # Alternative structural change: use simulated annealing with a smart move set.
    # State: a set S. Moves: add one element, remove one element, swap two elements.
    # Energy: number of violations (not just valid/invalid).

    def count_violations(S_list):
        """Count the number of constraint violations in S_list."""
        S_set = set(S_list)
        n_s = len(S_list)
        violations = 0

        # For each x in S, count triples (a,b,c) in S\{x} with x | (a+b+c)
        for x in S_list:
            if n_s >= 4:  # need at least 3 others
                for i in range(n_s):
                    a = S_list[i]
                    if a == x:
                        continue
                    for j in range(i + 1, n_s):
                        b = S_list[j]
                        if b == x:
                            continue
                        for k in range(j + 1, n_s):
                            c = S_list[k]
                            if c == x:
                                continue
                            if (a + b + c) % x == 0:
                                violations += 1

        # For each a in S, count pairs (u,v) in S\{a} with a | (x+u+v) for some x in S\{a}
        # Wait, the constraint is: no element a in S divides (x + u + v) where x,u,v are
        # three DISTINCT OTHER elements. So for each a, for each triple (x,u,v) in S\{a},
        # check if a | (x+u+v).
        for a in S_list:
            others = [y for y in S_list if y != a]
            n_o = len(others)
            if n_o >= 3:
                for i in range(n_o):
                    x = others[i]
                    for j in range(i + 1, n_o):
                        u = others[j]
                        for k in range(j + 1, n_o):
                            v = others[k]
                            if (x + u + v) % a == 0:
                                violations += 1

        return violations

    # Simulated annealing approach
    def simulated_annealing(initial_set, iterations_per_phase=200):
        """Run simulated annealing to improve the set."""
        current = list(initial_set)
        current_set = set(current)
        current_violations = count_violations(current)
        best = list(current)
        best_violations = current_violations
        best_size = len(current)

        # We want to maximize size while having 0 violations.
        # Use a two-phase approach: first reduce violations to 0, then maximize size.

        all_elements = list(range(1, n + 1))
        temp = 10.0
        cooling = 0.995

        for iteration in range(iterations_per_phase):
            if time.time() > deadline - 5:
                break

            # Choose a random move
            move_type = random.random()
            if move_type < 0.3 and len(current) < n:
                # Add a random element
                candidates = [x for x in all_elements if x not in current_set]
                if candidates:
                    x = random.choice(candidates)
                    new_set = current + [x]
                    new_set_set = current_set | {x}
                    new_violations = count_violations(new_set)
                    delta = new_violations - current_violations
                    if delta <= 0 or random.random() < pow(2.718, -delta / max(temp, 0.01)):
                        current = new_set
                        current_set = new_set_set
                        current_violations = new_violations
            elif move_type < 0.5 and len(current) > 1:
                # Remove a random element
                idx = random.randint(0, len(current) - 1)
                x = current[idx]
                new_set = current[:idx] + current[idx+1:]
                new_set_set = current_set - {x}
                new_violations = count_violations(new_set)
                delta = new_violations - current_violations
                if delta <= 0 or random.random() < pow(2.718, -delta / max(temp, 0.01)):
                    current = new_set
                    current_set = new_set_set
                    current_violations = new_violations
            elif move_type < 0.8 and len(current) > 0:
                # Swap: remove one, add another
                idx = random.randint(0, len(current) - 1)
                x = current[idx]
                candidates = [y for y in all_elements if y not in current_set]
                if candidates:
                    y = random.choice(candidates)
                    new_set = current[:idx] + [y] + current[idx+1:]
                    new_set_set = current_set - {x} | {y}
                    new_violations = count_violations(new_set)
                    delta = new_violations - current_violations
                    if delta <= 0 or random.random() < pow(2.718, -delta / max(temp, 0.01)):
                        current = new_set
                        current_set = new_set_set
                        current_violations = new_violations
            else:
                # Just add (if possible) or do nothing
                candidates = [x for x in all_elements if x not in current_set]
                if candidates:
                    x = random.choice(candidates)
                    new_set = current + [x]
                    new_set_set = current_set | {x}
                    new_violations = count_violations(new_set)
                    delta = new_violations - current_violations
                    if delta <= 0 or random.random() < pow(2.718, -delta / max(temp, 0.01)):
                        current = new_set
                        current_set = new_set_set
                        current_violations = new_violations

            # Update best
            if current_violations == 0 and len(current) > best_size:
                best_size = len(current)
                best = list(current)
                best_violations = 0
            elif current_violations < best_violations:
                best_violations = current_violations
                best = list(current)
                best_size = len(current)

            temp *= cooling

        return best, best_violations

    # Greedy construction with the fast check
    def greedy_construct(order):
        """Greedy construction with a given ordering."""
        S_list = []
        S_set = set()
        for x in order:
            if time.time() > deadline - 5:
                break
            if can_add_fast(x, S_list):
                S_list.append(x)
                S_set.add(x)
                if len(S_list) >= max_k:
                    break
        return S_list

    # Phase 1: Multiple greedy constructions with different orderings
    best_set = []
    best_size = 0

    # Descending order (large numbers first - they're harder to satisfy)
    s = greedy_construct(range(n, 0, -1))
    if len(s) > best_size:
        best_size = len(s)
        best_set = s

    # Ascending order
    s = greedy_construct(range(1, n + 1))
    if len(s) > best_size:
        best_size = len(s)
        best_set = s

    # Random orderings
    for trial in range(30):
        if time.time() > deadline - 10:
            break
        order = list(range(1, n + 1))
        random.shuffle(order)
        s = greedy_construct(order)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s

    # Phase 2: Local search - try to add any element
    if time.time() < deadline - 15:
        current = list(best_set)
        current_set = set(current)
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            for x in range(1, n + 1):
                if x not in current_set:
                    if can_add_fast(x, current):
                        current.append(x)
                        current_set.add(x)
                        improved = True
                        if len(current) > best_size:
                            best_size = len(current)
                            best_set = list(current)
                        break

    # Phase 3: Swap-based local search
    if time.time() < deadline - 10:
        current = list(best_set)
        current_set = set(current)
        no_improve_count = 0
        max_no_improve = 20

        while no_improve_count < max_no_improve and time.time() < deadline - 5:
            improved = False
            # Try all possible swaps: remove one, add another
            for i, x in enumerate(current):
                if time.time() > deadline - 5:
                    break
                for y in range(1, n + 1):
                    if y == x or y in current_set:
                        continue
                    # Try swap
                    test_set = set(current) - {x} | {y}
                    test_list = list(test_set)
                    if can_add_fast(y, list(test_set - {y})):
                        # Check if this improves things (size is same, but maybe enables further additions)
                        # For now, accept swaps that maintain validity
                        current = list(test_set)
                        current_set = test_set
                        improved = True
                        break
                if improved:
                    break

            if not improved:
                no_improve_count += 1
            else:
                no_improve_count = 0
                # After a swap, try to add more
                for x in range(1, n + 1):
                    if x not in current_set:
                        if can_add_fast(x, current):
                            current.append(x)
                            current_set.add(x)
                            if len(current) > best_size:
                                best_size = len(current)
                                best_set = list(current)

    # Phase 4: Simulated annealing from best
    if time.time() < deadline - 10:
        sa_result, sa_violations = simulated_annealing(best_set, iterations_per_phase=500)
        if sa_violations == 0 and len(sa_result) > best_size:
            best_size = len(sa_result)
            best_set = sa_result

    # Final verification
    final_set = list(set(best_set))
    # Verify the set is valid
    valid = True
    S_set = set(final_set)
    for x in final_set:
        others = [y for y in final_set if y != x]
        # Check x doesn't divide sum of any 3 distinct others
        for i in range(len(others)):
            for j in range(i + 1, len(others)):
                for k in range(j + 1, len(others)):
                    if (others[i] + others[j] + others[k]) % x == 0:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        if not valid:
            break
        # Check x doesn't divide (a+b+c) for any 3 distinct others - already checked
        # Check that no other element divides (x + u + v) for distinct u,v in S\{other}
    for a in final_set:
        others = [y for y in final_set if y != a]
        for i in range(len(others)):
            for j in range(i + 1, len(others)):
                for k in range(j + 1, len(others)):
                    if (others[i] + others[j] + others[k]) % a == 0:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        if not valid:
            break

    if not valid:
        # Fallback: use a simple valid set
        # All numbers > 96: sum of 3 is > 288, max element is 192, 
        # 192 can divide 288 (288=192+96, no... 288/192=1.5, not integer)
        # Actually let's just return the greedy descending result which should be valid
        final_set = greedy_construct(range(n, 0, -1))

    return final_set
