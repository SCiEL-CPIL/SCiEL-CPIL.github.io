"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations_with_replacement, combinations

def construct():
    start_time = time.time()
    deadline = start_time + 75  # safety margin before 90s limit

    n = 256
    max_size = 64

    def is_sidon(s):
        """Check if s is a Sidon set (all pair sums including a+a are unique)."""
        sums = set()
        for i in range(len(s)):
            for j in range(i, len(s)):
                val = s[i] + s[j]
                if val in sums:
                    return False
                sums.add(val)
        return True

    def get_diffs(s):
        """Get nonzero differences of set s."""
        return {b - a for a, b in combinations(s, 2)}

    def try_add_greedy(sets, order, randomize=False):
        """
        Greedily build two Sidon sets with disjoint nonzero differences.
        'order' is the list of candidates to try in sequence.
        Returns the best (A, B) found during this greedy pass.
        """
        A = []
        B = []
        diffs_A = set()
        diffs_B = set()
        sums_A = set()
        sums_B = set()

        for x in order:
            # Try adding to A
            can_add_A = True
            # Check Sidon condition for A
            for a in A:
                if (x + x) in sums_A or (x + a) in sums_A:
                    can_add_A = False
                    break
            if can_add_A:
                # Check difference disjointness
                new_diffs_A = set()
                for a in A:
                    d = x - a
                    if d in diffs_B or d in diffs_A:
                        can_add_A = False
                        break
                    new_diffs_A.add(d)
                if can_add_A:
                    # Also check that existing A elements don't create diff conflicts
                    # (already handled by checking d in diffs_A above for new diffs)
                    # But we also need to ensure no existing pair in A has diff in diffs_B
                    # That's already maintained by invariant
                    A.append(x)
                    diffs_A.update(new_diffs_A)
                    sums_A.add(2 * x)
                    for a in A[:-1]:
                        sums_A.add(x + a)
                    continue

            # Try adding to B
            can_add_B = True
            for b in B:
                if (x + x) in sums_B or (x + b) in sums_B:
                    can_add_B = False
                    break
            if can_add_B:
                new_diffs_B = set()
                for b in B:
                    d = x - b
                    if d in diffs_A or d in diffs_B:
                        can_add_B = False
                        break
                    new_diffs_B.add(d)
                if can_add_B:
                    B.append(x)
                    diffs_B.update(new_diffs_B)
                    sums_B.add(2 * x)
                    for b in B[:-1]:
                        sums_B.add(x + b)

        return A, B

    def try_add_greedy_v2(sets, order):
        """
        More careful greedy: at each step, try adding x to A or B,
        checking full constraints.
        """
        A = []
        B = []
        diffs_A = set()
        diffs_B = set()
        sums_A = set()
        sums_B = set()

        for x in order:
            placed = False

            # Try A first
            ok = True
            # Sidon check for A with x
            if (2 * x) in sums_A:
                ok = False
            else:
                for a in A:
                    if (x + a) in sums_A:
                        ok = False
                        break
            if ok:
                # Difference check
                for a in A:
                    d = x - a
                    if d in diffs_A or d in diffs_B:
                        ok = False
                        break
            if ok:
                # Commit to A
                A.append(x)
                diffs_A.update({x - a for a in A[:-1]})
                sums_A.add(2 * x)
                for a in A[:-1]:
                    sums_A.add(x + a)
                placed = True

            if not placed:
                # Try B
                ok = True
                if (2 * x) in sums_B:
                    ok = False
                else:
                    for b in B:
                        if (x + b) in sums_B:
                            ok = False
                            break
                if ok:
                    for b in B:
                        d = x - b
                        if d in diffs_A or d in diffs_B:
                            ok = False
                            break
                if ok:
                    B.append(x)
                    diffs_B.update({x - b for b in B[:-1]})
                    sums_B.add(2 * x)
                    for b in B[:-1]:
                        sums_B.add(x + b)
                    placed = True

        return A, B

    best_total = 0
    best_result = ([1], [2])  # fallback valid result

    def validate(A, B):
        """Validate that A and B are valid Sidon sets with disjoint nonzero diffs."""
        if not A or not B:
            return False
        if len(A) > max_size or len(B) > max_size:
            return False
        if not is_sidon(A) or not is_sidon(B):
            return False
        dA = get_diffs(A)
        dB = get_diffs(B)
        if dA & dB:
            return False
        return True

    # Run multiple greedy passes with different orderings
    rng = random.Random(42)

    for restart in range(200):
        if time.time() > deadline:
            break

        # Create a randomized ordering
        order = list(range(1, n + 1))
        rng.shuffle(order)

        A, B = try_add_greedy_v2(None, order)

        if A and B:
            total = len(A) + len(B)
            if total > best_total:
                if validate(A, B):
                    best_total = total
                    best_result = (A, B)
                    if best_total >= 32:
                        break

    # Try a few more targeted strategies
    for restart in range(100):
        if time.time() > deadline:
            break

        # Try with a bias toward smaller numbers first
        order = list(range(1, n + 1))
        rng.shuffle(order)
        # Sort by a random key to create varied orderings
        order.sort(key=lambda x: rng.random())

        A, B = try_add_greedy_v2(None, order)

        if A and B:
            total = len(A) + len(B)
            if total > best_total:
                if validate(A, B):
                    best_total = total
                    best_result = (A, B)
                    if best_total >= 32:
                        break

    return list(best_result[0]), list(best_result[1])
# EVOLVE-BLOCK-END
