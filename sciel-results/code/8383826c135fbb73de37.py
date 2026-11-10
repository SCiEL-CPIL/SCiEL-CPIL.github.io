import math
import random
import time
from itertools import combinations

def construct():
    """
    Return a nonempty subset of [1..22] such that ALL reciprocal subset sums,
    including the empty sum, are distinct. Maximize cardinality.
    
    A set {a_1, ..., a_n} is "dissociated" for reciprocal subset sums iff
    the only integer combination sum c_i * (1/a_i) = 0 with c_i in {-1,0,1}
    is the trivial one. Equivalently, all 2^n subset sums of {1/a_i} are distinct.
    
    We use exact rational arithmetic via LCM to avoid floating-point errors.
    """
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit

    # LCM of 1..22
    def lcm(a, b):
        return abs(a * b) // math.gcd(a, b)

    L = 1
    for i in range(1, 23):
        L = lcm(L, i)

    # A[i] = L / (i+1) for i in 0..21, so 1/(i+1) = A[i] / L
    A = [L // (i + 1) for i in range(22)]

    def check_dissociated(indices):
        """
        Check if the set of indices (0-based, representing elements 1..22)
        has all distinct reciprocal subset sums.
        Returns True if dissociated, False otherwise.
        """
        k = len(indices)
        if k == 0:
            return True

        vals = [A[i] for i in indices]

        # Generate all subset sums and check for duplicates
        seen = {0}
        current_sums = [0]

        for i in range(k):
            v = vals[i]
            new_sums = []
            for s in current_sums:
                ns = s + v
                if ns in seen:
                    return False
                seen.add(ns)
                new_sums.append(ns)
            current_sums.extend(new_sums)

        return True

    # Greedy approach: start with all 22, remove elements causing collisions
    # Try multiple strategies and pick the best (largest cardinality)

    best_set = [0]  # At minimum, a single element is always valid

    def greedy_remove(heuristic_func):
        """
        Start with all 22 elements, repeatedly find a collision and remove
        one element involved, using the given heuristic.
        """
        if time.time() > deadline:
            return None

        S = list(range(22))

        while True:
            if time.time() > deadline:
                return S if S else None

            k = len(S)
            if k == 0:
                return None

            # Check if S is dissociated
            if check_dissociated(S):
                return S

            # Find a collision: two different subsets with the same sum
            # We need to identify which elements are involved
            vals = [A[i] for i in S]

            # Build subset sums with tracking of masks
            sum_to_mask = {0: 0}
            current = [(0, 0)]

            collision_found = False
            coll_elements = None

            for i in range(k):
                v = vals[i]
                new_entries = []
                for s, m in current:
                    ns = s + v
                    nm = m | (1 << i)
                    if ns in sum_to_mask:
                        # Collision!
                        m1 = nm
                        m2 = sum_to_mask[ns]
                        # Elements involved are those in the symmetric difference
                        diff_mask = m1 ^ m2
                        coll_local = []
                        for j in range(k):
                            if (diff_mask >> j) & 1:
                                coll_local.append(S[j])
                        coll_elements = coll_local
                        collision_found = True
                        break
                    else:
                        sum_to_mask[ns] = nm
                        new_entries.append((ns, nm))
                current.extend(new_entries)
                if collision_found:
                    break

            if not collision_found or not coll_elements:
                # Shouldn't happen, but return current set
                return S

            # Choose which element to remove
            x = heuristic_func(coll_elements)
            if x in S:
                S.remove(x)

        return S if S else None

    # Heuristic functions
    def remove_smallest_idx(coll):
        """Remove element with smallest value (largest reciprocal)"""
        return min(coll)

    def remove_largest_idx(coll):
        """Remove element with largest value (smallest reciprocal)"""
        return max(coll)

    def remove_random(coll):
        """Remove a random element from collision"""
        return random.choice(coll)

    # Try multiple heuristics
    strategies = [
        ("smallest", remove_smallest_idx),
        ("largest", remove_largest_idx),
    ]

    for name, func in strategies:
        if time.time() > deadline:
            break
        result = greedy_remove(func)
        if result and len(result) > len(best_set):
            best_set = result

    # Try some random restarts
    for _ in range(20):
        if time.time() > deadline:
            break
        result = greedy_remove(remove_random)
        if result and len(result) > len(best_set):
            best_set = result

    # Try a different approach: build up from scratch using greedy addition
    # Start empty, try to add each element if it maintains dissociation
    if time.time() < deadline - 10:
        def greedy_build(order):
            S = []
            for idx in order:
                if time.time() > deadline:
                    break
                new_S = S + [idx]
                if check_dissociated(new_S):
                    S = new_S
            return S if S else None

        # Try different orderings
        # 1. Ascending
        result = greedy_build(list(range(22)))
        if result and len(result) > len(best_set):
            best_set = result

        # 2. Descending
        result = greedy_build(list(range(21, -1, -1)))
        if result and len(result) > len(best_set):
            best_set = result

        # 3. Some random orderings
        for _ in range(10):
            if time.time() > deadline - 5:
                break
            order = list(range(22))
            random.shuffle(order)
            result = greedy_build(order)
            if result and len(result) > len(best_set):
                best_set = result

    # Final safety: ensure best_set is nonempty
    if not best_set:
        best_set = [0]

    # Return 1-based indices
    return [i + 1 for i in sorted(best_set)]

# EVOLVE-BLOCK-END
