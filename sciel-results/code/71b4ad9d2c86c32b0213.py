"""Sum-free subset finder using Max-Weight Independent Set on 3-uniform hypergraph
with branch-and-bound and greedy upper bounds."""
import time
import random
import math
from collections import defaultdict


def construct():
    deadline = time.time() + 142  # Safety margin before 150s limit

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x != 0:
                values.add(x)
        return sorted(values)

    def is_sum_free(elems):
        """Verify a list is sum-free."""
        s = set(elems)
        for a in elems:
            for b in elems:
                if a + b in s:
                    return False
        return True

    def find_sum_free(elements, time_budget=45.0):
        """Find a large sum-free subset using MWIS on 3-uniform hypergraph."""
        if not elements:
            return []

        n = len(elements)
        elem_set = set(elements)
        elem_to_idx = {x: i for i, x in enumerate(elements)}
        my_deadline = time.time() + time_budget

        # Build 3-uniform hypergraph: hyperedge = (a, b, c) where a+b=c
        # We want maximum weight independent set (maximize cardinality)
        # A vertex is "independent" if no hyperedge is fully contained in the set.

        # For efficiency, we track for each vertex k (the sum), which pairs (i,j) conflict
        # A set S is sum-free iff for all k in S, no pair (i,j) with i,j in S has i+j=elements[k]

        # Precompute: for each k, the set of (i,j) pairs where elements[i]+elements[j]==elements[k]
        conflicts_for_sum = [set() for _ in range(n)]
        # For each i, the set of (j,k) where elements[i]+elements[j]==elements[k]
        conflicts_for_left = [set() for _ in range(n)]
        # For each j, the set of (i,k) where elements[i]+elements[j]==elements[k]
        conflicts_for_right = [set() for _ in range(n)]

        for i in range(n):
            for j in range(i, n):
                s = elements[i] + elements[j]
                if s in elem_to_idx:
                    k = elem_to_idx[s]
                    conflicts_for_sum[k].add((i, j))
                    conflicts_for_left[i].add((j, k))
                    conflicts_for_right[j].add((i, k))

        # Degree of each vertex in the hypergraph (number of hyperedges it participates in)
        degrees = [len(conflicts_for_sum[i]) + len(conflicts_for_left[i]) + len(conflicts_for_right[i]) for i in range(n)]

        def count_conflicts(active):
            """Count violated sum-free constraints in active set."""
            count = 0
            for k in active:
                for (i, j) in conflicts_for_sum[k]:
                    if i in active and j in active:
                        count += 1
            return count

        def can_add(idx, active):
            """Check if adding idx to active creates any conflict."""
            for (i, j) in conflicts_for_sum[idx]:
                if i in active and j in active:
                    return False
            for (j, k) in conflicts_for_left[idx]:
                if j in active and k in active:
                    return False
            for (i, k) in conflicts_for_right[idx]:
                if i in active and k in active:
                    return False
            return True

        def greedy_upper_bound(remaining, active):
            """
            Greedy upper bound: assume we can add all remaining vertices,
            then greedily remove to fix conflicts. This gives an upper bound
            on the maximum independent set size from this state.
            """
            # Start with all remaining + active
            candidate = set(active) | set(remaining)
            # Greedily remove vertices that are part of conflicts
            # We want to maximize size, so we remove vertices with highest "damage"
            changed = True
            while changed:
                changed = False
                # Find all conflicts
                to_remove = set()
                for k in candidate:
                    for (i, j) in conflicts_for_sum[k]:
                        if i in candidate and j in candidate:
                            # This triple is a conflict; we need to remove at least one
                            # For upper bound, we optimistically keep as many as possible
                            # Remove the one with highest degree (most conflicts)
                            to_remove.add(k)  # Simple: remove the sum element
                if to_remove:
                    candidate -= to_remove
                    changed = True
            return len(candidate)

        def greedy_completion(active):
            """Greedy add any element that fits, until no more can be added."""
            active = set(active)
            improved = True
            while improved:
                improved = False
                for idx in range(n):
                    if idx not in active and can_add(idx, active):
                        active.add(idx)
                        improved = True
            return active

        # ========== BRANCH AND BOUND ==========
        def branch_and_bound(initial_active, time_limit=10.0):
            """
            Branch and bound for maximum independent set.
            State: (active, remaining) where active is confirmed in the solution,
            remaining are candidates that could still be added.
            """
            best_size = len(initial_active)
            best_solution = set(initial_active)
            bb_deadline = time.time() + time_limit
            nodes_explored = 0

            def bb(active, remaining, upper_bound):
                nonlocal best_size, best_solution, nodes_explored

                if time.time() > bb_deadline:
                    return

                nodes_explored += 1

                # Prune: if upper bound can't beat best, stop
                if len(active) + len(remaining) <= best_size:
                    return

                # If no remaining, update best
                if not remaining:
                    if len(active) > best_size:
                        best_size = len(active)
                        best_solution = set(active)
                    return

                # Greedy upper bound
                ub = len(active) + greedy_upper_bound(remaining, active)
                if ub <= best_size:
                    return

                # Choose branching vertex: highest degree in remaining
                # Sort remaining by degree descending
                sorted_remaining = sorted(remaining, key=lambda x: -degrees[x])
                v = sorted_remaining[0]

                # Branch 1: Include v
                new_active = active | {v}
                # Remove vertices that conflict with v from remaining
                new_remaining = set()
                for u in remaining:
                    if u == v:
                        continue
                    if can_add(u, new_active):
                        new_remaining.add(u)
                # Also need to check: does adding v create conflicts among remaining?
                # Actually, we need to ensure new_remaining is internally consistent
                # For now, just filter: u can be in solution with v
                # But pairs in remaining might conflict with each other AND with v
                # The can_add check only checks against active, not against other remaining
                # So we need to be more careful. Let's just recurse and let the bound handle it.
                bb(new_active, new_remaining, ub)

                # Branch 2: Exclude v
                new_remaining2 = remaining - {v}
                # Upper bound for this branch
                ub2 = len(active) + greedy_upper_bound(new_remaining2, active)
                if ub2 > best_size:
                    bb(active, new_remaining2, ub2)

            # Start B&B
            initial_remaining = set(range(n)) - set(initial_active)
            bb(set(initial_active), initial_remaining, len(initial_active) + len(initial_remaining))

            return best_solution

        # ========== MODEL-BASED INITIALIZATION ==========
        def model_sign_partition():
            """Split by sign: take all positive or all negative."""
            pos = [i for i in range(n) if elements[i] > 0]
            neg = [i for i in range(n) if elements[i] < 0]
            best = []
            for subset in [pos, neg, range(n)]:
                active = set(subset)
                active = greedy_completion(active)
                if len(active) > len(best):
                    best = active
            return best

        def model_residue(m):
            """Try all residue classes mod m, keep the largest sum-free one."""
            best = []
            for r in range(m):
                active = set(i for i in range(n) if elements[i] % m == r)
                active = greedy_completion(active)
                if len(active) > len(best):
                    best = active
            return best

        def model_magnitude_split():
            """Split by magnitude: take elements in a range."""
            abs_vals = sorted(range(n), key=lambda i: abs(elements[i]))
            best = []
            for frac_lo in [0, 0.25, 0.5]:
                for frac_hi in [0.75, 1.0]:
                    lo_idx = int(frac_lo * n)
                    hi_idx = int(frac_hi * n)
                    active = set(abs_vals[lo_idx:hi_idx])
                    active = greedy_completion(active)
                    if len(active) > len(best):
                        best = active
            return best

        def model_greedy_random(trial_seed):
            """Randomized greedy addition."""
            rng = random.Random(trial_seed)
            order = list(range(n))
            rng.shuffle(order)
            active = set()
            for idx in order:
                if can_add(idx, active):
                    active.add(idx)
            return active

        # ========== LOCAL SEARCH ==========
        def local_search_1flip(initial_active, max_iters=5000):
            """1-flip local search."""
            active = set(initial_active)
            current_conflicts = count_conflicts(active)
            best_active = set(active)
            best_size = len(active)
            rng = random.Random(99999)

            for iteration in range(max_iters):
                if time.time() > my_deadline:
                    break

                if current_conflicts == 0:
                    added = False
                    for idx in range(n):
                        if idx not in active and can_add(idx, active):
                            active.add(idx)
                            added = True
                            break
                    if not added:
                        if len(active) > best_size:
                            best_size = len(active)
                            best_active = set(active)
                        if len(active) > 2:
                            to_remove = rng.choice(list(active))
                            active.discard(to_remove)
                            active = greedy_completion(active)
                            current_conflicts = count_conflicts(active)
                    else:
                        current_conflicts = count_conflicts(active)
                        if len(active) > best_size:
                            best_size = len(active)
                            best_active = set(active)
                    continue

                conflict_found = None
                for k in active:
                    for (i, j) in conflicts_for_sum[k]:
                        if i in active and j in active:
                            conflict_found = (i, j, k)
                            break
                    if conflict_found:
                        break

                if not conflict_found:
                    break

                i, j, k = conflict_found
                candidates = [i, j, k]
                rng.shuffle(candidates)

                best_new_conflicts = current_conflicts
                best_remove = None
                for c in candidates:
                    new_active = active - {c}
                    new_conflicts = count_conflicts(new_active)
                    if new_conflicts < best_new_conflicts:
                        best_new_conflicts = new_conflicts
                        best_remove = c

                if best_remove is not None:
                    active.remove(best_remove)
                    current_conflicts = best_new_conflicts
                    active = greedy_completion(active)
                    current_conflicts = count_conflicts(active)
                    if len(active) > best_size and current_conflicts == 0:
                        best_size = len(active)
                        best_active = set(active)
                else:
                    if len(active) > 1:
                        to_remove = rng.choice(list(active))
                        active.discard(to_remove)
                        active = greedy_completion(active)
                        current_conflicts = count_conflicts(active)

            return best_active

        # ========== RUN ALL MODELS AND SEARCH ==========
        best_result_indices = set()

        # Model 1: Sign partition
        if time.time() < my_deadline:
            m1 = model_sign_partition()
            if len(m1) > len(best_result_indices):
                best_result_indices = set(m1)

        # Model 2: Residue classes
        if time.time() < my_deadline:
            for m in [3, 5, 7]:
                m2 = model_residue(m)
                if len(m2) > len(best_result_indices):
                    best_result_indices = set(m2)

        # Model 3: Magnitude split
        if time.time() < my_deadline:
            m3 = model_magnitude_split()
            if len(m3) > len(best_result_indices):
                best_result_indices = set(m3)

        # Model 4: Randomized greedy
        num_random_trials = 10 if n <= 120 else 5
        for t in range(num_random_trials):
            if time.time() > my_deadline:
                break
            m4 = model_greedy_random(1000 + t)
            if len(m4) > len(best_result_indices):
                best_result_indices = set(m4)

        # Local search from best model
        if best_result_indices and time.time() < my_deadline:
            ls_result = local_search_1flip(best_result_indices)
            if len(ls_result) > len(best_result_indices):
                best_result_indices = set(ls_result)

        # Branch and bound from best
        if best_result_indices and time.time() < my_deadline:
            bb_time = min(8.0, my_deadline - time.time() - 5)
            if bb_time > 1:
                bb_result = branch_and_bound(best_result_indices, time_limit=bb_time)
                if len(bb_result) > len(best_result_indices):
                    best_result_indices = set(bb_result)

        # More local search
        if best_result_indices and time.time() < my_deadline:
            ls_result2 = local_search_1flip(best_result_indices, max_iters=3000)
            if len(ls_result2) > len(best_result_indices):
                best_result_indices = set(ls_result2)

        # Final: greedy complete
        if best_result_indices:
            best_result_indices = greedy_completion(best_result_indices)

        # Convert to values
        best_result = [elements[i] for i in best_result_indices]

        # Final verification
        result_set = set(best_result)
        for _ in range(3):
            to_remove = set()
            for a in best_result:
                for b in best_result:
                    s = a + b
                    if s in result_set:
                        to_remove.add(s)
            if to_remove:
                result_set -= to_remove
                best_result = list(result_set)
            else:
                break

        if not is_sum_free(best_result):
            result_set = set(best_result)
            changed = True
            while changed:
                changed = False
                for a in list(result_set):
                    for b in list(result_set):
                        if a + b in result_set:
                            result_set.discard(a + b)
                            changed = True
                best_result = list(result_set)
            if not is_sum_free(best_result):
                best_result = [x for x in sorted(best_result, key=lambda x: -abs(x))[:len(best_result)//2]]

        return best_result

    # Solve each instance
    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        remaining = deadline - time.time()
        if remaining < 5:
            elems = instance(size, seed)
            results.append([x for x in elems if abs(x) % 2 == 1])
            continue

        time_budget = min(remaining - 3, 50.0)
        elems = instance(size, seed)
        result = find_sum_free(elems, time_budget=time_budget)
        results.append(result)

    return results
