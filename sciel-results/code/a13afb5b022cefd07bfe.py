import time
import random
from itertools import combinations

def construct():
    start = time.time()
    DEADLINE = 145.0  # Leave 5s margin for safety

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x != 0:
                values.add(x)
        return sorted(values)

    def is_sum_free(S):
        Sset = set(S)
        for a in S:
            for b in S:
                if a + b in Sset:
                    return False
        return True

    def can_add(x, Sset):
        for a in Sset:
            if (x + a) in Sset or (x - a) in Sset:
                return False
        return True

    def compute_conflicts(S):
        """Compute conflict count for each element in S."""
        Sset = set(S)
        n = len(S)
        conflicts = [0] * n
        idx_map = {v: i for i, v in enumerate(S)}
        for i in range(n):
            a = S[i]
            for j in range(i, n):
                b = S[j]
                s = a + b
                if s in Sset:
                    conflicts[i] += 1
                    if j != i:
                        conflicts[j] += 1
                    k = idx_map.get(s)
                    if k is not None:
                        conflicts[k] += 1
        return conflicts

    def repair_sum_free(S, time_limit, start_time):
        """Remove elements until S is sum-free, removing the most-conflicting element each time."""
        S = list(set(S))
        Sset = set(S)
        while not is_sum_free(S):
            if time.time() - start_time > time_limit:
                break
            conflicts = compute_conflicts(S)
            if max(conflicts) == 0:
                break
            max_conf = max(conflicts)
            idx = conflicts.index(max_conf)
            S.pop(idx)
            Sset = set(S)
        return S

    def greedy_add(S_base, V, time_limit, start_time):
        """Greedily add elements from V \\ S_base to S_base, maintaining sum-freeness."""
        S = list(set(S_base))
        Sset = set(S)
        remaining = [x for x in V if x not in Sset]
        best_S = S
        best_size = len(S)
        orders = [
            lambda lst: sorted(lst, key=lambda x: abs(x), reverse=True),
            lambda lst: sorted(lst, key=lambda x: x),
            lambda lst: sorted(lst, key=lambda x: -x),
            lambda lst: sorted(lst, key=lambda x: random.Random(42).randint(0, 1) * abs(x), reverse=True),
        ]
        for order_fn in orders:
            if time.time() - start_time > time_limit:
                break
            S_try = list(best_S)
            Sset_try = set(S_try)
            rem = [x for x in V if x not in Sset_try]
            rem = order_fn(rem)
            for x in rem:
                if time.time() - start_time > time_limit:
                    break
                if can_add(x, Sset_try):
                    S_try.append(x)
                    Sset_try.add(x)
            if len(S_try) > best_size:
                best_size = len(S_try)
                best_S = S_try
        return best_S

    def local_search(S, V, time_limit, start_time):
        """Local search: try swaps and additions to improve the sum-free set."""
        S = list(set(S))
        Sset = set(S)
        best_size = len(S)
        improved = True
        while improved and time.time() - start_time < time_limit:
            improved = False
            # Try adding elements
            for x in V:
                if time.time() - start_time > time_limit:
                    break
                if x not in Sset and can_add(x, Sset):
                    S.append(x)
                    Sset.add(x)
                    improved = True
                    break
            if improved:
                continue
            # Try swaps: remove one element, add another
            for i in range(len(S)):
                if time.time() - start_time > time_limit:
                    break
                S_without_i = S[:i] + S[i+1:]
                Sset_without_i = set(S_without_i)
                for x in V:
                    if x in Sset:
                        continue
                    if can_add(x, Sset_without_i):
                        # Check if this swap improves or maintains size
                        S = S_without_i + [x]
                        Sset = Sset_without_i | {x}
                        improved = True
                        break
                if improved:
                    break
        return S

    def branch_and_bound(V, initial_solution, time_limit, start_time):
        """
        Branch and bound for maximum sum-free subset.
        Formulates as finding maximum independent set in conflict hypergraph.
        """
        V = list(set(V))
        n = len(V)
        if n == 0:
            return list(initial_solution)
        
        # Build conflict structure: for each element, which other elements conflict with it
        # Two elements a, b conflict if there exists c in V such that a+b=c or a+c=b or b+c=a
        # Actually, for sum-free: we cannot have a, b, c all in S where a+b=c
        # So the "conflict" is that if we include a and b, we cannot include a+b (if a+b in V)
        # And if we include a and a+b, we cannot include b
        
        # Let's build a pairwise conflict graph:
        # Edge between i and j if including both i and j forces exclusion of some k
        # More precisely: i and j are "incompatible" if there's no way to have both
        # in a sum-free set. But that's not quite right either.
        
        # Better approach: Use the fact that a set is sum-free iff for all a,b in S, a+b not in S.
        # We can model this as: choose a subset S of V such that no triple (a,b,c) with a+b=c is fully in S.
        
        # Branch and bound:
        # State: (included, excluded, remaining)
        # Upper bound: |included| + |remaining|
        
        included = set(initial_solution) if initial_solution else set()
        best_solution = list(included)
        best_size = len(included)
        
        # Precompute: for each pair (a,b), what is a+b?
        V_set = set(V)
        
        def upper_bound(included, remaining):
            return len(included) + len(remaining)
        
        def is_valid_set(S):
            Sset = set(S)
            for a in S:
                for b in S:
                    if a + b in Sset:
                        return False
            return True
        
        # Sort V by some heuristic
        # Elements that are "more central" (involved in more triples) should be decided first
        triple_count = [0] * n
        for i in range(n):
            a = V[i]
            for j in range(n):
                b = V[j]
                s = a + b
                if s in V_set:
                    # Find index of s
                    for k in range(n):
                        if V[k] == s:
                            triple_count[i] += 1
                            triple_count[j] += 1
                            triple_count[k] += 1
                            break
        
        # Order by triple count descending
        order = sorted(range(n), key=lambda i: -triple_count[i])
        
        # Branch and bound with memoization on (frozen remaining set)
        # But that's expensive. Instead, use a simpler recursive B&B.
        
        node_count = [0]
        
        def bb(idx, included, excluded, remaining):
            nonlocal best_solution, best_size
            if time.time() - start_time > time_limit:
                return
            node_count[0] += 1
            
            if node_count[0] % 1000 == 0 and time.time() - start_time > time_limit:
                return
            
            # Upper bound
            ub = len(included) + len(remaining)
            if ub <= best_size:
                return
            
            if idx == len(order):
                if len(included) > best_size:
                    best_size = len(included)
                    best_solution = list(included)
                return
            
            elem = V[order[idx]]
            
            # Option 1: Include elem if possible
            new_included = included | {elem}
            # Check if adding elem creates a violation
            valid = True
            for a in included:
                if (a + elem) in included or (a - elem) in included:
                    valid = False
                    break
                if (elem - a) in included:
                    valid = False
                    break
            if valid:
                # Also check that no a+b=elem for a,b in included
                for a in included:
                    for b in included:
                        if a + b == elem:
                            valid = False
                            break
                    if not valid:
                        break
            if valid:
                new_remaining = [v for v in remaining if v != elem]
                # Remove from remaining any element that would conflict with elem
                # Actually, we just filter remaining in the next step
                bb(idx + 1, new_included, excluded, new_remaining)
            
            # Option 2: Exclude elem
            new_remaining = [v for v in remaining if v != elem]
            bb(idx + 1, included, excluded | {elem}, new_remaining)
        
        # Start B&B
        remaining = [v for v in V if v not in included]
        bb(0, included, set(), remaining)
        
        return best_solution

    def direct_odds(V):
        return [x for x in V if x % 2 != 0]

    def direct_upper_magnitude(V):
        if not V:
            return []
        max_abs = max(abs(x) for x in V)
        threshold = max_abs / 2.0
        return [x for x in V if abs(x) > threshold]

    def direct_upper_third(V):
        if not V:
            return []
        max_abs = max(abs(x) for x in V)
        threshold = max_abs * 2.0 / 3.0
        return [x for x in V if abs(x) > threshold]

    def direct_modular(V, M):
        if not V or M <= 0:
            return []
        min_v = min(V)
        lower = M // 3 + 1
        upper = (2 * M) // 3
        result = []
        for x in V:
            r = (x - min_v) % M
            if lower <= r <= upper:
                result.append(x)
        return result

    def direct_positive(V):
        return [x for x in V if x > 0]

    def direct_negative(V):
        return [x for x in V if x < 0]

    def solve_instance(V, time_limit, start_time):
        """Solve one instance, return the best sum-free subset."""
        # Generate direct structural candidates
        candidates = [
            direct_odds(V),
            direct_upper_magnitude(V),
            direct_upper_third(V),
            direct_positive(V),
            direct_negative(V),
        ]
        max_abs = max(abs(x) for x in V) if V else 0
        for M in [max_abs, max_abs * 2, max(3, len(V)), max(3, len(V)) * 2]:
            if M > 0:
                candidates.append(direct_modular(V, M))

        best_set = set()
        best_size = 0

        for cand in candidates:
            if time.time() - start_time > time_limit:
                break
            cand_set = set(cand)
            if not is_sum_free(cand_set):
                cand_list = repair_sum_free(list(cand_set), 1.0, time.time())
                cand_set = set(cand_list)
            if time.time() - start_time > time_limit - 0.5:
                if len(cand_set) > best_size:
                    best_size = len(cand_set)
                    best_set = cand_set
                continue
            expanded = greedy_add(cand_set, V, time_limit - (time.time() - start_time) * 0.2, start_time)
            if is_sum_free(expanded):
                if len(expanded) > best_size:
                    best_size = len(expanded)
                    best_set = set(expanded)

        # Local search to improve
        if time.time() - start_time < time_limit - 5.0:
            ls_result = local_search(best_set, V, time_limit - 5.0, start_time)
            if is_sum_free(ls_result) and len(ls_result) > best_size:
                best_size = len(ls_result)
                best_set = set(ls_result)

        # Branch and bound if we have time and the instance is small enough
        if time.time() - start_time < time_limit - 10.0 and len(V) <= 130:
            bnb_time = time_limit - (time.time() - start_time) - 10.0
            if bnb_time > 2.0:
                bnb_result = branch_and_bound(V, best_set, bnb_time, time.time())
                if is_sum_free(bnb_result) and len(bnb_result) > best_size:
                    best_size = len(bnb_result)
                    best_set = set(bnb_result)

        # Final safety
        if not is_sum_free(best_set):
            best_set = set(direct_odds(V))
            if not is_sum_free(best_set):
                best_set = set()
                for x in V:
                    if can_add(x, best_set):
                        best_set.add(x)

        return sorted(best_set)

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        V = instance(size, seed)
        elapsed = time.time() - start
        remaining = DEADLINE - elapsed
        if remaining < 10:
            results.append(direct_odds(V))
            continue
        # Allocate more time to smaller instances for B&B
        time_alloc = min(remaining - 2.0, 40.0) if size <= 120 else min(remaining - 2.0, 20.0)
        results.append(solve_instance(V, time_alloc, start))
        # Update remaining for next instance
    return results
