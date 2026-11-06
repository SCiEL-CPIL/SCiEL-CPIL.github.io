import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 118  # 32s safety margin before 150s kill

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x != 0:
                values.add(x)
        return sorted(values)

    def is_sum_free(S):
        if len(S) < 3:
            return True
        S_set = set(S)
        for a in S:
            for b in S:
                if a + b in S_set:
                    return False
        return True

    def can_add(S_set, candidate):
        """Check if candidate can be added to S_set to keep it sum-free."""
        if candidate in S_set:
            return False
        for x in S_set:
            if (candidate + x) in S_set:
                return False
            if (candidate - x) in S_set:
                return False
        return True

    def greedy_add(S_list, candidates, S_set=None):
        if S_set is None:
            S_set = set(S_list)
        result = list(S_list)
        for c in candidates:
            if c in S_set:
                continue
            if can_add(S_set, c):
                result.append(c)
                S_set.add(c)
        return result, S_set

    def build_graph(values):
        n = len(values)
        val_to_idx = {v: i for i, v in enumerate(values)}
        adj = np.zeros((n, n), dtype=int)
        
        for i in range(n):
            vi = values[i]
            for j in range(i + 1, n):
                vj = values[j]
                # Check sum
                s = vi + vj
                if s in val_to_idx:
                    adj[i, j] = 1
                    adj[j, i] = 1
                # Check doubling
                if 2 * vi == vj or 2 * vj == vi:
                    adj[i, j] = 1
                    adj[j, i] = 1
        return adj

    def spectral_independent_set(values, adj):
        n = len(values)
        if n == 0:
            return []
        
        # Compute Laplacian L = D - A
        D = np.diag(np.sum(adj, axis=1))
        L = D - adj
        
        # Find eigenvalues and eigenvectors
        try:
            eigvals, eigvecs = np.linalg.eigh(L)
            # Smallest eigenvalue is usually 0 for connected graph.
            # We want the Fiedler vector (2nd smallest) or just the smallest non-zero?
            # Actually, for clustering/bisection, the eigenvector of the smallest non-zero eigenvalue is used.
            # But for ordering, any eigenvector with spread can be used.
            # Let's use the eigenvector corresponding to the smallest eigenvalue > 1e-6.
            # If graph is disconnected, smallest is 0.
            
            # Heuristic: Use the eigenvector of the smallest eigenvalue that is not close to 0.
            # If all are 0 (empty graph), any order works.
            idx = 0
            for i in range(len(eigvals)):
                if eigvals[i] > 1e-6:
                    idx = i
                    break
            
            vec = eigvecs[:, idx]
            
            # Sort indices by vector values
            order = np.argsort(vec)
            
            # Greedy independent set selection
            S_set = set()
            S_list = []
            for i in order:
                vi = values[i]
                # Check if vi conflicts with any in S_set
                conflict = False
                for j in S_set:
                    # j is a value, need index?
                    # We have adj matrix.
                    # We need to check if adj[i, j_idx] == 1
                    # But S_set stores values.
                    # Let's store indices in S_idx_set
                    pass
                
            # Better: work with indices
            S_idx_set = set()
            for i in order:
                conflict = False
                for j in S_idx_set:
                    if adj[i, j] == 1:
                        conflict = True
                        break
                if not conflict:
                    S_idx_set.add(i)
            
            S_list = [values[i] for i in S_idx_set]
            return S_list

        except Exception:
            # Fallback to simple heuristic
            return [x for x in values if x % 2 != 0]

    def local_search(S_list, values, time_limit=5.0):
        S_set = set(S_list)
        values_set = set(values)
        start = time.time()
        S_list = list(S_list)

        # Phase 1: Try adding any missing element
        missing = list(values_set - S_set)
        if missing:
            S_list, S_set = greedy_add(S_list, missing, S_set)

        # Phase 2: Iterative improvement - remove one element, greedily add all possible
        improved = True
        while improved and (time.time() - start) < time_limit:
            improved = False
            S_copy = list(S_list)
            S_set_copy = set(S_set)

            for x in S_copy:
                if (time.time() - start) >= time_limit:
                    break

                # Remove x
                S_test = S_set_copy - {x}
                S_test_list = [y for y in S_copy if y != x]

                # Try to add all missing elements greedily
                cand_list = [y for y in values if y not in S_test]
                S_new_list, S_new_set = greedy_add(S_test_list, cand_list, S_test)

                if len(S_new_list) > len(S_list):
                    S_list = S_new_list
                    S_set = S_new_set
                    improved = True
                    break

        # Phase 3: Try swap - remove one, add up to 2 specific elements
        if (time.time() - start) < time_limit:
            improved = True
            while improved and (time.time() - start) < time_limit:
                improved = False
                S_copy = list(S_list)
                S_set_copy = set(S_set)

                for x in S_copy:
                    if (time.time() - start) >= time_limit:
                        break

                    # Remove x
                    S_test = S_set_copy - {x}
                    S_test_list = [y for y in S_copy if y != x]

                    # Try adding pairs of missing elements
                    missing = list(values_set - S_test)
                    added = 0
                    for i, a in enumerate(missing):
                        if (time.time() - start) >= time_limit:
                            break
                        if a in S_test:
                            continue
                        if can_add(S_test, a):
                            S_test.add(a)
                            S_test_list.append(a)
                            added += 1
                            if added >= 2:
                                break

                    if len(S_test_list) > len(S_list):
                        S_list = S_test_list
                        S_set = S_test
                        improved = True
                        break

        # Phase 4: Final greedy add pass
        missing = list(values_set - S_set)
        if missing:
            S_list, S_set = greedy_add(S_list, missing, S_set)

        return S_list

    def solve_one(values, time_budget=15.0):
        if time.time() > deadline:
            return []

        n = len(values)
        if n == 0:
            return []

        V_set = set(values)
        max_v = max(values)
        min_v = min(values)

        best = []
        best_set = set()

        candidates = []

        # Strategy 1: Odd numbers
        s1 = [x for x in values if x % 2 != 0]
        candidates.append(s1)

        # Strategy 2: Large positive numbers
        thresh_pos = max_v / 2.0
        s2 = [x for x in values if x > thresh_pos]
        candidates.append(s2)

        # Strategy 3: Large negative numbers
        thresh_neg = min_v / 2.0
        s3 = [x for x in values if x < thresh_neg]
        candidates.append(s3)

        # Strategy 4: Union of large pos and neg
        s4 = sorted(set(s2).union(s3))
        candidates.append(s4)

        # Strategy 5: Spectral Independent Set
        if n > 10:
            adj = build_graph(values)
            s5 = spectral_independent_set(values, adj)
            candidates.append(s5)

        # Strategy 6: Greedy by absolute value
        s6_list, s6_set = greedy_add([], sorted(values, key=abs, reverse=True))
        candidates.append(s6_list)

        # Strategy 7: Greedy by value (smallest first)
        s7_list, s7_set = greedy_add([], sorted(values))
        candidates.append(s7_list)

        # Strategy 8: Greedy by value (largest first)
        s8_list, s8_set = greedy_add([], sorted(values, reverse=True))
        candidates.append(s8_list)

        # Strategy 9: Random shuffles
        rng = random.Random(42)
        for trial in range(5):
            shuffled = list(values)
            rng.shuffle(shuffled)
            sl, ss = greedy_add([], shuffled)
            candidates.append(sl)

        # Evaluate and keep best
        for c in candidates:
            if not is_sum_free(c):
                # Repair: remove conflicting elements
                c_valid = list(c)
                c_set = set(c_valid)
                changed = True
                while changed:
                    changed = False
                    i = 0
                    while i < len(c_valid):
                        x = c_valid[i]
                        conflict = False
                        for y in c_set:
                            if y == x:
                                continue
                            if (x + y) in c_set:
                                conflict = True
                                break
                            if (x - y) in c_set:
                                conflict = True
                                break
                        if conflict:
                            c_valid.pop(i)
                            c_set.remove(x)
                            changed = True
                        else:
                            i += 1
                c = c_valid

            if len(c) > len(best):
                best = c
                best_set = set(best)

        # Local search on best
        remaining_time = deadline - time.time()
        if remaining_time > 5:
            ts = min(time_budget, remaining_time - 3)
            improved = local_search(best, values, time_limit=ts)
            if len(improved) > len(best):
                best = improved
                best_set = set(best)

        # Try additional random restarts if time permits
        remaining_time = deadline - time.time()
        if remaining_time > 8:
            for _ in range(10):
                if time.time() > deadline - 5:
                    break
                shuffled = list(values)
                rng.shuffle(shuffled)
                trial_list, trial_set = greedy_add([], shuffled)

                if len(trial_list) >= len(best):
                    ts = min(3.0, deadline - time.time() - 3)
                    if ts > 1:
                        improved = local_search(trial_list, values, time_limit=ts)
                        if len(improved) > len(best):
                            best = improved
                            best_set = set(best)

        return best

    results = []
    time_alloc = {60: 12.0, 120: 25.0, 240: 70.0}

    for size, seed in ((60, 801), (120, 802), (240, 803)):
        if time.time() > deadline:
            results.append([])
            continue

        values = instance(size, seed)
        budget = time_alloc.get(size, 15.0)
        remaining = deadline - time.time()
        budget = min(budget, max(2.0, remaining - 3.0))

        subset = solve_one(values, time_budget=budget)

        if not is_sum_free(subset):
            subset = [x for x in values if x % 2 != 0]

        results.append(subset)

    return results
