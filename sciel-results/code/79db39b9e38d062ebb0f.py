import time
import random
import numpy as np
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s kill

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

    def build_conflict_structure(values):
        """Build conflict structure for local search.
        conflicts_as_sum[k] = set of (i, j) where values[i] + values[j] == values[k]
        conflicts_as_left[i] = set of (j, k) where values[i] + values[j] == values[k]
        conflicts_as_right[j] = set of (i, k) where values[i] + values[j] == values[k]
        """
        n = len(values)
        val_to_idx = {v: i for i, v in enumerate(values)}
        conflicts_as_sum = defaultdict(set)
        conflicts_as_left = defaultdict(set)
        conflicts_as_right = defaultdict(set)

        for i in range(n):
            for j in range(i, n):
                s = values[i] + values[j]
                if s in val_to_idx:
                    k = val_to_idx[s]
                    conflicts_as_sum[k].add((i, j))
                    conflicts_as_left[i].add((j, k))
                    conflicts_as_right[j].add((i, k))

        return conflicts_as_sum, conflicts_as_left, conflicts_as_right

    def count_conflicts(active, conflicts_as_sum):
        """Count number of violated sum-free constraints."""
        count = 0
        for k in active:
            for (i, j) in conflicts_as_sum[k]:
                if i in active and j in active:
                    count += 1
        return count

    def can_add_idx(idx, active, conflicts_as_sum, conflicts_as_left, conflicts_as_right):
        """Check if adding idx to active creates any new conflict."""
        for (i, j) in conflicts_as_sum[idx]:
            if i in active and j in active:
                return False
        for (j, k) in conflicts_as_left[idx]:
            if j in active and k in active:
                return False
        for (i, k) in conflicts_as_right[idx]:
            if i in active and k in active:
                return False
        return True

    def local_search_idx(initial_active, n, conflicts_as_sum, conflicts_as_left, conflicts_as_right, time_limit=5.0):
        """Improve active set (as set of indices) using conflict-based local search."""
        active = set(initial_active)
        current_conflicts = count_conflicts(active, conflicts_as_sum)
        best_active = set(active)
        best_size = len(active)
        rng = random.Random(12345)
        start = time.time()

        for iteration in range(5000):
            if time.time() - start > time_limit:
                break

            if current_conflicts == 0:
                # Try to add more elements
                improved = False
                for idx in range(n):
                    if idx not in active:
                        if can_add_idx(idx, active, conflicts_as_sum, conflicts_as_left, conflicts_as_right):
                            active.add(idx)
                            current_conflicts = count_conflicts(active, conflicts_as_sum)
                            if len(active) > best_size:
                                best_size = len(active)
                                best_active = set(active)
                            improved = True
                            break
                if not improved:
                    break
                continue

            # Find a conflict
            conflict_found = None
            for k in active:
                for (i, j) in conflicts_as_sum[k]:
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
                new_conflicts = count_conflicts(new_active, conflicts_as_sum)
                if new_conflicts < best_new_conflicts:
                    best_new_conflicts = new_conflicts
                    best_remove = c

            if best_remove is not None:
                active.remove(best_remove)
                current_conflicts = best_new_conflicts
                if len(active) > best_size and current_conflicts == 0:
                    best_size = len(active)
                    best_active = set(active)
            else:
                # Try swap: remove one from conflict, add a random inactive
                for c in candidates:
                    new_active = active - {c}
                    inactive = [x for x in range(n) if x not in new_active]
                    if not inactive:
                        continue
                    for _ in range(min(5, len(inactive))):
                        add_candidate = rng.choice(inactive)
                        test_active = new_active | {add_candidate}
                        new_conflicts = count_conflicts(test_active, conflicts_as_sum)
                        if new_conflicts < current_conflicts:
                            active = test_active
                            current_conflicts = new_conflicts
                            if len(active) > best_size and current_conflicts == 0:
                                best_size = len(active)
                                best_active = set(active)
                            break
                    else:
                        continue
                    break

        return best_active

    def greedy_removal_idx(n, conflicts_as_sum, time_limit=3.0):
        """Greedy removal: start with all elements, remove those causing most conflicts."""
        active = set(range(n))
        start = time.time()

        for _ in range(n * 2):
            if time.time() - start > time_limit:
                break
            # Find all conflicts
            conflict_elements = defaultdict(int)
            for k in active:
                for (i, j) in conflicts_as_sum[k]:
                    if i in active and j in active:
                        conflict_elements[i] += 1
                        conflict_elements[j] += 1
                        conflict_elements[k] += 1
            if not conflict_elements:
                break
            to_remove = max(conflict_elements, key=conflict_elements.get)
            active.remove(to_remove)

        return active

    def build_conflict_graph(values):
        """Build adjacency list for conflict graph.
        Edge between i,j if values[i]+values[j] is in values, or 2*values[i]==values[j]."""
        n = len(values)
        val_to_idx = {v: i for i, v in enumerate(values)}
        adj = [set() for _ in range(n)]

        for i in range(n):
            for j in range(i+1, n):
                vi, vj = values[i], values[j]
                s = vi + vj
                if s in val_to_idx:
                    k = val_to_idx[s]
                    adj[i].add(k)
                    adj[k].add(i)
                    adj[j].add(k)
                    adj[k].add(j)
                if 2 * vi == vj:
                    adj[i].add(j)
                    adj[j].add(i)
                elif 2 * vj == vi:
                    adj[i].add(j)
                    adj[j].add(i)

        return adj

    def greedy_independent_set(adj, order):
        """Greedy: add vertices in given order if no neighbor already added."""
        n = len(adj)
        in_set = [False] * n
        result = []
        for v in order:
            if in_set[v]:
                continue
            conflict = False
            for nb in adj[v]:
                if in_set[nb]:
                    conflict = True
                    break
            if not conflict:
                in_set[v] = True
                result.append(v)
        return result

    def spectral_partition(adj, values):
        """Use spectral bisection to find a large independent set."""
        n = len(adj)
        if n == 0:
            return []

        deg = [len(adj[i]) for i in range(n)]
        L = np.zeros((n, n))
        for i in range(n):
            L[i, i] = deg[i]
            for j in adj[i]:
                L[i, j] = -1

        try:
            eigvals, eigvecs = np.linalg.eigh(L)
            fiedler = eigvecs[:, 1]
        except:
            return []

        pos_indices = [i for i in range(n) if fiedler[i] >= 0]
        neg_indices = [i for i in range(n) if fiedler[i] < 0]

        best = []
        for side in [pos_indices, neg_indices]:
            side_sorted = sorted(side, key=lambda v: deg[v])
            indep = greedy_independent_set(adj, side_sorted)
            if len(indep) > len(best):
                best = indep

        pos_indices2 = [i for i in range(n) if fiedler[i] <= 0]
        neg_indices2 = [i for i in range(n) if fiedler[i] > 0]
        for side in [pos_indices2, neg_indices2]:
            side_sorted = sorted(side, key=lambda v: deg[v])
            indep = greedy_independent_set(adj, side_sorted)
            if len(indep) > len(best):
                best = indep

        return [values[i] for i in best]

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

        # Strategy 5: Greedy by absolute value
        s5_list, s5_set = greedy_add([], sorted(values, key=abs, reverse=True))
        candidates.append(s5_list)

        # Strategy 6: Greedy by value (smallest first)
        s6_list, s6_set = greedy_add([], sorted(values))
        candidates.append(s6_list)

        # Strategy 7: Greedy by value (largest first)
        s7_list, s7_set = greedy_add([], sorted(values, reverse=True))
        candidates.append(s7_list)

        # Strategy 8: Random shuffles
        rng = random.Random(42)
        for trial in range(5):
            shuffled = list(values)
            rng.shuffle(shuffled)
            sl, ss = greedy_add([], shuffled)
            candidates.append(sl)

        # Strategy 9: Greedy removal
        if n <= 300:
            try:
                cas, cal, car = build_conflict_structure(values)
                gr_active = greedy_removal_idx(n, cas, time_limit=2.0)
                gr_result = [values[i] for i in gr_active]
                if len(gr_result) > 0:
                    candidates.append(gr_result)
            except:
                pass

        # Strategy 10: Spectral partition
        if n <= 300:
            try:
                adj = build_conflict_graph(values)
                spec_result = spectral_partition(adj, values)
                if len(spec_result) > 0:
                    candidates.append(spec_result)
            except:
                pass

        # Evaluate and keep best
        for c in candidates:
            if not is_sum_free(c):
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
                best_set = set(c)

        # Local search on best using conflict structure
        remaining_time = deadline - time.time()
        if remaining_time > 5:
            try:
                cas, cal, car = build_conflict_structure(values)
                val_to_idx = {v: i for i, v in enumerate(values)}
                best_idx = set(val_to_idx[x] for x in best)
                ts = min(time_budget, remaining_time - 3)
                improved_idx = local_search_idx(best_idx, n, cas, cal, car, time_limit=ts)
                improved = [values[i] for i in improved_idx]
                if len(improved) > len(best):
                    best = improved
                    best_set = set(best)
            except:
                pass

        # Try additional random restarts if time permits
        remaining_time = deadline - time.time()
        if remaining_time > 8:
            for _ in range(15):
                if time.time() > deadline - 5:
                    break
                shuffled = list(values)
                rng.shuffle(shuffled)
                trial_list, trial_set = greedy_add([], shuffled)

                if len(trial_list) >= len(best):
                    ts = min(3.0, deadline - time.time() - 3)
                    if ts > 1:
                        try:
                            cas, cal, car = build_conflict_structure(values)
                            val_to_idx = {v: i for i, v in enumerate(values)}
                            trial_idx = set(val_to_idx[x] for x in trial_list)
                            improved_idx = local_search_idx(trial_idx, n, cas, cal, car, time_limit=ts)
                            improved = [values[i] for i in improved_idx]
                            if len(improved) > len(best):
                                best = improved
                                best_set = set(best)
                        except:
                            pass

        return best

    results = []
    time_alloc = {60: 10.0, 120: 22.0, 240: 75.0}

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
