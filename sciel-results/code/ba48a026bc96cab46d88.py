"""Improved sum-free subset finder using spectral initialization + greedy + local search + polish."""
import time
import random
import numpy as np
from collections import defaultdict

def construct():
    deadline = time.time() + 145  # Safety margin before 150s limit

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x != 0:
                values.add(x)
        return sorted(values)

    def find_sum_free(elements):
        """Find a large sum-free subset using spectral init + greedy + local search + polish."""
        if not elements:
            return []

        n = len(elements)
        elem_set = set(elements)
        elem_to_idx = {x: i for i, x in enumerate(elements)}

        # Precompute conflict structure
        conflicts_as_sum = defaultdict(set)
        conflicts_as_left = defaultdict(set)
        conflicts_as_right = defaultdict(set)

        for i in range(n):
            for j in range(i, n):
                s = elements[i] + elements[j]
                if s in elem_to_idx:
                    k = elem_to_idx[s]
                    conflicts_as_sum[k].add((i, j))
                    conflicts_as_left[i].add((j, k))
                    conflicts_as_right[j].add((i, k))

        def count_conflicts(active):
            """Count number of violated sum-free constraints."""
            count = 0
            for k in active:
                for (i, j) in conflicts_as_sum[k]:
                    if i in active and j in active:
                        count += 1
            return count

        def can_add(idx, active):
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

        # Strategy 1: Spectral bisection on conflict graph
        def spectral_bisection():
            """Use eigenvector of Laplacian to find a good initial partition."""
            # Build adjacency matrix of conflict graph
            # Two elements conflict if they appear together in any sum triple
            adj = np.zeros((n, n))
            for k in range(n):
                for (i, j) in conflicts_as_sum[k]:
                    adj[i][j] = 1
                    adj[j][i] = 1
                    adj[i][k] = 1
                    adj[k][i] = 1
                    adj[j][k] = 1
                    adj[k][j] = 1
            # Make symmetric
            adj = (adj + adj.T) / 2.0
            # Compute degree
            degree = adj.sum(axis=1)
            # Laplacian L = D - A
            L = np.diag(degree) - adj
            
            # Find eigenvector corresponding to smallest non-zero eigenvalue
            try:
                eigvals, eigvecs = np.linalg.eigh(L)
                # Fiedler vector is the second smallest eigenvalue's eigenvector
                fiedler = eigvecs[:, 1]
            except:
                # Fallback: random
                fiedler = np.random.randn(n)
            
            # Partition: positive values in set A, negative in set B
            active_a = set(i for i in range(n) if fiedler[i] >= 0)
            active_b = set(i for i in range(n) if fiedler[i] < 0)
            
            # Use the larger partition as starting point
            if len(active_a) >= len(active_b):
                return active_a
            else:
                return active_b

        # Strategy 2: Greedy removal from full set
        def greedy_removal():
            active = set(range(n))
            for _ in range(n * 2):
                if time.time() > deadline - 5:
                    break
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

        # Strategy 3: Greedy addition with various orderings
        def greedy_addition(order):
            active = set()
            for idx in order:
                if can_add(idx, active):
                    active.add(idx)
            return active

        # Strategy 4: Local search with swaps
        def local_search(initial_active, max_iters=3000):
            active = set(initial_active)
            current_conflicts = count_conflicts(active)
            best_active = set(active)
            best_size = len(active)
            rng = random.Random(12345)

            for iteration in range(max_iters):
                if time.time() > deadline - 5:
                    break

                if current_conflicts == 0:
                    improved = False
                    for idx in range(n):
                        if idx not in active:
                            if can_add(idx, active):
                                active.add(idx)
                                current_conflicts = count_conflicts(active)
                                if len(active) > best_size:
                                    best_size = len(active)
                                    best_active = set(active)
                                improved = True
                                break
                    if not improved:
                        break
                    continue

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
                    new_conflicts = count_conflicts(new_active)
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
                    for c in candidates:
                        new_active = active - {c}
                        inactive = [x for x in range(n) if x not in new_active]
                        if not inactive:
                            continue
                        for _ in range(min(5, len(inactive))):
                            add_candidate = rng.choice(inactive)
                            test_active = new_active | {add_candidate}
                            new_conflicts = count_conflicts(test_active)
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

        # Strategy 5: Polish - try to improve a sum-free set by swaps
        def polish(active_set, max_iters=2000):
            active = set(active_set)
            best_active = set(active)
            best_size = len(active)
            rng = random.Random(67890)

            for _ in range(max_iters):
                if time.time() > deadline - 5:
                    break

                improved = False
                inactive = [x for x in range(n) if x not in active]
                if len(inactive) < 2:
                    break

                for remove_c in list(active):
                    if time.time() > deadline - 5:
                        break
                    new_active = active - {remove_c}
                    added = False
                    for idx1 in inactive:
                        if added:
                            break
                        if not can_add(idx1, new_active):
                            continue
                        new_active2 = new_active | {idx1}
                        for idx2 in inactive:
                            if idx2 == idx1:
                                continue
                            if can_add(idx2, new_active2):
                                new_active3 = new_active2 | {idx2}
                                if count_conflicts(new_active3) == 0:
                                    active = new_active3
                                    if len(active) > best_size:
                                        best_size = len(active)
                                        best_active = set(active)
                                    improved = True
                                    added = True
                                    break
                    if improved:
                        break

                if not improved:
                    break

            return best_active

        # Run strategies
        best_result = []

        # Strategy 1: Spectral bisection
        if n >= 10:
            spectral_active = spectral_bisection()
            # Clean up spectral result with local search
            cleaned = local_search(spectral_active, max_iters=1500)
            if len(cleaned) > len(best_result):
                best_result = [elements[i] for i in cleaned]

        # Strategy 2: Greedy removal
        active2 = greedy_removal()
        if len(active2) > len(best_result):
            best_result = [elements[i] for i in active2]

        # Strategy 3: Multiple greedy additions
        rng = random.Random(42)
        for trial in range(20):
            if time.time() > deadline - 10:
                break
            order = list(range(n))
            rng.shuffle(order)
            active3 = greedy_addition(order)
            if len(active3) > len(best_result):
                best_result = [elements[i] for i in active3]

        # Also try sorted orders
        for order_func in [
            lambda: list(range(n)),
            lambda: sorted(range(n), key=lambda i: elements[i]),
            lambda: sorted(range(n), key=lambda i: abs(elements[i])),
            lambda: sorted(range(n), key=lambda i: -elements[i]),
        ]:
            if time.time() > deadline - 10:
                break
            order = order_func()
            active4 = greedy_addition(order)
            if len(active4) > len(best_result):
                best_result = [elements[i] for i in active4]

        # Strategy 4: Local search from best
        if best_result:
            best_indices = {elements.index(x) for x in best_result}
            max_iters = 2500 if n <= 120 else 1500
            improved = local_search(best_indices, max_iters)
            if len(improved) > len(best_result):
                best_result = [elements[i] for i in improved]

        # Strategy 5: Polish
        if best_result:
            best_indices = {elements.index(x) for x in best_result}
            polished = polish(best_indices, max_iters=800)
            if len(polished) > len(best_result):
                best_result = [elements[i] for i in polished]

        # Final verification
        result_set = set(best_result)
        for a in best_result:
            for b in best_result:
                if a + b in result_set:
                    result_set.discard(a + b)
        best_result = list(result_set)

        return best_result

    # Solve each instance
    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        if time.time() > deadline - 5:
            elems = instance(size, seed)
            results.append([x for x in elems if x % 2])
            continue

        elems = instance(size, seed)
        result = find_sum_free(elems)
        results.append(result)

    return results
