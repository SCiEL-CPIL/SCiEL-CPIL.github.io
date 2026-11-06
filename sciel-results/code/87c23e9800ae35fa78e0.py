"""Sum-free subset finder using Schur graph MIS formulation with degree-based local search."""
import time
import random
import math
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

    def is_sum_free(elems):
        """Verify a list is sum-free."""
        s = set(elems)
        for a in elems:
            for b in elems:
                if a + b in s:
                    return False
        return True

    def find_sum_free(elements, time_budget=45.0):
        """Find a large sum-free subset using Schur graph MIS with degree-based local search."""
        if not elements:
            return []

        n = len(elements)
        elem_set = set(elements)
        elem_to_idx = {x: i for i, x in enumerate(elements)}
        my_deadline = time.time() + time_budget

        # Build Schur graph: edges represent sum conflicts
        # Vertex i conflicts with j if there exists k such that elements[i] + elements[j] = elements[k]
        # Or more precisely, the conflict graph has an edge (i,j) if i+j creates a triple
        # Actually, let's think of it differently:
        # A set is sum-free if for all a,b,c in the set, a+b != c
        # So the "forbidden" configurations are triples (a,b,c) where a+b=c
        # The conflict graph: vertex i is in conflict with vertices j,k if elements[i]+elements[j]=elements[k]
        # For MIS, we want a set with no such triples
        # 
        # Better formulation: Define a 3-uniform hypergraph where each edge is {i,j,k} with elements[i]+elements[j]=elements[k]
        # We want a set that contains no hyperedge
        # 
        # For local search, track conflicts per vertex:
        # conf[i] = number of triples (a,b,c) with i in {a,b,c} and a+b=c where all three are in current set

        # Precompute: for each pair (i,j), what is i+j? If i+j is in the set, it conflicts with vertex k=elem_to_idx[i+j]
        # Also handle i=j case: 2*i = elements[k]

        # Build conflict structure
        # For each vertex k, store pairs (i,j) such that elements[i] + elements[j] = elements[k]
        sum_pairs = [[] for _ in range(n)]
        # For each vertex i, store (j, k) pairs where elements[i] + elements[j] = elements[k]
        # This tells us: if i and j are both in the set, then k cannot be

        for k in range(n):
            for i in range(n):
                s = elements[k] - elements[i]
                if s in elem_to_idx:
                    j = elem_to_idx[s]
                    # elements[i] + elements[j] = elements[k]
                    sum_pairs[k].append((i, j))

        # Degree: number of conflicts vertex k is involved in
        # deg[k] = len(sum_pairs[k])

        def count_conflicts_in_set(selected):
            """Count violated triples in the selected set."""
            count = 0
            sel_set = set(selected)
            for k in selected:
                for (i, j) in sum_pairs[k]:
                    if i in sel_set and j in sel_set:
                        count += 1
            return count

        def can_add(selected_set, k):
            """Check if element k can be added without creating conflicts."""
            for (i, j) in sum_pairs[k]:
                if i in selected_set and j in selected_set:
                    return False
            return True

        def greedy_maximal(selected_set):
            """Greedy: add any element that doesn't conflict."""
            changed = True
            while changed:
                changed = False
                for k in range(n):
                    if k not in selected_set and can_add(selected_set, k):
                        selected_set.add(k)
                        changed = True
            return selected_set

        def greedy_by_degree(selected_set):
            """Greedy: add elements in order of fewest conflicts first."""
            candidates = list(range(n))
            candidates.sort(key=lambda k: len(sum_pairs[k]))
            for k in candidates:
                if k not in selected_set and can_add(selected_set, k):
                    selected_set.add(k)
            return selected_set

        def local_search_degree_swap(selected_set, max_iters=5000):
            """
            Local search targeting high-degree vertices for removal,
            replacing with low-degree vertices that fit.
            """
            current_conflicts = count_conflicts_in_set(selected_set)
            best_set = set(selected_set)
            best_size = len(selected_set)
            rng = random.Random(424242)

            for _ in range(max_iters):
                if time.time() > my_deadline:
                    break

                if current_conflicts == 0:
                    # Try to add more
                    added = False
                    for k in range(n):
                        if k not in selected_set and can_add(selected_set, k):
                            selected_set.add(k)
                            added = True
                            break
                    if not added:
                        # Try swapping: remove a high-degree element, add a low-degree one
                        active = list(selected_set)
                        if not active:
                            break
                        # Pick a high-degree element to try removing
                        active_sorted = sorted(active, key=lambda x: len(sum_pairs[x]), reverse=True)
                        rem = active_sorted[0]
                        selected_set.discard(rem)
                        # Try to add back something that fits
                        found_add = False
                        for k in range(n):
                            if k not in selected_set and k != rem and can_add(selected_set, k):
                                selected_set.add(k)
                                found_add = True
                                break
                        if not found_add:
                            # Put it back
                            selected_set.add(rem)
                            break
                        current_conflicts = count_conflicts_in_set(selected_set)
                        if len(selected_set) > best_size and current_conflicts == 0:
                            best_size = len(selected_set)
                            best_set = set(selected_set)
                    else:
                        current_conflicts = count_conflicts_in_set(selected_set)
                        if len(selected_set) > best_size and current_conflicts == 0:
                            best_size = len(selected_set)
                            best_set = set(selected_set)
                    continue

                # Find a conflict
                conflict_found = None
                for k in list(selected_set):
                    for (i, j) in sum_pairs[k]:
                        if i in selected_set and j in selected_set:
                            conflict_found = (i, j, k)
                            break
                    if conflict_found:
                        break

                if not conflict_found:
                    break

                i, j, k = conflict_found
                # Try removing one of the three and adding a low-degree element
                candidates = sorted([i, j, k], key=lambda x: len(sum_pairs[x]), reverse=True)
                improved = False
                for rem in candidates:
                    selected_set.discard(rem)
                    # Try to add a low-degree element that fits
                    inactive = [x for x in range(n) if x not in selected_set]
                    inactive.sort(key=lambda x: len(sum_pairs[x]))
                    for add in inactive:
                        if can_add(selected_set, add):
                            selected_set.add(add)
                            break
                    new_conflicts = count_conflicts_in_set(selected_set)
                    if new_conflicts < current_conflicts or (new_conflicts == current_conflicts and len(selected_set) > len(best_set)):
                        current_conflicts = new_conflicts
                        improved = True
                        if current_conflicts == 0 and len(selected_set) > best_size:
                            best_size = len(selected_set)
                            best_set = set(selected_set)
                        break
                    else:
                        # Revert
                        selected_set.discard(add) if add in selected_set else None
                        selected_set.add(rem)

                if not improved:
                    # Just remove the worst offender
                    selected_set.discard(candidates[0])
                    current_conflicts = count_conflicts_in_set(selected_set)

            return best_set

        def local_search_pairwise(selected_set, max_iters=3000):
            """
            Pairwise swap local search: remove two high-conflict elements, 
            try to add two low-conflict elements.
            """
            current_conflicts = count_conflicts_in_set(selected_set)
            best_set = set(selected_set)
            best_size = len(selected_set)
            rng = random.Random(55555)

            for _ in range(max_iters):
                if time.time() > my_deadline:
                    break

                if current_conflicts == 0:
                    added = False
                    for k in range(n):
                        if k not in selected_set and can_add(selected_set, k):
                            selected_set.add(k)
                            added = True
                            break
                    if not added:
                        active = list(selected_set)
                        if len(active) < 2:
                            break
                        active_sorted = sorted(active, key=lambda x: len(sum_pairs[x]), reverse=True)
                        rem1, rem2 = active_sorted[0], active_sorted[1]
                        selected_set.discard(rem1)
                        selected_set.discard(rem2)
                        inactive = [x for x in range(n) if x not in selected_set]
                        inactive.sort(key=lambda x: len(sum_pairs[x]))
                        added_any = False
                        for add in inactive:
                            if can_add(selected_set, add):
                                selected_set.add(add)
                                added_any = True
                                break
                        if not added_any:
                            selected_set.add(rem1)
                            selected_set.add(rem2)
                            break
                        current_conflicts = count_conflicts_in_set(selected_set)
                        if current_conflicts == 0 and len(selected_set) > best_size:
                            best_size = len(selected_set)
                            best_set = set(selected_set)
                    else:
                        current_conflicts = count_conflicts_in_set(selected_set)
                        if current_conflicts == 0 and len(selected_set) > best_size:
                            best_size = len(selected_set)
                            best_set = set(selected_set)
                    continue

                # Find conflict
                conflict_found = None
                for k in list(selected_set):
                    for (i, j) in sum_pairs[k]:
                        if i in selected_set and j in selected_set:
                            conflict_found = (i, j, k)
                            break
                    if conflict_found:
                        break

                if not conflict_found:
                    break

                i, j, k = conflict_found
                # Remove the highest-degree one
                to_remove = max([i, j, k], key=lambda x: len(sum_pairs[x]))
                selected_set.discard(to_remove)
                current_conflicts = count_conflicts_in_set(selected_set)
                # Try to add a low-degree element
                inactive = [x for x in range(n) if x not in selected_set]
                inactive.sort(key=lambda x: len(sum_pairs[x]))
                for add in inactive[:10]:
                    if can_add(selected_set, add):
                        selected_set.add(add)
                        current_conflicts = count_conflicts_in_set(selected_set)
                        if current_conflicts == 0 and len(selected_set) > best_size:
                            best_size = len(selected_set)
                            best_set = set(selected_set)
                        break

            return best_set

        # ========== RUN SEARCH ==========
        best_set = set()

        # 1. Sign partition (all positive or all negative)
        pos_set = {i for i in range(n) if elements[i] > 0}
        neg_set = {i for i in range(n) if elements[i] < 0}

        for s in [pos_set, neg_set]:
            gs = greedy_maximal(set(s))
            if len(gs) > len(best_set):
                best_set = gs

        # 2. Residue classes mod 3 and mod 5
        for mod in [3, 5, 7]:
            for r in range(mod):
                s = {i for i in range(n) if elements[i] % mod == r}
                gs = greedy_maximal(set(s))
                if len(gs) > len(best_set):
                    best_set = gs

        # 3. Degree-ordered greedy
        gs = greedy_by_degree(set())
        if len(gs) > len(best_set):
            best_set = gs

        # 4. Randomized greedy with multiple seeds
        for t in range(10):
            rng = random.Random(2000 + t)
            order = list(range(n))
            rng.shuffle(order)
            s = set()
            for idx in order:
                if can_add(s, idx):
                    s.add(idx)
            if len(s) > len(best_set):
                best_set = s

        # 5. Local search with degree-based swaps
        if best_set:
            ls = local_search_degree_swap(set(best_set))
            if len(ls) > len(best_set):
                best_set = ls

        # 6. Pairwise local search
        if best_set and time.time() < my_deadline - 5:
            ls2 = local_search_pairwise(set(best_set))
            if len(ls2) > len(best_set):
                best_set = ls2

        # 7. More restarts if time allows
        if time.time() < my_deadline - 10:
            for t in range(20, 60):
                if time.time() > my_deadline - 3:
                    break
                rng = random.Random(3000 + t)
                # Random initial set
                s = set()
                for idx in range(n):
                    if rng.random() < 0.4:
                        s.add(idx)
                # Clean conflicts
                changed = True
                while changed:
                    changed = False
                    for k in list(s):
                        for (i, j) in sum_pairs[k]:
                            if i in s and j in s:
                                # Remove one
                                to_rem = max([i, j, k], key=lambda x: len(sum_pairs[x]))
                                s.discard(to_rem)
                                changed = True
                                break
                        if changed:
                            break
                # Greedy complete
                gs = greedy_maximal(s)
                # Local search
                if len(gs) > len(best_set) - 2:
                    ls = local_search_degree_swap(gs)
                    if len(ls) > len(best_set):
                        best_set = ls

        # Final polish
        best_set = greedy_maximal(best_set)

        # Convert to values
        best_result = [elements[i] for i in best_set]

        # Final Verification
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
