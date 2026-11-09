import numpy as np
import time
import random
from functools import lru_cache

def furstenberg_sarkozy_205() -> np.ndarray:
    m = 205
    start_time = time.time()
    deadline = start_time + 570  # Safety margin (600s limit, leave 30s buffer)

    # Compute forbidden differences: nonzero squares mod 205
    forbidden = set()
    for x in range(1, m):
        sq = pow(x, 2, m)
        if sq != 0:
            forbidden.add(sq)

    # Build conflict masks: conflict_mask[i] has bit j set iff i conflicts with j
    conflict_mask = [0] * m
    for i in range(m):
        mask = 0
        for d in forbidden:
            j1 = (i + d) % m
            j2 = (i - d) % m
            mask |= (1 << j1)
            mask |= (1 << j2)
        conflict_mask[i] = mask

    # Precompute degree (number of conflicts) for each vertex
    degree = [bin(conflict_mask[i]).count('1') for i in range(m)]

    def greedy_construct(order):
        """Greedy construction given an order of candidates."""
        selected_mask = 0
        selected_list = []
        for candidate in order:
            if (conflict_mask[candidate] & selected_mask) == 0:
                selected_mask |= (1 << candidate)
                selected_list.append(candidate)
        return selected_list, selected_mask

    def greedy_construct_weighted():
        """Weighted greedy: at each step pick the available candidate with lowest degree."""
        selected_mask = 0
        selected_list = []
        available = list(range(m))
        while available:
            # Pick candidate with minimum degree among available
            best_idx = 0
            best_deg = degree[available[0]]
            for i in range(1, len(available)):
                if degree[available[i]] < best_deg:
                    best_idx = i
                    best_deg = degree[available[i]]
            candidate = available.pop(best_idx)
            if (conflict_mask[candidate] & selected_mask) == 0:
                selected_mask |= (1 << candidate)
                selected_list.append(candidate)
                # Remove all conflicts from available
                cm = conflict_mask[candidate]
                available = [v for v in available if (cm & (1 << v)) == 0]
        return selected_list, selected_mask

    def local_search(subset, mask, max_rounds=50):
        """
        Local search: try to increase size by removing elements and greedily adding back.
        """
        subset = list(subset)
        mask = mask
        improved = True
        rounds = 0
        while improved and rounds < max_rounds:
            improved = False
            rounds += 1
            n = len(subset)

            # Try removing 1 element and greedily adding back
            for i in range(n):
                if time.time() > deadline - 5:
                    return subset, mask
                u = subset[i]
                trial_mask = mask & ~(1 << u)
                trial_subset = subset[:i] + subset[i+1:]

                new_mask = trial_mask
                new_subset = list(trial_subset)
                for v in range(m):
                    if (new_mask & (1 << v)) == 0 and (conflict_mask[v] & new_mask) == 0:
                        new_mask |= (1 << v)
                        new_subset.append(v)

                if len(new_subset) > len(subset):
                    subset = new_subset
                    mask = new_mask
                    improved = True
                    break

            # Try removing 2 elements and greedily adding back
            if not improved:
                for i in range(n):
                    if time.time() > deadline - 5:
                        return subset, mask
                    for j in range(i+1, n):
                        if time.time() > deadline - 5:
                            return subset, mask
                        u, v = subset[i], subset[j]
                        trial_mask = mask & ~(1 << u) & ~(1 << v)
                        trial_subset = [x for x in subset if x != u and x != v]

                        new_mask = trial_mask
                        new_subset = list(trial_subset)
                        for w in range(m):
                            if (new_mask & (1 << w)) == 0 and (conflict_mask[w] & new_mask) == 0:
                                new_mask |= (1 << w)
                                new_subset.append(w)

                        if len(new_subset) > len(subset):
                            subset = new_subset
                            mask = new_mask
                            improved = True
                            break
                    if improved:
                        break

        return subset, mask

    # Branch-and-bound maximum independent set solver
    # State: (available_mask, selected_count) -> best additional vertices we can add
    # We use memoization on the available set to exploit symmetry

    # First, get an initial solution from greedy to set a lower bound
    initial_subset, initial_mask = greedy_construct_weighted()
    initial_subset, initial_mask = local_search(initial_subset, initial_mask)
    best_size = len(initial_subset)
    best_subset = list(initial_subset)

    # Branch and bound with memoization
    # Key insight: the graph is vertex-transitive (Cayley graph), so we can canonicalize states
    # For simplicity, we'll use the available mask as the key (up to 205 bits)

    # To make this tractable, we'll use a recursive B&B that picks vertices in a smart order
    # and uses the current best as a pruning bound

    # Precompute: for each vertex, its "closed neighborhood" (itself + all conflicts)
    closed_neighborhood = [0] * m
    for i in range(m):
        closed_neighborhood[i] = conflict_mask[i] | (1 << i)

    # Sort vertices by degree (ascending) for better pruning
    vertices_by_degree = sorted(range(m), key=lambda x: degree[x])

    def branch_and_bound(available_mask, current_count, current_selected):
        """
        Recursive branch-and-bound to find maximum independent set.
        Returns (best_count, best_selected) from this state.
        """
        nonlocal best_size, best_subset, deadline

        if time.time() > deadline - 10:
            return current_count, list(current_selected)

        # Upper bound: current_count + number of available vertices
        # Better upper bound: use a greedy packing
        if available_mask == 0:
            return current_count, list(current_selected)

        # Quick upper bound: if current_count + popcount(available_mask) <= best_size, prune
        avail_count = bin(available_mask).count('1')
        if current_count + avail_count <= best_size:
            return current_count, list(current_selected)

        # Pick a vertex to branch on
        # Strategy: pick the available vertex with minimum degree in the available subgraph
        # This tends to prune more effectively
        best_vertex = -1
        min_deg = float('inf')
        for v in vertices_by_degree:
            if (available_mask >> v) & 1:
                # Count conflicts within available set
                deg_in_avail = bin(conflict_mask[v] & available_mask).count('1')
                if deg_in_avail < min_deg:
                    min_deg = deg_in_avail
                    best_vertex = v
                if deg_in_avail == 0:
                    break

        if best_vertex == -1:
            return current_count, list(current_selected)

        # Branch 1: Include best_vertex
        new_available = available_mask & ~closed_neighborhood[best_vertex]
        new_selected = list(current_selected) + [best_vertex]
        count1, selected1 = branch_and_bound(new_available, current_count + 1, new_selected)

        # Update global best if improved
        if count1 > best_size:
            best_size = count1
            best_subset = selected1

        # Branch 2: Exclude best_vertex
        new_available2 = available_mask & ~(1 << best_vertex)
        count2, selected2 = branch_and_bound(new_available2, current_count, current_selected)

        if count2 > best_size:
            best_size = count2
            best_subset = selected2

        # Return the better of the two branches
        if count1 >= count2:
            return count1, selected1
        else:
            return count2, selected2

    # Run branch-and-bound from the full set
    full_mask = (1 << m) - 1
    bnb_count, bnb_selected = branch_and_bound(full_mask, 0, [])

    # If B&B found something better, use it
    if bnb_count > len(best_subset):
        best_subset = bnb_selected
        best_size = bnb_count

    # Final validation
    if not best_subset:
        subset, mask = greedy_construct(list(range(m)))
        best_subset = subset
        best_size = len(best_subset)

    # Verify no forbidden pairs
    best_set = set(best_subset)
    for a in best_subset:
        for b in best_subset:
            if a == b:
                continue
            diff = (a - b) % m
            if diff in forbidden:
                # Should not happen, but if it does, remove b
                best_subset.remove(b)
                best_set.discard(b)
                break

    return np.array(best_subset, dtype=np.int64)
