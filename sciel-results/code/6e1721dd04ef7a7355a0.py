"""Rewritten program: maximize comparable pairs among 40 distinct subsets of {0..11}."""
import time
import random
import itertools

def construct():
    n = 12
    m = 40
    deadline = time.time() + 85  # safety margin before 90s kill

    # Key insight: to maximize comparable pairs, we want a "near-chain" structure.
    # The maximum possible is C(40,2) = 780 (a full chain).
    # A chain of length 40 on 12 elements: we need 40 distinct subsets where each is subset of next.
    # Max chain length is 13 (empty set through full set). So we can't have a pure chain of 40.
    # 
    # Strategy: use a small number of "active" elements to create many comparable pairs.
    # If we restrict to k elements, we have 2^k subsets. For k=6, we have 64 subsets.
    # We need to pick 40 from these 64 to maximize comparable pairs.
    #
    # Better approach: use a combination of a long chain and additional sets that are comparable
    # to many others.
    #
    # Let's think about this more carefully. The comparability graph on the Boolean lattice B_12.
    # We want a 40-vertex induced subgraph maximizing edges (comparability pairs).
    #
    # A greedy/local search approach:
    # 1. Start with a good initial configuration (e.g., all subsets of a 6-element subset, pick best 40)
    # 2. Use local search (swap a set for another) to improve

    # First, let's try a structured approach:
    # Use all subsets of {0,1,2,3,4,5} (64 sets), pick the 40 with maximum internal comparability.
    # Then try to improve by swapping in sets involving other elements.

    active = [0, 1, 2, 3, 4, 5]
    all_subsets = []
    for mask in range(1 << 6):
        s = tuple(sorted(b for b in range(6) if (mask >> b) & 1))
        all_subsets.append(s)

    # Convert to frozensets for comparison
    all_frozens = [frozenset(s) for s in all_subsets]

    def is_comparable(a, b):
        """Check if a and b are comparable (one strict subset of other)."""
        if a == b:
            return False
        return a.issubset(b) or b.issubset(a)

    # Precompute comparability matrix for the 64 subsets
    comp = [[False] * 64 for _ in range(64)]
    for i in range(64):
        for j in range(i + 1, 64):
            if is_comparable(all_frozens[i], all_frozens[j]):
                comp[i][j] = True
                comp[j][i] = True

    # Greedy: pick 40 subsets to maximize comparable pairs
    # Start with all, then remove the 24 that contribute least
    selected = set(range(64))

    def count_comparable(sel):
        cnt = 0
        sel_list = list(sel)
        for i in range(len(sel_list)):
            for j in range(i + 1, len(sel_list)):
                if comp[sel_list[i]][sel_list[j]]:
                    cnt += 1
        return cnt

    # Greedy removal: repeatedly remove the element whose removal loses the fewest edges
    current_score = count_comparable(selected)

    while len(selected) > 40:
        best_remove = None
        best_loss = float('inf')
        for elem in selected:
            # Count edges involving elem
            loss = sum(1 for other in selected if other != elem and comp[elem][other])
            if loss < best_loss:
                best_loss = loss
                best_remove = elem
        selected.remove(best_remove)

    initial_selected = list(selected)
    initial_score = count_comparable(selected)

    # Now do local search: try swapping one selected element with one unselected
    unselected = set(range(64)) - selected

    best_config = initial_selected[:]
    best_score = initial_score

    # Local search with restarts
    random.seed(42)

    def evaluate_config(config):
        """Count comparable pairs in a config (list of indices into all_frozens)."""
        cnt = 0
        for i in range(len(config)):
            for j in range(i + 1, len(config)):
                if comp[config[i]][config[j]]:
                    cnt += 1
        return cnt

    # Hill climbing: try all single swaps
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        config = best_config[:]
        for i in range(m):
            for j in unselected:
                if j in config:
                    continue
                # Try swapping config[i] with j
                new_config = config[:]
                new_config[i] = j
                new_score = evaluate_config(new_config)
                if new_score > best_score:
                    best_score = new_score
                    best_config = new_config
                    improved = True
                    # Update unselected
                    unselected = set(range(64)) - set(best_config)

    # Now expand the search space: allow sets involving elements 6-11
    # Build a larger candidate pool
    # Consider all subsets of {0..11} that are "close" to the active set
    # For efficiency, consider subsets that use at most 2 elements from {6,7,8,9,10,11}

    extra_elements = [6, 7, 8, 9, 10, 11]
    extended_pool = []
    extended_pool_indices = {}  # frozenset -> index in extended_pool

    # Add all 64 base subsets
    for i in range(64):
        fs = frozenset(all_subsets[i])
        extended_pool.append(fs)
        extended_pool_indices[fs] = i

    # Add subsets that use 1 extra element
    for extra in extra_elements:
        for base_idx in range(64):
            fs = frozenset(all_subsets[base_idx]) | {extra}
            if fs not in extended_pool_indices:
                extended_pool_indices[fs] = len(extended_pool)
                extended_pool.append(fs)

    # Add subsets that use 2 extra elements
    for e1, e2 in itertools.combinations(extra_elements, 2):
        for base_idx in range(64):
            fs = frozenset(all_subsets[base_idx]) | {e1, e2}
            if fs not in extended_pool_indices:
                extended_pool_indices[fs] = len(extended_pool)
                extended_pool.append(fs)

    num_extended = len(extended_pool)

    # Precompute comparability for extended pool (might be large, so do it lazily or in batches)
    # For now, let's just do local search on the extended pool using direct comparison

    def is_comparable_ext(a, b):
        """Check if frozensets a and b are comparable."""
        if a == b:
            return False
        return a.issubset(b) or b.issubset(a)

    def evaluate_config_ext(config_indices):
        """Count comparable pairs using extended pool."""
        cnt = 0
        for i in range(len(config_indices)):
            for j in range(i + 1, len(config_indices)):
                if is_comparable_ext(extended_pool[config_indices[i]], extended_pool[config_indices[j]]):
                    cnt += 1
        return cnt

    # Convert best_config from base indices to extended pool indices
    # Base indices 0..63 map directly to extended pool indices 0..63
    best_config_ext = best_config[:]

    # Local search on extended pool
    if time.time() < deadline - 10:
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            current_score = evaluate_config_ext(best_config_ext)
            for i in range(m):
                if time.time() > deadline - 10:
                    break
                for j in range(num_extended):
                    if j in best_config_ext:
                        continue
                    new_config = best_config_ext[:]
                    new_config[i] = j
                    new_score = evaluate_config_ext(new_config)
                    if new_score > current_score:
                        current_score = new_score
                        best_config_ext = new_config
                        improved = True
                        break

    # Convert back to lists of integers
    result = [sorted(extended_pool[idx]) for idx in best_config_ext]

    # Ensure types are correct: list of lists of Python ints
    for i in range(len(result)):
        result[i] = [int(x) for x in result[i]]

    return result
