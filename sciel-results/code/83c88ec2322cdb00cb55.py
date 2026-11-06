"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    n = 512
    deadline = time.time() + 85.0  # Safety margin for 90s limit

    # Precompute all 4-APs
    all_4aps = []
    for a in range(1, n + 1):
        for d in range(1, (n - a) // 3 + 1):
            if a + 3*d <= n:
                all_4aps.append((a, a+d, a+2*d, a+3*d))

    # For each element, precompute which 4-APs it's part of and the other elements
    element_to_aps = {x: [] for x in range(1, n + 1)}
    for idx, (a, b, c, d_val) in enumerate(all_4aps):
        for elem in (a, b, c, d_val):
            others = [x for x in (a, b, c, d_val) if x != elem]
            element_to_aps[elem].append((others[0], others[1], others[2], idx))

    # Build conflict bitmask: for each x, a numpy bool array marking elements y
    # such that {x, y} appears together in some 4-AP with two other specific elements.
    # Actually, for can_add we need: if x is added, check if any 4-AP containing x
    # has all other 3 already present.
    #
    # Key insight: For element x in 4-AP (p, q, r, s), we need p,q,r,s all present.
    # If we're testing adding x, we need the other 3 to be present.
    # We can precompute for each x a list of "conflict triples": (o1, o2, o3)
    # such that if o1, o2, o3 are all present, x cannot be added.
    #
    # For speed: precompute a numpy array present (bool, size n+1)
    # can_add(x) = all NOT (present[o1] and present[o2] and present[o3]) for each triple

    # Precompute conflict triples per element as a 2D numpy array for vectorized checks
    # conflict_triples[x] = array of shape (k, 3) containing the other 3 elements
    conflict_triples = {}
    for x in range(1, n + 1):
        triples = element_to_aps[x]
        if triples:
            conflict_triples[x] = np.array([[t[0], t[1], t[2]] for t in triples], dtype=np.int32)
        else:
            conflict_triples[x] = np.empty((0, 3), dtype=np.int32)

    def can_add_fast(x, present):
        """Check if x can be added given current present array."""
        ct = conflict_triples[x]
        if ct.shape[0] == 0:
            return True
        # Vectorized check: for each triple (o1,o2,o3), check if all are present
        o1 = present[ct[:, 0]]
        o2 = present[ct[:, 1]]
        o3 = present[ct[:, 2]]
        return not np.any(o1 & o2 & o3)

    def greedy_construct(order, present=None):
        """Greedy construction given an ordering of candidates."""
        if present is None:
            present = np.zeros(n + 1, dtype=bool)
        selected = set()
        for x in order:
            if not present[x] and can_add_fast(x, present):
                present[x] = True
                selected.add(x)
        return selected, present

    def try_add_all(present, selected):
        """Try to add any element that can be added. Returns number added."""
        added = 0
        for x in range(1, n + 1):
            if not present[x] and can_add_fast(x, present):
                present[x] = True
                selected.add(x)
                added += 1
        return added

    def local_search(selected, present, max_iters=50):
        """Try to improve selected by adding/removing elements."""
        current_size = len(selected)

        for it in range(max_iters):
            if time.time() > deadline:
                break

            # First, try to add any available element
            added = try_add_all(present, selected)
            if added > 0:
                current_size += added
                continue

            # No single element can be added. Try 1-removal + re-add
            improved = False
            candidates_to_remove = list(selected)
            random.shuffle(candidates_to_remove)

            for x in candidates_to_remove[:20]:
                present[x] = False
                selected.discard(x)
                trial_size = current_size - 1
                added = try_add_all(present, selected)
                if added > 1:
                    current_size = trial_size + added
                    improved = True
                    break
                else:
                    # Undo
                    present[x] = True
                    selected.add(x)
                    current_size += 1

            if improved:
                continue

            # Try 2-removal + re-add (broader neighborhood)
            improved = False
            candidates_to_remove = list(selected)
            random.shuffle(candidates_to_remove)
            num_pairs = min(10, len(candidates_to_remove) // 2)
            for _ in range(num_pairs):
                i = random.randrange(len(candidates_to_remove))
                j = random.randrange(len(candidates_to_remove) - 1)
                if j >= i:
                    j += 1
                x1, x2 = candidates_to_remove[i], candidates_to_remove[j]
                if x1 == x2:
                    continue
                present[x1] = False
                present[x2] = False
                selected.discard(x1)
                selected.discard(x2)
                trial_size = current_size - 2
                added = try_add_all(present, selected)
                if added > 2:
                    current_size = trial_size + added
                    improved = True
                    break
                else:
                    # Undo
                    present[x1] = True
                    present[x2] = True
                    selected.add(x1)
                    selected.add(x2)
                    current_size += 2

            if not improved:
                break

        return selected, present

    # Phase 1: Multiple randomized greedy passes
    best_set = set()
    best_present = np.zeros(n + 1, dtype=bool)
    best_size = 0

    # Deterministic baseline
    selected, present = greedy_construct(range(1, n + 1))
    if len(selected) > best_size:
        best_size = len(selected)
        best_set = set(selected)
        best_present = present.copy()

    # Randomized greedy with local search
    candidates = list(range(1, n + 1))

    while time.time() < deadline - 15:  # Reserve 15s for final polish
        random.shuffle(candidates)
        selected, present = greedy_construct(candidates)

        # Local search refinement
        selected, present = local_search(selected, present, max_iters=40)

        if len(selected) > best_size:
            best_size = len(selected)
            best_set = set(selected)
            best_present = present.copy()

    # Phase 2: More aggressive local search on best found
    if time.time() < deadline - 3:
        improved_set, improved_present = local_search(best_set, best_present, max_iters=300)
        if len(improved_set) > best_size:
            best_size = len(improved_set)
            best_set = improved_set
            best_present = improved_present

    # Final: try adding any remaining elements
    try_add_all(best_present, best_set)

    return sorted(best_set)

# EVOLVE-BLOCK-END
