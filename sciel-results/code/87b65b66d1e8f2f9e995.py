import time
import itertools
import random
import numpy as np

N = 16

# EVOLVE-BLOCK-START

def _dist2(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2

def _is_iso(p, a, b):
    """Check if triangle (p, a, b) is isosceles (any two sides equal)."""
    dp_a = _dist2(p, a)
    dp_b = _dist2(p, b)
    da_b = _dist2(a, b)
    return dp_a == dp_b or dp_a == da_b or dp_b == da_b

def construct():
    start_time = time.time()
    deadline = start_time + 320

    points = list(itertools.product(range(N), repeat=2))
    n_pts = N * N

    # Precompute for each point q: the set of pairs (a,b) such that (q,a,b) is isosceles
    # Represent as a list of (a_idx, b_idx)
    iso_pairs = [[] for _ in range(n_pts)]
    for q_idx in range(n_pts):
        q = points[q_idx]
        for a_idx in range(q_idx):
            a = points[a_idx]
            for b_idx in range(a_idx + 1, n_pts):
                if b_idx == q_idx:
                    continue
                b = points[b_idx]
                if _is_iso(q, a, b):
                    iso_pairs[q_idx].append((a_idx, b_idx))
        for a_idx in range(q_idx + 1, n_pts):
            a = points[a_idx]
            for b_idx in range(a_idx + 1, n_pts):
                if b_idx == q_idx:
                    continue
                b = points[b_idx]
                if _is_iso(q, a, b):
                    iso_pairs[q_idx].append((a_idx, b_idx))

    # For bitset operations: for each point p, precompute the set of pairs that would
    # form an isosceles triangle with p. We'll use this for independence checking.
    # conflict_mask[p] is a list of (a, b) pairs - when both a and b are in S, p cannot be added

    def is_independent_add(p_idx, S_mask):
        """Check if p_idx can be added to S_mask while maintaining independence."""
        for a, b in iso_pairs[p_idx]:
            if (S_mask >> a) & 1 and (S_mask >> b) & 1:
                return False
        return True

    def check_maximal(S_mask):
        """Check if S_mask is a maximal no-isosceles set."""
        for q_idx in range(n_pts):
            if (S_mask >> q_idx) & 1:
                continue
            if is_independent_add(q_idx, S_mask):
                return False
        return True

    def greedy_build(seed):
        """Greedy construction: build a maximal independent set."""
        random.seed(seed)
        order = list(range(n_pts))
        random.shuffle(order)

        S_mask = 0
        S_list = []

        for p_idx in order:
            if is_independent_add(p_idx, S_mask):
                S_mask |= (1 << p_idx)
                S_list.append(p_idx)

        # Make maximal by adding all points that can be added
        for p_idx in range(n_pts):
            if (S_mask >> p_idx) & 1:
                continue
            if is_independent_add(p_idx, S_mask):
                S_mask |= (1 << p_idx)
                S_list.append(p_idx)

        return S_mask, S_list

    def local_search_remove(S_mask, S_list, seed):
        """Try to remove points while maintaining maximality.
        
        Key insight: after removing point p, the set remains maximal iff
        no point q not in the new set can be added. We check this efficiently.
        """
        current_mask = S_mask
        current_list = S_list[:]
        random.seed(seed)

        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            random.shuffle(current_list)

            for p_idx in current_list:
                new_mask = current_mask & ~(1 << p_idx)

                # Check if new_mask is still maximal:
                # For every q not in new_mask, q cannot be added to new_mask
                can_add = False
                for q_idx in range(n_pts):
                    if (new_mask >> q_idx) & 1:
                        continue
                    if is_independent_add(q_idx, new_mask):
                        can_add = True
                        break

                if not can_add:
                    current_mask = new_mask
                    current_list.remove(p_idx)
                    improved = True
                    break

        return current_mask, current_list

    def local_search_swap(S_mask, S_list, seed):
        """Try to swap points to reduce set size while maintaining maximality."""
        current_mask = S_mask
        current_list = S_list[:]
        random.seed(seed)

        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            random.shuffle(current_list)

            for p_idx in current_list:
                # Try removing p_idx and see if we can maintain maximality
                new_mask = current_mask & ~(1 << p_idx)

                # Check if new_mask is maximal
                can_add_any = False
                for q_idx in range(n_pts):
                    if (new_mask >> q_idx) & 1:
                        continue
                    if is_independent_add(q_idx, new_mask):
                        can_add_any = True
                        break

                if not can_add_any:
                    # new_mask is already maximal, just remove
                    current_mask = new_mask
                    current_list.remove(p_idx)
                    improved = True
                    break

        return current_mask, current_list

    def randomized_rebuild(seed):
        """Rebuild with different randomization strategy."""
        random.seed(seed)

        # Strategy: start with a random subset, then fill in to make maximal
        # Try different initial densities
        density = random.choice([0.1, 0.15, 0.2, 0.25, 0.3])
        initial_size = int(n_pts * density)

        # Random initial set
        candidates = list(range(n_pts))
        random.shuffle(candidates)

        S_mask = 0
        S_list = []

        # Try to build from random candidates
        for p_idx in candidates:
            if is_independent_add(p_idx, S_mask):
                S_mask |= (1 << p_idx)
                S_list.append(p_idx)

        # Make maximal
        for p_idx in range(n_pts):
            if (S_mask >> p_idx) & 1:
                continue
            if is_independent_add(p_idx, S_mask):
                S_mask |= (1 << p_idx)
                S_list.append(p_idx)

        return S_mask, S_list

    def multi_start_local_search(seed):
        """Multi-start approach with local search."""
        random.seed(seed)

        # Try multiple strategies
        strategies = [
            greedy_build,
            randomized_rebuild,
        ]

        best_mask = None
        best_list = None
        best_size = n_pts + 1

        for strategy in strategies:
            if time.time() > deadline - 10:
                break

            S_mask, S_list = strategy(seed)
            S_mask, S_list = local_search_remove(S_mask, S_list, seed)
            S_mask, S_list = local_search_swap(S_mask, S_list, seed + 1)

            if len(S_list) < best_size:
                best_size = len(S_list)
                best_mask = S_mask
                best_list = S_list[:]

        return best_mask, best_list

    # Main search loop
    best_size = n_pts + 1
    best_set = None
    best_mask = 0

    # Phase 1: Multiple random restarts with local search
    num_trials = 500
    for trial in range(num_trials):
        if time.time() > deadline - 30:
            break

        S_mask, S_list = multi_start_local_search(trial)

        if S_list and len(S_list) < best_size:
            best_size = len(S_list)
            best_set = S_list[:]
            best_mask = S_mask

    # Phase 2: More aggressive local search on best solution
    if best_set and time.time() < deadline - 20:
        # Try harder local search
        for iteration in range(100):
            if time.time() > deadline - 10:
                break

            S_mask = best_mask
            S_list = best_set[:]

            S_mask, S_list = local_search_remove(S_mask, S_list, 10000 + iteration)

            if len(S_list) < best_size:
                best_size = len(S_list)
                best_set = S_list[:]
                best_mask = S_mask
            else:
                # No improvement, try different approach
                S_mask, S_list = greedy_build(20000 + iteration)
                S_mask, S_list = local_search_remove(S_mask, S_list, 30000 + iteration)
                if len(S_list) < best_size:
                    best_size = len(S_list)
                    best_set = S_list[:]
                    best_mask = S_mask

    # Ensure maximality of final result
    if best_set:
        changed = True
        while changed:
            changed = False
            S_mask = 0
            for idx in best_set:
                S_mask |= (1 << idx)
            for p_idx in range(n_pts):
                if p_idx in best_set:
                    continue
                if is_independent_add(p_idx, S_mask):
                    best_set.append(p_idx)
                    changed = True

    if best_set is None:
        S_mask, S_list = greedy_build(0)
        best_set = S_list

    return [points[idx] for idx in best_set]

# EVOLVE-BLOCK-END
