import time
import random
import sys
from itertools import combinations

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0

    # Theoretical lower bound
    num_diffs = n_rows * k * (k + 1) // 2
    lower_bound = num_diffs

    # 1. Fast greedy baseline
    best_solution = greedy_construct(n_rows, k)
    if not best_solution:
        return []
    best_scope = max(max(row) for row in best_solution)

    if best_scope <= lower_bound:
        return best_solution

    # 2. Local improvement to reduce scope
    best_solution = _improve_scope(best_solution, n_rows, k, deadline)
    best_scope = max(max(row) for row in best_solution)

    if best_scope <= lower_bound:
        return best_solution

    # 3. Clique-based search: precompute patterns and find clique of size n_rows
    # Try scopes from lower_bound up to best_scope - 1
    if time.time() < deadline - 10.0:
        clique_solution = _clique_search(n_rows, k, best_scope - 1, lower_bound, deadline)
        if clique_solution:
            clique_scope = max(max(row) for row in clique_solution)
            if clique_scope < best_scope:
                best_scope = clique_scope
                best_solution = clique_solution
                # Polish with local search
                improved = _improve_scope(best_solution, n_rows, k, deadline)
                new_scope = max(max(row) for row in improved)
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_solution = improved
                if best_scope <= lower_bound:
                    return best_solution

    # 4. Continue trying smaller scopes with clique search
    if time.time() < deadline - 5.0:
        target = best_scope - 1
        while target >= lower_bound and time.time() < deadline - 5.0:
            found = False
            for attempt in range(20):
                if time.time() > deadline - 5.0:
                    break
                sol = _clique_search(n_rows, k, target, target, deadline)
                if sol:
                    new_scope = max(max(row) for row in sol)
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best_solution = sol
                        found = True
                        improved = _improve_scope(best_solution, n_rows, k, deadline)
                        new_scope = max(max(row) for row in improved)
                        if new_scope < best_scope:
                            best_scope = new_scope
                            best_solution = improved
                        if best_scope <= lower_bound:
                            return best_solution
                        break
            if not found:
                target -= 1
            else:
                target = best_scope - 1

    return best_solution


def greedy_construct(n_rows, k):
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if not new_diffs & used:
                row.append(x)
                used |= new_diffs
        rows.append(row)
    return rows


def _clique_search(n_rows, k, max_scope, min_scope, deadline):
    """
    Find n_rows Sidon sequences of length k+1 starting at 0, all within [0, max_scope],
    with all pairwise differences distinct across rows, using clique search.
    """
    k_plus_1 = k + 1
    rng = random.Random(int(time.time() * 1000) % 100000)

    # Step 1: Enumerate all valid single-row patterns within max_scope
    # A pattern is a tuple of k+1 increasing integers starting at 0
    # whose internal pairwise differences are all distinct
    patterns = []
    pattern_diffs = []  # list of frozenset of differences for each pattern

    def enumerate_patterns(current, used_diffs):
        if time.time() > deadline:
            return
        if len(current) == k_plus_1:
            # Check all differences are distinct (they should be by construction)
            patterns.append(tuple(current))
            pattern_diffs.append(frozenset(used_diffs))
            return

        # Need to add more elements; last element is current[-1]
        last = current[-1] if current else 0
        remaining = k_plus_1 - len(current)
        # Max value we can use: need room for remaining-1 more elements
        max_val = max_scope - (remaining - 1)
        if max_val <= last:
            return

        for x in range(last + 1, max_val + 1):
            # Check new differences against used_diffs
            new_diffs = set()
            for a in current:
                d = x - a
                if d in used_diffs:
                    break
                new_diffs.add(d)
            else:
                enumerate_patterns(current + [x], used_diffs | new_diffs)
                if time.time() > deadline:
                    return

    enumerate_patterns([0], frozenset())

    if not patterns:
        return None

    # If we have fewer patterns than n_rows, impossible
    if len(patterns) < n_rows:
        return None

    # Step 2: Build conflict graph using bitsets
    # For each pattern, represent its difference set as a Python int bitmask
    # Two patterns are compatible iff (mask_i & mask_j) == 0
    masks = []
    for diffset in pattern_diffs:
        mask = 0
        for d in diffset:
            mask |= (1 << d)
        masks.append(mask)

    n_patterns = len(patterns)

    # Build adjacency: adj[i] is a set of indices j > i such that masks[i] & masks[j] == 0
    # For efficiency, use bitsets: adj_bits[i] is an int with bit j set if i and j are adjacent
    adj_bits = [0] * n_patterns
    for i in range(n_patterns):
        for j in range(i + 1, n_patterns):
            if (masks[i] & masks[j]) == 0:
                adj_bits[i] |= (1 << j)
                adj_bits[j] |= (1 << i)

    # Step 3: Find a clique of size n_rows using branch-and-bound with bitsets
    # We want to find any clique of size n_rows; if found, return the corresponding patterns

    def find_clique(candidates, clique_size, current_clique):
        """
        candidates: int bitmask of candidate vertices
        clique_size: how many more vertices we need
        current_clique: list of vertices in the current partial clique
        Returns list of vertices in the clique, or None
        """
        if time.time() > deadline:
            return None

        if clique_size == 0:
            return list(current_clique)

        # If not enough candidates, fail
        if candidates.bit_count() < clique_size:
            return None

        # If clique_size == 1, just pick any candidate
        if clique_size == 1:
            # Pick the first set bit
            v = (candidates & -candidates).bit_length() - 1
            return current_clique + [v]

        # Get list of candidates
        cand_list = []
        c = candidates
        while c:
            v = (c & -c).bit_length() - 1
            cand_list.append(v)
            c &= c - 1

        # Heuristic: order candidates by degree within the candidate set (descending)
        # to find cliques faster
        def degree_in_cand(v):
            return (adj_bits[v] & candidates).bit_count()

        cand_list.sort(key=degree_in_cand, reverse=True)

        for v in cand_list:
            if not (candidates >> v) & 1:
                continue
            # v must be in candidates
            new_candidates = candidates & adj_bits[v] & ((1 << (n_patterns - 1)) - (1 << v))
            # Actually, we need candidates that are adjacent to v and have index > v for symmetry breaking?
            # No, we need all candidates adjacent to v. But to avoid duplicates, we can restrict to candidates with index > v
            # Actually for clique finding, we just need to intersect with adj_bits[v] and remove v
            new_candidates = candidates & adj_bits[v]
            new_candidates &= ~(1 << v)  # remove v

            res = find_clique(new_candidates, clique_size - 1, current_clique + [v])
            if res is not None:
                return res

        return None

    # Initialize: all patterns are candidates
    all_candidates = (1 << n_patterns) - 1

    # Try to find a clique of size n_rows
    # Use randomized restarts with different orderings
    for restart in range(50):
        if time.time() > deadline:
            break

        # Shuffle the candidate ordering by permuting the bitmask
        # We'll do this by trying different starting vertices
        if restart == 0:
            # Natural order
            result = find_clique(all_candidates, n_rows, [])
        else:
            # Pick a random starting vertex and build from there
            start_v = rng.randint(0, n_patterns - 1)
            # Candidates that are adjacent to start_v and have index != start_v
            candidates = all_candidates & adj_bits[start_v]
            candidates &= ~(1 << start_v)
            result = find_clique(candidates, n_rows - 1, [start_v])

        if result is not None:
            # Convert vertex indices to patterns
            solution = [list(patterns[v]) for v in result]
            # Verify
            all_diffs = set()
            for row in solution:
                for i in range(len(row)):
                    for j in range(i + 1, len(row)):
                        d = row[j] - row[i]
                        if d in all_diffs:
                            return None  # Should not happen
                        all_diffs.add(d)
            return solution

    return None


def _improve_scope(solution, n_rows, k, deadline):
    """
    Reduce scope by shifting elements down. Uses efficient difference tracking.
    """
    if not solution:
        return solution

    best = [list(row) for row in solution]
    best_scope = max(max(row) for row in best)

    # Build difference set
    used_diffs = set()
    for row in best:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                used_diffs.add(row[j] - row[i])

    improved = True
    while improved and time.time() < deadline:
        improved = False
        for row_idx in range(n_rows):
            row = best[row_idx]
            # Process from end to start
            for pos in range(len(row) - 1, 0, -1):
                original = row[pos]
                if original <= row[pos - 1] + 1:
                    continue

                # Try shifting down
                for new_val in range(row[pos - 1] + 1, original):
                    # Compute new differences
                    new_diffs = set()
                    for i in range(pos):
                        new_diffs.add(new_val - row[i])
                    for j in range(pos + 1, len(row)):
                        new_diffs.add(row[j] - new_val)

                    # Old differences
                    old_diffs = set()
                    for i in range(pos):
                        old_diffs.add(original - row[i])
                    for j in range(pos + 1, len(row)):
                        old_diffs.add(row[j] - original)

                    # Check conflicts
                    if new_diffs & (used_diffs - old_diffs):
                        continue

                    # Apply
                    used_diffs.difference_update(old_diffs)
                    used_diffs.update(new_diffs)
                    row[pos] = new_val
                    improved = True
                    break

    new_scope = max(max(row) for row in best)
    if new_scope < best_scope:
        return best
    else:
        return [list(row) for row in solution]


def _randomized_improve(solution, n_rows, k, deadline, current_scope):
    """
    Randomized local search: try to reduce scope by random perturbations.
    """
    best = [list(row) for row in solution]
    best_scope = current_scope
    rng = random.Random(int(time.time() * 1000) % 100000)

    for iteration in range(200):
        if time.time() > deadline:
            break

        # Pick a random row and position
        row_idx = rng.randint(0, n_rows - 1)
        row = best[row_idx]
        pos = rng.randint(1, len(row) - 1)

        # Try a random new value
        original = row[pos]
        min_val = row[pos - 1] + 1
        max_val = min(original, best_scope)

        if max_val <= min_val:
            continue

        new_val = rng.randint(min_val, max_val)
        if new_val == original:
            continue

        # Compute differences
        new_diffs = set()
        for i in range(pos):
            new_diffs.add(new_val - row[i])
        for j in range(pos + 1, len(row)):
            new_diffs.add(row[j] - new_val)

        old_diffs = set()
        for i in range(pos):
            old_diffs.add(original - row[i])
        for j in range(pos + 1, len(row)):
            old_diffs.add(row[j] - original)

        # Build global used set
        used_diffs = set()
        for r in best:
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    used_diffs.add(r[j] - r[i])

        if new_diffs & (used_diffs - old_diffs):
            continue

        # Apply
        row[pos] = new_val
        new_scope = max(max(r) for r in best)

        if new_scope < best_scope:
            best_scope = new_scope
        else:
            # Revert
            row[pos] = original

    return best, best_scope
