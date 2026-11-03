import time
import random

def construct(n_rows, k):
    start_time = time.time()
    # 150s limit; use 118s for safety margin
    deadline = start_time + 118.0

    # Bitset mask for differences up to a generous bound
    # Differences can't exceed scope; scope <= greedy_scope
    greedy = greedy_construct(n_rows, k)
    if not greedy:
        return []
    best_scope = max(max(row) for row in greedy)
    best_solution = greedy

    # Total distinct differences required
    num_diffs = n_rows * k * (k + 1) // 2
    lower_bound = num_diffs

    if best_scope <= lower_bound:
        return best_solution

    # Bitset mask: bit d set means difference d is used.
    # We'll use differences in [1, max_scope].
    # Represent as Python int; bit position = difference value.

    # Targeted scope reduction: try to find a solution with max <= target
    # for decreasing targets.
    for target_scope in range(best_scope - 1, lower_bound - 1, -1):
        if time.time() > deadline - 2.0:
            break
        # Bitset mask for differences in [1, target_scope]
        # (we only care about diffs <= target_scope for validity within this scope)
        sol = backtracking_search(n_rows, k, target_scope, deadline)
        if sol:
            new_scope = max(max(row) for row in sol)
            if new_scope < best_scope:
                best_scope = new_scope
                best_solution = sol
                # Update: continue trying to reduce further
                # (loop already goes downward; we found one at target_scope,
                #  but actual new_scope may be < target_scope, so continue)
                if best_scope <= lower_bound:
                    break

    # Polish: try to reduce the largest entry of each row individually
    if time.time() < deadline - 1.0:
        best_solution = polish_rows(best_solution, n_rows, k, deadline)
        best_scope = max(max(row) for row in best_solution)

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


def backtracking_search(n_rows, k, max_scope, deadline):
    """
    Find n_rows rows of k+1 increasing ints starting at 0,
    with all pairwise differences distinct across all rows,
    and max entry <= max_scope.
    Uses bitset (Python int) for used differences.
    """
    used_bits = 0  # bit d set => difference d is used
    rows = []

    def build_row(pos, current_mark, row_so_far):
        nonlocal used_bits
        if time.time() > deadline:
            return None
        if pos == k + 1:
            return row_so_far

        # Find valid candidates: x > current_mark, x <= max_scope,
        # and {x - a for a in row_so_far} disjoint from used_bits.
        # Build candidate list, then sample from smallest ones.
        valid_candidates = []
        # Precompute row_so_far for speed
        rsf = row_so_far
        for x in range(current_mark + 1, max_scope + 1):
            # Compute new diffs bitset
            nb = 0
            ok = True
            for a in rsf:
                d = x - a
                bit = 1 << d
                if nb & bit:  # duplicate within this row (shouldn't happen since increasing)
                    ok = False
                    break
                if used_bits & bit:
                    ok = False
                    break
                nb |= bit
            if ok:
                valid_candidates.append((x, nb))

        if not valid_candidates:
            return None

        # Sample from the smallest candidates to encourage small scope
        pool = valid_candidates[:30]
        sample_size = min(6, len(pool))
        chosen = random.sample(pool, sample_size)
        random.shuffle(chosen)

        for x, nb in chosen:
            used_bits |= nb
            res = build_row(pos + 1, x, row_so_far + [x])
            if res is not None:
                return res
            used_bits &= ~nb

        return None

    for i in range(n_rows):
        row = build_row(1, 0, [0])
        if row is None:
            # Undo any partial state (used_bits is global per call)
            return None
        rows.append(row)

    return rows


def polish_rows(solution, n_rows, k, deadline):
    """
    Try to reduce the largest entry of each row by replacing that row
    with a new row that fits within the current used-differences set
    (minus the old row's diffs) and has a smaller max entry.
    """
    # Compute the set of used differences (as a bitset) from the current solution
    used_bits = 0
    for row in solution:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                used_bits |= (1 << d)

    new_solution = list(solution)
    for r in range(n_rows):
        if time.time() > deadline:
            break
        old_row = solution[r]
        old_max = old_row[-1]
        # Remove old row's diffs from used_bits
        old_diffs_bits = 0
        for i in range(len(old_row)):
            for j in range(i + 1, len(old_row)):
                d = old_row[j] - old_row[i]
                old_diffs_bits |= (1 << d)
        remaining_bits = used_bits & ~old_diffs_bits

        # Try to find a new row with max < old_max
        # Search for max in range [old_max - 1, lower_bound_for_row]
        # Lower bound for a single row of k+1 elements: at least k*(k+1)/2 diffs
        # but we need them distinct and not in remaining_bits.
        # Just try decreasing max targets.
        found = False
        for target_max in range(old_max - 1, 0, -1):
            if time.time() > deadline:
                break
            new_row = find_row_with_max(k, target_max, remaining_bits)
            if new_row is not None:
                new_solution[r] = new_row
                # Update used_bits: add new row's diffs
                for i in range(len(new_row)):
                    for j in range(i + 1, len(new_row)):
                        d = new_row[j] - new_row[i]
                        used_bits |= (1 << d)
                found = True
                break
        # If not found, keep old row (used_bits unchanged for this row)

    return new_solution


def find_row_with_max(k, max_val, forbidden_bits):
    """
    Find a row of k+1 increasing ints starting at 0, max <= max_val,
    such that all pairwise diffs are not in forbidden_bits and distinct.
    Returns the row or None.
    """
    def build(pos, current_mark, row_so_far, local_used):
        if pos == k + 1:
            return row_so_far
        rsf = row_so_far
        valid = []
        for x in range(current_mark + 1, max_val + 1):
            nb = 0
            ok = True
            for a in rsf:
                d = x - a
                bit = 1 << d
                if local_used & bit or forbidden_bits & bit:
                    ok = False
                    break
                nb |= bit
            if ok:
                valid.append((x, nb))
        if not valid:
            return None
        # Try smallest first for polish (we want to minimize max)
        for x, nb in valid:
            local_used |= nb
            res = build(pos + 1, x, row_so_far + [x], local_used)
            if res is not None:
                return res
            local_used &= ~nb
        return None

    return build(1, 0, [0], 0)
