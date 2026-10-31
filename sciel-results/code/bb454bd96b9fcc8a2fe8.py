"""
Difference triangle set (4,7) solver.

Strategy:
- Phase 1: Greedy construction to get an initial valid set.
- Phase 2: Iterated Local Search (ILS) with:
    * Feasibility repair (resolve conflicts by moving elements)
    * Optimization (minimize the maximum entry)
    * Kick perturbations to escape local minima
- Deadline guard to ensure we return within 120s.
"""

import time
import random
import copy

def _get_scope(rows):
    """Return the largest entry across all rows."""
    return max(max(row) for row in rows) if rows else 0

def _is_valid(rows):
    """Check if all positive differences within rows are distinct across all rows."""
    used = set()
    for row in rows:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in used:
                    return False
                used.add(d)
    return True

def _count_conflicts(rows):
    """Count the number of duplicate differences."""
    used = set()
    conflicts = 0
    for row in rows:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in used:
                    conflicts += 1
                else:
                    used.add(d)
    return conflicts

def _greedy_construct(n_rows, k, seed=0):
    """Greedy ruler construction."""
    rng = random.Random(seed)
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
        rows.append(row)
    return rows

def _local_search(rows, n_rows, k, max_iters=5000, time_limit=None, start_time=None):
    """
    Local search to minimize the scope while maintaining validity.
    Uses simulated annealing-like acceptance.
    """
    if start_time is None:
        start_time = time.time()
    if time_limit is None:
        time_limit = 100.0

    rows = [list(r) for r in rows]
    best_rows = [list(r) for r in rows]
    best_scope = _get_scope(rows)
    current_scope = best_scope

    # Track used differences for quick conflict checking
    def get_used_set(rs):
        used = set()
        for row in rs:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    used.add(row[j] - row[i])
        return used

    def conflicts_for_move(rs, row_idx, pos, old_val, new_val):
        """Check if moving row[row_idx][pos] to new_val creates conflicts."""
        # Remove old differences involving pos
        row = rs[row_idx]
        old_diffs = set()
        for j in range(len(row)):
            if j != pos:
                d = abs(row[j] - old_val)
                old_diffs.add(d)
        
        # New differences
        new_diffs = set()
        for j in range(len(row)):
            if j != pos:
                d = abs(row[j] - new_val)
                new_diffs.add(d)
        
        # Check if new_diffs conflict with any other row's diffs or within same row
        # We need to check against all other differences
        all_other_diffs = set()
        for ri, r in enumerate(rs):
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    if ri == row_idx and (i == pos or j == pos):
                        continue
                    all_other_diffs.add(r[j] - r[i])
        
        # Also check within the same row for new value
        for j in range(len(row)):
            if j != pos:
                d = abs(row[j] - new_val)
                if d in all_other_diffs:
                    return 1  # conflict
        
        # Check if new_diffs have internal conflicts (shouldn't for a single move)
        return 0

    current_temp = 10.0
    cooling = 0.999

    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break

        # Pick a random row and position to modify
        row_idx = random.randint(0, n_rows - 1)
        pos = random.randint(1, k)  # Don't move the 0
        row = rows[row_idx]
        old_val = row[pos]

        # Try to reduce the value if possible
        if pos == k and old_val > 1:
            # Try to decrease the last element
            new_val = old_val - random.randint(1, 3)
        else:
            # Try a small random perturbation
            delta = random.randint(-3, 3)
            new_val = old_val + delta
            # Ensure increasing order
            if pos > 0 and new_val <= row[pos - 1]:
                new_val = row[pos - 1] + 1
            if pos < k and new_val >= row[pos + 1]:
                new_val = row[pos + 1] - 1
            if new_val <= row[pos - 1] or new_val >= row[pos + 1]:
                continue

        if new_val == old_val:
            continue

        # Apply move
        row[pos] = new_val

        # Check validity
        if _is_valid(rows):
            new_scope = _get_scope(rows)
            if new_scope < current_scope:
                current_scope = new_scope
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = [list(r) for r in rows]
            # Accept improvement or stay
        else:
            # Reject and revert
            row[pos] = old_val
            current_temp *= cooling
            continue

        current_temp *= cooling

    return best_rows, best_scope

def _kick_perturb(rows, n_rows, k, strength=3):
    """Apply a random perturbation to escape local minima."""
    rows = [list(r) for r in rows]
    for _ in range(strength):
        row_idx = random.randint(0, n_rows - 1)
        pos = random.randint(1, k)
        row = rows[row_idx]
        delta = random.randint(-5, 5)
        new_val = row[pos] + delta
        if pos > 0 and new_val <= row[pos - 1]:
            new_val = row[pos - 1] + 1
        if pos < k and new_val >= row[pos + 1]:
            new_val = row[pos + 1] - 1
        if new_val > 0 and new_val != row[pos]:
            row[pos] = new_val
    return rows

def construct(n_rows, k):
    """
    Main entry point. Constructs n_rows rows of k+1 increasing integers starting at 0
    such that all differences are distinct, minimizing the largest entry.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

    # Phase 1: Greedy initial construction
    best_rows = _greedy_construct(n_rows, k, seed=42)
    best_scope = _get_scope(best_rows)

    # Phase 2: Iterated Local Search
    current_rows = [list(r) for r in best_rows]
    current_scope = best_scope

    kick_count = 0
    max_kicks = 20

    while time.time() < deadline and kick_count < max_kicks:
        # Local search
        improved_rows, improved_scope = _local_search(
            current_rows, n_rows, k,
            max_iters=2000,
            time_limit=min(30.0, deadline - time.time()),
            start_time=start_time
        )

        if improved_scope < best_scope:
            best_scope = improved_scope
            best_rows = [list(r) for r in improved_rows]
            current_rows = [list(r) for r in improved_rows]
            current_scope = best_scope
            kick_count = 0  # Reset kick counter on improvement
        else:
            # No improvement, apply kick
            current_rows = _kick_perturb(current_rows, n_rows, k, strength=2)
            kick_count += 1

    # Final local search with remaining time
    if time.time() < deadline - 5.0:
        final_rows, final_scope = _local_search(
            best_rows, n_rows, k,
            max_iters=10000,
            time_limit=deadline - time.time() - 5.0,
            start_time=start_time
        )
        if final_scope < best_scope:
            best_scope = final_scope
            best_rows = [list(r) for r in final_rows]

    return best_rows
