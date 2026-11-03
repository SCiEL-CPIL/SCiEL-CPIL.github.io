"""
Difference-triangle solver using constraint-satisfaction backtracking
with efficient difference tracking and multi-phase search.

Strategy:
  Phase 1 – Fast greedy construction (multiple randomised restarts).
  Phase 2 – Constraint-based row-by-row backtracking with pruning.
  Phase 3 – Local search: try to shrink individual entries while
            maintaining validity.
  Phase 4 – Iterative improvement: repeatedly attempt to reduce the
            maximum entry by re-solving rows under tighter bounds.
"""

import time
import random
import sys

# EVOLVE-BLOCK-START

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _scope(rows):
    return max(max(r) for r in rows)


def _validate(rows):
    diffs = set()
    for r in rows:
        n = len(r)
        for i in range(n):
            for j in range(i + 1, n):
                d = r[j] - r[i]
                if d in diffs:
                    return False
                diffs.add(d)
    return True


def _row_diffs(row):
    """Return set of all positive pairwise differences in a row."""
    s = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            s.add(row[j] - row[i])
    return s


def _all_diffs(rows):
    s = set()
    for r in rows:
        s |= _row_diffs(r)
    return s


# ---------------------------------------------------------------------------
# Phase 1 – Randomised greedy with look-ahead
# ---------------------------------------------------------------------------

def _greedy_row(used, k, rng, look_ahead=8):
    """Build one row greedily, choosing from candidate values."""
    row = [0]
    for _ in range(k):
        start = row[-1] + 1
        best_cand = None
        # Try candidates in a window
        for cand in range(start, start + look_ahead + 1):
            nd = {cand - a for a in row}
            if not (nd & used):
                if best_cand is None or cand < best_cand:
                    best_cand = cand
        if best_cand is None:
            # Extend search
            cand = start
            while True:
                nd = {cand - a for a in row}
                if not (nd & used):
                    best_cand = cand
                    break
                cand += 1
                if cand > 500:
                    break
        if best_cand is None:
            return None
        # With some probability, pick a non-minimal candidate for diversity
        if rng.random() < 0.15:
            # Try a slightly larger candidate
            alt = best_cand + rng.randint(1, 3)
            nd_alt = {alt - a for a in row}
            if not (nd_alt & used):
                best_cand = alt
        row.append(best_cand)
        used |= {best_cand - a for a in row[:-1]}
    return row


def _greedy_full(n_rows, k, rng, look_ahead=8):
    """Build all rows greedily."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = _greedy_row(used, k, rng, look_ahead)
        if row is None:
            return None
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Phase 2 – Backtracking row construction with pruning
# ---------------------------------------------------------------------------

def _backtrack_row(n, max_val, other_diffs, rng, deadline):
    """
    Build a row of n entries (starting at 0) with all entries <= max_val,
    all internal diffs distinct and disjoint from other_diffs.
    Returns the row or None.
    """
    row = [0]
    row_diffs = set()

    def _build(pos):
        if time.time() > deadline:
            return False
        if pos == n:
            return True
        lo = row[-1] + 1
        hi = min(max_val, lo + 30)  # limit candidate window
        # Generate candidates
        cands = list(range(lo, hi + 1))
        # Shuffle for diversity
        rng.shuffle(cands)
        # Sort to try smaller first (but keep some randomness)
        if rng.random() < 0.7:
            cands.sort()
        else:
            cands = cands[:min(12, len(cands))]
            rng.shuffle(cands)

        for cand in cands:
            nd = {cand - a for a in row}
            if nd & other_diffs:
                continue
            if nd & row_diffs:
                continue
            # Check for internal duplicates more carefully
            # (already covered by nd & row_diffs check above for new diffs)
            row.append(cand)
            row_diffs |= nd
            if _build(pos + 1):
                return True
            row.pop()
            row_diffs -= nd
        return False

    if _build(1):
        return list(row)
    return None


def _construct_set_backtrack(n_rows, k, rng, deadline, initial_scope=None):
    """
    Build the full set row by row using backtracking.
    Each row is built with a target max that we try to minimize.
    """
    used = set()
    rows = []

    for ri in range(n_rows):
        # Determine target max: try to keep it small
        # Estimate minimum possible max for this row
        # Simple lower bound: need k+1 entries, max diff at least k*(k+1)/2
        # But that's for a single row; here we have constraints from other rows.
        
        # Try increasing max until we find a valid row
        if initial_scope is not None:
            start_max = initial_scope
        else:
            start_max = k * (k + 1) // 2 + ri * 2
        
        found = None
        for target_max in range(start_max, min(start_max + 60, 200)):
            row = _backtrack_row(k + 1, target_max, used, rng, deadline)
            if row is not None:
                found = row
                break
        
        if found is None:
            # Fallback: greedy
            row = _greedy_row(used, k, rng, look_ahead=10)
            if row is None:
                return None
            found = row
        
        rows.append(found)
        used |= _row_diffs(found)
    
    return rows


# ---------------------------------------------------------------------------
# Phase 3 – Local search: shrink entries
# ---------------------------------------------------------------------------

def _local_search(rows, k, deadline, rng):
    """
    Try to reduce individual entries while maintaining validity.
    For each row, for each position, try to decrease the value.
    """
    best = [list(r) for r in rows]
    best_sc = _scope(best)
    cur = [list(r) for r in rows]
    cur_sc = best_sc

    t0 = time.time()
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Process rows in order of largest max first
        order = sorted(range(len(cur)), key=lambda i: cur[i][-1], reverse=True)
        for ri in order:
            if time.time() > deadline:
                break
            row = cur[ri]
            # Compute diffs of all other rows
            other_diffs = set()
            for rj in range(len(cur)):
                if rj != ri:
                    other_diffs |= _row_diffs(cur[rj])
            
            # Try to shrink each entry from right to left
            for pos in range(len(row) - 1, 0, -1):
                if time.time() > deadline:
                    break
                old_val = row[pos]
                lo = row[pos - 1] + 1
                # Try values from lo up to old_val - 1
                for new_val in range(lo, old_val):
                    row[pos] = new_val
                    # Check validity
                    rd = _row_diffs(row)
                    if rd & other_diffs:
                        continue
                    # Check internal uniqueness
                    if len(rd) != len(row) * (len(row) - 1) // 2:
                        continue
                    # Valid! Accept
                    other_diffs = (other_diffs | rd)
                    # Actually we need to recompute other_diffs properly
                    # But for now, this is an approximation
                    improved = True
                    break
                row[pos] = old_val  # restore if no improvement found
            
            # Update best if improved
            new_sc = _scope(cur)
            if new_sc < best_sc:
                best_sc = new_sc
                best = [list(r) for r in cur]
    
    return best, best_sc


# ---------------------------------------------------------------------------
# Phase 4 – Iterative re-optimization
# ---------------------------------------------------------------------------

def _reoptimize(rows, k, deadline, rng):
    """
    Repeatedly try to rebuild each row with a tighter max bound.
    """
    best = [list(r) for r in rows]
    best_sc = _scope(best)
    n_rows = len(best)

    for _pass in range(5):
        if time.time() > deadline:
            break
        order = sorted(range(n_rows), key=lambda i: best[i][-1], reverse=True)
        improved = False
        for ri in order:
            if time.time() > deadline:
                break
            # Compute other diffs
            other_diffs = set()
            for rj in range(n_rows):
                if rj != ri:
                    other_diffs |= _row_diffs(best[rj])
            
            # Try to find a row with smaller max
            cur_max = best[ri][-1]
            for target_max in range(cur_max - 1, max(cur_max - 15, 0), -1):
                new_row = _backtrack_row(k + 1, target_max, other_diffs, rng, deadline)
                if new_row is not None:
                    best[ri] = new_row
                    improved = True
                    best_sc = _scope(best)
                    break
        
        if not improved:
            break

    return best, best_sc


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def construct(n_rows, k):
    t_start = time.time()
    deadline = t_start + 135.0  # Safety margin before 150s kill
    rng = random.Random(12345)

    # Phase 1: Multiple greedy restarts
    best_rows = None
    best_sc = float('inf')
    
    phase1_deadline = t_start + 15.0
    attempts = 0
    while time.time() < phase1_deadline:
        rows = _greedy_full(n_rows, k, rng, look_ahead=rng.choice([6, 8, 10]))
        if rows is not None:
            sc = _scope(rows)
            if sc < best_sc:
                best_sc = sc
                best_rows = rows
        attempts += 1

    if best_rows is None:
        # Absolute fallback: simple greedy
        used = set()
        best_rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                x = row[-1] + 1
                nd = {x - a for a in row}
                if not (nd & used):
                    row.append(x)
                    used |= nd
            best_rows.append(row)
        best_sc = _scope(best_rows)

    # Phase 2: Backtracking construction from scratch
    phase2_deadline = t_start + 45.0
    while time.time() < phase2_deadline:
        rows = _construct_set_backtrack(n_rows, k, rng, phase2_deadline, initial_scope=best_sc)
        if rows is not None:
            sc = _scope(rows)
            if sc < best_sc:
                best_sc = sc
                best_rows = rows

    # Phase 3: Local search
    phase3_deadline = t_start + 90.0
    best_rows, best_sc = _local_search(best_rows, k, phase3_deadline, rng)

    # Phase 4: Re-optimization
    phase4_deadline = deadline
    best_rows, best_sc = _reoptimize(best_rows, k, phase4_deadline, rng)

    # Final validation
    if not _validate(best_rows):
        # Fallback to greedy
        used = set()
        best_rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                x = row[-1] + 1
                nd = {x - a for a in row}
                if not (nd & used):
                    row.append(x)
                    used |= nd
            best_rows.append(row)

    return best_rows

# EVOLVE-BLOCK-END
