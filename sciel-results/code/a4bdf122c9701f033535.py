"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import sys

# EVOLVE-BLOCK-START
def _greedy_construct(n_rows, k, rng, max_scope=None):
    """Greedy construction with randomized tie-breaking. Returns (rows, scope) or None."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            if max_scope is not None and x > max_scope:
                return None, max_scope
            new = {x - a for a in row}
            if not (new & used):
                row.append(x)
                used |= new
        rows.append(row)
    scope = max(max(r) for r in rows)
    return rows, scope


def _try_backtrack(n_rows, k, max_scope, rng, time_limit):
    """
    Backtracking search: try to place rows one by one, choosing the next row
    from pre-generated candidate rows (all valid rulers of length k+1 with max <= max_scope)
    using minimum-remaining-values heuristic.
    """
    start_time = time.time()
    
    # Pre-generate all valid "frames" (rulers) of length k+1 starting at 0 with max <= max_scope
    # A frame is a list [0, a1, a2, ..., ak] where 0 < a1 < ... < ak <= max_scope
    # and all positive differences within the frame are distinct.
    # For k=6, max_scope=112, this could be large, so we limit.
    
    # Actually, for k=6, the number of 6-element subsets of {1,...,max_scope} such that
    # the 15 pairwise differences are all distinct is the count of "Golomb rulers" of order 7.
    # For max_scope=112, this is manageable. Let's generate them.
    
    # Generate all Golomb rulers of order k+1 with max <= max_scope
    def generate_frames(k, max_scope):
        frames = []
        diffs_needed = k * (k + 1) // 2  # number of differences
        # We need to choose k elements from 1..max_scope (plus 0)
        # Use recursive generation
        def gen(start, current, diffs_used):
            if len(current) == k:
                # Check all differences are distinct
                all_diffs = set()
                for i in range(len(current)):
                    for j in range(i + 1, len(current)):
                        d = current[j] - current[i]
                        if d in all_diffs:
                            return
                        all_diffs.add(d)
                # Also check differences with 0
                for a in current:
                    if a in all_diffs:
                        return
                    all_diffs.add(a)
                frames.append([0] + current)
                return
            for x in range(start, max_scope + 1):
                # Prune: check if x - a for all a in current are in diffs_used
                valid = True
                for a in current:
                    if x - a in diffs_used:
                        valid = False
                        break
                if not valid:
                    continue
                # Also check x itself (difference from 0)
                if x in diffs_used:
                    continue
                new_diffs = set(diffs_used)
                new_diffs.add(x)
                for a in current:
                    new_diffs.add(x - a)
                gen(x + 1, current + [x], new_diffs)
        
        gen(1, [], set())
        return frames
    
    # For k=6 and max_scope up to ~112, generating all frames might be slow.
    # Let's use a smarter approach: just do randomized greedy with multiple restarts,
    # and use a local improvement step.
    
    best_rows = None
    best_scope = max_scope + 1
    deadline = start_time + time_limit
    
    # Try multiple random orderings / random greedy
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        # Randomized greedy
        rng.shuffle = None  # reset
        rows, scope = _greedy_construct(n_rows, k, rng, max_scope)
        if rows is not None and scope < best_scope:
            best_scope = scope
            best_rows = [r[:] for r in rows]
            if best_scope <= 105:  # counting bound
                break
    
    return best_rows, best_scope


def _local_search(rows, k, n_rows, rng, time_limit):
    """
    Local search: try to reduce the scope by modifying individual entries.
    For each row, try to reduce the largest element or rearrange.
    """
    start_time = time.time()
    best_rows = [r[:] for r in rows]
    best_scope = max(max(r) for r in best_rows)
    
    deadline = start_time + time_limit
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Compute used differences
        used = set()
        for row in best_rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    used.add(row[j] - row[i])
        
        # Try to improve each row
        for ri in range(n_rows):
            if time.time() > deadline:
                break
            row = best_rows[ri]
            # Try to reduce the last element
            last = row[-1]
            for new_last in range(last - 1, 0, -1):
                if new_last <= row[-2]:
                    break
                # Check if replacing last with new_last is valid
                new_row = row[:-1] + [new_last]
                # Check internal differences
                valid = True
                row_diffs = set()
                for i in range(len(new_row)):
                    for j in range(i + 1, len(new_row)):
                        d = new_row[j] - new_row[i]
                        if d in row_diffs:
                            valid = False
                            break
                        row_diffs.add(d)
                if not valid:
                    continue
                # Check against other rows' differences
                other_diffs = set()
                for oi in range(n_rows):
                    if oi == ri:
                        continue
                    orow = best_rows[oi]
                    for i in range(len(orow)):
                        for j in range(i + 1, len(orow)):
                            other_diffs.add(orow[j] - orow[i])
                if row_diffs & other_diffs:
                    continue
                # Valid!
                best_rows[ri] = new_row
                new_scope = max(max(r) for r in best_rows)
                if new_scope < best_scope:
                    best_scope = new_scope
                    improved = True
                break  # only try the first valid reduction
    
    return best_rows, best_scope


def construct(n_rows, k):
    start_time = time.time()
    total_deadline = start_time + 140  # safety margin before 150s kill
    
    # The task is for (7,5): n_rows=7, k=5, so each row has 6 elements.
    # But the function should be general.
    
    # Start with a good initial solution using greedy
    rng = random.Random(42)
    best_rows = None
    best_scope = float('inf')
    
    # Phase 1: Multiple greedy restarts
    phase1_deadline = min(start_time + 30, total_deadline - 30)
    for _ in range(1000):
        if time.time() > phase1_deadline:
            break
        rows, scope = _greedy_construct(n_rows, k, rng)
        if rows is not None and scope < best_scope:
            best_scope = scope
            best_rows = [r[:] for r in rows]
            if best_scope <= 105:
                break
    
    if best_rows is None:
        # Fallback: deterministic greedy
        rows, scope = _greedy_construct(n_rows, k, random.Random(0))
        if rows is not None:
            best_rows = rows
            best_scope = scope
    
    # Phase 2: Local search
    if best_rows is not None and time.time() < total_deadline - 5:
        phase2_deadline = min(total_deadline - 5, time.time() + 60)
        best_rows, best_scope = _local_search(best_rows, k, n_rows, rng, phase2_deadline - time.time())
    
    # Phase 3: More randomized greedy with tighter bound
    if best_rows is not None and time.time() < total_deadline - 2:
        remaining = total_deadline - time.time()
        for _ in range(500):
            if time.time() > total_deadline - 2:
                break
            rows, scope = _greedy_construct(n_rows, k, rng, max_scope=best_scope - 1)
            if rows is not None and scope < best_scope:
                best_scope = scope
                best_rows = [r[:] for r in rows]
                if best_scope <= 105:
                    break
    
    # Final local search polish
    if best_rows is not None and time.time() < total_deadline - 1:
        remaining = total_deadline - time.time() - 1
        if remaining > 2:
            best_rows, best_scope = _local_search(best_rows, k, n_rows, rng, remaining)
    
    return best_rows
# EVOLVE-BLOCK-END
