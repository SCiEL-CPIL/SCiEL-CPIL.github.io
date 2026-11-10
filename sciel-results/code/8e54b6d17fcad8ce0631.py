"""Rewritten program using randomized backtracking with bitmask difference tracking."""
import time
import random
import sys

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0,
    minimizing the largest entry, such that all positive differences
    within rows are distinct across all rows.
    
    Uses randomized backtracking with bitmask difference tracking
    and multiple restarts to explore a wider search space.
    """
    start_time = time.time()
    deadline = start_time + 115  # 15s safety margin before 130s limit
    
    # Total number of differences needed: n_rows * k*(k+1)/2
    total_diffs = n_rows * k * (k + 1) // 2
    
    # Simple counting bound: we need at least total_diffs distinct differences
    # in [1, M], so M >= total_diffs
    # But we also need each row to fit, so M must be large enough for k+1 elements
    # starting at 0 with k gaps summing to <= M
    
    # Lower bound on M from counting
    lb = total_diffs
    
    # Try to find a solution with M as small as possible
    # Start from a reasonable upper bound and try to decrease
    
    # Greedy upper bound first (fast)
    greedy_result = _greedy_construct(n_rows, k)
    best_m = max(max(row) for row in greedy_result) if greedy_result else lb
    
    # Try to improve using randomized backtracking with smaller M
    # Search from lb upward, but limit total time
    best_result = greedy_result
    best_score = best_m
    
    # Try randomized backtracking for smaller M values
    # Start near the counting bound and increase
    # But cap the number of restarts based on time remaining
    
    max_search_m = best_m - 1  # try to beat greedy
    if max_search_m < lb:
        max_search_m = lb
    
    # Try M values from lb to best_m-1
    # But be smart: start from a reasonable point
    # For small problems, we can try many M values
    # For larger problems, focus on a few
    
    # Determine search strategy based on problem size
    if n_rows <= 4 and k <= 8:
        # Small problem: try many M values with multiple restarts
        m_start = max(lb, best_m - 20)
        m_end = best_m - 1
    else:
        # Larger problem: fewer restarts, focus on promising M
        m_start = max(lb, best_m - 10)
        m_end = best_m - 1
    
    # Try M values from best_m-1 downward (most promising first)
    # Actually, let's try from a lower bound upward to find the minimum
    # But that could be slow. Let's try a few values near the bound.
    
    # Strategy: try M values from lb to best_m-1, but limit total restarts
    # and time. Use random restarts for each M.
    
    restarts_per_m = 50
    max_total_restarts = 2000
    
    total_restarts = 0
    found_improvement = False
    
    # Try M values in a smart order: start from a reasonable lower bound
    # and go up, but also try values near the greedy solution
    m_values_to_try = []
    
    # Include the counting bound and values near it
    for m in range(lb, min(lb + 30, best_m)):
        if m not in m_values_to_try:
            m_values_to_try.append(m)
    
    # Also include values just below the greedy solution
    for m in range(max(lb, best_m - 15), best_m):
        if m not in m_values_to_try:
            m_values_to_try.append(m)
    
    # Sort by M (smaller M is better)
    m_values_to_try.sort()
    
    # Remove duplicates and sort
    m_values_to_try = sorted(set(m_values_to_try))
    
    # If best_m is small, we might have already tried everything
    if not m_values_to_try:
        return best_result
    
    # For each M, try to find a valid configuration using backtracking
    for M in m_values_to_try:
        if time.time() > deadline:
            break
        if total_restarts >= max_total_restarts:
            break
        
        # Skip if M is not large enough
        if M < lb:
            continue
        
        # Try multiple random restarts for this M
        for restart in range(restarts_per_m):
            if time.time() > deadline:
                break
            if total_restarts >= max_total_restarts:
                break
            
            total_restarts += 1
            result = _backtracking_construct(n_rows, k, M, restart)
            if result is not None:
                current_max = max(max(row) for row in result)
                if current_max < best_score:
                    best_score = current_max
                    best_result = result
                    found_improvement = True
                    # If we found the counting bound, we're done
                    if current_max == lb:
                        return best_result
                # If we found a good solution, we can try smaller M
                # but for now, continue searching
                break  # Found a solution for this M, try next M
    
    return best_result


def _greedy_construct(n_rows, k):
    """Greedy construction: each row starts at 0 and takes the smallest
    next mark whose differences to earlier marks are all unused."""
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


def _backtracking_construct(n_rows, k, M, restart_seed):
    """
    Backtracking construction with a maximum value M.
    Returns a list of n_rows rows if successful, None otherwise.
    
    Uses randomized element ordering to explore different search paths.
    """
    rng = random.Random(restart_seed * 1000003 + M * 7919)
    
    # Bitmask for differences: we need to track which differences [1..M] are used
    # M can be up to ~150, so we use a Python int as a bitset
    # Bit i (0-indexed) represents difference i+1
    
    def diff_bit(d):
        """Return the bitmask for difference d (1-indexed)."""
        return 1 << (d - 1)
    
    # Track used differences as a bitmask
    used_mask = 0
    
    # For each row, we need to place k+1 elements starting at 0
    # The remaining k elements must be chosen from [1..M]
    
    rows = []
    
    # Backtracking state: we build rows one at a time
    # For each row, we build it element by element
    
    def build_row(row_idx, current_row, current_max):
        """
        Build the next row starting from current_row.
        Returns (success, new_row, new_max) or (False, None, None).
        """
        if len(current_row) == k + 1:
            return True, current_row, current_max
        
        # We need to add more elements to this row
        # Try adding elements in a randomized order
        # Elements must be > current_max (strictly increasing)
        # and <= M
        
        # Generate candidate positions
        candidates = list(range(current_max + 1, M + 1))
        
        # Randomize the order for diversity
        rng.shuffle(candidates)
        
        # But we want to try smaller values first for efficiency
        # Use a weighted random: smaller values more likely
        # Actually, let's just shuffle and try
        
        for pos in candidates:
            # Check if adding pos creates any conflict
            # New differences: pos - a for a in current_row
            new_diffs = set()
            conflict = False
            for a in current_row:
                d = pos - a
                if d > M:
                    conflict = True
                    break
                bit = diff_bit(d)
                if used_mask & bit:
                    conflict = True
                    break
                new_diffs.add(d)
            
            if conflict:
                continue
            
            # Check for internal conflicts within the new row
            # (differences between new pos and existing elements should be distinct
            #  and not conflict with each other)
            # Since current_row is strictly increasing, pos - a for different a
            # are all different (because a are different)
            # So no internal conflict within this addition
            
            # Recurse
            new_row = current_row + [pos]
            success, completed_row, final_max = build_row(row_idx, new_row, pos)
            if success:
                return True, completed_row, final_max
        
        return False, None, None
    
    # Build each row
    for r in range(n_rows):
        if time.time() > start_time + 115:
            return None
        
        # Build row r
        # Start with [0]
        success, completed_row, final_max = build_row(r, [0], 0)
        
        if not success:
            return None
        
        # Update used_mask with differences from this row
        for i in range(len(completed_row)):
            for j in range(i + 1, len(completed_row)):
                d = completed_row[j] - completed_row[i]
                if d > M:
                    return None
                used_mask |= diff_bit(d)
        
        rows.append(completed_row)
    
    return rows


# EVOLVE-BLOCK-END
