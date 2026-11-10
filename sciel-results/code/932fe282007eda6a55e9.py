"""Difference triangle set (4,7): minimize largest entry via randomized backtracking with bitmask tracking."""
import time
import random
import sys

sys.setrecursionlimit(10000)


def construct(n_rows, k):
    """
    Find n_rows rows of k+1 increasing integers starting at 0 such that all
    positive differences within rows are distinct across all rows.
    Minimize the largest entry.
    """
    deadline = time.time() + 110.0  # safety margin before 120s kill

    # Total number of differences needed: n_rows * k*(k+1)/2
    num_diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * num_diffs_per_row

    # Upper bound on M (largest entry) from simple counting:
    # We need total_diffs distinct differences, each in [1, M]
    # So M >= total_diffs (since differences are from 1 to M)
    # Actually differences range from 1 to M, so we need at least total_diffs values in [1,M]
    # Thus M >= total_diffs
    M_lower = total_diffs
    # Better bound: each row has k+1 elements in [0,M], so max difference is M.
    # The differences per row are k*(k+1)/2 distinct values in [1,M].
    # Across all rows, all are distinct, so total_diffs <= M.
    # So M >= total_diffs.
    
    # Start searching from a reasonable upper bound
    # Greedy gives us a baseline
    greedy_result = _greedy(n_rows, k)
    best_M = max(max(row) for row in greedy_result)
    best_rows = greedy_result

    # Try to improve with randomized backtracking
    # We'll try increasing M from M_lower upward, but start near greedy result
    # Actually, let's do a binary-search-like or incremental approach
    
    # Use randomized backtracking to find valid configurations for given M
    # and try to minimize M
    
    # Strategy: try M values from lower bound up, using randomized backtracking
    # with time limit per attempt
    
    # For efficiency, use bitmask for tracking used differences when M is small enough
    # If M > 64, use a set instead
    
    # Let's do iterative deepening on M
    # Start from a slightly improved greedy and work down, or start from lower bound and go up
    
    # Given time constraints, let's try a few M values with randomized backtracking
    # Start from min(M_lower, best_M) and go up, but also try values below best_M
    
    # Actually, let's try to find the minimum M by trying values in increasing order
    # But that could be slow. Instead, let's do randomized search trying to minimize M.
    
    # Better approach: use randomized backtracking with a fixed M, and try multiple M values
    # in parallel (sequentially with time budget)
    
    # Let's try M values: start from M_lower, go up by 1 until we find a valid config
    # But limit total time
    
    # For the specific case n_rows=4, k=7:
    # num_diffs_per_row = 28, total_diffs = 112
    # M_lower = 112
    # Best known = 134, greedy might give something around 140-160
    
    # Let's try randomized backtracking for M from M_lower upward
    # But to save time, try a few random M values in range [M_lower, best_M]
    
    # Actually, the most effective approach: randomized backtracking for a given M
    # Try M = M_lower, M_lower+1, ... up to best_M-1
    # Use multiple random restarts for each M
    
    # Time budget allocation:
    # - 20s for greedy (already done)
    # - Remaining ~90s for backtracking
    
    # Let's be smart: try M values in order from M_lower, with a few random restarts each
    # If we can't find a solution for M, try M+1
    
    # For bitmask: if M <= 60, use int bitmask; else use set
    # For M up to ~200, we can use Python int as bitmask (arbitrary precision)
    
    # Let's implement backtracking that builds rows one at a time
    # For each row, we need to choose k+1 positions in [0, M] starting with 0
    # such that all pairwise differences are unused
    
    # Key insight: the order of elements in a row matters for backtracking efficiency
    # We build the row element by element, choosing the next element
    
    # To make this efficient, we can use the fact that differences must be distinct
    # across ALL rows, so we track a global set of used differences
    
    # Let's implement a cleaner randomized backtracking:
    
    def backtrack_row(used_diffs, M, row_so_far, k):
        """
        Build a single row of k+1 elements starting at 0, within [0, M].
        used_diffs: set of already-used differences (global)
        Returns: (row, new_used_diffs) or (None, None)
        """
        if len(row_so_far) == k + 1:
            return list(row_so_far), used_diffs
        
        last = row_so_far[-1]
        # Try next element from last+1 to M
        # Randomize order for diversity
        candidates = list(range(last + 1, M + 1))
        random.shuffle(candidates)
        
        for x in candidates:
            # Check all differences x - a for a in row_so_far
            new_diffs = set()
            valid = True
            for a in row_so_far:
                d = x - a
                if d in used_diffs:
                    valid = False
                    break
                new_diffs.add(d)
            if valid:
                new_used = used_diffs | new_diffs
                row_so_far.append(x)
                result = backtrack_row(new_used, M, row_so_far, k)
                row_so_far.pop()
                if result[0] is not None:
                    return result
        return None, None
    
    def find_configuration(M, n_rows, k, max_restarts=5):
        """
        Try to find a valid configuration with all entries <= M.
        Uses randomized backtracking with multiple restarts.
        Returns list of rows or None.
        """
        for restart in range(max_restarts):
            if time.time() > deadline:
                return None
            
            used_diffs = set()
            rows = []
            success = True
            
            for r in range(n_rows):
                if time.time() > deadline:
                    success = False
                    break
                result, new_used = backtrack_row(used_diffs, M, [0], k)
                if result is None:
                    success = False
                    break
                rows.append(result)
                used_diffs = new_used
            
            if success:
                return rows
        
        return None
    
    # Try M values from lower bound upward
    # But to be efficient, start from a few below best_M and also from M_lower
    # Given time, let's try a strategic set of M values
    
    # First, try to beat the greedy result
    # Try M values: M_lower to best_M - 1, but sample strategically
    
    # For n_rows=4, k=7: M_lower=112, best known=134
    # Let's try M from M_lower up to best_M-1
    # But with time limit, we can only try a few
    
    # Strategy: try M = M_lower, M_lower+1, ..., but limit restarts
    # Also try some random M values in [M_lower, best_M-1]
    
    # Let's be more aggressive: try M values in increasing order
    # with decreasing number of restarts as we go higher
    
    # Actually, for this problem, the answer is likely close to M_lower or slightly above
    # Let's try M_lower first with more restarts, then M_lower+1, etc.
    
    # Time budget per M value: ~5-10 seconds
    # With 90 seconds, we can try ~10-15 M values
    
    # Let's try M from M_lower to min(M_lower + 20, best_M - 1)
    # with 3 restarts each
    
    # But also, let's interleave: try a few random M values too
    
    # Simpler approach: iterate M from M_lower upward, try 2 restarts each
    # Stop when we find a solution or run out of time
    
    # To speed up backtracking, use a better heuristic:
    # When building a row, prefer smaller x values (to keep entries small)
    # But randomize for diversity
    
    # Let's enhance the backtracking with a better ordering:
    # Sort candidates by some heuristic (e.g., number of new differences that are "rare")
    
    # For now, let's use the simple randomized approach but with a twist:
    # Try candidates in a mix of ascending order with random perturbation
    
    # Let's rewrite backtrack_row to be more efficient:
    
    def backtrack_row_v2(used_diffs, M, row_so_far, k, time_limit=None):
        """
        More efficient backtracking with time checking.
        """
        if time_limit and time.time() > time_limit:
            return None, None
        
        if len(row_so_far) == k + 1:
            return list(row_so_far), used_diffs
        
        last = row_so_far[-1]
        remaining_slots = k + 1 - len(row_so_far)
        
        # The smallest possible last element if we pick consecutive values
        # We need at least remaining_slots - 1 more elements after last
        # So x must be at most M - (remaining_slots - 1 - 1) = M - remaining_slots + 2
        # Actually, we need remaining_slots - 1 more elements, each at least 1 apart
        # So the last element must be at most M - (remaining_slots - 1)
        # Wait, we're picking one element at a time. The next element x must satisfy:
        # x >= last + 1, and we need remaining_slots - 1 more elements after x
        # So x <= M - (remaining_slots - 1)
        
        max_x = M - (remaining_slots - 1)
        if max_x < last + 1:
            return None, None
        
        candidates = list(range(last + 1, max_x + 1))
        
        # Heuristic: prefer candidates that create "less common" differences
        # or simply randomize
        random.shuffle(candidates)
        
        # Also try a few in ascending order (greedy bias)
        # Mix: 70% random, 30% ascending
        if len(candidates) > 3:
            # Take first few in order, then randomize the rest
            n_greedy = max(1, len(candidates) // 5)
            greedy_part = candidates[:n_greedy]
            random_part = candidates[n_greedy:]
            random.shuffle(random_part)
            candidates = greedy_part + random_part
        
        for x in candidates:
            if time_limit and time.time() > time_limit:
                return None, None
            
            # Check differences
            new_diffs = set()
            valid = True
            for a in row_so_far:
                d = x - a
                if d in used_diffs:
                    valid = False
                    break
                new_diffs.add(d)
            
            if not valid:
                continue
            
            # Additional pruning: check if it's possible to complete the row
            # The new last element is x, and we need remaining_slots - 1 more elements
            # All future differences must be unused. This is hard to check in advance.
            
            new_used = used_diffs | new_diffs
            row_so_far.append(x)
            result = backtrack_row_v2(new_used, M, row_so_far, k, time_limit)
            row_so_far.pop()
            
            if result[0] is not None:
                return result
        
        return None, None
    
    # Now let's search over M values
    # Try from M_lower up to best_M - 1
    # With time budget
    
    search_start = M_lower
    search_end = best_M - 1
    
    if search_start > search_end:
        # Already at lower bound, nothing to improve
        return best_rows
    
    # Try M values in increasing order
    # Allocate time: total remaining / number of M values to try
    # But we don't know how many we'll try, so use a per-attempt time limit
    
    # Let's try each M with a time limit of ~5 seconds
    # and a few restarts
    
    for M in range(search_start, search_end + 1):
        if time.time() > deadline - 5:  # 5s safety margin
            break
        
        # Try to find a configuration for this M
        # Use multiple restarts with different random seeds
        max_restarts = 3
        found = None
        
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break
            
            t_start = time.time()
            used_diffs = set()
            rows = []
            success = True
            
            for r in range(n_rows):
                if time.time() > deadline - 5:
                    success = False
                    break
                
                # Time limit per row: ~2 seconds
                row_time_limit = time.time() + 2.0
                result, new_used = backtrack_row_v2(used_diffs, M, [0], k, row_time_limit)
                
                if result is None:
                    success = False
                    break
                
                rows.append(result)
                used_diffs = new_used
            
            if success:
                found = rows
                break
        
        if found is not None:
            best_M = M
            best_rows = found
            # We found a solution for this M; since we're going in increasing order,
            # this is the minimum M we've found so far. But maybe a smaller M works?
            # We started from M_lower, so if we found it at M, all smaller M's failed.
            # So this is optimal (within our search).
            break
    
    return best_rows


def _greedy(n_rows, k):
    """Simple greedy construction for baseline."""
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
