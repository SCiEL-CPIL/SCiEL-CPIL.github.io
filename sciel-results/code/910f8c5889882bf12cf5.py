"""Difference triangle set (7,5): minimize the largest entry.
Structural redesign: difference-level assignment with row reconstruction,
combined with multi-seed greedy initialization and simulated annealing polish."""
import time
import random
import math
from collections import defaultdict

# EVOLVE-BLOCK-START

def _row_diffs(row):
    """Return the set of all positive differences in a row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs


def _all_diffs(rows):
    """Return the complete set of all differences across all rows."""
    used = set()
    for row in rows:
        used |= _row_diffs(row)
    return used


def _valid(rows):
    """Check rows are valid: start at 0, strictly increasing, globally distinct diffs."""
    all_d = set()
    for row in rows:
        if not row or row[0] != 0:
            return False
        for i in range(len(row) - 1):
            if row[i + 1] <= row[i]:
                return False
        rd = _row_diffs(row)
        if rd & all_d:
            return False
        all_d |= rd
    return True


def _scope(rows):
    """Return the largest entry across all rows."""
    if not rows:
        return 0
    return max(max(r) for r in rows)


def _reconstruct_row_from_diffs(diffs, k):
    """
    Given a set of C(k+1,2) positive integers that should be the differences
    of a (k+1)-element set starting at 0, try to reconstruct the set.
    Returns the row or None if reconstruction fails.
    """
    n = k + 1
    expected_count = n * (n - 1) // 2
    if len(diffs) != expected_count:
        return None
    
    diffs_sorted = sorted(diffs)
    
    # The smallest difference must be the gap between consecutive elements
    # Strategy: the set is {0, a1, a2, ..., ak} with 0 < a1 < ... < ak
    # The differences are {aj - ai : i < j}
    # The smallest difference is min(aj - a(j-1)) for j=1..k
    
    # Key insight: if we know the set, we can verify. If we don't, we need to search.
    # Use the fact that the largest difference is ak (the max element).
    max_diff = max(diffs)
    
    # Try to reconstruct by assuming the max element is max_diff
    # The set must contain 0 and max_diff.
    # The remaining k-1 elements must be chosen such that all differences match.
    
    # A better approach: use the structure that differences form a "difference set"
    # For a set {0, x1, x2, ..., x_{k-1}, M} where M = max_diff:
    # The differences include {M, M-x1, M-x2, ..., M-x_{k-1}}
    # So {M - d : d in diffs, d < M} should give us candidates for the other elements
    # Actually, the elements are: 0, and values v where M-v is in diffs for some v
    
    # Let's try: the elements (other than 0 and M) are values v where:
    # - v is in the range (0, M)
    # - For each such v, the set {v, M-v} plus all differences between v and other elements must be in diffs
    
    # Simpler reconstruction: 
    # Sort diffs. The smallest diff d_min must be a gap between consecutive elements.
    # This is complex. Let's use a different approach:
    # 
    # We know the set has n elements. Let's denote them as a_0=0 < a_1 < ... < a_{n-1}
    # The differences are all a_j - a_i for i<j.
    # 
    # Key property: a_{n-1} = max(diffs)
    # a_{n-2} = a_{n-1} - min{d in diffs : d > 0 and a_{n-1}-d is also consistent}
    # 
    # This is still complex. Let's use a search-based reconstruction.
    
    # For small k (k=5, so n=6), we can do a targeted search.
    # The set is {0, x1, x2, x3, x4, x5} with 0 < x1 < x2 < x3 < x4 < x5 = M
    # Differences: x1, x2, x3, x4, x5, x2-x1, x3-x1, x4-x1, x5-x1, x3-x2, x4-x2, x5-x2, x4-x3, x5-x3, x5-x4
    # That's 15 differences.
    
    M = max_diff
    if M not in diffs:
        return None
    
    # x5 = M
    # The differences involving x5 are: M, M-x1, M-x2, M-x3, M-x4
    # So {M-x1, M-x2, M-x3, M-x4} must be in diffs (and distinct from M)
    # Thus x1, x2, x3, x4 are in {M - d : d in diffs, d < M}
    
    candidates = set()
    for d in diffs:
        if d < M:
            candidates.add(M - d)
    candidates.discard(0)
    candidates = {c for c in candidates if 0 < c < M}
    
    # We need to choose 4 values from candidates such that all 15 differences match.
    # For k=5, this is manageable with backtracking.
    
    def try_reconstruct(cand_list, k_val, M_val, diff_set):
        """Try to find k_val values from cand_list (plus 0 and M_val) that produce exactly diff_set."""
        n_elem = k_val + 1  # total elements including 0 and M
        # We need to choose k_val-1 intermediate values
        # Actually, we have n_elem - 2 = k_val - 1 intermediate values to choose
        
        if k_val <= 1:
            # Only 0 and M
            if k_val == 1:
                # n=2, diffs = {M}
                return {0, M}
            return None
        
        # For n=6 (k=5), we need 4 intermediate values
        # This could be expensive. Let's use a smarter approach.
        
        # Actually, let's just return None for now and use a different reconstruction method.
        return None
    
    # Alternative: use the fact that for a valid difference set, we can reconstruct
    # by sorting and using the structure.
    # 
    # For a set {a_0=0, a_1, ..., a_{n-1}}, the differences can be grouped.
    # The n-1 smallest differences include the consecutive gaps.
    # 
    # Let's try a direct approach for small n:
    # We know a_0 = 0, a_{n-1} = M.
    # The set of differences must equal the given diff_set.
    # 
    # We can use the following: for each element a_i (i=1..n-2), 
    # the differences a_i - a_0 = a_i and a_{n-1} - a_i = M - a_i must both be in diff_set.
    # So a_i must be in diff_set AND M - a_i must be in diff_set.
    
    valid_intermediates = []
    for d in diffs:
        if 0 < d < M and (M - d) in diffs:
            valid_intermediates.append(d)
    
    valid_intermediates.sort()
    
    # We need to choose n-2 = k-1 values from valid_intermediates
    # such that all C(n,2) differences match.
    
    if k == 5:  # n=6, need 4 intermediates
        # Try all combinations? That's C(len(valid_intermediates), 4) which could be large.
        # Instead, use a heuristic: try the smallest valid intermediates first.
        
        # For efficiency, limit the search.
        # The intermediates must be such that all pairwise differences are in diff_set.
        
        # Let's try a greedy approach: pick the smallest valid intermediate,
        # then the next smallest that's compatible, etc.
        
        def check_set(elements, diff_set):
            """Check if the set of elements produces exactly diff_set."""
            computed = set()
            n = len(elements)
            for i in range(n):
                for j in range(i + 1, n):
                    computed.add(elements[j] - elements[i])
            return computed == diff_set
        
        # Greedy reconstruction
        elements = [0]
        remaining_intermediates = list(valid_intermediates)
        
        for _ in range(k - 1):  # need k-1 intermediates for n=k+1
            if not remaining_intermediates:
                return None
            
            best_candidate = None
            best_score = float('inf')
            
            for cand in remaining_intermediates[:20]:  # Limit search
                # Check if cand is compatible with current elements
                new_diffs = set()
                for e in elements:
                    new_diffs.add(cand - e)
                
                # Also check that M - cand is in diff_set (already ensured by valid_intermediates)
                # Check that all new diffs are in diff_set
                if not new_diffs.issubset(diff_set):
                    continue
                
                # Score: prefer candidates that create "dense" differences
                score = cand
                if score < best_score:
                    best_score = score
                    best_candidate = cand
            
            if best_candidate is None:
                # Try a broader search
                for cand in remaining_intermediates:
                    new_diffs = set()
                    for e in elements:
                        new_diffs.add(cand - e)
                    if new_diffs.issubset(diff_set):
                        best_candidate = cand
                        break
            
            if best_candidate is None:
                return None
            
            elements.append(best_candidate)
            remaining_intermediates.remove(best_candidate)
        
        elements.append(M)
        
        # Verify
        if check_set(elements, diff_set):
            return elements
        else:
            return None
    
    return None


def _greedy_construct(n_rows, k, seed):
    """Multi-seed greedy construction with fragmentation-aware heuristic."""
    rng = random.Random(seed)
    used = set()
    rows = []
    
    for _ in range(n_rows):
        row = [0]
        for _ in range(k):
            last_x = row[-1]
            best_x = None
            best_score = float('inf')
            
            # Search for good candidates
            x_start = last_x + 1
            x_end = x_start + 300
            
            for x in range(x_start, x_end + 1):
                new_diffs = {x - a for a in row}
                if new_diffs & used:
                    continue
                
                # Score: prefer smaller x, less fragmentation
                frag = 0
                for d in new_diffs:
                    if (d - 1) not in used and (d + 1) not in used:
                        frag += 1
                
                score = x + 3 * frag
                if score < best_score:
                    best_score = score
                    best_x = x
            
            if best_x is None:
                # Fallback: find any valid x
                x = last_x + 1
                while True:
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used:
                        best_x = x
                        break
                    x += 1
            
            row.append(best_x)
            new_diffs = {best_x - a for a in row[:-1]}
            used |= new_diffs
        
        rows.append(row)
    
    return rows


def _local_search_improve(rows, k, deadline, rng=None):
    """Local search with simulated annealing: try to reduce the maximum entry."""
    if rng is None:
        rng = random.Random()
    
    rows = [list(r) for r in rows]
    best_scope = _scope(rows)
    used = _all_diffs(rows)
    best_rows = [list(r) for r in rows]
    
    # Simulated annealing parameters
    T = 50.0
    T_min = 0.1
    cooling = 0.985
    
    while time.time() < deadline and T > T_min:
        current_scope = _scope(rows)
        
        # Pick a random row and position to modify
        ri = rng.randint(0, len(rows) - 1)
        row = rows[ri]
        pos = rng.randint(1, len(row) - 1)
        
        old_val = row[pos]
        low = row[pos - 1] + 1
        high = row[pos + 1] - 1 if pos + 1 < len(row) else old_val + 15
        
        if low >= high:
            T *= cooling
            continue
        
        # Generate candidates
        candidates = set()
        for delta in range(-20, 21):
            v = old_val + delta
            if low <= v <= high:
                candidates.add(v)
        if old_val >= current_scope:
            for dv in range(1, 25):
                v = current_scope - dv
                if low <= v <= high:
                    candidates.add(v)
        
        if not candidates:
            T *= cooling
            continue
        
        # Evaluate candidates
        best_new_val = None
        best_new_scope = current_scope
        best_new_diffs = None
        best_old_diffs = None
        
        for new_val in candidates:
            if new_val == old_val:
                continue
            
            old_diffs = set()
            for i in range(pos):
                old_diffs.add(old_val - row[i])
            for i in range(pos + 1, len(row)):
                old_diffs.add(row[i] - old_val)
            
            new_diffs = set()
            for i in range(pos):
                new_diffs.add(new_val - row[i])
            for i in range(pos + 1, len(row)):
                new_diffs.add(row[i] - new_val)
            
            conflict = new_diffs & (used - old_diffs)
            if conflict:
                continue
            
            temp_row = row[:pos] + [new_val] + row[pos + 1:]
            new_scope = max(max(r) if r is not row else max(temp_row) for r in rows)
            
            if new_scope < best_new_scope:
                best_new_scope = new_scope
                best_new_val = new_val
                best_new_diffs = new_diffs
                best_old_diffs = old_diffs
        
        if best_new_val is not None:
            delta = best_new_scope - current_scope
            
            if delta <= 0 or rng.random() < (2.718281828 ** (-delta / max(T, 1e-9))):
                row[pos] = best_new_val
                used -= best_old_diffs
                used |= best_new_diffs
                
                if best_new_scope < best_scope:
                    best_scope = best_scope
                    best_rows = [list(r) for r in rows]
        
        T *= cooling
    
    return best_rows


def _polish_max(rows, k, deadline, rng=None):
    """Targeted polish: focus on reducing the maximum element."""
    if rng is None:
        rng = random.Random()
    
    rows = [list(r) for r in rows]
    used = _all_diffs(rows)
    best_scope = _scope(rows)
    best_rows = [list(r) for r in rows]
    
    while time.time() < deadline:
        current_scope = _scope(rows)
        
        # Find the row and position with the maximum value
        max_pos = None
        max_val = -1
        for ri, row in enumerate(rows):
            for pi, val in enumerate(row):
                if val > max_val:
                    max_val = val
                    max_pos = (ri, pi)
        
        if max_pos is None:
            break
        
        ri, pi = max_pos
        row = rows[ri]
        old_val = row[pi]
        
        low = row[pi - 1] + 1 if pi > 0 else 1
        high = row[pi + 1] - 1 if pi + 1 < len(row) else old_val
        
        if low >= high:
            break
        
        # Try to reduce the max
        best_new_val = None
        best_new_scope = current_scope
        
        for new_val in range(high, low - 1, -1):  # Try smaller values first
            if new_val == old_val:
                continue
            
            old_diffs = set()
            for i in range(pi):
                old_diffs.add(old_val - row[i])
            for i in range(pi + 1, len(row)):
                old_diffs.add(row[i] - old_val)
            
            new_diffs = set()
            for i in range(pi):
                new_diffs.add(new_val - row[i])
            for i in range(pi + 1, len(row)):
                new_diffs.add(row[i] - new_val)
            
            conflict = new_diffs & (used - old_diffs)
            if conflict:
                continue
            
            temp_row = row[:pi] + [new_val] + row[pi + 1:]
            new_scope = max(max(r) if r is not row else max(temp_row) for r in rows)
            
            if new_scope < best_new_scope:
                best_new_scope = new_scope
                best_new_val = new_val
        
        if best_new_val is not None:
            old_diffs = set()
            for i in range(pi):
                old_diffs.add(old_val - row[i])
            for i in range(pi + 1, len(row)):
                old_diffs.add(row[i] - old_val)
            
            new_diffs = set()
            for i in range(pi):
                new_diffs.add(best_new_val - row[i])
            for i in range(pi + 1, len(row)):
                new_diffs.add(row[i] - best_new_val)
            
            row[pi] = best_new_val
            used -= old_diffs
            used |= new_diffs
            
            if best_new_scope < best_scope:
                best_scope = best_new_scope
                best_rows = [list(r) for r in rows]
        else:
            # No improvement possible with current approach
            break
    
    return best_rows


def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Phase 1: Multi-seed greedy to find diverse initial solutions
    best_rows = None
    best_scope = float('inf')
    
    max_seeds = 30
    for seed in range(max_seeds):
        if time.time() > deadline - 60:
            break
        
        rows = _greedy_construct(n_rows, k, seed)
        if rows is None:
            continue
        
        scope = _scope(rows)
        if scope < best_scope:
            best_scope = scope
            best_rows = [list(r) for r in rows]
    
    if best_rows is None:
        # Fallback
        best_rows = _greedy_construct(n_rows, k, 0)
        if best_rows is None:
            # Last resort: simple construction
            rows = []
            used = set()
            for _ in range(n_rows):
                row = [0]
                for _ in range(k):
                    x = row[-1] + 1
                    while True:
                        new_diffs = {x - a for a in row}
                        if not new_diffs & used:
                            break
                        x += 1
                    row.append(x)
                    used |= new_diffs
                rows.append(row)
            best_rows = rows
            best_scope = _scope(best_rows)
    
    # Phase 2: Local search with simulated annealing
    if time.time() < deadline - 30:
        rng = random.Random(42)
        improved = _local_search_improve(best_rows, k, deadline - 30, rng)
        new_scope = _scope(improved)
        if new_scope < best_scope:
            best_rows = improved
            best_scope = new_scope
    
    # Phase 3: Targeted polish on the maximum element
    if time.time() < deadline - 5:
        rng = random.Random(123)
        polished = _polish_max(best_rows, k, deadline - 5, rng)
        new_scope = _scope(polished)
        if new_scope < best_scope:
            best_rows = polished
            best_scope = new_scope
    
    # Phase 4: Final local search pass with different parameters
    if time.time() < deadline - 2:
        rng = random.Random(999)
        improved = _local_search_improve(best_rows, k, deadline - 2, rng)
        new_scope = _scope(improved)
        if new_scope < best_scope:
            best_rows = improved
            best_scope = new_scope
    
    return best_rows

# EVOLVE-BLOCK-END
