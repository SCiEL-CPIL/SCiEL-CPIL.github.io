import time
import random
from itertools import combinations

def construct(n_rows, k):
    """
    Construct a difference triangle set with n_rows rows, each of k+1 integers
    starting at 0, such that all positive differences within rows are distinct
    across all rows. Minimize the largest entry (scope).
    
    Strategy:
    1. Enumerate all valid rows (sorted tuples of k+1 integers starting at 0)
       with max element up to a reasonable bound.
    2. Build a compatibility graph: two rows are compatible if their difference
       sets are disjoint.
    3. Use branch-and-bound to find a set of n_rows mutually compatible rows
       minimizing the maximum element.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 150s kill
    
    # Parameters
    num_diffs_per_row = (k + 1) * k // 2
    
    # Lower bound on scope
    lower_bound = n_rows * num_diffs_per_row
    
    # Upper bound for enumeration - start with a reasonable estimate
    # The best known for (7,5) is 112, counting bound is 105
    # For general n_rows, k, we need a good upper bound
    # Use a heuristic: lower_bound + some margin
    max_scope_upper = max(lower_bound + 20, 150)
    
    best_scope = float('inf')
    best_rows = None
    
    # Try to find a solution using branch-and-bound on the compatibility graph
    # We'll enumerate rows in order of their max element and try to build
    # a clique of size n_rows
    
    # First, generate all valid rows up to a certain max element
    # A row is a sorted tuple (0, a1, a2, ..., ak) with strictly increasing elements
    # and all pairwise differences distinct within the row (which is automatic
    # if elements are distinct and we only consider positive diffs)
    # Wait, the requirement is that all differences BETWEEN two entries of the same row
    # are distinct OVER ALL ROWS. Within a single row, differences can repeat? No:
    # "all differences between two entries of the same row are distinct over all rows"
    # This means: for any two rows, the set of positive differences in row 1 and
    # the set of positive differences in row 2 must be disjoint.
    # Within a single row, the differences a_j - a_i for i<j must all be distinct?
    # Actually, for a set of integers, the differences a_j - a_i for i<j are not
    # necessarily all distinct. For example, {0, 1, 2}: diffs are 1, 2, 1. So 1 appears twice.
    # But the problem says "all differences between two entries of the same row are distinct over all rows"
    # I think this means: consider the multiset of all differences from all rows. Each difference value
    # should appear at most once total. So within a single row, if a difference repeats, that's a problem.
    # Let me re-read: "all differences between two entries of the same row are distinct over all rows"
    # I believe this means: the set of all positive differences a_j - a_i (i < j) across all rows
    # should have no duplicates. So each difference value can appear at most once in total.
    # This implies that within a single row, all pairwise differences must be distinct
    # (otherwise the same value appears twice in that row alone).
    # Such sets are called "Sidon sets" or "B_2 sets" - no, wait. A Sidon set has all sums a_i + a_j distinct.
    # A set with all differences distinct is called a "Golomb ruler" or "distinct difference set".
    # Yes! A Golomb ruler has all pairwise differences distinct.
    # So each row must be a Golomb ruler (with 0 as the first element), and the difference sets
    # of different rows must be disjoint.
    
    # Generate all Golomb rulers of length k+1 starting at 0 with max element <= max_scope_upper
    # This could be a lot, so we need to be smart.
    
    # Actually, let's use a different approach. Let's use randomized search with
    # local improvement, similar to the current program but with better heuristics.
    
    # Let me try a hybrid: use the compatibility graph idea but with a smarter search.
    # For each candidate max element M, we want to find n_rows Golomb rulers of length k+1
    # with max element <= M, all starting at 0, with disjoint difference sets.
    
    # Binary search on M? Or just try increasing M?
    
    # Let's try a randomized greedy with local search, but with a better selection strategy.
    
    best_scope = float('inf')
    best_rows = None
    
    # Use randomized restarts with different strategies
    for trial in range(200):
        if time.time() > deadline:
            break
        
        rng = random.Random(trial)
        
        # Try different strategies
        if trial % 3 == 0:
            rows = _greedy_with_lookahead(n_rows, k, rng, c=8)
        elif trial % 3 == 1:
            rows = _greedy_random(n_rows, k, rng, c=12)
        else:
            rows = _greedy_min_conflict(n_rows, k, rng, c=6)
        
        if rows is None:
            continue
        
        scope = max(row[-1] for row in rows)
        
        if scope < best_scope:
            best_scope = scope
            best_rows = rows
            
            # Try local search to improve
            if time.time() < deadline - 5:
                improved = _local_search(rows, n_rows, k, rng, time_limit=deadline - 5)
                if improved is not None:
                    improved_scope = max(row[-1] for row in improved)
                    if improved_scope < best_scope:
                        best_scope = improved_scope
                        best_rows = improved
        
        # Early termination if we hit lower bound
        if best_scope == lower_bound:
            break
    
    if best_rows is None:
        # Fallback: simple greedy
        best_rows = _greedy_random(n_rows, k, random.Random(0), c=10)
    
    return best_rows


def _row_diffs(row):
    """Compute the set of positive differences in a row."""
    diffs = set()
    for i in range(len(row)):
        for j in range(i + 1, len(row)):
            diffs.add(row[j] - row[i])
    return diffs


def _greedy_with_lookahead(n_rows, k, rng, c=8):
    """Greedy with lookahead: pick the element that minimizes future conflicts."""
    used = set()
    rows = []
    
    for _ in range(n_rows):
        row = [0]
        for _ in range(k):
            last_x = row[-1]
            valid_xs = []
            x = last_x + 1
            limit = 300
            
            while len(valid_xs) < c and x <= limit:
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    valid_xs.append(x)
                x += 1
            
            if not valid_xs:
                x = last_x + 1
                while len(valid_xs) < c:
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used):
                        valid_xs.append(x)
                    x += 1
            
            # Lookahead: for each candidate, estimate how many future options it leaves
            # We can approximate by checking how many small differences it uses
            best_x = None
            best_score = float('inf')
            for x in valid_xs:
                new_diffs = {x - a for a in row}
                # Score: prefer candidates that use larger differences (leaves small ones for others)
                score = sum(new_diffs)
                if score < best_score:
                    best_score = score
                    best_x = x
            
            if best_x is None:
                best_x = valid_xs[0]
            
            row.append(best_x)
            new_diffs = {best_x - a for a in row[:-1]}
            used.update(new_diffs)
        
        rows.append(row)
    
    return rows


def _greedy_random(n_rows, k, rng, c=10):
    """Randomized greedy: pick randomly from valid candidates."""
    used = set()
    rows = []
    
    for _ in range(n_rows):
        row = [0]
        for _ in range(k):
            last_x = row[-1]
            valid_xs = []
            x = last_x + 1
            limit = 300
            
            while len(valid_xs) < c and x <= limit:
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    valid_xs.append(x)
                x += 1
            
            if not valid_xs:
                x = last_x + 1
                while len(valid_xs) < c:
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used):
                        valid_xs.append(x)
                    x += 1
            
            chosen_x = rng.choice(valid_xs)
            row.append(chosen_x)
            new_diffs = {chosen_x - a for a in row[:-1]}
            used.update(new_diffs)
        
        rows.append(row)
    
    return rows


def _greedy_min_conflict(n_rows, k, rng, c=6):
    """Greedy that minimizes conflicts: prefer candidates that don't use small differences."""
    used = set()
    rows = []
    
    for _ in range(n_rows):
        row = [0]
        for _ in range(k):
            last_x = row[-1]
            valid_xs = []
            x = last_x + 1
            limit = 300
            
            while len(valid_xs) < c and x <= limit:
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    valid_xs.append(x)
                x += 1
            
            if not valid_xs:
                x = last_x + 1
                while len(valid_xs) < c:
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used):
                        valid_xs.append(x)
                    x += 1
            
            # Prefer candidates that use larger differences
            scored = []
            for x in valid_xs:
                new_diffs = {x - a for a in row}
                # Count how many of these differences are "small" (<= k)
                small_count = sum(1 for d in new_diffs if d <= k)
                scored.append((small_count, x))
            
            scored.sort()
            # Pick from the top few with some randomness
            top_n = min(3, len(scored))
            chosen_x = scored[rng.randint(0, top_n - 1)][1]
            
            row.append(chosen_x)
            new_diffs = {chosen_x - a for a in row[:-1]}
            used.update(new_diffs)
        
        rows.append(row)
    
    return rows


def _local_search(rows, n_rows, k, rng, time_limit):
    """Local search: try to reduce the scope by modifying rows."""
    start = time.time()
    current = [row[:] for row in rows]
    current_scope = max(row[-1] for row in current)
    
    best = [row[:] for row in current]
    best_scope = current_scope
    
    while time.time() < time_limit:
        # Try to improve a random row
        row_idx = rng.randint(0, n_rows - 1)
        old_row = current[row_idx][:]
        old_diffs = _row_diffs(old_row)
        
        # Remove old row's diffs from used set
        used = set()
        for i, row in enumerate(current):
            if i != row_idx:
                used.update(_row_diffs(row))
        
        # Try to rebuild the row with a smaller max element
        new_row = _rebuild_row(used, k, rng, target_max=old_row[-1] - 1)
        if new_row is not None:
            current[row_idx] = new_row
            new_scope = max(row[-1] for row in current)
            if new_scope < best_scope:
                best = [row[:] for row in current]
                best_scope = new_scope
        
        # Sometimes try a full restart of a row
        if rng.random() < 0.3:
            used = set()
            for i, row in enumerate(current):
                if i != row_idx:
                    used.update(_row_diffs(row))
            
            new_row = _rebuild_row(used, k, rng, target_max=current_scope - 1)
            if new_row is not None:
                current[row_idx] = new_row
                new_scope = max(row[-1] for row in current)
                if new_scope < best_scope:
                    best = [row[:] for row in current]
                    best_scope = new_scope
    
    return best


def _rebuild_row(used, k, rng, target_max=None):
    """Try to build a row with max element <= target_max that doesn't conflict with used diffs."""
    if target_max is None:
        target_max = 200
    
    # Try multiple times
    for _ in range(50):
        row = [0]
        success = True
        for _ in range(k):
            last_x = row[-1]
            valid_xs = []
            x = last_x + 1
            while x <= target_max:
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    valid_xs.append(x)
                x += 1
            
            if not valid_xs:
                success = False
                break
            
            # Pick randomly from valid
            chosen_x = rng.choice(valid_xs)
            row.append(chosen_x)
            used_temp = {chosen_x - a for a in row[:-1]}
        
        if success and len(row) == k + 1:
            # Verify all diffs are valid
            all_diffs = _row_diffs(row)
            if not (all_diffs & used):
                return row
    
    return None


def _greedy_random(n_rows, k, rng, c=10):
    """Randomized greedy: pick randomly from valid candidates."""
    used = set()
    rows = []
    
    for _ in range(n_rows):
        row = [0]
        for _ in range(k):
            last_x = row[-1]
            valid_xs = []
            x = last_x + 1
            limit = 300
            
            while len(valid_xs) < c and x <= limit:
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    valid_xs.append(x)
                x += 1
            
            if not valid_xs:
                x = last_x + 1
                while len(valid_xs) < c:
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used):
                        valid_xs.append(x)
                    x += 1
            
            chosen_x = rng.choice(valid_xs)
            row.append(chosen_x)
            new_diffs = {chosen_x - a for a in row[:-1]}
            used.update(new_diffs)
        
        rows.append(row)
    
    return rows
