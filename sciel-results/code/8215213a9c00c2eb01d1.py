import time
import random
import math
from collections import Counter

_BEST = None
_DEADLINE = 0.0

def _scope(rows):
    return max(a for r in rows for a in r[1:])

def _valid(rows, k):
    seen = set()
    for r in rows:
        if r[0] != 0 or len(r) != k + 1:
            return False
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                d = r[j] - r[i]
                if d <= 0 or d in seen:
                    return False
                seen.add(d)
    return True

def _construct_row(used_diffs, k, max_scope, rng, strategy='tight'):
    """
    Construct a single row of k+1 elements starting at 0.
    used_diffs: set of differences already used by other rows.
    max_scope: upper bound for the largest element in this row.
    strategy: 'tight' (always smallest valid), 'random' (random valid within range), 'spread' (biased towards larger)
    """
    row = [0]
    local_diffs = set()
    
    for pos in range(1, k + 1):
        found = False
        # Determine search range
        # We want to keep the row as tight as possible, but allow some slack
        # The minimum possible next value is row[-1] + 1
        min_val = row[-1] + 1
        
        # Heuristic: if we are close to the end of the row, we might need to be tighter
        # to ensure we don't exceed max_scope
        remaining = k - pos + 1
        # Rough estimate of minimum space needed: remaining * 1
        # But differences must be unique, so we need more space.
        
        # Search for a valid next value
        # To speed up, we can limit the search window
        # If strategy is tight, we search from min_val upwards
        # If random, we might pick a random candidate from a window
        
        search_limit = min(max_scope, min_val + 20) # Heuristic window
        
        if strategy == 'tight':
            candidates = range(min_val, search_limit + 1)
        elif strategy == 'random':
            # Pick a random subset or window
            window_size = 10 + remaining * 2
            end = min(search_limit, min_val + window_size)
            candidates = list(range(min_val, end + 1))
            rng.shuffle(candidates)
        elif strategy == 'spread':
            # Bias towards larger values to spread out differences
            window_size = 10 + remaining * 2
            end = min(search_limit, min_val + window_size)
            candidates = list(range(min_val, end + 1))
            # Weight towards the end
            weights = [1] * len(candidates)
            # Simple bias: higher index = higher weight
            for i in range(len(candidates)):
                weights[i] = i + 1
            # This is O(N^2) if done naively, but N is small
            total_w = sum(weights)
            r = rng.random() * total_w
            cum = 0
            chosen_idx = 0
            for i, w in enumerate(weights):
                cum += w
                if r <= cum:
                    chosen_idx = i
                    break
            # To keep it simple and fast, let's just pick a random one from the back half
            # or just use a random window.
            # Let's stick to a simple random window for 'spread' to avoid overhead
            candidates = list(range(min_val, end + 1))
            rng.shuffle(candidates)
            # Actually, let's just try a few random ones
            candidates = sorted(candidates, key=lambda x: rng.random())
        else:
            candidates = range(min_val, search_limit + 1)
            
        for val in candidates:
            # Check if val is valid
            # 1. Must be > row[-1] (guaranteed by min_val)
            # 2. All new differences (val - a for a in row) must be in local_diffs? No, unique within row.
            # 3. All new differences must NOT be in used_diffs.
            
            valid = True
            new_diffs = []
            for a in row:
                d = val - a
                if d in used_diffs or d in local_diffs:
                    valid = False
                    break
                new_diffs.append(d)
            
            if valid:
                row.append(val)
                for d in new_diffs:
                    local_diffs.add(d)
                    used_diffs.add(d)
                found = True
                break
        
        if not found:
            # Backtrack? For a constructive approach without full backtracking, we just fail.
            # We need to restore state if we fail.
            # Since we modify used_diffs and local_diffs, we need to undo.
            # It's easier to build the row in a temporary structure and only commit if successful.
            # Let's refactor to build in temp.
            pass
            
    # The above implementation modifies used_diffs in-place, which is bad for backtracking/restart.
    # Let's rewrite _construct_row to be pure.
    
    return row

def _construct_row_pure(used_diffs, k, max_scope, rng, strategy='tight'):
    """
    Purely construct a row. Does not modify used_diffs.
    Returns the row or None if it couldn't be constructed within max_scope.
    """
    row = [0]
    local_diffs = set()
    
    for pos in range(1, k + 1):
        min_val = row[-1] + 1
        found = False
        
        # Heuristic search limit
        # We need to leave enough room for remaining elements.
        # Worst case, each remaining element adds at least 1 to the max.
        # So max_val - min_val should be at least (k - pos)
        # Also, differences must be unique.
        
        # Let's try a window of candidates
        # If tight, try smallest first.
        # If random, try random order.
        
        # To ensure we don't search too much, limit the range.
        # A good heuristic: the gap between consecutive elements in a tight difference set is small.
        # But we have global constraints.
        
        search_end = min(max_scope, min_val + 50) # Fixed window for speed
        
        if strategy == 'tight':
            candidates = range(min_val, search_end + 1)
        else:
            candidates = list(range(min_val, search_end + 1))
            rng.shuffle(candidates)
            
        for val in candidates:
            valid = True
            new_diffs = []
            for a in row:
                d = val - a
                if d in used_diffs or d in local_diffs:
                    valid = False
                    break
                new_diffs.append(d)
            
            if valid:
                row.append(val)
                for d in new_diffs:
                    local_diffs.add(d)
                found = True
                break
        
        if not found:
            return None
            
    return row

def _construct_set(n_rows, k, max_scope, rng, strategies):
    """
    Try to construct a full set of n_rows rows with scope <= max_scope.
    strategies: list of strategies to try for each row, or a single strategy.
    """
    used_diffs = set()
    rows = []
    
    for i in range(n_rows):
        strat = strategies[i % len(strategies)] if isinstance(strategies, list) else strategies
        row = _construct_row_pure(used_diffs, k, max_scope, rng, strat)
        if row is None:
            return None
        rows.append(row)
        # Update used_diffs
        for a in row:
            for b in row:
                if a != b:
                    used_diffs.add(abs(a - b))
                    
    return rows

def construct(n_rows, k):
    global _BEST, _DEADLINE
    _BEST = None
    _DEADLINE = time.time() + 108.0 # Safety margin
    
    rng = random.Random(42)
    
    # Lower bound for scope:
    # Total differences needed: n_rows * k * (k+1) / 2
    # Max difference is scope.
    # Counting bound: scope * (scope+1)/2 >= total_diffs
    total_diffs = n_rows * k * (k + 1) // 2
    # scope^2/2 >= total_diffs -> scope >= sqrt(2 * total_diffs)
    lb = int(math.ceil(math.sqrt(2 * total_diffs)))
    
    # Best known for (7,5) is 112.
    # Let's start from a reasonable estimate and go down.
    # Or start from a greedy upper bound and go down.
    
    # 1. Get an initial upper bound via a simple greedy
    # Use the pure constructor with 'tight' strategy and a large scope
    initial_scope = lb * 2
    rows = _construct_set(n_rows, k, initial_scope, rng, 'tight')
    
    if rows is None:
        # If tight fails, try random
        rows = _construct_set(n_rows, k, initial_scope, rng, 'random')
        
    if rows is None:
        # Fallback to a very loose greedy
        # Just pick smallest available
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                ok = True
                for a in row:
                    if x - a in used:
                        ok = False
                        break
                if ok:
                    row.append(x)
                    for a in row[:-1]:
                        used.add(x - a)
            rows.append(row)
    
    current_scope = _scope(rows)
    _BEST = (current_scope, [list(r) for r in rows])
    
    # 2. Binary search / Iterative deepening to lower the scope
    # We know the scope must be at least lb.
    # We know the current best is current_scope.
    # Let's try to find the minimum scope by trying values from current_scope down to lb.
    
    # To be efficient, we can use a randomized restart approach for each target scope.
    # For each target scope S, try N times to construct a valid set.
    
    # We have ~100 seconds.
    # Let's estimate how many attempts we can afford.
    # _construct_row_pure is O(k * window * k) roughly.
    # k=5, window=50. 5*50*5 = 1250 ops per row. 7 rows = ~10k ops.
    # 100s / 10k ops = 10k attempts? Maybe less due to Python overhead.
    # Let's say 1000 attempts per second.
    
    # Strategy:
    # 1. Determine a range of scopes to test.
    # 2. For each scope, try a few random constructions.
    # 3. Keep the best valid one.
    
    # Since we already have a valid one at current_scope, we can try to improve it.
    # We'll iterate downwards from current_scope - 1 to lb.
    # But that might be too many steps if current_scope is much larger than lb.
    # Let's jump.
    
    # Actually, a better strategy:
    # Use the "tight" constructor but with a tighter max_scope.
    # The "tight" constructor is deterministic given the used_diffs set.
    # So we need to randomize the order of rows or the random choices within the row.
    
    # Let's modify _construct_set to allow randomizing the order of rows?
    # No, the order doesn't matter for validity, but it matters for the "tight" construction
    # because earlier rows consume "small" differences, which might force later rows to use larger ones.
    # So shuffling the order of row construction is a good randomization.
    
    def try_construct_with_scope(target_scope, n_attempts, rng):
        for _ in range(n_attempts):
            # Shuffle the order in which we build rows?
            # The function _construct_set builds them in order 0..n-1.
            # We can pass a permutation of row indices?
            # Or just rely on the 'random' strategy inside _construct_row_pure.
            
            # Let's use a mix of strategies
            strategies = ['tight', 'random', 'tight', 'random']
            
            # Randomize the seed for each attempt? No, rng is shared.
            
            rows = _construct_set(n_rows, k, target_scope, rng, strategies)
            if rows is not None:
                return rows
        return None
    
    # Determine search range
    # Start from current_scope - 1 down to lb
    # But we can skip some values if we are confident.
    
    # Let's do a linear search downwards, but limit the total attempts.
    # Or, binary search?
    # If scope S is feasible, is S+1 feasible? Yes.
    # So we can binary search for the minimum feasible scope.
    
    low = lb
    high = current_scope - 1
    
    best_scope = current_scope
    best_rows = _BEST[1]
    
    # Time budget allocation
    # We have ~100s.
    # Let's do a binary search.
    # Each step of binary search requires trying to construct a set for a specific scope.
    # If we fail, we increase low. If we succeed, we decrease high and update best.
    
    # How many attempts per scope in binary search?
    # If we use 'random' strategy, we need multiple attempts to be confident.
    # Let's say 20 attempts per scope.
    # Binary search takes log2(range) steps. Range is ~100. So ~7 steps.
    # 7 * 20 = 140 attempts.
    # Plus a final polish phase with more attempts on the best scope.
    
    # Let's refine:
    # Phase 1: Binary search to find a good upper bound.
    # Phase 2: Intensive search on the best scope found to try to find a valid construction
    #           (since binary search might have failed due to randomness).
    # Phase 3: Try to improve (lower) the scope further with more attempts.
    
    def binary_search_scope(low, high, rng, time_limit):
        best_s = high + 1
        best_r = None
        start_time = time.time()
        
        while low <= high:
            if time.time() > time_limit:
                break
            mid = (low + high) // 2
            # Try to construct with scope mid
            # Use a moderate number of attempts
            attempts = 10
            rows = try_construct_with_scope(mid, attempts, rng)
            if rows is not None:
                best_s = mid
                best_r = rows
                high = mid - 1
            else:
                low = mid + 1
                
        return best_s, best_r
    
    # Run binary search with a time limit
    # Reserve 20s for final polish
    bs_deadline = time.time() + 60
    s, r = binary_search_scope(lb, current_scope - 1, rng, bs_deadline)
    
    if r is not None:
        if s < best_scope:
            best_scope = s
            best_rows = r
            _BEST = (best_scope, [list(x) for x in best_rows])
    
    # Now, intensive search on the best scope found
    # Try many more attempts to find a valid construction for this scope
    # Or, if we didn't find one, try the next higher scope.
    
    target_scope = best_scope
    # If we didn't find a valid one for target_scope, we should have found one for target_scope+1?
    # In binary search, if we fail at mid, we go up. So best_s is always feasible (or current_scope).
    # Wait, if try_construct fails, we assume it's infeasible. But it might be feasible with more attempts.
    # So we should be careful.
    
    # Let's do a final polish:
    # Try to construct a set with scope = best_scope using many attempts.
    # If we fail, try best_scope + 1, etc.
    
    polish_deadline = time.time() + 100
    current_target = best_scope
    
    while current_target < current_scope and time.time() < polish_deadline:
        # Try many attempts for this target
        attempts = 100
        rows = try_construct_with_scope(current_target, attempts, rng)
        if rows is not None:
            if current_target < best_scope:
                best_scope = current_target
                best_rows = rows
                _BEST = (best_scope, [list(x) for x in best_rows])
            # If we found it, can we go lower?
            # Yes, try current_target - 1
            current_target -= 1
        else:
            # If we failed, maybe it's infeasible, or we just didn't find it.
            # Let's try a few more times?
            # Or assume infeasible and break?
            # Given the time, let's break if we fail a few times.
            break
            
    # One last check: if we haven't improved, return the initial one.
    if _BEST is not None:
        return _BEST[1]
    return best_rows
