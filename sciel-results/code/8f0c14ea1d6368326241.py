import time
import random
import sys
import math
from itertools import combinations
from collections import defaultdict

def construct(n_rows, k):
    if n_rows <= 0:
        return []
    if k <= 0:
        return [[0] * (k + 1) for _ in range(n_rows)]

    deadline = time.time() + 145.0
    m = k + 1  # entries per row
    diffs_per_row = k * (k + 1) // 2
    
    # Theoretical lower bound on scope: total distinct differences needed
    lb = n_rows * diffs_per_row
    
    # We will search for a solution with scope M.
    # Known best is 134. Counting bound is 112.
    # Let's try to find a solution with M close to lb, increasing M if necessary.
    
    def get_diffs(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    def is_valid(rows):
        if len(rows) != n_rows:
            return False
        seen = set()
        for row in rows:
            if len(row) != m or row[0] != 0:
                return False
            for i in range(1, len(row)):
                if row[i] <= row[i - 1]:
                    return False
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in seen:
                        return False
                    seen.add(d)
        return True

    def generate_patterns(scope, max_patterns=50000):
        """
        Generate a set of valid rows (patterns) with max entry <= scope.
        A valid row has distinct internal differences.
        We use randomized backtracking to generate diverse patterns.
        """
        patterns = []
        seen_pattern_sigs = set() # To avoid duplicates
        
        rng = random.Random(42) # Fixed seed for reproducibility of pool, or random? 
        # Using random seed per call might be better for diversity in restarts, 
        # but let's use a global RNG passed in or just random.
        
        attempts = 0
        max_attempts = 200000 # Safety cap for generation
        
        while len(patterns) < max_patterns and attempts < max_attempts:
            attempts += 1
            if time.time() > deadline - 20:
                break
            
            # Randomized backtracking to build one row
            row = [0]
            valid = True
            
            # Heuristic: try to keep gaps small to stay within scope
            # But randomize to explore space
            
            for pos in range(1, m):
                # Determine range for next value
                # Must be > row[-1] and <= scope
                # Also, new diffs must not collide with existing internal diffs
                
                lo = row[-1] + 1
                hi = scope
                
                # If hi < lo, fail
                if hi < lo:
                    valid = False
                    break
                
                # Generate candidates
                # To make it faster, we can sample a few candidates
                # If scope is small, iterate all. If large, sample.
                
                if hi - lo < 50:
                    candidates = list(range(lo, hi + 1))
                    random.shuffle(candidates)
                else:
                    # Sample 50 candidates
                    candidates = [random.randint(lo, hi) for _ in range(50)]
                    random.shuffle(candidates)
                    # Also add lo and hi to be safe
                    candidates.append(lo)
                    candidates.append(hi)

                found_val = None
                current_diffs = set()
                for a in row:
                    for b in row:
                        if a != b:
                            current_diffs.add(abs(a - b))
                
                # Wait, calculating current_diffs every step is O(k^2). 
                # Better to maintain a set.
                
                # Let's refactor the inner loop to maintain internal_diffs
                
                # Actually, let's just check against `used_internal` set
                # which we maintain as we build the row.
                
                # Let's restart the row building logic with proper state tracking
                
                # The previous attempt was a bit messy. Let's do a clean recursive/iterative backtracking for ONE row.
                
                # Since we are in a loop for `patterns`, let's just use a helper function.
                
                # Helper to build one row with randomization
                # This is essentially the same as the seed's greedy but randomized and bounded by scope.
                
                # For efficiency in pattern generation, we can just use the greedy method with randomization.
                
                pass # Skip this block, use a dedicated function below
                
            if valid:
                # Check if row is valid (internal diffs distinct)
                # We assumed we checked during construction, but let's verify
                # Also check if it's a new pattern
                sig = tuple(row)
                if sig not in seen_pattern_sigs:
                    # Verify internal distinctness
                    d_set = get_diffs(row)
                    if len(d_set) == diffs_per_row:
                        patterns.append(row)
                        seen_pattern_sigs.add(sig)
                        
        return patterns

    def build_row_randomized(scope, rng, max_restarts=50):
        """
        Try to build a single valid row with max entry <= scope using randomized greedy/backtracking.
        Returns row or None.
        """
        for _ in range(max_restarts):
            row = [0]
            internal_diffs = set()
            success = True
            
            for pos in range(1, m):
                lo = row[-1] + 1
                hi = scope
                if hi < lo:
                    success = False
                    break
                
                # Try random candidates
                # Prioritize smaller values to stay compact? Or random?
                # Random is better for diversity.
                
                # To speed up, we can try a few random candidates, then fallback to sequential if needed?
                # No, if we go sequential we might exceed scope or get stuck.
                
                # Let's try up to 20 random candidates
                found = False
                for _ in range(20):
                    val = rng.randint(lo, hi)
                    # Check if val - a is in internal_diffs for all a in row
                    ok = True
                    new_diffs = []
                    for a in row:
                        d = val - a
                        if d in internal_diffs:
                            ok = False
                            break
                        new_diffs.append(d)
                    
                    if ok:
                        # Check if new_diffs have duplicates among themselves?
                        # Since val > a for all a, d = val - a.
                        # If a1 < a2, then val - a1 > val - a2. So distinct a's give distinct diffs.
                        # So no internal duplicates in new_diffs.
                        # Just need to check against existing internal_diffs.
                        
                        row.append(val)
                        for d in new_diffs:
                            internal_diffs.add(d)
                        found = True
                        break
                
                if not found:
                    success = False
                    break
            
            if success and len(row) == m:
                return row
                
        return None

    def pack_rows(patterns, n_rows_needed, rng):
        """
        Greedy set packing: select n_rows_needed patterns from `patterns` such that their difference sets are disjoint.
        Heuristic: pick the pattern that uses the most "rare" differences (differences that appear in few other patterns).
        This helps leave common differences for others? Or vice versa?
        Actually, if we pick a pattern with rare diffs, we don't block others.
        If we pick a pattern with common diffs, we block many others.
        So we should prefer patterns with rare diffs?
        Wait. If I pick a pattern with rare diffs, I consume rare diffs. The remaining pool still has common diffs.
        If I pick a pattern with common diffs, I consume common diffs. The remaining pool has fewer common diffs, which might be good if I need to pack many rows?
        
        Let's think: We want to maximize the chance of finding a disjoint set.
        This is the Set Packing problem.
        A standard heuristic is to pick the set that has the smallest "overlap" with others? 
        Or the one that is "most constrained"?
        
        Let's try: Pick the pattern that minimizes the number of other patterns it conflicts with.
        i.e., for each candidate pattern P, count how many other patterns Q in the pool have non-empty intersection with P.
        Pick P with the minimum count.
        
        This is O(N^2 * D) which might be slow if N is large.
        Let's limit the pool size to ~5000 for this heuristic.
        """
        if len(patterns) < n_rows_needed:
            return None
        
        # Precompute diff sets for all patterns
        pattern_diffs = [get_diffs(p) for p in patterns]
        
        # Count frequency of each difference across all patterns?
        # diff_count[d] = number of patterns containing d
        
        # Actually, let's just do a simple greedy:
        # 1. Calculate "conflict score" for each pattern: number of other patterns it conflicts with.
        # 2. Pick the one with lowest score.
        # 3. Remove it and all patterns it conflicts with from the pool.
        # 4. Repeat.
        
        # This is O(N^2) per pick, so O(N^3) total. With N=5000, that's 125 billion. Too slow.
        
        # Alternative: 
        # Randomized greedy. Pick a random pattern, check if it conflicts with already selected.
        # If yes, discard. If no, select.
        # To improve, sort patterns by "rarity" (sum of inverse frequencies of their diffs).
        
        # Let's try: 
        # 1. Count frequency of each diff in the pool.
        # 2. Score each pattern by sum of 1/freq[d] for d in pattern. Higher score = rarer diffs.
        # 3. Sort patterns by score descending.
        # 4. Iterate through sorted list. If pattern is disjoint from selected, add it.
        
        selected = []
        selected_diffs = set()
        
        # Frequency count
        freq = defaultdict(int)
        for d_set in pattern_diffs:
            for d in d_set:
                freq[d] += 1
        
        # Score patterns
        scored_patterns = []
        for i, p in enumerate(patterns):
            score = 0
            for d in pattern_diffs[i]:
                score += 1.0 / freq[d]
            scored_patterns.append((score, i, p))
        
        # Sort by score descending (rarest first)
        scored_patterns.sort(key=lambda x: x[0], reverse=True)
        
        # Greedy select
        for score, idx, p in scored_patterns:
            if len(selected) == n_rows_needed:
                break
            
            # Check conflict
            conflict = False
            for d in pattern_diffs[idx]:
                if d in selected_diffs:
                    conflict = True
                    break
            
            if not conflict:
                selected.append(p)
                for d in pattern_diffs[idx]:
                    selected_diffs.add(d)
        
        if len(selected) == n_rows_needed:
            return selected
        return None

    def local_search(rows, patterns, rng, deadline):
        """
        Improve the selected rows by swapping with other patterns from the pool.
        """
        if not rows:
            return rows
            
        current_scope = max(max(r) for r in rows)
        best_rows = [list(r) for r in rows]
        best_scope = current_scope
        
        # Map row to its index in selected? 
        # We just need to know which patterns are NOT selected.
        
        selected_diffs = set()
        for r in rows:
            selected_diffs.update(get_diffs(r))
            
        # Create a pool of candidate replacements
        # We can use the original `patterns` list.
        
        iterations = 0
        while time.time() < deadline - 5:
            iterations += 1
            if iterations % 1000 == 0:
                # Check if we are stuck? 
                pass
                
            # Pick a random row to replace
            row_idx = rng.randint(0, len(rows) - 1)
            old_row = rows[row_idx]
            old_diffs = get_diffs(old_row)
            
            # Remove old row's diffs from selected_diffs
            for d in old_diffs:
                selected_diffs.discard(d)
            
            # Try to find a replacement from patterns
            # Prefer patterns with smaller max entry?
            # Or just random?
            # Let's try a few random patterns.
            
            found = False
            # Shuffle patterns to try different ones
            # To be efficient, we can pre-filter patterns by scope?
            # For now, just sample.
            
            for _ in range(50):
                if not patterns:
                    break
                cand_idx = rng.randint(0, len(patterns) - 1)
                cand = patterns[cand_idx]
                cand_diffs = get_diffs(cand)
                
                # Check conflict with other selected rows
                conflict = False
                for d in cand_diffs:
                    if d in selected_diffs:
                        conflict = True
                        break
                
                if not conflict:
                    # Accept replacement
                    rows[row_idx] = cand
                    for d in cand_diffs:
                        selected_diffs.add(d)
                    
                    new_scope = max(max(r) for r in rows)
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best_rows = [list(r) for r in rows]
                        
                    found = True
                    break
            
            # If we didn't find a replacement, we put the old row back
            if not found:
                for d in old_diffs:
                    selected_diffs.add(d)
            # If we did find one, the old diffs are already removed, and new ones added.
            # Wait, if we found one, we didn't add old diffs back. Correct.
            # If we didn't find one, we added old diffs back. Correct.
            
            # Additional move: try to compact a row?
            # If we have time, try to shift elements down.
            if rng.random() < 0.1:
                # Pick a row and try to reduce its max
                row_idx = rng.randint(0, len(rows) - 1)
                row = rows[row_idx]
                # Try to reduce last element
                # This is complex. Let's stick to swapping for now.
                pass

        return best_rows

    # Main Logic
    
    # 1. Determine a starting scope.
    # Start from lb, increase if no solution found.
    # We will generate patterns for a specific scope.
    
    # To be safe and fast, let's try a few scopes.
    # Scope 112 (lb), 115, 120, 125, 130, 134.
    
    # But generating patterns for small scopes might yield few patterns.
    # Let's try to generate a large pool for a slightly larger scope (e.g., 140) 
    # and then pack them. This ensures we have enough candidates.
    # Then we can filter/swap to reduce scope.
    
    # Actually, if we generate patterns with max entry <= 140, we might pick 4 that have max 140.
    # We want to minimize the max.
    
    # Strategy:
    # 1. Generate patterns up to scope M_max = 140.
    # 2. Pack them.
    # 3. Local search to minimize scope.
    
    # If we fail to pack with M_max=140, increase M_max.
    
    best_solution = None
    
    # Try increasing scopes
    # Start with a scope that is likely to have enough patterns.
    # lb = 112.
    # Let's try scopes: 112, 114, 116, 118, 120, 125, 130, 134, 140, 150
    
    scopes_to_try = [112, 114, 116, 118, 120, 122, 125, 128, 130, 132, 134, 136, 140, 145, 150]
    
    # If we find a solution at scope S, we can try to improve it.
    # But generating patterns for smaller scopes is harder.
    
    # Let's iterate through scopes.
    for scope in scopes_to_try:
        if time.time() > deadline - 30:
            break
            
        # Generate patterns for this scope
        # We need enough patterns to pack 4 rows.
        # Let's aim for 2000 patterns.
        
        patterns = []
        rng = random.Random(int(time.time() * 1000) % 10000) # Random seed for diversity
        
        # Generate patterns
        # We'll use build_row_randomized
        # To ensure diversity, we run it many times.
        
        max_attempts = 50000
        attempts = 0
        seen = set()
        
        while len(patterns) < 2000 and attempts < max_attempts:
            attempts += 1
            if time.time() > deadline - 10:
                break
                
            row = build_row_randomized(scope, rng, max_restarts=10)
            if row:
                sig = tuple(row)
                if sig not in seen:
                    seen.add(sig)
                    patterns.append(row)
        
        if len(patterns) < n_rows:
            continue # Not enough patterns, try larger scope
        
        # Pack
        packed = pack_rows(patterns, n_rows, rng)
        
        if packed:
            # Validate
            if is_valid(packed):
                current_scope = max(max(r) for r in packed)
                
                # Local search to improve
                improved = local_search(packed, patterns, rng, deadline - 10)
                if is_valid(improved):
                    improved_scope = max(max(r) for r in improved)
                    if improved_scope < current_scope:
                        packed = improved
                        current_scope = improved_scope
                
                if best_solution is None or current_scope < max(max(r) for r in best_solution):
                    best_solution = packed
                    
                # If we reached the lower bound, we can stop
                if current_scope == lb:
                    break
                # If we found a good solution, we might still try smaller scopes?
                # No, we are iterating from small to large. 
                # If we found a solution at scope=114, we don't need to try 116.
                # But we might want to try to find a solution at 113? 
                # We started at 112. If we failed at 112, we try 114.
                # If we succeed at 114, we stop? 
                # Yes, because we want the smallest scope.
                # But wait, `pack_rows` might not find the *best* packing for that scope.
                # It finds *a* packing.
                # So we might find a packing with max entry 114, but there might be one with 113?
                # No, the patterns are generated with max entry <= scope.
                # So any pattern has max entry <= scope.
                # If scope=114, all patterns have max <= 114.
                # So the packed solution has max <= 114.
                # If we wanted max <= 113, we should have generated patterns with scope=113.
                # So if we failed to pack at 112, and succeeded at 114, the best we can do with these patterns is 114.
                # But maybe a different set of patterns (generated with scope 113) would work?
                # We skipped 113 in the list. Let's add more granular steps.
                
                # Let's restart the loop with finer steps if we find a solution?
                # Or just break and trust the local search?
                # Local search swaps patterns. It doesn't generate new patterns.
                # So it can't reduce the scope below the minimum max-entry of the patterns used.
                # But it can reduce the scope if some patterns have max < scope.
                # E.g., if we generated patterns with scope 120, we might pick 4 that all have max 115.
                # So current_scope would be 115.
                
                # So we should continue to try smaller scopes?
                # No, we are iterating 112, 114...
                # If we find a solution at 114 (meaning we generated patterns with max<=114 and packed them),
                # the resulting scope is <= 114.
                # If we had tried 113, we might have found a solution with scope 113.
                # So we should try 113 before 114?
                # My list was [112, 114...]. I skipped 113.
                # Let's use a dynamic loop: try lb, lb+1, lb+2...
                
                break # Break the scope loop, we found a solution. 
                      # But we might want to try to improve it further?
                      # The local search already ran.
                      # Let's just return it.
        
    if best_solution is None:
        # Fallback: use the seed's greedy method if we failed completely
        # This shouldn't happen if scopes go up to 150.
        # Let's implement a simple greedy fallback.
        used = set()
        best_solution = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < m:
                x += 1
                ok = True
                for a in row:
                    if (x - a) in used:
                        ok = False
                        break
                if ok:
                    for a in row:
                        used.add(x - a)
                    row.append(x)
            best_solution.append(row)
            
    return best_solution

if __name__ == "__main__":
    sol = construct(4, 7)
    used = set()
    valid = True
    for row in sol:
        if len(row) != 8 or row[0] != 0:
            valid = False
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in used:
                    valid = False
                used.add(d)
    print("n_rows=4 k=7 valid:", valid, "scope:", max(max(r) for r in sol))
