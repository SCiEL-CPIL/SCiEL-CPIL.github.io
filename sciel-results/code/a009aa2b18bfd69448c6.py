"""Sidon set: minimal diameter covering differences 1..30."""
import time
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = 145.0
    
    # We need: Sidon set S ⊆ [0, max] containing 0,
    # all positive pairwise differences distinct,
    # every d in 1..30 appears as a difference,
    # minimize max(S).
    
    # With |S| = k, we have C(k,2) positive differences. Need >= 30, so k >= 9.
    
    # Strategy: For each candidate max element M (starting from lower bound),
    # do a backtracking search for a valid Sidon set in [0, M].
    
    # Lower bound: need difference 30, so M >= 30.
    # Also, with 9 elements in [0,M], the sum of all positive differences is 
    # sum_{i<j} (s_j - s_i) = sum_i s_i * (2i - k + 1) (for sorted s).
    # The minimum sum of 30 distinct positive differences is 1+2+...+30 = 465.
    # This gives a lower bound on M.
    
    # Let's just search M from 30 upward.
    
    def try_find(M, time_limit):
        """Find a Sidon set in [0, M] covering 1..30, or None."""
        
        # Use backtracking with the following state:
        # - chosen: list of chosen elements (sorted)
        # - used_diffs: set of differences already used
        # - covered: set of required differences covered
        
        # Key optimization: we MUST cover all of 1..30.
        # At each step, determine which differences are still needed.
        
        # We'll use a recursive search with strong pruning.
        
        # Precompute: for a partial set, what's the maximum number of new
        # required differences we can still cover?
        
        chosen = [0]
        used_diffs = set()
        covered = set()
        
        REQUIRED = set(range(1, 31))
        
        def dfs(chosen, used_diffs, covered, depth):
            if time.time() - start_time > time_limit:
                return None
            
            if REQUIRED.issubset(covered):
                return list(chosen)
            
            rem = REQUIRED - covered
            if not rem:
                return list(chosen)
            
            # Pruning 1: Can we still cover all remaining differences?
            # For each d in rem, we need a pair (a,b) in final set with |a-b|=d.
            # The final set extends chosen. New elements are in [max(chosen)+1, M].
            # (We build in increasing order for determinism.)
            
            max_chosen = max(chosen)
            
            for d in rem:
                # Can d be formed?
                # Option A: one endpoint in chosen, one new
                possible = False
                for x in chosen:
                    if x + d <= M:
                        possible = True
                        break
                    if x - d >= 0 and x - d not in chosen:
                        # x-d could be added if it's not already there and doesn't conflict
                        # But we're building in increasing order, so x-d < x <= max_chosen
                        # This would mean we should have added it earlier. Skip.
                        pass
                # Option B: both endpoints new, > max_chosen
                if M - max_chosen >= d:
                    possible = True
                
                if not possible:
                    return None
            
            # Pruning 2: Upper bound on how many more differences we can add
            # If we add t more elements, we get at most t * len(chosen) + C(t,2) new differences.
            # We need at least |rem| more.
            # Max additional elements: M - max_chosen (roughly)
            # This is a weak bound, skip for now.
            
            # Generate candidates: elements x > max_chosen such that adding x
            # doesn't violate the Sidon property and helps cover rem.
            
            # For efficiency, only consider x that create at least one new required difference
            candidates = []
            for x in range(max_chosen + 1, M + 1):
                # Check Sidon property with all chosen elements
                valid = True
                new_diffs = []
                for y in chosen:
                    d = x - y  # x > y since x > max_chosen >= y
                    if d in used_diffs:
                        valid = False
                        break
                    new_diffs.append(d)
                
                if not valid:
                    continue
                
                # Count how many required differences this covers
                benefit = len(set(new_diffs) & rem)
                if benefit > 0:
                    candidates.append((x, benefit))
            
            # Sort by benefit (descending), then by x (ascending for smaller diameter)
            candidates.sort(key=lambda t: (-t[1], t[0]))
            
            # Limit branching factor for speed
            for x, benefit in candidates[:20]:
                new_chosen = chosen + [x]
                new_used = used_diffs | set(x - y for y in chosen)
                new_covered = covered | (set(x - y for y in chosen) & rem)
                
                result = dfs(new_chosen, new_used, new_covered, depth + 1)
                if result is not None:
                    return result
            
            return None
        
        return dfs(chosen, used_diffs, covered, 0)
    
    # Search with increasing M
    for M in range(30, 600):
        if time.time() - start_time > DEADLINE - 5:
            break
        
        result = try_find(M, DEADLINE)
        if result is not None:
            # Verify
            assert 0 in result
            assert len(result) <= 32
            assert max(result) <= M
            
            diffs = []
            for i in range(len(result)):
                for j in range(i + 1, len(result)):
                    diffs.append(result[j] - result[i])
            
            assert len(set(diffs)) == len(diffs)
            assert all(d in diffs for d in range(1, 31))
            
            return sorted(result)
    
    # Fallback: greedy
    a = [0]
    def get_diffs(s):
        return [s[j] - s[i] for i in range(len(s)) for j in range(i + 1, len(s))]
    
    for d in range(1, 31):
        if d in get_diffs(sorted(a)):
            continue
        found = False
        for x in sorted(set(v + s for v in a for s in (d, -d) if v + s >= 0) - set(a)):
            trial = sorted(a + [x])
            if len(set(get_diffs(trial))) == len(get_diffs(trial)):
                a = trial
                found = True
                break
        if not found:
            y = 2 * max(a) + 3 * d + 1
            while True:
                trial = sorted(a + [y, y + d])
                if len(set(get_diffs(trial))) == len(get_diffs(trial)):
                    a = trial
                    break
                y += 1
    return a

