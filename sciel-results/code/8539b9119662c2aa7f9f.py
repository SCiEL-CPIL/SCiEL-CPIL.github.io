import time
import random

def construct():
    start_time = time.time()
    deadline = 110  # Safety margin
    
    # Target differences
    target_diffs = set(range(1, 31))
    full_mask = (1 << 30) - 1  # Bits 0..29 correspond to diffs 1..30
    
    def get_diffs(S):
        S = sorted(S)
        return set(y - x for i, x in enumerate(S) for y in S[i+1:])
        
    def is_sidon(S):
        if not S:
            return True
        diffs = get_diffs(S)
        return len(diffs) == len(S) * (len(S) - 1) // 2
        
    best_max = float('inf')
    best_set = None
    
    # State for DFS
    # S: current sorted list
    # used_diffs: set of all differences generated so far (to check Sidon property)
    # covered_mask: bitmask of target differences covered
    # current_max: max(S)
    
    # We will use a recursive DFS with a time limit check.
    # To avoid recursion depth issues, we can use a stack or increase recursion limit.
    # Given max size 32, recursion is fine.
    
    import sys
    sys.setrecursionlimit(10000)
    
    def dfs(S, used_diffs, covered_mask, current_max):
        nonlocal best_max, best_set, start_time, deadline
        
        # Time check
        if time.time() - start_time > deadline:
            return False
        
        # Pruning: If we have a solution, check if it's better
        if (covered_mask == full_mask):
            if current_max < best_max:
                best_max = current_max
                best_set = S[:]
            return True # Found a solution, backtrack to find smaller max? 
            # Actually, we want the MINIMUM max. 
            # If we found one, we can stop exploring branches that lead to larger max?
            # Not necessarily, a smaller max might be found by a different path.
            # But since we explore in increasing order of candidates, the first solution 
            # found for a specific prefix isn't necessarily global min.
            # However, if we order candidates by value, the first solution found 
            # with a specific set of previous elements is likely good.
            # To be safe, we continue searching unless we hit time limit, 
            # but we can prune if current_max >= best_max.
        
        if current_max >= best_max:
            return False
            
        if len(S) > 32:
            return False
            
        # Calculate missing diffs
        # We need to cover all 1..30.
        
        # Generate candidates for the next element
        # The next element $x$ must be $> S[-1]$ (to maintain sorted order and avoid duplicates in S)
        # Wait, does S have to be strictly increasing? Yes, distinct integers.
        # If we pick $x$, new diffs are $x - s$ for all $s \in S$.
        # These new diffs must not be in $used_diffs$.
        # Also, if any new diff is in 1..30, it updates covered_mask.
        
        # Heuristic: 
        # We want to cover missing diffs.
        # Let missing = [d for d in 1..30 if not covered]
        # Candidates $x$ are such that $x - s$ is "useful".
        # A good candidate $x$ should be small (to keep max low) but cover many missing diffs.
        
        # Optimization: 
        # Instead of generating all possible $x$, we can generate candidates based on missing diffs.
        # For each missing diff $d$, and each $s \in S$, $x = s + d$ is a candidate that covers $d$.
        # Also $x$ could be such that $x - s$ is a diff that is NOT in 1..30 (neutral) but helps structure?
        # No, we only care about covering 1..30. Neutral diffs just consume Sidon "slots".
        # So we should prioritize $x$ that covers missing diffs.
        
        candidates = set()
        for d in range(1, 31):
            if not (covered_mask & (1 << (d - 1))):
                for s in S:
                    x = s + d
                    candidates.add(x)
        
        # Also, we might need to add an element that doesn't immediately cover a missing diff 
        # to enable future coverage? 
        # In Sidon sets, differences are unique. Adding an element that creates "waste" diffs 
        # might block future necessary diffs. 
        # It is generally optimal to pick candidates that cover missing diffs.
        # If no candidate covers a missing diff, we are stuck?
        # Actually, if we can't cover a missing diff $d$ with any $x > S[-1]$, 
        # maybe we needed a smaller $x$ earlier? 
        # Since we are doing DFS, we backtrack.
        
        # If candidates is empty, we can't cover any missing diff with a larger element.
        # We should return False to backtrack.
        
        if not candidates:
            return False

        # Filter candidates
        # 1. x > S[-1] (strictly increasing)
        # 2. x - s not in used_diffs for all s in S
        # 3. x - s <= 10^6 (limit)
        
        valid_candidates = []
        for x in candidates:
            if x <= S[-1]:
                continue
            if x > 10**6:
                continue
            
            # Check Sidon property for new diffs
            new_diffs = []
            valid = True
            for s in S:
                d = x - s
                if d in used_diffs:
                    valid = False
                    break
                new_diffs.append(d)
            
            if not valid:
                continue
                
            # Check if any new diff is > 10^6? No, x is bounded.
            
            # Calculate new covered mask
            new_mask = covered_mask
            for d in new_diffs:
                if 1 <= d <= 30:
                    new_mask |= (1 << (d - 1))
            
            # Score for sorting: 
            # We want to try candidates that cover more new diffs first.
            # Tie-breaker: smaller x.
            new_covered_count = bin(new_mask ^ covered_mask).count('1')
            
            valid_candidates.append((x, new_mask, new_diffs, new_covered_count))
            
        if not valid_candidates:
            return False

        # Sort candidates
        # Primary: new_covered_count (descending) -> cover more first
        # Secondary: x (ascending) -> keep max small
        valid_candidates.sort(key=lambda item: (-item[3], item[0]))
        
        # Optional: Limit branching factor to speed up?
        # If we limit to top K, we might miss optimal. 
        # Given small N, let's try all, but with time limit.
        
        for x, new_mask, new_diffs, _ in valid_candidates:
            if time.time() - start_time > deadline:
                return False
                
            new_S = S + [x]
            new_used_diffs = used_diffs.union(new_diffs)
            
            # Recurse
            dfs(new_S, new_used_diffs, new_mask, x)
            
            # If we found a solution, we might want to stop early if it's very good?
            # But we want global min. 
            # However, if best_max is small, we can prune more.
            
    # Initialize
    S = [0]
    used_diffs = set()
    covered_mask = 0
    current_max = 0
    
    # Run DFS
    # To handle cases where simple DFS from 0 gets stuck or takes too long,
    # we can add a small randomization or multiple restarts with different initial "hints"?
    # Since 0 is fixed, the first step is choosing $a_1$.
    # The DFS above explores all valid $a_1, a_2, \dots$
    
    # One issue: The DFS might explore a huge tree.
    # We can add a heuristic: if we have a solution, we can use it to prune.
    # Also, we can limit the search depth or width if it's taking too long.
    
    # Let's run the DFS.
    dfs(S, used_diffs, covered_mask, current_max)
    
    if best_set:
        return best_set
    else:
        # Fallback: Return a valid set even if not optimal
        # Greedy without backtracking
        S = [0]
        covered = set()
        while not covered.issuperset(target_diffs) and len(S) <= 32:
            missing = target_diffs - covered
            found = False
            for d in missing:
                for s in S:
                    c = s + d
                    if c in S or c < 0:
                        continue
                    new_S = S + [c]
                    new_S.sort()
                    if is_sidon(new_S):
                        S = new_S
                        covered = get_diffs(S)
                        found = True
                        break
                if found:
                    break
            if not found:
                # Add a large number to unblock?
                c = 2 * max(S) + 3
                while True:
                    new_S = S + [c]
                    new_S.sort()
                    if is_sidon(new_S):
                        S = new_S
                        covered = get_diffs(S)
                        break
                    c += 1
        return S
