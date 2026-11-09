"""CSP-based construction of a Sidon set containing 0 with all differences 1..30."""
import sys
sys.setrecursionlimit(10000)

def construct():
    # We need a Sidon set S containing 0 such that every d in 1..30 is a difference.
    # Sidon condition: all positive differences y-x (x<y in S) are distinct.
    # We want to minimize max(S).
    
    # Key insight: If S is a Sidon set with |S| = n, then the number of positive differences is n*(n-1)/2.
    # We need at least 30 differences (for d=1..30), so n*(n-1)/2 >= 30 => n >= 9.
    
    # Strategy: CSP with Backtracking and Forward Checking.
    # State: 
    #   elements: list of current elements (sorted, includes 0)
    #   all_diffs: set of all differences generated so far
    #   mask: bitmask of which differences in 1..30 are covered
    #   current_max: max element in elements
    
    # Goal: mask == (1 << 30) - 1 (bits 0..29 set, representing diffs 1..30)
    # Minimize: current_max at the solution.
    
    FULL_MASK = (1 << 30) - 1  # Bits 0 to 29 correspond to diffs 1 to 30
    
    def diff_to_bit(d):
        return 1 << (d - 1)
        
    best_solution = None
    best_max_val = float('inf')
    
    def dfs(elements, all_diffs, mask, current_max, depth, target_n):
        nonlocal best_solution, best_max_val
        
        # Pruning: if current max is already worse than best, stop
        if current_max >= best_max_val:
            return
            
        # Check if we have reached target size
        if len(elements) == target_n:
            if mask == FULL_MASK:
                if current_max < best_max_val:
                    best_max_val = current_max
                    best_solution = list(elements)
            return
            
        # Heuristic Pruning: 
        # Remaining elements to add: k = target_n - len(elements)
        # Remaining missing diffs: m = 30 - popcount(mask)
        # Each new element adds len(elements) new diffs.
        # Total new diffs possible = sum_{i=0}^{k-1} (len(elements) + i)
        # If this is < m, it's impossible.
        k = target_n - len(elements)
        current_len = len(elements)
        max_new_diffs = sum(current_len + i for i in range(k))
        if max_new_diffs < 30 - bin(mask).count('1'):
            return
            
        # Generate Candidates
        # We want to cover missing diffs.
        missing_diffs = [d for d in range(1, 31) if not (mask & diff_to_bit(d))]
        
        candidates = set()
        
        # 1. Candidates derived from covering missing diffs directly
        # If we add x, and y in elements, and |x-y| = d (missing), then we cover d.
        # So x = y + d or x = y - d.
        for y in elements:
            for d in missing_diffs:
                x1 = y + d
                x2 = y - d
                if x1 >= 0:
                    candidates.add(x1)
                if x2 >= 0:
                    candidates.add(x2)
                    
        # 2. Also consider small integers near 0 or existing elements that might cover multiple diffs?
        # The above usually generates enough. Let's also add a few "exploratory" candidates
        # like current_max + 1, current_max + 2, etc., in case we missed a combination.
        # But the "missing diffs" logic is strong.
        
        # Filter and Score Candidates
        valid_candidates = []
        for x in candidates:
            if x in elements:
                continue
            if x < 0:
                continue
                
            # Check Sidon condition: new diffs must not be in all_diffs
            valid = True
            new_diffs = []
            for y in elements:
                d = abs(x - y)
                if d in all_diffs:
                    valid = False
                    break
                new_diffs.append(d)
                
            if not valid:
                continue
                
            # Calculate score: how many missing diffs does this x cover?
            covered_missing = 0
            for d in new_diffs:
                if 1 <= d <= 30 and not (mask & diff_to_bit(d)):
                    covered_missing += 1
                    
            # Also, we prefer smaller x to keep max low? 
            # Score: (covered_missing, -x) -> Maximize covered, Minimize x
            # Actually, we want to try high coverage first.
            valid_candidates.append((covered_missing, x))
            
        # Sort by coverage (desc), then by value (asc)
        valid_candidates.sort(key=lambda t: (-t[0], t[1]))
        
        # Limit candidates to avoid explosion?
        # If there are too many, we might want to prune.
        # But with forward checking, the branching factor is usually manageable.
        # Let's try top 50 or so.
        for _, x in valid_candidates[:50]:
            new_elements = elements + [x]
            new_elements.sort()
            
            new_diffs_set = set(all_diffs)
            new_mask = mask
            for y in elements:
                d = abs(x - y)
                new_diffs_set.add(d)
                if 1 <= d <= 30:
                    new_mask |= diff_to_bit(d)
                    
            new_max = max(new_elements)
            
            dfs(new_elements, new_diffs_set, new_mask, new_max, depth + 1, target_n)
            
            # If we found a solution, we can return early if we want the first one?
            # No, we want the global best. But if we are doing iterative deepening on n,
            # and for a fixed n we find a solution, we should continue to see if a smaller max exists?
            # Actually, for a fixed n, different orderings might lead to different maxes.
            # But since we sort candidates by value, we are exploring smaller x first.
            # However, a smaller x now might lead to a larger max later.
            # So we must explore all valid branches for this n to find the min max for that n.
            
    # Try increasing n
    for n in range(9, 13):
        # Initial state: {0}
        dfs([0], set(), 0, 0, 0, n)
        if best_solution is not None:
            # We found a solution for this n.
            # Should we try larger n? 
            # Larger n allows more flexibility, potentially smaller max?
            # Usually, more elements -> larger max due to Sidon constraint (more diffs).
            # But with more elements, we can "pack" differences better?
            # For Sidon sets, the minimal max element for size n grows roughly as n^2.
            # So n=9 should have a smaller max than n=10.
            # Let's check if n=9 yields a solution. If yes, it's likely optimal.
            # But to be safe, we can check n=9, 10, 11 and take the best.
            pass
            
    if best_solution is None:
        # Fallback: should not happen
        return [0]
        
    return best_solution
