import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # We want to minimize the largest element.
    # Theoretical lower bound: 
    # We need to cover 1..30.
    # A Sidon set of size k has k(k-1)/2 differences.
    # We need at least 30 differences.
    # k=9 gives 36 diffs.
    # The minimal diameter for a Sidon set covering 1..N is related to Golomb rulers.
    # Let's search M from 30 upwards.
    
    target_diffs = set(range(1, 31))
    target_mask = 0
    for d in target_diffs:
        target_mask |= (1 << d)
        
    # We need to track all used differences to ensure Sidon property.
    # Differences can go up to M.
    # If M is small (e.g., 40), we can use a bitmask of size M+1.
    
    def solve_for_M(M):
        """
        Find a Sidon set in [0, M] covering 1..30.
        Returns the set if found, None otherwise.
        """
        # used_diffs_mask tracks all differences used.
        # We only care about differences up to M.
        # If M is large, bitmask might be big, but M is expected to be small (< 50).
        
        def dfs(current_set, used_diffs_mask, covered_mask, depth):
            # Check if all targets are covered
            if (covered_mask & target_mask) == target_mask:
                return current_set[:]
            
            # If we run out of elements allowed
            if len(current_set) >= 32:
                return None
                
            # Time check
            if time.time() > deadline:
                return None
                
            last = current_set[-1] if current_set else -1
            
            # Try adding x from last+1 to M
            # Optimization: We can restrict x.
            # The smallest uncovered difference is min_uncovered.
            # x must be at least min_uncovered (if 0 is in set, which it is).
            # Actually, if 0 is in set, x itself is a difference.
            
            # Heuristic: Try x that cover many new targets.
            # But first, filter valid x.
            
            valid_xs = []
            # We can bound x.
            # If we have k elements left to place, the max difference we can generate is M.
            # This doesn't restrict x much directly.
            
            # Iterate x
            for x in range(last + 1, M + 1):
                # Check Sidon constraint
                conflict = False
                new_covered = 0
                for a in current_set:
                    d = x - a
                    # d is always > 0 since x > a (current_set is sorted)
                    if used_diffs_mask & (1 << d):
                        conflict = True
                        break
                    if d <= 30: # Only track coverage for 1..30
                        if covered_mask & (1 << d) == 0:
                            new_covered |= (1 << d)
                
                if not conflict:
                    valid_xs.append((x, new_covered))
            
            if not valid_xs:
                return None
                
            # Sort candidates: prioritize those covering more new targets, then smaller x
            # Smaller x is better for minimizing diameter? 
            # Actually, we are fixing M. We just want *a* solution.
            # But finding a solution with smaller intermediate steps might help prune?
            # Let's sort by -new_count, then x.
            valid_xs.sort(key=lambda t: (-bin(t[1]).count('1'), t[0]))
            
            for x, new_cov in valid_xs:
                # Update state
                new_used_mask = used_diffs_mask
                for a in current_set:
                    d = x - a
                    new_used_mask |= (1 << d)
                
                new_cov_mask = covered_mask | new_cov
                # Note: new_cov only has bits for 1..30. 
                # We OR it.
                
                res = dfs(current_set + [x], new_used_mask, new_cov_mask, depth + 1)
                if res is not None:
                    return res
                    
            return None
            
        # Initial state: [0]
        # used_diffs: 0 (no diffs yet)
        # covered: 0
        return dfs([0], 0, 0, 0)

    # Search for minimal M
    # Start from a reasonable lower bound.
    # For N=30, M=30 is the absolute minimum (since diff 30 must exist).
    for M in range(30, 60):
        if time.time() > deadline - 5:
            break
            
        # Allocate time per M
        # If M is small, search space is small.
        # If M is larger, search space grows.
        
        result = solve_for_M(M)
        if result is not None:
            return result
            
    # Fallback if no solution found in range (should not happen)
    # Return a valid set even if not optimal, to avoid score 0
    # Greedy construction
    result = [0]
    used = set()
    for x in range(1, 1000):
        valid = True
        for a in result:
            d = x - a
            if d in used:
                valid = False
                break
        if valid:
            result.append(x)
            for a in result[:-1]:
                used.add(x - a)
            if len(result) >= 32:
                break
    return result
