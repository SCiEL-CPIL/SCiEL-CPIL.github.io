"""CSP-based construction of a Sidon set containing 0 with all differences 1..30."""
import sys
sys.setrecursionlimit(10000)

def construct():
    # We need a Sidon set S containing 0 such that every d in 1..30 is a difference.
    # Sidon condition: all positive differences y-x (x<y in S) are distinct.
    # Minimize max(S).
    
    target_diffs = set(range(1, 31))
    
    def solve():
        best_solution = None
        best_max_val = float('inf')
        
        # S is a sorted list of elements
        # used_diffs is a set of all positive differences generated so far
        def backtrack(S, used_diffs):
            nonlocal best_solution, best_max_val
            
            # Check if we have covered all target diffs
            if target_diffs.issubset(used_diffs):
                if max(S) < best_max_val:
                    best_max_val = max(S)
                    best_solution = list(S)
                return True
            
            # Pruning: if we have too many elements
            if len(S) >= 32:
                return False
                
            # Pruning: if current max is already worse than best found
            if max(S) >= best_max_val:
                return False
                
            # Determine candidates for the next element x
            # x must be > 0, not in S
            # To minimize max(S), we try small x first
            # However, x can be less than max(S) if we haven't filled gaps?
            # Actually, since we start with 0 and add elements, if we always add x > max(S), 
            # then max(S) is the last added element.
            # But we could add an element smaller than current max?
            # If we add x < max(S), the max doesn't change. This might be useful to fill gaps in differences.
            # But typically, Sidon sets are constructed by adding larger elements.
            # Let's allow x to be any integer not in S, x > 0.
            # To keep search space small, let's restrict x to be > max(S) initially?
            # No, consider {0, 5, 1}. Diff 1 is covered by 5-? No, 5-1=4, 1-0=1, 5-0=5.
            # If we add 1 after 5, max is still 5.
            # Let's try candidates x in range(1, best_max_val) if best_max_val is finite, else some upper bound.
            # But we don't know the upper bound.
            # Let's use iterative deepening or a reasonable upper bound.
            # A known perfect difference set for 1..30 doesn't exist in small integers easily?
            # Wait, the problem says "truncated perfect difference set".
            # Let's try x from 1 upwards.
            
            # Heuristic: try x values that cover missing diffs.
            # For a candidate x, the new diffs are {abs(x - s) for s in S}.
            # These must all be in target_diffs (ideally) or at least not conflict with used_diffs.
            # Actually, they just must not be in used_diffs. They can be > 30.
            # But to cover 1..30 efficiently, we should prefer x such that abs(x-s) is in target_diffs.
            
            # Let's generate candidates x from 1 to some limit.
            # What limit? If best_max_val is inf, we need a heuristic.
            # Let's try up to 100 first, then increase if needed?
            # Or just try all x from 1 to 1000?
            # Given the constraints, a solution likely exists with max < 100.
            
            # To be safe and efficient, let's sort candidates by "potential".
            # Potential: number of missing diffs covered by this x.
            
            candidates = []
            # We can't iterate to infinity. Let's bound x by best_max_val - 1 if known, else a heuristic bound.
            # Let's try a dynamic bound. If we fail to find a solution with x < 100, we might need larger.
            # But let's assume a solution exists with max < 200.
            
            # Better: Iterate x from 1 to best_max_val - 1 (if best found) or 200 (if not).
            # Actually, if we are searching for the *first* solution with minimal max, we can just try x in increasing order.
            # If we find a solution, we update best_max_val. Then we restart or prune?
            # Backtracking with global best pruning works.
            
            # Let's try x from 1 to 100. If no solution, we might miss one.
            # Let's use a wider range, say 1..500.
            # To speed up, we can pre-filter x.
            
            for x in range(1, 500):
                if x in S:
                    continue
                
                new_diffs = set()
                valid = True
                for s in S:
                    d = abs(x - s)
                    if d in used_diffs:
                        valid = False
                        break
                    new_diffs.add(d)
                
                if not valid:
                    continue
                
                # Check if new_diffs are distinct (they are, as d is derived from unique s? No, abs(x-s) can be same for s1, s2 if x is midpoint. But S is Sidon, so differences are distinct. Does that imply abs(x-s) are distinct?
                # If x - s1 = x - s2 => s1 = s2.
                # If s1 - x = s2 - x => s1 = s2.
                # If x - s1 = s2 - x => 2x = s1 + s2.
                # If 2x = s1 + s2, then x - s1 = s2 - x. This means the difference d is repeated.
                # So we must check if all values in new_diffs are distinct.
                if len(new_diffs) != len(S):
                    continue
                
                # Also, we should prefer x that covers missing target diffs.
                # Let's just add it and recurse.
                # To find minimal max, we try small x first.
                
                new_S = sorted(S + [x])
                new_used_diffs = used_diffs | new_diffs
                
                if backtrack(new_S, new_used_diffs):
                    return True
            
            return False
        
        # Start with {0}
        backtrack([0], set())
        
        if best_solution:
            return best_solution
        else:
            # Fallback: return a valid set even if not minimal? 
            # The problem implies a solution exists.
            # If search fails, we might need to expand the range or use a different heuristic.
            # Let's try a greedy construction as a fallback.
            S = [0]
            used_diffs = set()
            for x in range(1, 10000):
                new_diffs = set()
                valid = True
                for s in S:
                    d = abs(x - s)
                    if d in used_diffs:
                        valid = False
                        break
                    new_diffs.add(d)
                if valid and len(new_diffs) == len(S):
                    S.append(x)
                    S.sort()
                    used_diffs |= new_diffs
                    if target_diffs.issubset(used_diffs):
                        return S
            return S # Should not happen

    return solve()
