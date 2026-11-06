"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    def all_diffs(s):
        """Return list of all positive differences in sorted set s."""
        n = len(s)
        return [s[j] - s[i] for i in range(n) for j in range(i+1, n)]
    
    def is_sidon(s):
        """Check if sorted set s has all positive differences distinct."""
        d = all_diffs(s)
        return len(d) == len(set(d))
    
    def diff_set(s):
        """Return set of all positive differences."""
        return set(all_diffs(s))
    
    # Backtracking search: build set element by element
    # State: (current_set, current_diffs_set, covered_targets)
    # We want to minimize max element
    
    best_solution = None
    best_max_val = float('inf')
    
    def search(current_set, current_diffs, covered, depth):
        nonlocal best_solution, best_max_val
        
        if time.time() > deadline:
            return
        
        # If we've covered all 1..30, check if better
        if covered >= 30:
            max_val = current_set[-1] if current_set else 0
            if max_val < best_max_val:
                best_max_val = max_val
                best_solution = list(current_set)
            return
        
        # Pruning: if current max already >= best, no point continuing
        if current_set and current_set[-1] >= best_max_val:
            return
        
        # Try to find next element
        # Lower bound for next element
        if not current_set:
            next_val = 0
        else:
            next_val = current_set[-1] + 1
        
        # Upper bound: don't exceed best_max_val
        upper_bound = best_max_val - 1 if best_max_val < float('inf') else 200
        
        # Determine which targets are still uncovered
        uncovered = [d for d in range(1, 31) if d not in covered]
        if not uncovered:
            return
        
        # For each candidate next value, check if it's valid
        for nv in range(next_val, upper_bound + 1):
            if time.time() > deadline:
                return
            
            # Check Sidon constraint with new element
            new_diffs = []
            valid = True
            for existing in current_set:
                d = nv - existing
                if d in current_diffs:
                    valid = False
                    break
                new_diffs.append(d)
            if not valid:
                continue
            
            # Also check new diffs don't collide with each other
            if len(new_diffs) != len(set(new_diffs)):
                continue
            
            # Update covered targets
            new_covered = set(covered)
            for d in new_diffs:
                if 1 <= d <= 30:
                    new_covered.add(d)
            
            new_diffs_set = current_diffs | set(new_diffs)
            new_set = current_set + [nv]
            
            # Additional pruning: check if remaining targets can be covered
            # with remaining elements (we have at most 32 total)
            remaining_slots = 32 - len(new_set)
            remaining_targets = set(range(1, 31)) - new_covered
            if len(remaining_targets) > 0:
                # Each new element can cover at most (len(new_set)) new differences
                # but we need at least len(remaining_targets) more differences
                # This is a weak bound, skip for now
                pass
            
            search(new_set, new_diffs_set, new_covered, depth + 1)
    
    # Start with 0
    search([0], set(), set(), 0)
    
    if best_solution is not None:
        return best_solution
    
    # Fallback: use the original greedy approach
    a = [0]
    def diffs(s):
        return [y - x for i, x in enumerate(s) for y in s[i + 1:]]
    for d in range(1, 31):
        if d in diffs(sorted(a)):
            continue
        found = False
        for x in sorted(set(v + s for v in a for s in (d, -d) if v + s >= 0) - set(a)):
            trial = sorted(a + [x])
            if len(set(diffs(trial))) == len(diffs(trial)):
                a = trial
                found = True
                break
        if not found:
            y = 2 * max(a) + 3 * d + 1
            while True:
                trial = sorted(a + [y, y + d])
                if len(set(diffs(trial))) == len(diffs(trial)):
                    a = trial
                    break
                y += 1
    return a

# EVOLVE-BLOCK-END
