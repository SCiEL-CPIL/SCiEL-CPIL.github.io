"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
from functools import lru_cache

def construct():
    deadline = time.time() + 115
    
    # Maximum clique approach:
    # Vertices are candidate integers 1..MAX_CAND
    # Edge between u,v if {0,u,v} forms a Sidon set (i.e., u, v, v-u all distinct and non-zero)
    # Actually, we need the full set to be Sidon, so we need incremental compatibility.
    # Better: build a graph where we track which differences are used.
    # 
    # Alternative structural approach: enumerate small Sidon sets by diameter.
    # For each diameter D, check if there exists a Sidon set of size >= 9 
    # (need 36 diffs for 9 elements, but we only need 30) within [0,D]
    # that covers all of 1..30.
    #
    # Key insight: use bitsets for fast difference checking.
    # 
    # New structural approach: 
    # 1. Precompute all valid 2-element extensions of {0} 
    # 2. Use branch-and-bound with bitset difference tracking
    # 3. Prioritize by coverage of missing differences
    
    MAX_CAND = 200  # Search up to this diameter
    
    def find_minimal_sidon():
        """Find minimal-diameter Sidon set covering 1..30 using branch-and-bound."""
        best_solution = [None]
        best_diameter = [float('inf')]
        
        # We'll search by increasing diameter
        # For each diameter D, try to find a valid set
        # Use bitsets for efficiency
        
        for D in range(30, MAX_CAND + 1):
            if time.time() > deadline - 5:
                break
            
            # Try to find a Sidon set in [0, D] covering 1..30
            solution = _search_diameter(D, best_diameter[0])
            if solution is not None:
                best_solution[0] = solution
                best_diameter[0] = D
                # We found one for this D, try smaller? No, we're going up.
                # Actually we want the minimum, so first D that works is optimal
                # unless we can prove no smaller works
                # Since we're iterating D from 30 upward, first success is minimal
                return solution
        
        return best_solution[0]
    
    def _search_diameter(D, upper_bound):
        """Find a Sidon set in [0, D] covering all diffs 1..30.
        Uses backtracking with bitset optimization."""
        
        # Precompute: for each candidate x in 1..D, 
        # and for each potential current set (represented by used_diffs bitmask),
        # check if x can be added
        # 
        # Since D is small (<=200), we can use a 32-bit int to represent diffs 1..31
        # and a larger structure for all diffs up to D
        
        # Strategy: build the set incrementally
        # Start with {0}
        # At each step, choose the next element that:
        #   1. Creates no duplicate differences
        #   2. Is in [1, D]
        #   3. Maximizes coverage of missing diffs 1..30
        
        elements = [0]
        used_diffs = set()
        target_diffs = set(range(1, 31))
        covered = set()
        
        # Precompute candidate positions
        candidates = list(range(1, D + 1))
        
        def can_add(x, current_elements, used_diffs_set):
            """Check if x can be added to current_elements without creating duplicate diffs."""
            if x <= current_elements[-1]:
                return False
            for a in current_elements:
                d = x - a
                if d in used_diffs_set:
                    return False
            return True
        
        def get_new_diffs(x, current_elements):
            """Get all new differences created by adding x."""
            return [x - a for a in current_elements]
        
        def coverage_gain(x, current_elements, used_diffs_set, target_set):
            """How many target diffs does adding x cover?"""
            gain = 0
            for a in current_elements:
                d = x - a
                if d in target_set and d not in used_diffs_set:
                    gain += 1
            return gain
        
        def backtrack():
            if time.time() > deadline - 2:
                return False
            
            # Check if all target diffs are covered
            missing = target_diffs - covered
            if not missing:
                # Found a solution!
                return True
            
            current_size = len(elements)
            
            # Pruning: even if we add all remaining candidates, can we cover missing?
            # Each new element adds at most current_size new diffs
            # But many may not be in target
            # Simple bound: need at least ceil(len(missing) / current_size) more elements
            # And we have D - elements[-1] possible positions
            min_more = -(-len(missing) // current_size) if current_size > 0 else len(missing)
            if min_more > D - elements[-1]:
                return False
            
            # Get all valid next elements
            valid_next = []
            for x in range(elements[-1] + 1, D + 1):
                if can_add(x, elements, used_diffs):
                    gain = coverage_gain(x, elements, used_diffs, target_diffs)
                    valid_next.append((x, gain))
            
            if not valid_next:
                return False
            
            # Sort by coverage gain (descending), then by value (ascending for minimization)
            valid_next.sort(key=lambda t: (-t[1], t[0]))
            
            # Try each valid next element
            for x, gain in valid_next:
                if time.time() > deadline - 2:
                    return False
                
                # Add x
                new_diffs = get_new_diffs(x, elements)
                added_to_covered = set()
                for d in new_diffs:
                    used_diffs.add(d)
                    if d in target_diffs:
                        covered.add(d)
                        added_to_covered.add(d)
                
                elements.append(x)
                
                if backtrack():
                    # Undo before returning
                    elements.pop()
                    for d in new_diffs:
                        used_diffs.discard(d)
                    for d in added_to_covered:
                        covered.discard(d)
                    return True
                
                # Undo
                elements.pop()
                for d in new_diffs:
                    used_diffs.discard(d)
                for d in added_to_covered:
                    covered.discard(d)
            
            return False
        
        if backtrack():
            return list(elements)
        return None
    
    # First, try to find a solution with a reasonable diameter
    # Lower bound: we need 30 distinct positive differences from a set of n elements
    # n*(n-1)/2 >= 30 => n >= 8 (8*7/2=28 < 30), so n >= 9
    # For 9 elements, minimum diameter is at least... 
    # The Golomb ruler of order 9 has length 42
    # But we don't need a full Golomb ruler, just need diffs 1..30 to appear
    # 
    # Let's try diameters from 30 upward
    result = find_minimal_sidon()
    
    if result is not None:
        return result
    
    # Fallback: use the original approach if the new one fails
    a = [0]
    def get_diffs(s):
        return [y - x for i, x in enumerate(s) for y in s[i + 1:]]
    
    for _ in range(100):
        covered = set(get_diffs(sorted(a)))
        missing = [d for d in range(1, 31) if d not in covered]
        if not missing:
            break
        d = missing[0]
        found = False
        for x in range(1, 10000):
            if x in a:
                continue
            trial = sorted(a + [x])
            diffs = get_diffs(trial)
            if len(set(diffs)) == len(diffs) and d in diffs:
                a = trial
                found = True
                break
        if not found:
            x = max(a) + d + 1
            while x in a or any((x - ai) in get_diffs(a) for ai in a):
                x += 1
            a = sorted(a + [x])
    
    return a
# EVOLVE-BLOCK-END
