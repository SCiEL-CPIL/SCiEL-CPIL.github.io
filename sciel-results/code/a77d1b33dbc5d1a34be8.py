"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    # We need a Sidon set containing 0, with all positive differences distinct,
    # and every d in 1..30 occurring as a difference.
    # Minimize the largest element.
    
    # Key insight: We can search for the minimum diameter D such that
    # there exists a Sidon set {0 = a0 < a1 < ... < a_{k-1}} with a_{k-1} <= D
    # and {a_j - a_i : 0 <= i < j < k} covers {1, 2, ..., 30}.
    
    # The set must have at least enough elements to produce 30 distinct differences.
    # With k elements, we have C(k,2) differences. We need C(k,2) >= 30, so k >= 8.
    # But we also need the differences to be exactly 1..30 (subset, not necessarily all C(k,2)).
    # Actually, we need 1..30 to be a subset of the differences. The set can have more differences.
    
    # Let's use a DFS/backtracking approach with bitmask for covered differences.
    # We build the set incrementally, adding elements one by one.
    # At each step, we try candidate values for the next element.
    
    # To minimize the diameter, we do a binary search or iterative deepening on D.
    
    # But first, let's try a direct search with good pruning.
    
    # State: current set (sorted list), set of used differences (bitmask for 1..30),
    #        and the set of all differences so far (to check Sidon property).
    
    # For efficiency, we'll represent the set of all pairwise differences as a set.
    
    # Let's try to find the optimal solution by searching for small diameters.
    
    # Lower bound: We need 30 distinct differences from 1..30. The minimum possible
    # largest element: if we have k elements, the max difference is at most D.
    # The number of distinct differences is at most D (since they're in 1..D).
    # We need at least 30 differences, so D >= 30. But that's trivial.
    
    # A better lower bound: Consider that we need differences 1, 2, ..., 30.
    # The element that gives difference 1 must be adjacent to another element.
    # Let's just search.
    
    # We'll use a recursive backtracking search.
    # The set always contains 0. We build it by adding elements in increasing order.
    
    # To make this tractable, we limit the search to reasonable sizes.
    # A Sidon set with 30 consecutive differences likely has around 10-15 elements.
    
    # Let's try iterative deepening on the maximum element D.
    
    best_solution = None
    best_D = float('inf')
    
    # Precompute: for a given D, what's the max number of elements in a Sidon set in [0, D]?
    # This is related to the maximum size of a Sidon set in [1, D].
    # For small D, we can just search.
    
    # Let's do a DFS that builds the set, trying to minimize the max element.
    
    # We'll search for solutions with max element up to some limit, increasing if needed.
    
    # Actually, let's just do a well-structured DFS with the goal of finding ANY solution
    # with a small max element, and keep the best one found.
    
    # State representation:
    # - elements: sorted tuple of elements so far (always includes 0)
    # - diff_set: set of all pairwise differences
    # - covered: bitmask of which differences in 1..30 are covered
    
    # We'll use memoization on (frozenset of elements, covered_mask) but that's too large.
    # Instead, we'll use the structure of the problem.
    
    # Key pruning:
    # 1. If we've covered all 30 differences, we have a solution. Record it.
    # 2. If the next element to add would exceed our current best_D, prune.
    # 3. Sidon constraint: new element x must not create a duplicate difference.
    
    # Let's implement this with careful ordering and pruning.
    
    FULL_MASK = (1 << 30) - 1  # bits 0..29 represent differences 1..30
    
    def dfs(elements, diff_set, covered_mask, depth):
        nonlocal best_solution, best_D, deadline
        
        if time.time() > deadline:
            return
        
        # Check if we've covered all differences
        if covered_mask == FULL_MASK:
            max_elem = elements[-1]
            if max_elem < best_D:
                best_D = max_elem
                best_solution = list(elements)
            return
        
        # If we've exceeded our best known diameter, prune
        if elements[-1] >= best_D:
            return
        
        # Determine which differences are still missing
        missing = [d for d in range(1, 31) if not (covered_mask & (1 << (d - 1)))]
        if not missing:
            return
        
        # Try adding a new element
        # The new element must be greater than the current max
        # and must be within our diameter bound
        max_current = elements[-1]
        
        # Candidate values for the new element:
        # It must create at least one new difference that's in 1..30
        # and not create any duplicate differences
        
        # For efficiency, only consider candidates that are within a reasonable range
        # and that create at least one missing difference
        
        # The new element x must satisfy:
        # For each existing element a in elements:
        #   x - a must not be in diff_set (unless x - a > 30, in which case it's OK as long as it's not a duplicate)
        #   Actually, x - a must not be in diff_set at all (Sidon constraint)
        #   And if x - a is in 1..30, it helps cover that difference
        
        # Let's generate candidates
        # x must be > max_current
        # x must be < best_D (to improve on current best)
        
        # Also, x must be such that for all a in elements, (x - a) not in diff_set
        
        # To limit the search, we'll only consider x values that are within some range
        # and that are "close" to creating missing differences
        
        # For each missing difference d, x could be a + d for some a in elements
        # Let's collect candidate x values
        
        candidates = set()
        for d in missing:
            for a in elements:
                x = a + d
                if x > max_current and x < best_D:
                    candidates.add(x)
        
        # Also try some other candidates to be safe
        # But for now, let's stick with these
        
        if not candidates:
            return
        
        # Sort candidates by value (smaller first, to find small diameter solutions)
        for x in sorted(candidates):
            if time.time() > deadline:
                return
            
            # Check Sidon constraint
            valid = True
            new_diffs = []
            for a in elements:
                d = x - a
                if d <= 0:
                    continue
                if d in diff_set:
                    valid = False
                    break
                new_diffs.append(d)
            
            if not valid:
                continue
            
            # Check if x is within bounds
            if x >= best_D:
                continue
            
            # Update state
            new_diff_set = diff_set | set(new_diffs)
            new_covered = covered_mask
            for d in new_diffs:
                if 1 <= d <= 30:
                    new_covered |= (1 << (d - 1))
            
            # Recurse
            dfs(elements + (x,), new_diff_set, new_covered, depth + 1)
            
            # If we found a solution, update best and prune
            if elements[-1] >= best_D:
                break
    
    # Initial state
    initial_elements = (0,)
    initial_diff_set = set()
    initial_covered = 0
    
    # We need an initial upper bound on D
    # Let's use the greedy solution as an upper bound
    # Run a quick greedy to get an initial solution
    
    def greedy_solution():
        a = [0]
        diff_set = set()
        covered = set()
        
        for d in range(1, 31):
            if d in covered:
                continue
            found = False
            # Try adding x such that x - a = d or a - x = d for some a in a
            candidates = set()
            for existing in a:
                candidates.add(existing + d)
                if existing - d >= 0:
                    candidates.add(existing - d)
            
            for x in sorted(candidates):
                if x in a:
                    continue
                # Check Sidon
                valid = True
                new_diffs = []
                for existing in a:
                    diff = abs(x - existing)
                    if diff == 0:
                        continue
                    if diff in diff_set:
                        valid = False
                        break
                    new_diffs.append(diff)
                
                if valid:
                    a.append(x)
                    a.sort()
                    diff_set.update(new_diffs)
                    covered.update(d for d in new_diffs if 1 <= d <= 30)
                    found = True
                    break
            
            if not found:
                # Add a new element far away
                y = 2 * max(a) + 3 * d + 1
                while True:
                    valid = True
                    new_diffs = []
                    for existing in a:
                        diff = abs(y - existing)
                        if diff in diff_set:
                            valid = False
                            break
                        new_diffs.append(diff)
                    
                    if valid:
                        a.append(y)
                        a.sort()
                        diff_set.update(new_diffs)
                        covered.update(d for d in new_diffs if 1 <= d <= 30)
                        break
                    y += 1
        
        return a
    
    # Get initial upper bound
    greedy_a = greedy_solution()
    initial_D = max(greedy_a)
    best_D = initial_D
    best_solution = greedy_a
    
    # Now run DFS to find better solutions
    dfs(initial_elements, initial_diff_set, initial_covered, 0)
    
    if best_solution is not None:
        return best_solution
    else:
        return greedy_a

# EVOLVE-BLOCK-END
