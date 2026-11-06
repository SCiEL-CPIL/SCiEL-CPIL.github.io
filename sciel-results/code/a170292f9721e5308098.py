"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools

def construct():
    deadline = time.time() + 115
    best_set = [None]
    best_max = [float('inf')]

    def is_valid_sidon(elements):
        """Check if a set of elements forms a Sidon set."""
        diffs = set()
        n = len(elements)
        for i in range(n):
            for j in range(i + 1, n):
                d = elements[j] - elements[i]
                if d in diffs:
                    return False
                diffs.add(d)
        return True

    def covers_all_diffs(elements):
        """Check if all differences 1..30 are covered."""
        diffs = set()
        n = len(elements)
        for i in range(n):
            for j in range(i + 1, n):
                diffs.add(elements[j] - elements[i])
        return all(d in diffs for d in range(1, 31))

    def solve_csp():
        """CSP-style solver: assign elements to cover missing differences."""
        # We build a Sidon set containing 0 that covers diffs 1..30
        # Key insight: in a Sidon set with n elements, there are n*(n-1)/2 differences
        # We need at least 30 differences, so n >= 8 (8*7/2=28 < 30, need n>=9: 9*8/2=36)
        
        # For each size n from 9 to 32, try to find a Sidon set of size n
        # that covers all diffs 1..30 and minimizes max element
        
        for n in range(9, 20):  # Try reasonable sizes
            if time.time() > deadline:
                return
            
            # In a Sidon set of size n, the minimum possible max element
            # is related to the Golomb ruler problem
            # Lower bound: for n elements with distinct differences,
            # max element >= n*(n-1)/2 approximately (but this is loose)
            # A tighter bound: the largest difference must be >= 30
            # Also, we need 30 specific small differences to appear
            
            # Use backtracking with strong pruning
            elements = [0]
            used_diffs = set()
            need = set(range(1, 31))
            
            # Precompute: for a given set of used diffs and current max,
            # what's the minimum additional elements needed?
            # If we have k elements left to add, we can add at most k*(k+1)/2 new diffs
            # (each new element x adds diffs x-a_i for each existing a_i)
            # Actually each new element adds exactly len(elements) new diffs
            
            def min_additional_needed(remaining_need, current_size, max_size):
                """Lower bound on additional elements needed."""
                if not remaining_need:
                    return 0
                # Each new element adds at most (current_size + i) diffs for the i-th new element
                # but some may overlap with existing diffs
                # Conservative: each new element adds at most current_size new useful diffs
                # Actually, element added when we have m elements adds m new diffs
                # So if we add t more elements, total new diffs <= m*(m+1)/2 where m = current_size
                # But we only need |remaining_need| more
                # Simple bound: need at least ceil(|remaining_need| / current_size) more
                # Better: solve m*(m+1)/2 >= |remaining_need|
                needed = len(remaining_need)
                m = current_size
                t = 0
                while m * (m + 1) // 2 < needed:
                    t += 1
                    m += 1
                return t
            
            # MRV-based: pick the most constrained uncovered difference
            # A difference d is "constrained" if it has few possible placements
            
            def get_placement_options(d, current_elements, used_diffs_set, upper_bound):
                """Find valid positions x such that x - a_i = d for some a_i in current_elements,
                or x - a_i covers d for new element x."""
                # For difference d to be covered by adding element x:
                # Either x - a_i = d for some existing a_i, meaning x = a_i + d
                # And x - a_i must not create duplicate diffs
                options = set()
                for a_i in current_elements:
                    x = a_i + d
                    if x <= current_elements[-1]:
                        continue
                    if x > upper_bound:
                        continue
                    # Check if adding x creates duplicate differences
                    valid = True
                    for a_j in current_elements:
                        diff = x - a_j
                        if diff in used_diffs_set:
                            valid = False
                            break
                    if valid:
                        options.add(x)
                return options
            
            def backtrack():
                if time.time() > deadline:
                    return False
                
                if not need:
                    mx = elements[-1]
                    if mx < best_max[0]:
                        best_max[0] = mx
                        best_set[0] = list(elements)
                    return True
                
                current_size = len(elements)
                # Pruning: if we can't fit more elements before deadline, stop
                remaining_need = len(need)
                # Each new element adds at most current_size new diffs (to existing elements)
                # But also, the new element itself can be at most upper_bound
                # Upper bound on max element
                upper_bound = best_max[0] - 1  # Must be strictly less than current best
                if upper_bound <= elements[-1]:
                    return False
                
                # Check if it's even possible to cover remaining_need diffs
                # With (upper_bound - elements[-1]) / 1 possible new positions... 
                # Simpler: each new element adds at most current_size diffs
                # We need remaining_need more diffs
                # So we need at least ceil(remaining_need / current_size) more elements
                # And each new element must be > elements[-1] and <= upper_bound
                min_more = -(-remaining_need // current_size)  # ceiling division
                if min_more > upper_bound - elements[-1]:
                    return False
                
                # MRV: pick the uncovered difference with fewest placement options
                best_d = None
                best_options = None
                best_count = float('inf')
                
                for d in sorted(need):
                    options = get_placement_options(d, elements, used_diffs, upper_bound)
                    if len(options) == 0:
                        return False  # Dead end
                    if len(options) < best_count:
                        best_count = len(options)
                        best_d = d
                        best_options = options
                        if best_count == 1:
                            break
                
                if best_d is None:
                    return False
                
                # Try each option for best_d
                for x in sorted(best_options):
                    if time.time() > deadline:
                        return False
                    
                    # Validate x more carefully
                    new_diffs = []
                    valid = True
                    for a_i in elements:
                        diff = x - a_i
                        if diff in used_diffs:
                            valid = False
                            break
                        new_diffs.append(diff)
                    
                    if not valid:
                        continue
                    
                    # Also check: x must be > elements[-1] (guaranteed since x = a_i + d > a_i >= 0)
                    # and x > elements[-1]
                    if x <= elements[-1]:
                        continue
                    
                    # Add x
                    elements.append(x)
                    added_diffs = set()
                    for diff in new_diffs:
                        used_diffs.add(diff)
                        added_diffs.add(diff)
                        if diff in need:
                            need.discard(diff)
                    
                    if backtrack():
                        # Undo
                        elements.pop()
                        for diff in added_diffs:
                            used_diffs.discard(diff)
                        return True
                    
                    # Undo
                    elements.pop()
                    for diff in added_diffs:
                        used_diffs.discard(diff)
                
                return False
            
            backtrack()
            
            if best_set[0] is not None:
                return best_set[0]
        
        return None

    result = solve_csp()
    
    if result is not None:
        return result
    
    # Fallback: greedy construction
    a = [0]
    def get_diffs(s):
        return [y - x for i, x in enumerate(s) for y in s[i + 1:]]
    
    for _ in range(100):  # Try to cover all diffs
        covered = set(get_diffs(sorted(a)))
        missing = [d for d in range(1, 31) if d not in covered]
        if not missing:
            break
        d = missing[0]
        found = False
        # Try to add a single element that covers d
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
            # Add two elements
            for x in range(1, 5000):
                if x in a:
                    continue
                for y in range(x + 1, x + 5000):
                    if y in a:
                        continue
                    trial = sorted(a + [x, y])
                    diffs = get_diffs(trial)
                    if len(set(diffs)) == len(diffs) and d in diffs:
                        a = trial
                        found = True
                        break
                if found:
                    break
        if not found:
            # Force add
            x = max(a) + d + 1
            while x in a or any((x - ai) in get_diffs(a) for ai in a):
                x += 1
            a = sorted(a + [x])
    
    return a
# EVOLVE-BLOCK-END
