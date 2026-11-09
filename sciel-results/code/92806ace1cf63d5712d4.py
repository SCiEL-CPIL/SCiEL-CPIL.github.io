import time
import sys
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    target_diffs = set(range(1, 31))
    full_mask = (1 << 30) - 1
    
    def get_diffs(S):
        S = sorted(S)
        return set(y - x for i, x in enumerate(S) for y in S[i+1:])
        
    def is_sidon(S):
        if not S:
            return True
        diffs = get_diffs(S)
        return len(diffs) == len(S) * (len(S) - 1) // 2

    def covers_all(S):
        diffs = get_diffs(S)
        return target_diffs.issubset(diffs)

    best_max = float('inf')
    best_set = None

    # Backtracking search with bitmask for used differences
    # We search for the minimum diameter D such that there exists a Sidon set
    # {0, x1, x2, ..., xk} with 0 < x1 < x2 < ... < xk <= D
    # covering all differences 1..30.
    
    # Key insight: use a bitset for used differences (up to D)
    # and backtrack over elements in increasing order.
    
    # Lower bound: need at least 9 elements (from known results)
    # For k elements, max distinct diffs = k*(k-1)/2
    # Need k*(k-1)/2 >= 30, so k >= 8 (8*7/2=28 < 30), k >= 9 (9*8/2=36 >= 30)
    
    # Try diameters from lower bound upward
    # Known: perfect difference sets exist; minimal for 1..30 is around 100-200
    
    def search_for_diameter(D, time_limit):
        """Find a Sidon set with max element <= D covering 1..30."""
        # Use backtracking
        # State: current set (sorted), used_diffs bitset, covered_mask
        # We build the set element by element, always adding the next smallest.
        
        # Precompute: for each pair of positions, what difference they'd create
        # Actually, we just track used differences in a set.
        
        # Optimization: use a list for the set, and a set for used diffs
        # The key pruning: if we can't cover all remaining diffs with remaining elements, prune.
        
        max_elements = 32  # at most 32 elements including 0
        
        # We'll use DFS with memoization-like pruning
        # State: (last_elem, num_elements, covered_mask, used_diffs_tuple)
        # But used_diffs can be large, so we'll use a different approach.
        
        # Better approach: iterative deepening on the number of elements
        # For k elements (including 0), we need k*(k-1)/2 >= 30
        # Minimum k = 9
        
        best_solution = None
        
        def backtrack(current_set, last_elem, used_diffs, covered_mask, depth, max_depth):
            nonlocal best_solution
            
            if time.time() > time_limit:
                return None
            
            if covered_mask == full_mask:
                if best_solution is None or len(current_set) < len(best_solution):
                    best_solution = current_set[:]
                return current_set[:]
            
            if len(current_set) >= max_depth:
                return None
            
            # Pruning: can we still cover all missing diffs?
            # Remaining elements: max_depth - len(current_set)
            # Max new diffs from r more elements: r * (len(current_set)) + r*(r-1)/2
            # (each new element makes diffs with all existing + with other new elements)
            remaining = max_depth - len(current_set)
            max_new_diffs = remaining * len(current_set) + remaining * (remaining - 1) // 2
            missing_count = 30 - bin(covered_mask).count('1')
            if max_new_diffs < missing_count:
                return None
            
            # Try adding next element x > last_elem, x <= D
            # To minimize diameter, try smaller x first
            # But also, we want to cover missing diffs efficiently
            
            # Candidate generation: focus on values that cover missing diffs
            missing_diffs = [d for d in range(1, 31) if not (covered_mask & (1 << (d - 1)))]
            
            # Generate candidates: for each existing element s and each missing diff d,
            # x = s + d is a candidate (if x > last_elem and x <= D)
            candidates = set()
            for s in current_set:
                for d in missing_diffs:
                    x = s + d
                    if last_elem < x <= D:
                        candidates.add(x)
            
            # Also try last_elem + 1, last_elem + 2, etc. to fill gaps
            for x in range(last_elem + 1, min(last_elem + 31, D + 1)):
                candidates.add(x)
            
            # Sort candidates: prefer those that cover more missing diffs
            def score_candidate(x):
                new_diffs = []
                valid = True
                for s in current_set:
                    d = x - s
                    if d in used_diffs:
                        return -1  # invalid
                    if 1 <= d <= 30:
                        new_diffs.append(d)
                return len(set(new_diffs) & set(missing_diffs))
            
            scored = []
            for x in candidates:
                sc = score_candidate(x)
                if sc >= 0:
                    scored.append((sc, x))
            
            # Sort by score (desc), then x (asc)
            scored.sort(key=lambda item: (-item[0], item[1]))
            
            # Limit candidates to avoid explosion
            max_cands = 50
            for sc, x in scored[:max_cands]:
                new_diffs = [x - s for s in current_set]
                new_used = used_diffs | set(new_diffs)
                new_mask = covered_mask
                for d in new_diffs:
                    if 1 <= d <= 30:
                        new_mask |= (1 << (d - 1))
                
                result = backtrack(current_set + [x], x, new_used, new_mask, depth + 1, max_depth)
                if result is not None:
                    return result
            
            return None
        
        # Try different numbers of elements
        for max_depth in range(9, 33):
            if time.time() > time_limit:
                break
            result = backtrack([0], 0, set(), 0, 1, max_depth)
            if result is not None:
                return result
        
        return None
    
    # Try diameters in increasing order
    # Start from a reasonable lower bound
    # For 9 elements, the minimal diameter for a (9,1,1) difference set is known to be around 100+
    # Let's start from 100 and go up
    
    lower_bound = 100
    upper_bound = 500
    
    # First, do a quick beam search to get a good initial solution
    # This gives us an upper bound on the diameter
    
    # Beam search for initial solution
    beam_width = 2000
    S_init = [0]
    used_diffs_init = set()
    covered_mask_init = 0
    
    beam = [(0, S_init, used_diffs_init, covered_mask_init)]
    
    rng = random.Random(42)
    
    for step in range(10):  # Add up to 10 more elements
        if time.time() > deadline - 20:
            break
        
        new_beam = []
        for max_val, S, used_diffs, covered_mask in beam:
            if time.time() > deadline - 20:
                break
                
            if covered_mask == full_mask:
                if max_val < best_max:
                    best_max = max_val
                    best_set = S[:]
                continue
            
            missing_diffs = [d for d in range(1, 31) if not (covered_mask & (1 << (d - 1)))]
            rng.shuffle(missing_diffs)
            
            candidates = set()
            for d in missing_diffs[:15]:
                for s in S:
                    x = s + d
                    if x > max(S):
                        candidates.add(x)
            
            # Add some random candidates
            last = max(S)
            for _ in range(5):
                rand_x = last + rng.randint(1, 20)
                candidates.add(rand_x)
            
            S_list = sorted(S)
            last = S_list[-1]
            
            valid_candidates = []
            for x in candidates:
                if x <= last or x > 10**6:
                    continue
                
                new_diffs = []
                valid = True
                for s in S_list:
                    d = x - s
                    if d in used_diffs:
                        valid = False
                        break
                    new_diffs.append(d)
                
                if not valid:
                    continue
                
                new_mask = covered_mask
                for d in new_diffs:
                    if 1 <= d <= 30:
                        new_mask |= (1 << (d - 1))
                
                valid_candidates.append((x, new_mask, new_diffs))
            
            valid_candidates.sort(key=lambda item: (item[0], -bin(item[1]).count('1')))
            for x, new_mask, new_diffs in valid_candidates[:10]:
                new_S = S + [x]
                new_used = used_diffs | set(new_diffs)
                new_beam.append((x, new_S, new_used, new_mask))
        
        if not new_beam:
            break
        
        new_beam.sort(key=lambda item: (item[0], -bin(item[3]).count('1')))
        
        seen = set()
        unique_beam = []
        for state in new_beam:
            s_tuple = tuple(state[1])
            if s_tuple not in seen:
                seen.add(s_tuple)
                unique_beam.append(state)
            if len(unique_beam) >= beam_width:
                break
        
        beam = unique_beam
        
        if best_max < float('inf'):
            beam = [s for s in beam if s[0] < best_max]
            if not beam:
                break
    
    # Now use the beam search result as an upper bound
    if best_max < float('inf'):
        upper_bound = min(upper_bound, best_max)
    
    # Now do systematic diameter search from lower_bound to upper_bound
    # Use the backtracking search with a time budget per diameter
    
    time_per_diameter = 30  # seconds per diameter value
    current_time = time.time()
    
    for D in range(lower_bound, upper_bound + 1):
        if time.time() > deadline - 10:
            break
        
        remaining_time = deadline - 10 - time.time()
        if remaining_time < 2:
            break
        
        # Allocate time for this diameter
        alloc_time = min(time_per_diameter, remaining_time)
        
        result = search_for_diameter(D, time.time() + alloc_time)
        
        if result is not None:
            # Verify
            if is_sidon(result) and covers_all(result) and max(result) <= D:
                best_max = D
                best_set = result
                # Update upper bound
                upper_bound = D
                # Try to find even smaller
                break  # Found solution for this D, try smaller D's
    
    # If we found a solution, try to polish it
    if best_set:
        S = sorted(best_set[:])
        
        # Try to reduce elements
        improved = True
        while improved and (time.time() - start_time) < deadline - 5:
            improved = False
            S.sort()
            
            for i in range(len(S) - 1, 0, -1):
                if (time.time() - start_time) > deadline - 5:
                    break
                
                s_i = S[i]
                S_rest = S[:i] + S[i+1:]
                lower_bound_i = S[i-1] + 1 if i > 0 else 1
                
                # Try to reduce s_i
                for new_val in range(s_i - 1, lower_bound_i - 1, -1):
                    S_new = S_rest + [new_val]
                    S_new.sort()
                    if is_sidon(S_new) and covers_all(S_new):
                        S = S_new
                        improved = True
                        break
                if improved:
                    break
        
        best_set = S
        best_max = max(S)
    
    if best_set:
        return best_set
    else:
        # Fallback: greedy construction
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
