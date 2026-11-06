# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Precompute masks for differences 1..30
    # Bit i-1 represents difference i
    # We need to cover bits 0..29
    
    def get_diffs_set(elements):
        """Get set of all positive differences."""
        diffs = set()
        for i in range(len(elements)):
            for j in range(i+1, len(elements)):
                diffs.add(elements[j] - elements[i])
        return diffs
    
    def is_valid_sidon(elements):
        """Check if elements form a Sidon set (all positive differences distinct)."""
        n = len(elements)
        if n < 2:
            return True
        seen = set()
        for i in range(n):
            for j in range(i+1, n):
                d = elements[j] - elements[i]
                if d in seen:
                    return False
                seen.add(d)
        return True
    
    def covers_all_diffs(elements):
        """Check if all differences 1..30 are present."""
        diffs = get_diffs_set(elements)
        return all(d in diffs for d in range(1, 31))
        
    def backtrack(elements, used_diffs_set, req_mask, max_val_bound):
        """
        Backtracking search.
        elements: tuple of sorted elements
        used_diffs_set: set of all differences formed
        req_mask: bitmask of required diffs 1..30 that are COVERED. 
                   Bit k-1 is 1 if diff k is in used_diffs_set.
        """
        nonlocal best_solution, best_max_val
        
        if time.time() > deadline:
            return (False, None)
        
        # Check if all required diffs are covered
        if req_mask == (1 << 30) - 1:
            current_max = elements[-1]
            if current_max < best_max_val:
                best_max_val = current_max
                best_solution = elements
            return (True, elements)
        
        if len(elements) > 32:
            return (False, None)
        
        last = elements[-1]
        
        if last + 1 > max_val_bound:
            return (False, None)
        
        candidates = []
        
        # Optimization: Iterate candidates
        for candidate in range(last + 1, max_val_bound + 1):
            if time.time() > deadline:
                break
            
            new_diffs = []
            valid = True
            new_mask_additions = 0
            
            for i in range(len(elements)):
                d = candidate - elements[i]
                if d <= 0:
                    continue
                if d in used_diffs_set or d in new_diffs:
                    valid = False
                    break
                new_diffs.append(d)
                if 1 <= d <= 30:
                    new_mask_additions |= (1 << (d - 1))
            
            if not valid:
                continue
            
            # Pruning: If this candidate doesn't add any new required differences,
            # it's only useful if we are forced to add more elements or if it helps structure.
            # However, adding "waste" elements consumes the size limit (32).
            # We should strongly prefer candidates that cover missing diffs.
            
            new_req_mask = req_mask | new_mask_additions
            
            # Heuristic: Prioritize candidates that cover more missing required diffs
            # and have smaller values (to keep diameter small)
            
            # Calculate how many NEW required diffs are covered
            new_covered_count = bin(new_req_mask ^ req_mask).count('1')
            
            candidates.append((candidate, new_diffs, new_req_mask, new_covered_count))
        
        # Sort by:
        # 1. Most new required diffs covered (descending)
        # 2. Smallest candidate value (ascending)
        candidates.sort(key=lambda x: (-x[3], x[0]))
        
        for candidate, new_diffs, new_req_mask, _ in candidates:
            if time.time() > deadline:
                break
            
            new_used_diffs_set = set(used_diffs_set)
            new_used_diffs_set.update(new_diffs)
            
            # Forward Checking:
            # Are there any required diffs that are now IMPOSSIBLE to form?
            # A required diff d is impossible if:
            # 1. d is not in new_used_diffs_set
            # 2. And we cannot form it with future elements.
            # Future elements y > candidate.
            # d = y - x_i.
            # y = x_i + d.
            # Since y > candidate, we need x_i + d > candidate => x_i > candidate - d.
            # Also y <= max_val_bound => x_i + d <= max_val_bound => x_i <= max_val_bound - d.
            # So if there is NO x_i in elements such that candidate - d < x_i <= max_val_bound - d,
            # then d CANNOT be formed by any future y > candidate.
            # Wait, d could be formed by y - x_j where x_j is a FUTURE element too?
            # No, differences are between any pair.
            # If d is not formed now, it must be formed later.
            # Later pairs are (y, z) with y < z, both > candidate? No, one can be from current set.
            # If d is not in new_used_diffs_set, it must be formed by a pair (a, b) where b > a.
            # If b <= candidate, it would have been formed already (since elements are sorted and we check all pairs with candidate).
            # So b must be > candidate.
            # Then a = b - d.
            # Since a must be in the set, and a < b, a can be in current elements or future elements.
            # If a is in future elements, a > candidate.
            # Then b = a + d > candidate + d.
            
            # This check is complex. Let's stick to a simpler bound:
            # If a required diff d is not covered, and d < candidate - elements[0], 
            # it doesn't mean it's dead, but it means the "leftmost" part of the set is too sparse?
            
            # Let's just rely on the size bound and time limit.
            
            success, sol = backtrack(elements + (candidate,), new_used_diffs_set, new_req_mask, max_val_bound)
            if success and sol is not None:
                return (True, sol)
        
        return (False, None)
    
    best_solution = None
    best_max_val = 10**6
    
    initial_elements = (0,)
    initial_diffs = set()
    initial_mask = 0
    
    # Try increasing bounds
    # We can start higher if 40 is too tight, but 40 is a good lower bound for 32 elements covering 1..30?
    # Actually, min diameter for 32 elements Sidon is approx sqrt(32)*something?
    # For perfect difference sets, size k, diameter ~ k^2.
    # Here we only need 1..30.
    # Let's try bounds starting from 35.
    
    for bound in range(35, 100, 1):
        if time.time() > deadline:
            break
        success, sol = backtrack(initial_elements, initial_diffs, initial_mask, bound)
        if success and sol is not None:
            if sol[-1] < best_max_val:
                best_max_val = sol[-1]
                best_solution = sol
            break
    
    # Fallback
    if best_solution is None or best_max_val > 200:
        a = [0]
        for d in range(1, 31):
            if d in get_diffs_set(a):
                continue
            found = False
            for x in sorted(set(v + s for v in a for s in (d, -d) if v + s >= 0) - set(a)):
                trial = sorted(a + [x])
                if is_valid_sidon(trial):
                    a = trial
                    found = True
                    break
            if not found:
                y = 2 * max(a) + 3 * d + 1
                while True:
                    trial = sorted(a + [y, y + d])
                    if is_valid_sidon(trial):
                        a = trial
                        break
                    y += 1
        
        if covers_all_diffs(a) and len(a) <= 32 and max(a) < best_max_val:
            best_solution = tuple(a)
            best_max_val = max(a)
    
    if best_solution is not None:
        sol = list(best_solution)
        diffs = get_diffs_set(sol)
        if (len(diffs) == len(sol) * (len(sol) - 1) // 2 and 
            all(d in diffs for d in range(1, 31)) and
            0 in sol and
            len(sol) <= 32 and
            max(sol) <= 10**6):
            return sol
    
    return a

# EVOLVE-BLOCK-END
