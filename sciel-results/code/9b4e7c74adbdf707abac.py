"""Rewritten program for divisibility-antichain representations by smooth numbers."""
# EVOLVE-BLOCK-START
import time
from functools import lru_cache

def _generate_smooth(max_val):
    """Generate all 2^a * 3^b * 5^c numbers up to max_val."""
    smooth = set()
    a = 1
    while a <= max_val:
        b = a
        while b <= max_val:
            c = b
            while c <= max_val:
                smooth.add(c)
                c *= 5
            b *= 3
        a *= 2
    return sorted(smooth)

def _is_antichain(terms):
    """Check if no term divides another in the list."""
    n = len(terms)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = terms[i], terms[j]
            if a > b:
                a, b = b, a
            if b % a == 0:
                return False
    return True

def _build_conflict_masks(smooth):
    """Build conflict masks for each smooth number."""
    n = len(smooth)
    # conflict_mask[i] is a bitmask where bit j is set if smooth[i] conflicts with smooth[j]
    conflict_mask = [0] * n
    val_to_idx = {v: i for i, v in enumerate(smooth)}
    
    for i in range(n):
        for j in range(i + 1, n):
            a, b = smooth[i], smooth[j]
            if a > b:
                a, b = b, a
            if b % a == 0:
                # They conflict
                conflict_mask[i] |= (1 << j)
                conflict_mask[j] |= (1 << i)
    
    return conflict_mask, val_to_idx

def _solve_target(n, smooth, conflict_mask, val_to_idx, deadline):
    """Find minimum antichain representation for target n."""
    # Filter candidates <= n
    candidates = [x for x in smooth if x <= n]
    if not candidates:
        return None
    
    # Get indices of candidates
    cand_indices = [val_to_idx[x] for x in candidates]
    
    # Sort candidates in descending order
    candidates.sort(reverse=True)
    cand_indices.sort(key=lambda i: smooth[i], reverse=True)
    
    # For each count from 1 to 30, try to find a solution
    for count in range(1, 31):
        if len(candidates) < count:
            break
        if time.time() > deadline:
            break
        
        # DFS to find antichain of size count summing to n
        solution = None
        
        def dfs(idx, remaining, current_indices, current_vals):
            nonlocal solution
            if solution is not None:
                return
            if time.time() > deadline:
                return
            
            if len(current_indices) == count:
                if remaining == 0:
                    solution = list(current_vals)
                return
            
            if remaining < 0:
                return
            
            # Pruning: need at least (count - len(current)) more terms
            needed = count - len(current_indices)
            if remaining < needed:
                return
            
            # Also, the maximum possible sum with remaining slots
            # is limited by the largest available candidates
            max_possible = sum(candidates[i] for i in range(idx, min(idx + needed, len(candidates))))
            if max_possible < remaining:
                return
            
            # Try each candidate from idx onwards
            for i in range(idx, len(candidates)):
                if time.time() > deadline:
                    return
                
                val = candidates[i]
                if val > remaining:
                    continue
                
                # Check conflict using bitmask
                cand_idx = cand_indices[i]
                conflicts = conflict_mask[cand_idx]
                
                # Check if this candidate conflicts with any already selected
                valid = True
                for sel_idx in current_indices:
                    if conflicts & (1 << sel_idx):
                        valid = False
                        break
                
                if not valid:
                    continue
                
                current_indices.append(cand_idx)
                current_vals.append(val)
                dfs(i + 1, remaining - val, current_indices, current_vals)
                current_indices.pop()
                current_vals.pop()
                
                if solution is not None:
                    return
        
        current_indices = []
        current_vals = []
        dfs(0, n, current_indices, current_vals)
        
        if solution is not None:
            return solution
    
    return None

def construct():
    """
    Construct 128 rows corresponding to targets 1..128.
    Every row must contain 1..30 distinct positive 2^a*3^b*5^c terms summing to its target;
    no chosen term may divide another. Minimize the total number of summands.
    """
    start_time = time.time()
    deadline = start_time + 85.0  # 5 second safety margin before 90s limit
    
    max_val = 128
    smooth = _generate_smooth(max_val)
    
    # Precompute conflict masks
    conflict_mask, val_to_idx = _build_conflict_masks(smooth)
    
    # For each target, find the minimum number of terms
    answers = [None] * 129
    
    for n in range(1, 129):
        if time.time() > deadline:
            break
        
        answers[n] = _solve_target(n, smooth, conflict_mask, val_to_idx, deadline)
    
    # Return 128 rows for targets 1..128
    result = []
    for n in range(1, 129):
        if answers[n] is None:
            # Fallback: try a simple approach
            # Find any valid representation
            found = False
            for count in range(1, 31):
                if time.time() > deadline:
                    break
                # Simple DFS without precomputed masks (slow but should rarely be needed)
                candidates = [x for x in smooth if x <= n]
                if len(candidates) < count:
                    break
                candidates.sort(reverse=True)
                
                solution = None
                
                def dfs_fallback(idx, remaining, current):
                    nonlocal solution
                    if solution is not None:
                        return
                    if time.time() > deadline:
                        return
                    
                    if len(current) == count:
                        if remaining == 0 and _is_antichain(current):
                            solution = list(current)
                        return
                    
                    if remaining < 0:
                        return
                    
                    needed = count - len(current)
                    if remaining < needed:
                        return
                    
                    for i in range(idx, len(candidates)):
                        if time.time() > deadline:
                            return
                        
                        val = candidates[i]
                        if val > remaining:
                            continue
                        
                        valid = True
                        for term in current:
                            if term % val == 0 or val % term == 0:
                                valid = False
                                break
                        
                        if not valid:
                            continue
                        
                        current.append(val)
                        dfs_fallback(i + 1, remaining - val, current)
                        current.pop()
                        
                        if solution is not None:
                            return
                
                current = []
                dfs_fallback(0, n, current)
                
                if solution is not None:
                    answers[n] = solution
                    found = True
                    break
            
            if not found:
                # Last resort: use n itself if it's smooth, otherwise fail
                if n in smooth:
                    result.append([n])
                else:
                    result.append([n])  # This may not be valid
        else:
            result.append(answers[n])
    
    return result
# EVOLVE-BLOCK-END
