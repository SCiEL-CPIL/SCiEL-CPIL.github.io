import time
import random

def _validate(colors):
    """Check that every nonconstant 4-term AP uses at least 3 distinct colors."""
    n = len(colors)
    for i in range(n):
        for d in range(1, (n - 1 - i) // 3 + 1):
            j = i + d
            k = i + 2 * d
            l = i + 3 * d
            if l >= n:
                break
            if len({colors[i], colors[j], colors[k], colors[l]}) < 3:
                return False
    return True

def _count_colors(colors):
    """Return the number of distinct color labels used."""
    return len(set(colors))

def _greedy_base():
    """
    Build a deterministic base coloring using the smallest valid color at each step.
    This gives a valid coloring that may use more colors than optimal.
    """
    n = 128
    colors = [0] * n
    for i in range(n):
        pos = i + 1
        valid = True
        # Try colors 0, 1, 2, ... until we find one that works
        for c in range(128):
            ok = True
            # Check all 4-APs where pos is the last element: pos, pos-d, pos-2d, pos-3d
            max_d = (pos - 1) // 3
            for d in range(1, max_d + 1):
                i1 = pos - 3 * d - 1  # 0-indexed
                i2 = pos - 2 * d - 1
                i3 = pos - d - 1
                if len({colors[i1], colors[i2], colors[i3], c}) < 3:
                    ok = False
                    break
            # Also check 4-APs where pos is not the last element
            # pos, pos+d, pos+2d, pos+3d
            max_d2 = (n - pos) // 3
            for d in range(1, max_d2 + 1):
                i1 = pos + d - 1
                i2 = pos + 2 * d - 1
                i3 = pos + 3 * d - 1
                if len({c, colors[i1], colors[i2], colors[i3]}) < 3:
                    ok = False
                    break
            # Check 4-APs where pos is the 2nd element: pos-3d, pos-2d, pos, pos+d
            max_d3 = (pos - 1) // 3
            for d in range(1, max_d3 + 1):
                i1 = pos - 3 * d - 1
                i2 = pos - 2 * d - 1
                i4 = pos + d - 1
                if pos + d - 1 < n:
                    if len({colors[i1], colors[i2], c, colors[i4]}) < 3:
                        ok = False
                        break
            # Check 4-APs where pos is the 3rd element: pos-3d, pos-2d, pos-d, pos
            # Already covered above (pos as last)
            
            if ok:
                colors[i] = c
                break
        else:
            # Should not happen
            colors[i] = 0
    return colors

def _local_search(colors, deadline):
    """
    Local search via simulated annealing to minimize the number of distinct colors.
    We allow temporary violations but penalize them heavily.
    """
    n = 128
    best = list(colors)
    best_score = _count_colors(best)
    current = list(colors)
    current_score = _count_colors(current)
    
    # Precompute all 4-APs for fast evaluation
    aps = []
    for i in range(n):
        for d in range(1, (n - 1 - i) // 3 + 1):
            j = i + d
            k = i + 2 * d
            l = i + 3 * d
            if l < n:
                aps.append((i, j, k, l))
    
    def evaluate(arr):
        """Return (violation_count, distinct_colors)."""
        violations = 0
        for (i, j, k, l) in aps:
            if len({arr[i], arr[j], arr[k], arr[l]}) < 3:
                violations += 1
        return violations, len(set(arr))
    
    cur_viol, cur_dist = evaluate(current)
    best_viol, best_dist = evaluate(best)
    
    # We want to minimize distinct colors, but must have 0 violations.
    # Score = violations * 1000 + distinct
    cur_score = cur_viol * 1000 + cur_dist
    best_score_full = best_viol * 1000 + best_dist
    
    T = 50.0
    T_min = 0.01
    cooling = 0.999
    
    while time.time() < deadline:
        if T < T_min:
            break
            
        # Pick a random position
        i = random.randint(0, n - 1)
        old_color = current[i]
        
        # Try to change to a different color
        # Choose from a small palette to encourage consolidation
        max_color = max(current) + 1
        if max_color > 10:
            max_color = 10
        
        new_color = random.randint(0, max_color - 1)
        if new_color == old_color:
            continue
            
        current[i] = new_color
        new_viol, new_dist = evaluate(current)
        new_score = new_viol * 1000 + new_dist
        
        delta = new_score - cur_score
        
        if delta < 0 or random.random() < pow(2.718281828, -delta / T):
            cur_score = new_score
            cur_viol = new_viol
            cur_dist = new_dist
            if new_score < best_score_full:
                best_score_full = new_score
                best = list(current)
        else:
            current[i] = old_color
            
        T *= cooling
        
    # Return the best valid solution found, or best overall if none valid
    if best_viol == 0:
        return best, _count_colors(best)
    else:
        # Try to repair: for each violating AP, try to fix it
        for _ in range(100):
            if time.time() > deadline:
                break
            # Find a violating AP
            violating = None
            for (i, j, k, l) in aps:
                if len({best[i], best[j], best[k], best[l]}) < 3:
                    violating = (i, j, k, l)
                    break
            if violating is None:
                break
            i, j, k, l = violating
            # Try changing one of the positions
            for pos in [i, j, k, l]:
                for c in range(10):
                    if c == best[pos]:
                        continue
                    best[pos] = c
                    if _validate(best):
                        return best, _count_colors(best)
                    best[pos] = best[pos]  # revert
        return best, _count_colors(best)

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    # Step 1: Generate a base coloring
    base = _greedy_base()
    
    # If base is already valid, great
    if not _validate(base):
        # Fallback: use the seed algorithm
        colors = []
        for x in range(1, 129):
            for color in range(128):
                if (all(len({color} | {colors[x-1-j*d] for j in (1,2,3)}) >= 3 for d in range(1, (x-1)//3+1))
                    and all(any(colors[x-1-j*d] != color for j in (1,2)) for d in range(1, (x-1)//2+1))):
                    colors.append(color)
                    break
        if _validate(colors):
            return colors
        # If even the seed fails, return base (shouldn't happen)
        return base
    
    best_colors = base
    best_count = _count_colors(base)
    
    # Step 2: Local search to reduce number of distinct colors
    # Run multiple restarts with different initial perturbations
    remaining = deadline - start_time
    per_restart = min(remaining / 5, 15.0)
    
    for restart in range(5):
        if time.time() > deadline:
            break
            
        # Perturb the base
        perturbed = list(base)
        for i in range(128):
            if random.random() < 0.3:
                perturbed[i] = random.randint(0, 5)
        
        # Ensure perturbed is valid by repairing
        if not _validate(perturbed):
            # Try to repair
            for i in range(128):
                for c in range(10):
                    perturbed[i] = c
                    if _validate(perturbed):
                        break
                if _validate(perturbed):
                    break
            if not _validate(perturbed):
                # Skip this restart
                continue
        
        # Run local search
        result, count = _local_search(perturbed, deadline)
        if count < best_count and _validate(result):
            best_colors = result
            best_count = count
    
    return best_colors
