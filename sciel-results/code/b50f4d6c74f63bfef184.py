"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import random

def construct():
    """
    Minimize the number of distinct color labels used on positions 1..128
    such that every non-constant 4-term arithmetic progression (a, a+d, a+2d, a+3d)
    uses at least 3 distinct colors.
    
    Strategy:
    1. Precompute all 4-term APs.
    2. Attempt to find a valid coloring using exactly 3 colors (0, 1, 2) via local search.
       A valid 3-coloring means every 4-AP contains all three colors.
    3. If 3 colors fail, attempt 4 colors.
    4. If all fail (unlikely), fall back to a greedy high-color solution.
    """
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    n = 128
    
    # Precompute all 4-term APs: (a, a+d, a+2d, a+3d) with 0-indexed
    ap_list = []
    for d in range(1, n // 3 + 1):
        for a in range(0, n - 3 * d):
            ap_list.append((a, a + d, a + 2 * d, a + 3 * d))
            
    num_aps = len(ap_list)
    
    # Map position to list of AP indices it belongs to
    pos_to_aps = [[] for _ in range(n)]
    for idx, ap in enumerate(ap_list):
        for pos in ap:
            pos_to_aps[pos].append(idx)
            
    def count_violations(colors):
        """
        Count the number of 4-APs that have fewer than 3 distinct colors.
        """
        violations = 0
        for ap in ap_list:
            c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
            # Check if distinct count is < 3
            # This is equivalent to: not (at least 3 distinct)
            # We can check: if len({c0,c1,c2,c3}) < 3: violations += 1
            # Optimization: 
            # If all 4 are same: bad
            # If 3 same, 1 diff: bad (2 distinct)
            # If 2,2: bad (2 distinct)
            # If 2,1,1: good (3 distinct)
            # If 1,1,1,1: good (4 distinct, impossible with 3 colors unless we allow >3, but we restrict to k colors)
            
            # Simple set check is fast enough for n=128 and ~1000 APs
            if len({c0, c1, c2, c3}) < 3:
                violations += 1
        return violations

    def try_find_coloring(k, time_limit):
        """
        Try to find a valid coloring using k colors (0..k-1).
        Returns list of colors if found, else None.
        """
        # Initialize randomly
        colors = np.random.randint(0, k, n)
        current_violations = count_violations(colors)
        
        if current_violations == 0:
            return colors.tolist()
            
        best_colors = colors.copy()
        best_violations = current_violations
        
        # Simulated Annealing
        T = 1.0
        cooling = 0.999
        min_T = 0.01
        max_steps = 100000
        
        steps = 0
        while time.time() < time_limit and T > min_T:
            steps += 1
            # Pick a random position
            pos = np.random.randint(0, n)
            old_color = colors[pos]
            
            # Pick a new color different from old
            new_color = np.random.randint(0, k)
            while new_color == old_color:
                new_color = np.random.randint(0, k)
                
            colors[pos] = new_color
            
            # Calculate delta violations
            # Instead of recounting all, we can just recount affected APs
            # But for n=128, full recount is ~1000 ops, which is fast.
            # Let's stick to full recount for simplicity and correctness, 
            # unless we optimize by tracking only APs involving 'pos'.
            
            new_violations = count_violations(colors)
            
            delta = new_violations - current_violations
            
            if delta <= 0 or np.random.rand() < np.exp(-delta / T):
                current_violations = new_violations
                if new_violations < best_violations:
                    best_violations = new_violations
                    best_colors = colors.copy()
                    if best_violations == 0:
                        return best_colors.tolist()
            else:
                # Revert
                colors[pos] = old_color
            
            T *= cooling
            
        return None

    # Try k=3 first
    # We know 2 is impossible.
    # Use a tighter time limit for k=3 to allow multiple restarts if needed?
    # Or just one long run?
    # Let's try a few restarts for k=3 if the first one fails quickly.
    
    result = None
    
    # Attempt 1 for k=3
    t_start = time.time()
    limit_per_attempt = 10.0
    result = try_find_coloring(3, t_start + limit_per_attempt)
    
    if result is not None:
        return result
        
    # If first attempt for k=3 fails, try again with different random seed
    # But we need to be careful with time.
    # Let's try k=3 for a total of 30s.
    
    while time.time() < start_time + 30.0:
        result = try_find_coloring(3, time.time() + 5.0)
        if result is not None:
            return result
            
    # If k=3 failed, try k=4
    # With 4 colors, the constraint "at least 3 distinct" is easier to satisfy.
    # Patterns like 0,1,2,0 are valid. 0,1,1,1 is invalid (2 distinct).
    # So we just need to avoid 4-APs with <= 2 distinct colors.
    
    result = try_find_coloring(4, time.time() + 15.0)
    if result is not None:
        return result
        
    # Fallback: Greedy construction with more colors
    # This is a safety net.
    colors = [-1] * n
    for i in range(n):
        # Try to assign smallest color that doesn't create an immediate violation
        # This greedy approach might use many colors, but it's a valid fallback.
        # Actually, the seed program's fallback was better.
        # Let's use the seed's fallback logic which is deterministic and known to work (though suboptimal).
        # But we want to minimize colors. 
        # If local search failed, it's likely because the structure is hard.
        # Let's just return a valid coloring with potentially more colors.
        
        # Simple greedy: assign color i % 128? No, that uses 128 colors.
        # We need to ensure validity.
        
        # Let's use the logic from the seed program's fallback as a last resort.
        # It constructs a valid coloring.
        
        pass

    # Re-implementing the seed fallback for safety
    colors = []
    for x in range(1, 129):
        found = False
        for color in range(128):
            # Check 4-term APs ending at x-1 (0-indexed)
            # x is 1-indexed. pos = x-1.
            pos = x - 1
            valid = True
            for d in range(1, (pos) // 3 + 1):
                # AP: (pos - 3d, pos - 2d, pos - d, pos)
                # But we only have colors for indices < pos.
                # So we check APs where pos is the LAST element?
                # The seed logic checked all APs involving the new position.
                
                # Let's just check all APs involving pos where other elements are assigned.
                pass
            
            # Actually, the seed logic was:
            # all(len({color} | {colors[x - 1 - j * d] for j in (1, 2, 3)}) >= 3 for d in range(1, (x - 1) // 3 + 1))
            # This checks APs ending at x-1.
            # AND
            # all(any(colors[x - 1 - j * d] != color for j in (1, 2)) for d in range(1, (x - 1) // 2 + 1))
            # This second part seems to check 3-term APs? Or part of 4-term?
            # "any(... != color ...)" ensures not all 3 previous are same as color?
            # If colors[pos-d] == color and colors[pos-2d] == color and colors[pos-3d] == color, then the AP (pos-3d, ..., pos) has 1 distinct color.
            # The first condition covers the 4-term AP ending at pos.
            # What about APs where pos is not the last element?
            # The seed logic is a bit opaque.
            
            # Let's use a simpler, guaranteed valid fallback: 
            # Use 4 colors in a pattern that is known to be valid?
            # Or just use the result of a very fast local search with k=10?
            
            if time.time() > deadline:
                break
                
            # Let's just try one more time with k=5 if we are here.
            if color == 0: 
                # We are in the fallback loop, let's break and do a k=5 search
                break
        
        if found:
            colors.append(color)
        else:
            # Fallback: assign a new unique color? No, max 128.
            # Assign 0? 
            colors.append(0)
            
    # This fallback is weak. Let's replace the entire fallback section with a robust k=5 search.
    
    # Reset and try k=5
    if time.time() < deadline - 5.0:
        result = try_find_coloring(5, deadline - 5.0)
        if result is not None:
            return result

    # Absolute last resort: return a random valid-ish coloring? 
    # No, must be valid.
    # If all else fails, return the best found so far? 
    # The function must return a valid coloring.
    # If we can't find one, we might return an invalid one and score 0.
    # But k=3 or k=4 should work.
    
    # If we reach here, something went wrong. Return a dummy valid coloring?
    # We can construct a valid 128-coloring easily (all distinct).
    # But we want to minimize.
    # Let's just return the best result from the k=4 attempt if it was close?
    # No, must be valid.
    
    # Let's assume k=3 or k=4 works.
    # If not, we return a list of 128 zeros? That is invalid.
    # We must ensure we return a valid list.
    
    # Let's use the greedy from the seed as the absolute final fallback, even if it uses many colors.
    # It is known to produce a valid coloring.
    
    colors = []
    for x in range(1, 129):
        for color in range(128):
            # Check condition 1: All 4-APs ending at x-1
            ok1 = True
            for d in range(1, (x - 1) // 3 + 1):
                idxs = [x - 1 - j * d for j in (1, 2, 3)]
                if any(i < 0 for i in idxs): continue
                s = {color}
                for i in idxs:
                    s.add(colors[i])
                if len(s) < 3:
                    ok1 = False
                    break
            if not ok1:
                continue
                
            # Check condition 2: All 4-APs where x-1 is NOT the last element?
            # The seed logic had a second condition.
            # Let's verify if just checking ending APs is sufficient for a left-to-right greedy.
            # No, because an AP like (0, 1, 2, 3) when assigning pos 2, we don't know pos 3 yet.
            # But when we assign pos 3, we check the AP (0,1,2,3).
            # So if we check all APs *ending* at the current position, we ensure that all APs are checked when their last element is assigned.
            # Is that sufficient?
            # Yes, every 4-AP has a unique last element (largest index).
            # So checking all APs ending at `pos` when assigning `pos` covers all APs exactly once.
            
            colors.append(color)
            break
            
    return colors

# EVOLVE-BLOCK-END
