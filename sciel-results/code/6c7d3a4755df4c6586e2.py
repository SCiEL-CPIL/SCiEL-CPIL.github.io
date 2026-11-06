"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    n = 128
    # Precompute all 4-term arithmetic progressions
    # (a, b, c, d) such that b-a = c-b = d-c = d > 0
    # Indices are 0-based.
    ap_indices = []
    for d in range(1, (n - 3) // 3 + 1):
        for start in range(0, n - 3 * d):
            ap_indices.append((start, start + d, start + 2 * d, start + 3 * d))
    
    # Convert to numpy arrays for potential vectorization, or just keep as list of tuples
    # For n=128, ~2000 APs. A python loop over 2000 tuples is fast enough.
    
    def count_violations(colors):
        """Count 4-APs with < 3 distinct colors."""
        v = 0
        # Using a local variable for speed
        for i1, i2, i3, i4 in ap_indices:
            c1 = colors[i1]
            c2 = colors[i2]
            c3 = colors[i3]
            c4 = colors[i4]
            # Check if < 3 distinct
            # Fast check:
            # If c1==c2==c3==c4 -> 1 distinct (violation)
            # If (c1==c2 and c3==c4) and c1!=c3 -> 2 distinct (violation)
            # If (c1==c3 and c2==c4) and c1!=c2 -> 2 distinct (violation)
            # If (c1==c4 and c2==c3) and c1!=c2 -> 2 distinct (violation)
            # If 3 same, 1 diff -> 2 distinct (violation)
            # It is faster to just use len(set()) for 4 elements?
            # len(set([c1,c2,c3,c4])) creates a set. 
            # Manual check might be faster in pure python for 4 items.
            if c1 == c2:
                if c3 == c4:
                    if c1 == c3: return v + 1 # All same
                    else: return v + 1 # 2 distinct (pairs)
                else:
                    if c3 == c1 or c4 == c1: return v + 1 # 2 distinct (3 same)
                    else: return v # 3 or 4 distinct? 
                    # If c1==c2, c3!=c4.
                    # Distincts: {c1, c3, c4}.
                    # If c3==c1 or c4==c1, then {c1, c3} (2 distinct) -> Violation
                    # If c3!=c1 and c4!=c1, then {c1, c3, c4} (3 distinct) -> OK
            # ... This manual check is error-prone. 
            # Let's just use the set. It's C-optimized.
            if len({c1, c2, c3, c4}) < 3:
                v += 1
        return v

    def reduce_colors(colors):
        """Try to reduce the number of distinct colors used, maintaining validity."""
        current_colors = list(colors)
        distinct = set(current_colors)
        
        # Try to merge each color into another existing color
        # We want to minimize the count, so we try to eliminate the least used color?
        # Or just try to eliminate any color.
        # Let's try to eliminate color K if it exists, by mapping it to 0 or 1.
        
        # First, normalize labels to 0..K-1
        unique_colors = sorted(list(distinct))
        mapping = {c: i for i, c in enumerate(unique_colors)}
        current_colors = [mapping[c] for c in current_colors]
        K = len(unique_colors)
        
        best_colors = current_colors
        best_K = K
        
        # Try to reduce K by 1
        # For each color 'target' to eliminate, try mapping it to 'source'
        for target in range(K):
            for source in range(target):
                new_colors = [source if c == target else (c if c < target else c - 1) for c in current_colors]
                # Note: if we map target to source, and target > source, 
                # other colors > target shift down by 1.
                # This effectively merges target into source.
                
                if count_violations(new_colors) == 0:
                    new_K = K - 1
                    if new_K < best_K:
                        best_K = new_K
                        best_colors = new_colors
                        current_colors = new_colors
                        K = new_K
                        # Restart the loop as K changed
                        break
            if K < best_K:
                break
        
        # Map back to 0..best_K-1 if needed (they are already)
        return best_colors

    best_solution = None
    best_score = float('inf')
    
    # Strategy: 
    # 1. Find ANY valid coloring (minimize violations).
    # 2. Then minimize distinct colors.
    
    # Use Randomized Local Search (Hill Climbing with Random Restarts)
    # Objective: violations * 10000 + distinct_count
    
    for attempt in range(100): # Max 100 restarts
        if time.time() > deadline:
            break
            
        rng = random.Random(attempt)
        # Start with random 3-coloring (since 3 is likely optimal)
        # If 3 is not possible, we might need 4. 
        # Let's start with random 4-coloring to be safe, then try to reduce.
        # Or start with 3 and if stuck, try 4.
        
        # Let's try K=3 first.
        K_init = 3
        colors = [rng.randint(0, K_init - 1) for _ in range(n)]
        
        current_violations = count_violations(colors)
        
        # Hill Climbing
        for step in range(20000):
            if time.time() > deadline:
                break
                
            if current_violations == 0:
                break
                
            # Pick random position
            pos = rng.randint(0, n - 1)
            old_c = colors[pos]
            
            # Try a new color
            new_c = rng.randint(0, K_init - 1)
            if new_c == old_c:
                continue
                
            colors[pos] = new_c
            new_violations = count_violations(colors)
            
            if new_violations < current_violations:
                current_violations = new_violations
            else:
                colors[pos] = old_c
                
        # If we found a valid 3-coloring, try to reduce it (it's already 3, so maybe it uses <3? No, random 3 uses all 3 usually).
        # Check distinct colors
        if current_violations == 0:
            distinct = len(set(colors))
            # Try to reduce distinct colors
            reduced = reduce_colors(colors)
            final_distinct = len(set(reduced))
            
            score = final_distinct
            if score < best_score:
                best_score = score
                best_solution = reduced
                
            if best_score == 3: # Can't do better than 3 for non-trivial APs? 
                # Actually, can we do 2? 
                # For n=128, 2 colors are NOT enough. (R(4,4) is large, but for 4-APs, 2 colors allow monochromatic 4-APs easily? 
                # No, the condition is "AT LEAST 3 distinct". 
                # If we use 2 colors, any 4-AP can have at most 2 distinct colors. 
                # So 2 colors ALWAYS violate the condition "AT LEAST 3 distinct". 
                # Therefore, the minimum number of colors is 3.
                return best_solution

    # If no 3-coloring found, try 4
    if best_solution is None:
        for attempt in range(100):
            if time.time() > deadline:
                break
                
            rng = random.Random(1000 + attempt)
            K_init = 4
            colors = [rng.randint(0, K_init - 1) for _ in range(n)]
            
            current_violations = count_violations(colors)
            
            for step in range(20000):
                if time.time() > deadline:
                    break
                if current_violations == 0:
                    break
                    
                pos = rng.randint(0, n - 1)
                old_c = colors[pos]
                new_c = rng.randint(0, K_init - 1)
                if new_c == old_c:
                    continue
                    
                colors[pos] = new_c
                new_violations = count_violations(colors)
                
                if new_violations < current_violations:
                    current_violations = new_violations
                else:
                    colors[pos] = old_c
                    
            if current_violations == 0:
                reduced = reduce_colors(colors)
                final_distinct = len(set(reduced))
                if final_distinct < best_score:
                    best_score = final_distinct
                    best_solution = reduced
                if best_score == 3:
                    return best_solution

    # Fallback: Return best found or a trivial valid solution (128 colors)
    if best_solution is None:
        # Trivial solution: all distinct
        best_solution = list(range(128))
    return best_solution
# EVOLVE-BLOCK-END
