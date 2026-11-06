import time
import random
import sys

# Precompute popcount for 0..255 to speed up mask checks
POP_COUNT = [0] * 256
for i in range(256):
    POP_COUNT[i] = bin(i).count('1')

# Precompute APs ending at each index (0-indexed for internal array)
# AP is (a, b, c, d) with d = i.
# a, b, c, d are 0-indexed.
# d = 3d_step + a => a = d - 3*step
# b = 2d_step + a = d - step
# c = d_step + a = d - 2*step
# Step must be >= 1.
# a >= 0 => d - 3*step >= 0 => step <= d // 3

APS_ENDING_AT = [[] for _ in range(128)]
for d in range(128):
    for step in range(1, d // 3 + 1):
        a = d - 3 * step
        b = d - 2 * step
        c = d - step
        # Store indices a, b, c
        APS_ENDING_AT[d].append((a, b, c))

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    n = 128
    
    # We prioritize finding a solution with K=3, then K=4, etc.
    # The minimum possible is 3.
    
    for K in range(3, 12):
        # Time budget per K: give more time to smaller K initially?
        # Let's just run as many restarts as possible until deadline or success.
        # If K=3 is feasible, we want to find it fast.
        
        # Number of restarts depends on K. Smaller K is harder, maybe need more?
        # Or just fixed large number?
        # Let's use a dynamic limit based on remaining time.
        
        found = False
        
        # For K=3, do extensive search. For larger K, it's easier, so fewer restarts needed.
        # But we want the MINIMUM K, so we must exhaust reasonable search for K=3 before trying K=4.
        
        max_restarts = 20000 if K == 3 else 5000
        
        for r in range(max_restarts):
            if time.time() > deadline:
                break
            
            rng = random.Random(r)
            colors = [-1] * n
            success = True
            
            for i in range(n):
                # Determine valid colors for position i
                # Valid if for all APs ending at i, the condition holds.
                
                # We can pre-filter the palette {0..K-1}
                valid_colors = list(range(K))
                
                # Check constraints from APs ending at i
                for a, b, c in APS_ENDING_AT[i]:
                    # Get colors of a, b, c
                    ca = colors[a]
                    cb = colors[b]
                    cc = colors[c]
                    
                    # Create mask of existing colors
                    mask = (1 << ca) | (1 << cb) | (1 << cc)
                    
                    # Filter valid_colors
                    # A color x is valid if POP_COUNT(mask | (1 << x)) >= 3
                    # We can optimize this:
                    # If POP_COUNT(mask) >= 3, all colors are valid (for this AP).
                    # If POP_COUNT(mask) == 2, colors in the mask are invalid? No.
                    # Let's just check. K is small.
                    
                    new_valid = []
                    for x in valid_colors:
                        if POP_COUNT[mask | (1 << x)] >= 3:
                            new_valid.append(x)
                    valid_colors = new_valid
                    
                    if not valid_colors:
                        success = False
                        break
                        
                if not success:
                    break
                    
                if not valid_colors:
                    success = False
                    break
                    
                # Choose a color
                # Random choice is fine, but maybe bias towards balancing?
                # Random is usually robust.
                colors[i] = rng.choice(valid_colors)
                
            if success:
                # Found a valid coloring with K colors
                return colors
        
        # If loop finished without success (either max_restarts or time), try next K
        if time.time() > deadline:
            break
            
    # Fallback: Deterministic greedy (Seed algorithm)
    # This ensures we return *something* valid, even if it uses many colors.
    colors = []
    for x in range(1, 129):
        for color in range(128):
            # Check 4-APs ending at x
            valid = True
            for d in range(1, (x-1)//3 + 1):
                i1 = x - 3*d
                i2 = x - 2*d
                i3 = x - d
                # Indices in colors list are 0-indexed, so x-1 is current pos
                # i1-1, i2-1, i3-1 are previous
                c1 = colors[i1-1]
                c2 = colors[i2-1]
                c3 = colors[i3-1]
                if len({c1, c2, c3, color}) < 3:
                    valid = False
                    break
            
            # Also need to check if this assignment creates issues for FUTURE?
            # The seed algorithm only checks past constraints.
            # Is that sufficient?
            # The seed algorithm:
            # all(len({color} | {colors[x-1-j*d] for j in (1,2,3)})>=3 for d in range(1,(x-1)//3+1))
            # and all(any(colors[x-1-j*d] != color for j in (1,2)) for d in range(1,(x-1)//2+1))
            # The second part `any(... != color ...)` for 2-term APs?
            # Wait, the task is 4-term APs.
            # Why does the seed check 2-term APs?
            # "Every nonconstant four-term progression ... must use AT LEAST three distinct colors."
            # If a 4-AP has colors A, A, A, B -> distinct {A, B} size 2. Invalid.
            # If a 4-AP has colors A, B, A, B -> distinct {A, B} size 2. Invalid.
            # The seed's second condition: `any(colors[x-1-j*d] != color for j in (1,2))`
            # This checks if color at x is different from x-d or x-2d.
            # This seems to prevent patterns like A, A, A, A or A, B, A, B?
            # Actually, if we just satisfy the 4-AP constraint greedily, we might paint a 4-AP as A, B, A, A.
            # Distinct {A, B} size 2. Invalid.
            # My greedy check `len({c1, c2, c3, color}) >= 3` handles this.
            # The seed's second condition is likely a heuristic to avoid getting stuck or a specific property of the seed's specific search space.
            # I will stick to my robust 4-AP check.
            
            if valid:
                colors.append(color)
                break
        else:
            # Should not happen if we have enough colors, but just in case
            colors.append(0)
            
    return colors

