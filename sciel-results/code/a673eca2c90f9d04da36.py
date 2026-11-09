import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 10s margin for safety
    
    n = 128
    
    # Precompute APs
    past_aps = [[] for _ in range(n)]
    third_aps = [[] for _ in range(n)]
    
    max_d = (n - 1) // 3
    for d in range(1, max_d + 1):
        for a in range(n - 3 * d):
            i1 = a
            i2 = a + d
            i3 = a + 2 * d
            i4 = a + 3 * d
            
            # i4 is the last term
            past_aps[i4].append((i1, i2, i3))
            
            # i3 is the third term
            # Store i1, i2, and i4 (the next index)
            third_aps[i3].append((i1, i2, i4))
            
    def is_valid_past(x, c, colors):
        """Check if assigning color c to x makes all past APs valid."""
        for i1, i2, i3 in past_aps[x]:
            c1 = colors[i1]
            c2 = colors[i2]
            c3 = colors[i3]
            if len({c1, c2, c3, c}) < 3:
                return False
        return True

    def can_complete_third(x, c, colors, K):
        """
        Check if assigning color c to x (3rd term) allows the 4th term to be satisfied.
        """
        for i1, i2, i4 in third_aps[x]:
            c1 = colors[i1]
            c2 = colors[i2]
            c3 = c
            
            distinct_vals = {c1, c2, c3}
            
            if len(distinct_vals) == 1:
                # 3 same colors, impossible to make size 3
                return False
            elif len(distinct_vals) == 2:
                # 2 distinct colors. Need a 3rd distinct color for c4.
                # Requires K >= 3
                if K < 3:
                    return False
            # else len == 3, always ok
            
        return True

    def try_greedy(K, randomize=False):
        colors = [-1] * n
        for x in range(n):
            if time.time() > deadline:
                return None
            
            valid_colors = []
            for c in range(K):
                if is_valid_past(x, c, colors) and can_complete_third(x, c, colors, K):
                    valid_colors.append(c)
            
            if not valid_colors:
                return None
            
            if randomize:
                colors[x] = random.choice(valid_colors)
            else:
                colors[x] = valid_colors[0]
        
        return colors

    # Try K=3, 4, 5...
    for K in range(3, 10):
        if time.time() > deadline:
            break
        
        # Try deterministic greedy first
        res = try_greedy(K, randomize=False)
        if res is not None:
            return res
        
        # If deterministic fails, try randomized greedy
        for restart in range(50):
            if time.time() > deadline:
                break
            res = try_greedy(K, randomize=True)
            if res is not None:
                return res

    # Fallback: Seed program logic
    colors = []
    for x in range(1, 129):
        for color in range(128):
            valid = True
            for d in range(1, (x - 1) // 3 + 1):
                if x - 1 - 3 * d < 0:
                    continue
                c0 = colors[x - 1 - 3 * d]
                c1 = colors[x - 1 - 2 * d]
                c2 = colors[x - 1 - d]
                c3 = color
                if len({c0, c1, c2, c3}) < 3:
                    valid = False
                    break
            if not valid:
                continue
            colors.append(color)
            break
        else:
            colors.append(0)
    return colors
