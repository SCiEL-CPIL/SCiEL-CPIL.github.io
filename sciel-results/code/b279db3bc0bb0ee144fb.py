import time

def construct():
        start_time = time.time()
        deadline = start_time + 85.0 # 5s margin
        
        # Precompute AP constraints
        # For each x (1-based), list of (i1, i2, i3) 0-based indices
        # such that x, i1+1, i2+1, i3+1 form a 4-AP?
        # No, x is the LAST element.
        # AP: a, a+d, a+2d, a+3d = x.
        # a = x - 3d.
        # Indices: a-1, a+d-1, a+2d-1, x-1.
        
        constraints = [[] for _ in range(128)]
        for x in range(1, 129):
            for d in range(1, (x - 1) // 3 + 1):
                a = x - 3 * d
                if a < 1:
                    break
                # Indices 0-based: a-1, a+d-1, a+2d-1
                i1 = a - 1
                i2 = a + d - 1
                i3 = a + 2 * d - 1
                constraints[x - 1].append((i1, i2, i3))
        
        def try_construct(max_c):
            colors = [None] * 128
            used = set()
            for x in range(128):
                if time.time() > deadline:
                    return None
                valid_colors = []
                for c in range(max_c):
                    # Check constraints
                    ok = True
                    for i1, i2, i3 in constraints[x]:
                        c1, c2, c3 = colors[i1], colors[i2], colors[i3]
                        # Set of 4 colors
                        # We need len >= 3
                        # Fast check:
                        # If c1==c2==c3:
                        #   then {c1, c} must have size >= 3 -> impossible since only 2 values.
                        #   So if c1==c2==c3, NO c works?
                        #   Wait. If c1==c2==c3=A, then set is {A, c}. Size is 1 or 2.
                        #   So if any constraint has 3 same colors, it's IMPOSSIBLE to satisfy with any 4th color?
                        #   Yes. {A, A, A, c} has size 1 (if c=A) or 2 (if c!=A).
                        #   So if we ever create a situation where 3 terms of an AP are same, we are stuck.
                        #   This means the greedy must avoid creating 3 same in a row (in AP sense) earlier.
                        
                        # Let's just compute set size.
                        if len({c1, c2, c3, c}) < 3:
                            ok = False
                            break
                    if ok:
                        valid_colors.append(c)
                
                if not valid_colors:
                    return None
                
                # Heuristic: prefer used colors
                # Partition valid_colors into used and new
                # Since max_c is small, just iterate
                # We want to minimize distinct count.
                # If we pick a color already in `used`, count doesn't increase.
                # If we pick new, it increases.
                
                # Sort valid_colors:
                # Key: (0 if c in used else 1, c)
                # This picks smallest used color first, then smallest new.
                
                valid_colors.sort(key=lambda c: (0 if c in used else 1, c))
                
                chosen = valid_colors[0]
                colors[x] = chosen
                used.add(chosen)
                
            return colors

        # Try 3 colors
        res = try_construct(3)
        if res:
            return res
            
        # Try 4 colors
        res = try_construct(4)
        if res:
            return res
            
        # Try 5 colors
        res = try_construct(5)
        if res:
            return res
            
        # ... up to some limit
        for mc in range(6, 12):
            if time.time() > deadline:
                break
            res = try_construct(mc)
            if res:
                return res
        
        # Fallback: Greedy without max_c limit (like seed but improved)
        # Or just return a valid solution if we found one with higher colors
        # If all failed, we might need backtracking. 
        # But 3 or 4 should work.
        
        # If we reach here, we failed. Return a dummy valid sequence?
        # We must return a valid sequence.
        # Let's try a randomized greedy with backtracking if simple fails.
        # But for now, assume 3 or 4 works.
        
        # Safety net: Use the seed logic if all else fails (it's known to be valid, score 1.0)
        colors=[]
        for x in range(1,129):
            for color in range(128):
                if (all(len({color}|{colors[x-1-j*d] for j in (1,2,3)})>=3 for d in range(1,(x-1)//3+1))
                    and all(any(colors[x-1-j*d]!=color for j in (1,2)) for d in range(1,(x-1)//2+1))):
                    colors.append(color);break
        return colors
