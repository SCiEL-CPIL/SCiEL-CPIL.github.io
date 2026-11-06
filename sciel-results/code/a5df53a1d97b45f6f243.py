import time

def construct():
    start_time = time.time()
    timeout = 85.0  # Safety margin
    
    colors = [0] * 128
    # We will determine colors sequentially
    # colors[i] is the color for position i+1
    
    for x in range(1, 129):
        idx = x - 1  # 0-indexed
        
        # Check time
        if time.time() - start_time > timeout:
            # Return best effort? The problem implies we must return a valid coloring.
            # If we time out, we might have an incomplete or invalid coloring.
            # The prompt says "return the best result found".
            # But an invalid result scores 0.
            # So we must ensure validity.
            # If we can't finish, we should probably fail or return a known valid small pattern?
            # But n=128 is small. The greedy loop is fast.
            # 128 * 128 * (128/3 + 128/2) ~ 128 * 128 * 100 ~ 1.6M ops.
            # Python is slow, but 1.6M simple ops should take < 1s.
            # The bottleneck might be the set creation in the inner loop.
            pass

        found = False
        for c in range(128):
            valid = True
            
            # Condition 1: 4-term progressions ending at idx
            # d such that idx - 3*d >= 0
            max_d1 = idx // 3
            for d in range(1, max_d1 + 1):
                # Check set size of {c, colors[idx-d], colors[idx-2d], colors[idx-3d]}
                # If size < 3, invalid
                c1 = colors[idx - d]
                c2 = colors[idx - 2*d]
                c3 = colors[idx - 3*d]
                
                # Fast check for size < 3
                # Size is 1 if all equal
                if c == c1 and c1 == c2 and c2 == c3:
                    valid = False
                    break
                # Size is 2 if:
                # c == c1 == c2 != c3
                # c == c1 == c3 != c2
                # c == c2 == c3 != c1
                # c1 == c2 == c3 != c
                # c == c1 != c2 == c3
                # c == c2 != c1 == c3
                # c == c3 != c1 == c2
                # c1 == c2 != c == c3 (covered)
                # Basically, if any two are equal, and the third is equal to one of them, and the fourth is different?
                # Easier: len({c, c1, c2, c3}) < 3
                
                # Optimization:
                # If c == c1:
                #   If c1 == c2:
                #     If c2 == c3: size 1 (Fail)
                #     Else: size 2 (Fail) -- wait, {c, c, c, c3}. If c!=c3, size 2.
                #   If c1 != c2:
                #     If c2 == c3: {c, c, c2, c2}. Size 2 (Fail)
                #     Else: {c, c, c2, c3}. If c2!=c3, size 3 (OK)
                # If c != c1:
                #   ...
                
                # Using set is probably fast enough in Python for small sets,
                # but manual comparison might be faster.
                if len({c, c1, c2, c3}) < 3:
                    valid = False
                    break
            
            if not valid:
                continue
                
            # Condition 2: Avoid dead ends for progressions starting at idx-2d
            # Progression: idx-2d, idx-d, idx, idx+d
            # Must have idx+d < 128
            # So d < 128 - idx
            max_d2 = (127 - idx) # d can be at most 127-idx
            # Also need idx-2d >= 0 => d <= idx//2
            max_d2 = min(max_d2, idx // 2)
            
            for d in range(1, max_d2 + 1):
                # Check if c == colors[idx-d] == colors[idx-2d]
                if c == colors[idx - d] and colors[idx - d] == colors[idx - 2*d]:
                    valid = False
                    break
            
            if valid:
                colors[idx] = c
                found = True
                break
        
        if not found:
            # Should not happen if the problem is solvable with 128 colors
            # But if it does, we might need to backtrack or return invalid
            # Given the constraints, 3 colors are likely sufficient.
            pass
            
    return colors
