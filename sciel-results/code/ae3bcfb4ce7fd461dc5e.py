import time
import random

def construct():
    n = 128
    
    def check_ap(c1, c2, c3, c4):
        return len({c1, c2, c3, c4}) >= 3

    def solve(k, timeout=5.0):
        colors = [-1] * n
        start_time = time.time()
        nodes = 0
        limit = 1000000 # Node limit
        
        def backtrack(pos):
            nonlocal nodes, start_time, limit
            nodes += 1
            if time.time() - start_time > timeout:
                return False
            if nodes > limit:
                return False
                
            if pos == n:
                return True
            
            # Heuristic: Try colors in a random order? Or 0..k-1?
            # Random can help escape bad local structures.
            # But deterministic is reproducible.
            # Let's try 0..k-1.
            
            for c in range(k):
                colors[pos] = c
                
                valid = True
                
                # Check APs ending at pos (pos is 4th term)
                # a+3d = pos
                for d in range(1, (pos + 1) // 3 + 1):
                    a = pos - 3 * d
                    if a < 0:
                        break
                    if colors[a] != -1 and colors[a+d] != -1 and colors[a+2*d] != -1:
                        if not check_ap(colors[a], colors[a+d], colors[a+2*d], colors[pos]):
                            valid = False
                            break
                            
                if not valid:
                    colors[pos] = -1
                    continue
                    
                # Check APs where pos is 3rd term (a+2d = pos)
                # a = pos - 2d, 4th = pos + d
                for d in range(1, (pos + 1) // 2 + 1):
                    a = pos - 2 * d
                    if a < 0:
                        break
                    t4 = pos + d
                    if t4 >= n:
                        continue
                    if colors[a] != -1 and colors[a+d] != -1:
                        # a, a+d, pos are assigned
                        if colors[a] == colors[a+d] == colors[pos]:
                            valid = False
                            break
                            
                if not valid:
                    colors[pos] = -1
                    continue
                    
                # Check APs where pos is 2nd term (a+d = pos)
                # a = pos - d, 3rd = pos + d, 4th = pos + 2d
                # Only 2 terms assigned (a, pos). No immediate mono-3 conflict.
                
                # Check APs where pos is 1st term.
                # No previous terms.
                
                if backtrack(pos + 1):
                    return True
                    
                colors[pos] = -1
                
            return False

        if backtrack(0):
            return colors
        return None

    # Try k=3, 4, 5, 6
    for k in [3, 4, 5, 6]:
        res = solve(k)
        if res is not None:
            # Verify result
            valid = True
            for a in range(n):
                for d in range(1, n):
                    if a + 3*d >= n:
                        break
                    if not check_ap(res[a], res[a+d], res[a+2*d], res[a+3*d]):
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                return res
            else:
                # Should not happen if backtrack is correct
                pass

    # Fallback: Use a simple pattern that is likely valid with more colors
    # e.g., c_i = i % 5
    # Check if i%5 works?
    # d=5: 0,0,0,0 -> Invalid.
    # We need a pattern that avoids monochromatic APs.
    # c_i = (i // 5) % 2 + 5 * (i % 5) ? No.
    
    # Just return a coloring with 128 distinct colors to be safe?
    # "Minimize the number of labels actually used"
    # If we fail to find a small one, returning 128 is valid but score is bad.
    # But better than 0 score.
    
    # Let's try a random coloring with 4 colors and verify.
    # If it works, return it.
    for _ in range(1000):
        res = [random.randint(0, 3) for _ in range(n)]
        valid = True
        for a in range(n):
            for d in range(1, n):
                if a + 3*d >= n:
                    break
                if not check_ap(res[a], res[a+d], res[a+2*d], res[a+3*d]):
                    valid = False
                    break
            if not valid:
                break
        if valid:
            return res
            
    # Last resort: distinct colors
    return list(range(n))

