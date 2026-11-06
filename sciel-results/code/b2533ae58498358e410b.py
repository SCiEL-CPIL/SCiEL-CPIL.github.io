import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50 seconds limit
    
    # Try k=3, 4, 5...
    for k in range(3, 10):
        if time.time() > deadline:
            break
            
        # Try to find a valid coloring with k colors
        # Using randomized greedy with backtracking is complex.
        # Let's try randomized assignment + repair (local search).
        
        # Initialize random coloring
        best_colors = None
        
        # Try multiple restarts
        for restart in range(100):
            if time.time() > deadline:
                break
                
            colors = [random.randint(0, k-1) for _ in range(128)]
            
            # Local search: minimize number of bad 4-APs
            # A 4-AP is bad if it uses < 3 distinct colors.
            
            # We can just check validity at the end of a few iterations
            # Or use a simple hill climbing.
            
            # Given the small N, maybe just random restarts and check validity?
            # Probability of random 3-coloring being valid is likely very low.
            
            # Better: Backtracking search with randomization.
            # Or: Use the greedy approach but with random tie-breaking and restarts.
            
            # Let's try a modified greedy that allows backtracking if stuck.
            # Since N=128, depth 128.
            # This might be too slow for pure backtracking.
            
            # Alternative: Use the fact that the structure is local.
            # We can use a SAT solver approach or CP-SAT, but only stdlib/np/scipy allowed.
            
            # Let's try a simple randomized greedy that picks a random valid color if multiple exist,
            # and if stuck, restarts.
            
            c = [None] * 128
            success = True
            for i in range(128):
                valid_colors = []
                for color in range(k):
                    # Check constraints with previous elements
                    ok = True
                    # Check all 4-APs ending at i
                    # d can be 1 to (i)//3
                    for d in range(1, (i)//3 + 1):
                        # AP: i, i-d, i-2d, i-3d
                        # Indices: i, i-d, i-2d, i-3d
                        # We have assigned c[0]...c[i-1]
                        # Need c[i-d], c[i-2d], c[i-3d] to be assigned.
                        # Since d <= i//3, i-3d >= 0.
                        prev1 = c[i-d]
                        prev2 = c[i-2*d]
                        prev3 = c[i-3*d]
                        
                        # If any is None, it's not assigned yet? 
                        # In greedy, we assign sequentially, so all j < i are assigned.
                        
                        s = {color, prev1, prev2, prev3}
                        if len(s) < 3:
                            ok = False
                            break
                    if ok:
                        valid_colors.append(color)
                
                if not valid_colors:
                    success = False
                    break
                
                # Pick randomly from valid colors to diversify
                c[i] = random.choice(valid_colors)
            
            if success:
                # Verify
                valid = True
                for i in range(128):
                    for d in range(1, (i)//3 + 1):
                        s = {c[i], c[i-d], c[i-2*d], c[i-3*d]}
                        if len(s) < 3:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    return c

    # If no small k found, fall back to a large k or the original greedy (if it works)
    # But original greedy might be broken.
    # Let's try k=10 with random greedy.
    for restart in range(100):
        if time.time() > deadline:
            break
        c = [None] * 128
        success = True
        for i in range(128):
            valid_colors = []
            for color in range(10):
                ok = True
                for d in range(1, (i)//3 + 1):
                    s = {color, c[i-d], c[i-2*d], c[i-3*d]}
                    if len(s) < 3:
                        ok = False
                        break
                if ok:
                    valid_colors.append(color)
            if not valid_colors:
                success = False
                break
            c[i] = random.choice(valid_colors)
        
        if success:
            return c
            
    # Fallback: return a list of 128 zeros (invalid, but better than crash?)
    # Or just return [0]*128
    return [0]*128
