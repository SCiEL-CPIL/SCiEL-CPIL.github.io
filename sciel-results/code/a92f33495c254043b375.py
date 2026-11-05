import random

N = 256

def construct():
    best_set = None
    best_size = N + 1
    
    # Run multiple trials to escape local minima or find better random tie-breaks
    # Increased trials to find a smaller maximal set
    for trial in range(500):
        S = set()
        sums = set()
        
        while True:
            # Identify U: elements that can be added
            U = []
            for x in range(1, N + 1):
                if x in S:
                    continue
                # Quick check: 2x
                if 2 * x in sums:
                    continue
                # Check x + a for a in S
                conflict = False
                for a in S:
                    if x + a in sums:
                        conflict = True
                        break
                if not conflict:
                    U.append(x)
            
            if not U:
                break
            
            # Find y in U that blocks the most others in U
            best_y = None
            max_blocked = -1
            candidates = []
            
            # Optimization: Pre-compute S list for faster iteration
            S_list = list(S)
            
            for y in U:
                # Simulate sums added by y
                new_sums_y = {2 * y}
                for a in S_list:
                    new_sums_y.add(y + a)
                
                # Count how many x in U (x != y) are blocked by adding these sums
                blocked_count = 0
                for x in U:
                    if x == y:
                        continue
                    # x is blocked if its new sums intersect (sums U new_sums_y)
                    # New sums for x: {2x} U {x+a for a in S} U {x+y}
                    # We know {2x} U {x+a for a in S} does not intersect sums.
                    # So check intersection with new_sums_y.
                    
                    blocked = False
                    if (x + y) in sums:
                        blocked = True
                    elif (x + y) in new_sums_y:
                        blocked = True
                    else:
                        # Check if any other sum of x is in new_sums_y
                        # new_sums_y = {2y} U {y+a for a in S}
                        # S_x\{x+y} = {2x} U {x+a for a in S}
                        if 2 * x in new_sums_y:
                            blocked = True
                        else:
                            for a in S_list:
                                if (x + a) in new_sums_y:
                                    blocked = True
                                    break
                    if blocked:
                        blocked_count += 1
                
                if blocked_count > max_blocked:
                    max_blocked = blocked_count
                    best_y = y
                    candidates = [y]
                elif blocked_count == max_blocked:
                    candidates.append(y)
            
            # Pick one
            chosen_y = random.choice(candidates)
            
            # Update
            S.add(chosen_y)
            sums.add(2 * chosen_y)
            for a in S:
                if a != chosen_y:
                    sums.add(chosen_y + a)
            
        if len(S) < best_size:
            best_size = len(S)
            best_set = list(S)
            
    return best_set
