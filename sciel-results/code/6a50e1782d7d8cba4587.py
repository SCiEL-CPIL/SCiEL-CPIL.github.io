import time
import random
import numpy as np

N = 256
DEADLINE_FRACTION = 0.85

def _precompute():
    """
    Precompute interaction matrices to speed up SA moves.
    
    gain3[x][y]: Number of 3-APs formed by x and y (where the third element is present).
                 This is the marginal gain in 3-APs if both x and y are in the set.
                 Note: This is symmetric. gain3[x][y] is the number of z such that
                 {x, y, z} is a 3-AP.
                 
    conflict4[x][y]: Number of 4-APs formed by x and y (where the other two elements are present).
                     This is the marginal increase in 4-APs if both x and y are in the set.
                     Note: This is symmetric. conflict4[x][y] is the number of pairs {u, v}
                     such that {x, y, u, v} is a 4-AP.
    """
    # Initialize matrices
    gain3 = np.zeros((N + 1, N + 1), dtype=np.int32)
    conflict4 = np.zeros((N + 1, N + 1), dtype=np.int32)
    
    # Count 3-AP contributions
    # A 3-AP is (a, a+d, a+2d).
    # For a pair (x, y), how many 3-APs do they complete?
    # Case 1: x, y are the first two: x, y, z -> z = 2y - x. If z in [1, N], then (x, y, z) is a 3-AP.
    # Case 2: x, y are the last two: z, x, y -> z = 2x - y. If z in [1, N], then (z, x, y) is a 3-AP.
    # Case 3: x, y are first and last: x, z, y -> z = (x+y)/2. If x+y is even and z in [1, N], then (x, z, y) is a 3-AP.
    
    for x in range(1, N + 1):
        for y in range(x + 1, N + 1):
            c = 0
            # x, y, z
            z = 2 * y - x
            if 1 <= z <= N and z != x and z != y:
                c += 1
            # z, x, y
            z = 2 * x - y
            if 1 <= z <= N and z != x and z != y:
                c += 1
            # x, z, y
            if (x + y) % 2 == 0:
                z = (x + y) // 2
                if 1 <= z <= N and z != x and z != y:
                    c += 1
            gain3[x][y] = c
            gain3[y][x] = c
            
    # Count 4-AP contributions
    # A 4-AP is (a, a+d, a+2d, a+3d).
    # For a pair (x, y), how many 4-APs do they complete?
    # We need to find u, v such that {x, y, u, v} forms a 4-AP.
    # There are C(4,2)=6 ways x and y can appear in the 4-AP.
    # Let the 4-AP be (p, q, r, s) with p < q < r < s and q-p = r-q = s-r = d.
    
    # We can iterate over all 4-APs and increment the conflict count for each pair within them.
    # Total 4-APs in [1, 256] is small enough to enumerate.
    for a in range(1, N + 1):
        max_d = (N - a) // 3
        for d in range(1, max_d + 1):
            p = a
            q = a + d
            r = a + 2 * d
            s = a + 3 * d
            if s > N:
                continue
            
            # Pairs in this 4-AP: (p,q), (p,r), (p,s), (q,r), (q,s), (r,s)
            pairs = [(p, q), (p, r), (p, s), (q, r), (q, s), (r, s)]
            for u, v in pairs:
                conflict4[u][v] += 1
                conflict4[v][u] += 1
                
    return gain3, conflict4

def _count_3aps(flags):
    """
    Count total 3-APs in the set defined by flags.
    Using precomputed gain3 is O(N^2) if we sum over pairs, but we can do better.
    Actually, summing gain3[i][j] for i<j where flags[i] and flags[j] are True gives the count.
    Wait, gain3[i][j] counts the number of k such that {i,j,k} is a 3-AP.
    So sum_{i<j, flags[i]&flags[j]} gain3[i][j] is exactly the number of 3-APs.
    """
    count = 0
    # Iterate over pairs
    for i in range(1, N + 1):
        if not flags[i]:
            continue
        for j in range(i + 1, N + 1):
            if flags[j]:
                count += GAIN3[i][j]
    return count

def _count_4aps(flags):
    """
    Count total 4-APs in the set defined by flags.
    Sum_{i<j, flags[i]&flags[j]} conflict4[i][j] is NOT the count of 4-APs.
    conflict4[i][j] is the number of 4-APs containing i and j.
    If we sum conflict4[i][j] over all pairs in the set, each 4-AP is counted C(4,2)=6 times.
    So total 4-APs = (1/6) * sum_{i<j, flags[i]&flags[j]} conflict4[i][j].
    """
    total_conflicts = 0
    for i in range(1, N + 1):
        if not flags[i]:
            continue
        for j in range(i + 1, N + 1):
            if flags[j]:
                total_conflicts += CONFLICT4[i][j]
    return total_conflicts // 6

def construct():
    start_time = time.time()
    deadline = start_time + 90 * DEADLINE_FRACTION
    
    global GAIN3, CONFLICT4
    # Precompute if not already done (though construct is called once usually)
    if 'GAIN3' not in globals():
        GAIN3, CONFLICT4 = _precompute()
    
    # Initialize with a random set or a greedy seed
    # Let's use a greedy seed to start reasonably well
    flags = np.zeros(N + 1, dtype=bool)
    # Greedy add: add element if it doesn't create a 4-AP and maximizes 3-AP gain
    # This is O(N^3) roughly, but N=256 is small.
    # Actually, let's just start with a random subset of density 0.5 to avoid bias,
    # or use the "destruction" seed from the parent if we want to be safe, 
    # but SA is better with a diverse start.
    # Let's try a deterministic start: all odds? No, that has 4-APs (1,5,9,13).
    # Let's just start empty and do a few greedy adds to get a valid base.
    
    # Simple greedy initialization
    for x in range(1, N + 1):
        # Check if adding x creates a 4-AP
        creates_4ap = False
        # Check all pairs (y, z) in current set to see if {x, y, z, w} is a 4-AP
        # This is expensive. Let's just use the conflict4 matrix.
        # Adding x creates a 4-AP if there exists y, z in set such that conflict4[x][y] > 0 and flags[y] and ...
        # Wait, conflict4[x][y] counts 4-APs containing x and y.
        # If we add x, we create a 4-AP if there is a pair (y, z) in the current set
        # such that {x, y, z, w} is a 4-AP.
        # This is equivalent to: sum_{y in set} (number of 4-APs containing x and y and at least one other element in set) > 0?
        # No, it's simpler:
        # A 4-AP is formed if x is part of a 4-AP where the other 3 elements are already in the set.
        # Let's just check explicitly for speed during init.
        
        for d in range(1, (N - x) // 3 + 1):
            if x + d <= N and x + 2*d <= N and x + 3*d <= N:
                if flags[x+d] and flags[x+2*d] and flags[x+3*d]:
                    creates_4ap = True
                    break
        if creates_4ap: continue
        
        for d in range(1, min(x-1, (N-x)//2) + 1):
            if x-d >= 1 and x+d <= N and x+2*d <= N:
                if flags[x-d] and flags[x+d] and flags[x+2*d]:
                    creates_4ap = True
                    break
        if creates_4ap: continue
        
        for d in range(1, min((x-1)//2, N-x) + 1):
            if x-2*d >= 1 and x-d >= 1 and x+d <= N:
                if flags[x-2*d] and flags[x-d] and flags[x+d]:
                    creates_4ap = True
                    break
        if creates_4ap: continue
        
        for d in range(1, (x-1)//3 + 1):
            if x-3*d >= 1 and x-2*d >= 1 and x-d >= 1:
                if flags[x-3*d] and flags[x-2*d] and flags[x-d]:
                    creates_4ap = True
                    break
        if creates_4ap: continue
        
        flags[x] = True
        
    current_score = _count_3aps(flags)
    best_score = current_score
    best_flags = flags.copy()
    
    # Simulated Annealing
    T = 10.0
    T_min = 0.01
    cooling_rate = 0.999
    
    # We need to estimate the max possible change in score to set T.
    # Max 3-AP gain from adding one element is ~N/2.
    
    while time.time() < deadline:
        # Generate a move
        # 1. Add a random element not in set
        # 2. Remove a random element in set
        # 3. Swap a random element in set with a random element not in set
        
        move_type = random.random()
        if move_type < 0.3:
            # Add
            candidates = [i for i in range(1, N + 1) if not flags[i]]
            if not candidates: continue
            x = random.choice(candidates)
            # Calculate delta
            # Delta 3-APs: sum of GAIN3[x][y] for y in set
            delta_3 = 0
            for y in range(1, N + 1):
                if flags[y]:
                    delta_3 += GAIN3[x][y]
            
            # Delta 4-APs: We need to check if adding x creates any 4-AP.
            # A 4-AP is created if there exist y, z in set such that {x, y, z, w} is a 4-AP.
            # This is equivalent to checking if x is in a 4-AP with other 3 elements in the set.
            # We can use the precomputed list of 4-APs or check on the fly.
            # On-the-fly check is O(N) per x.
            creates_4ap = False
            # x is first
            for d in range(1, (N - x) // 3 + 1):
                if x + d <= N and x + 2*d <= N and x + 3*d <= N:
                    if flags[x+d] and flags[x+2*d] and flags[x+3*d]:
                        creates_4ap = True
                        break
            if not creates_4ap:
                # x is second
                for d in range(1, min(x-1, (N-x)//2) + 1):
                    if x-d >= 1 and x+d <= N and x+2*d <= N:
                        if flags[x-d] and flags[x+d] and flags[x+2*d]:
                            creates_4ap = True
                            break
            if not creates_4ap:
                # x is third
                for d in range(1, min((x-1)//2, N-x) + 1):
                    if x-2*d >= 1 and x-d >= 1 and x+d <= N:
                        if flags[x-2*d] and flags[x-d] and flags[x+d]:
                            creates_4ap = True
                            break
            if not creates_4ap:
                # x is fourth
                for d in range(1, (x-1)//3 + 1):
                    if x-3*d >= 1 and x-2*d >= 1 and x-d >= 1:
                        if flags[x-3*d] and flags[x-2*d] and flags[x-d]:
                            creates_4ap = True
                            break
            
            if creates_4ap:
                # Hard constraint violation. Reject or penalize heavily.
                # For simplicity, reject 4-AP creating moves.
                continue
                
            new_score = current_score + delta_3
            if new_score > current_score or random.random() < np.exp((new_score - current_score) / T):
                flags[x] = True
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_flags = flags.copy()
                    
        elif move_type < 0.6:
            # Remove
            candidates = [i for i in range(1, N + 1) if flags[i]]
            if not candidates: continue
            x = random.choice(candidates)
            
            delta_3 = 0
            for y in range(1, N + 1):
                if flags[y] and y != x:
                    delta_3 -= GAIN3[x][y]
            
            new_score = current_score + delta_3
            # Removal never creates a 4-AP, so always valid.
            if new_score > current_score or random.random() < np.exp((new_score - current_score) / T):
                flags[x] = False
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_flags = flags.copy()
                    
        else:
            # Swap
            in_set = [i for i in range(1, N + 1) if flags[i]]
            out_set = [i for i in range(1, N + 1) if not flags[i]]
            if not in_set or not out_set: continue
            x = random.choice(in_set)
            y = random.choice(out_set)
            
            # Calculate delta for removing x and adding y
            # Remove x: lose GAIN3[x][z] for all z in set (z != x)
            delta_remove = 0
            for z in range(1, N + 1):
                if flags[z] and z != x:
                    delta_remove -= GAIN3[x][z]
            
            # Add y: gain GAIN3[y][z] for all z in set (z != x, since x is removed)
            # Note: GAIN3[y][x] is lost because x is removed.
            delta_add = 0
            for z in range(1, N + 1):
                if flags[z] and z != x:
                    delta_add += GAIN3[y][z]
            
            # Check if adding y creates a 4-AP (given x is removed)
            creates_4ap = False
            # x is removed, so we just check y against the rest of the set
            # Same logic as "Add" but flags[x] is False
            
            # x is first (relative to y)
            for d in range(1, (N - y) // 3 + 1):
                if y + d <= N and y + 2*d <= N and y + 3*d <= N:
                    if flags[y+d] and flags[y+2*d] and flags[y+3*d]:
                        creates_4ap = True
                        break
            if not creates_4ap:
                for d in range(1, min(y-1, (N-y)//2) + 1):
                    if y-d >= 1 and y+d <= N and y+2*d <= N:
                        if flags[y-d] and flags[y+d] and flags[y+2*d]:
                            creates_4ap = True
                            break
            if not creates_4ap:
                for d in range(1, min((y-1)//2, N-y) + 1):
                    if y-2*d >= 1 and y-d >= 1 and y+d <= N:
                        if flags[y-2*d] and flags[y-d] and flags[y+d]:
                            creates_4ap = True
                            break
            if not creates_4ap:
                for d in range(1, (y-1)//3 + 1):
                    if y-3*d >= 1 and y-2*d >= 1 and y-d >= 1:
                        if flags[y-3*d] and flags[y-2*d] and flags[y-d]:
                            creates_4ap = True
                            break
            
            if creates_4ap:
                continue
                
            total_delta = delta_remove + delta_add
            new_score = current_score + total_delta
            
            if new_score > current_score or random.random() < np.exp((new_score - current_score) / T):
                flags[x] = False
                flags[y] = True
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_flags = flags.copy()
        
        # Cooling
        T *= cooling_rate
        if T < T_min:
            T = T_min
    
    # Ensure validity of best_flags
    # The SA loop rejects 4-AP creating moves, so best_flags should be valid.
    # But let's double check and clean up if necessary.
    if _count_4aps(best_flags) > 0:
        # Fallback: remove elements from best_flags to break 4-APs
        # This shouldn't happen often if SA is well-tuned
        active_4aps = []
        for a in range(1, N + 1):
            if not best_flags[a]: continue
            max_d = (N - a) // 3
            for d in range(1, max_d + 1):
                if a + 3*d > N: break
                if best_flags[a+d] and best_flags[a+2*d] and best_flags[a+3*d]:
                    active_4aps.append((a, a+d, a+2*d, a+3*d))
        
        while active_4aps:
            # Remove first element of first AP
            x = active_4aps[0][0]
            best_flags[x] = False
            # Recalculate active 4-APs (expensive, but only if invalid)
            active_4aps = []
            for a in range(1, N + 1):
                if not best_flags[a]: continue
                max_d = (N - a) // 3
                for d in range(1, max_d + 1):
                    if a + 3*d > N: break
                    if best_flags[a+d] and best_flags[a+2*d] and best_flags[a+3*d]:
                        active_4aps.append((a, a+d, a+2*d, a+3*d))
                        
    best_set = [i for i in range(1, N + 1) if best_flags[i]]
    if not best_set:
        return [1]
    return best_set

# Precompute at module load time to save time in construct?
# No, construct is the entry point. Precompute inside or globally?
# Global is faster for multiple calls, but construct is called once.
# Let's precompute in construct to be safe, or just let it run.
# Actually, _precompute takes ~0.1s, which is fine.

# To ensure GAIN3 and CONFLICT4 are available, we define them as None initially
GAIN3 = None
CONFLICT4 = None

# Override construct to handle global vars properly
def construct():
    start_time = time.time()
    deadline = start_time + 90 * DEADLINE_FRACTION
    
    global GAIN3, CONFLICT4
    if GAIN3 is None:
        GAIN3, CONFLICT4 = _precompute()
    
    # ... rest of the code from above ...
    # (Copying the logic from the previous construct definition but ensuring globals are set)
    
    flags = np.zeros(N + 1, dtype=bool)
    
    # Greedy Initialization
    for x in range(1, N + 1):
        creates_4ap = False
        for d in range(1, (N - x) // 3 + 1):
            if x + d <= N and x + 2*d <= N and x + 3*d <= N:
                if flags[x+d] and flags[x+2*d] and flags[x+3*d]:
                    creates_4ap = True
                    break
        if creates_4ap: continue
        for d in range(1, min(x-1, (N-x)//2) + 1):
            if x-d >= 1 and x+d <= N and x+2*d <= N:
                if flags[x-d] and flags[x+d] and flags[x+2*d]:
                    creates_4ap = True
                    break
        if creates_4ap: continue
        for d in range(1, min((x-1)//2, N-x) + 1):
            if x-2*d >= 1 and x-d >= 1 and x+d <= N:
                if flags[x-2*d] and flags[x-d] and flags[x+d]:
                    creates_4ap = True
                    break
        if creates_4ap: continue
        for d in range(1, (x-1)//3 + 1):
            if x-3*d >= 1 and x-2*d >= 1 and x-d >= 1:
                if flags[x-3*d] and flags[x-2*d] and flags[x-d]:
                    creates_4ap = True
                    break
        if creates_4ap: continue
        flags[x] = True
        
    current_score = _count_3aps(flags)
    best_score = current_score
    best_flags = flags.copy()
    
    T = 10.0
    T_min = 0.01
    cooling_rate = 0.999
    
    while time.time() < deadline:
        move_type = random.random()
        if move_type < 0.3:
            candidates = [i for i in range(1, N + 1) if not flags[i]]
            if not candidates: continue
            x = random.choice(candidates)
            delta_3 = 0
            for y in range(1, N + 1):
                if flags[y]:
                    delta_3 += GAIN3[x][y]
            
            creates_4ap = False
            for d in range(1, (N - x) // 3 + 1):
                if x + d <= N and x + 2*d <= N and x + 3*d <= N:
                    if flags[x+d] and flags[x+2*d] and flags[x+3*d]:
                        creates_4ap = True
                        break
            if not creates_4ap:
                for d in range(1, min(x-1, (N-x)//2) + 1):
                    if x-d >= 1 and x+d <= N and x+2*d <= N:
                        if flags[x-d] and flags[x+d] and flags[x+2*d]:
                            creates_4ap = True
                            break
            if not creates_4ap:
                for d in range(1, min((x-1)//2, N-x) + 1):
                    if x-2*d >= 1 and x-d >= 1 and x+d <= N:
                        if flags[x-2*d] and flags[x-d] and flags[x+d]:
                            creates_4ap = True
                            break
            if not creates_4ap:
                for d in range(1, (x-1)//3 + 1):
                    if x-3*d >= 1 and x-2*d >= 1 and x-d >= 1:
                        if flags[x-3*d] and flags[x-2*d] and flags[x-d]:
                            creates_4ap = True
                            break
            
            if creates_4ap:
                continue
                
            new_score = current_score + delta_3
            if new_score > current_score or random.random() < np.exp((new_score - current_score) / T):
                flags[x] = True
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_flags = flags.copy()
                    
        elif move_type < 0.6:
            candidates = [i for i in range(1, N + 1) if flags[i]]
            if not candidates: continue
            x = random.choice(candidates)
            
            delta_3 = 0
            for y in range(1, N + 1):
                if flags[y] and y != x:
                    delta_3 -= GAIN3[x][y]
            
            new_score = current_score + delta_3
            if new_score > current_score or random.random() < np.exp((new_score - current_score) / T):
                flags[x] = False
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_flags = flags.copy()
                    
        else:
            in_set = [i for i in range(1, N + 1) if flags[i]]
            out_set = [i for i in range(1, N + 1) if not flags[i]]
            if not in_set or not out_set: continue
            x = random.choice(in_set)
            y = random.choice(out_set)
            
            delta_remove = 0
            for z in range(1, N + 1):
                if flags[z] and z != x:
                    delta_remove -= GAIN3[x][z]
            
            delta_add = 0
            for z in range(1, N + 1):
                if flags[z] and z != x:
                    delta_add += GAIN3[y][z]
            
            creates_4ap = False
            for d in range(1, (N - y) // 3 + 1):
                if y + d <= N and y + 2*d <= N and y + 3*d <= N:
                    if flags[y+d] and flags[y+2*d] and flags[y+3*d]:
                        creates_4ap = True
                        break
            if not creates_4ap:
                for d in range(1, min(y-1, (N-y)//2) + 1):
                    if y-d >= 1 and y+d <= N and y+2*d <= N:
                        if flags[y-d] and flags[y+d] and flags[y+2*d]:
                            creates_4ap = True
                            break
            if not creates_4ap:
                for d in range(1, min((y-1)//2, N-y) + 1):
                    if y-2*d >= 1 and y-d >= 1 and y+d <= N:
                        if flags[y-2*d] and flags[y-d] and flags[y+d]:
                            creates_4ap = True
                            break
            if not creates_4ap:
                for d in range(1, (y-1)//3 + 1):
                    if y-3*d >= 1 and y-2*d >= 1 and y-d >= 1:
                        if flags[y-3*d] and flags[y-2*d] and flags[y-d]:
                            creates_4ap = True
                            break
            
            if creates_4ap:
                continue
                
            total_delta = delta_remove + delta_add
            new_score = current_score + total_delta
            
            if new_score > current_score or random.random() < np.exp((new_score - current_score) / T):
                flags[x] = False
                flags[y] = True
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_flags = flags.copy()
        
        T *= cooling_rate
        if T < T_min:
            T = T_min
    
    if _count_4aps(best_flags) > 0:
        active_4aps = []
        for a in range(1, N + 1):
            if not best_flags[a]: continue
            max_d = (N - a) // 3
            for d in range(1, max_d + 1):
                if a + 3*d > N: break
                if best_flags[a+d] and best_flags[a+2*d] and best_flags[a+3*d]:
                    active_4aps.append((a, a+d, a+2*d, a+3*d))
        
        while active_4aps:
            x = active_4aps[0][0]
            best_flags[x] = False
            active_4aps = []
            for a in range(1, N + 1):
                if not best_flags[a]: continue
                max_d = (N - a) // 3
                for d in range(1, max_d + 1):
                    if a + 3*d > N: break
                    if best_flags[a+d] and best_flags[a+2*d] and best_flags[a+3*d]:
                        active_4aps.append((a, a+d, a+2*d, a+3*d))
                        
    best_set = [i for i in range(1, N + 1) if best_flags[i]]
    if not best_set:
        return [1]
    return best_set

