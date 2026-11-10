import time
import random

N = 256
DEADLINE = 60.0  # 60 seconds limit, keep margin

def _precompute_aps():
    """Precompute 3-APs and 4-APs and their element memberships."""
    # 3-APs: (a, a+d, a+2d)
    aps3 = []
    for d in range(1, N // 2 + 1):
        for a in range(1, N - 2 * d + 1):
            b = a + d
            c = a + 2 * d
            aps3.append((a, b, c))
    
    # 4-APs: (a, a+d, a+2d, a+3d)
    aps4 = []
    for d in range(1, N // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            aps4.append((a, b, c, e))
            
    # Map element to list of 3-AP indices
    elem_to_aps3 = {i: [] for i in range(1, N + 1)}
    for i, ap in enumerate(aps3):
        for x in ap:
            elem_to_aps3[x].append(i)
            
    # Map element to list of 4-AP indices
    elem_to_aps4 = {i: [] for i in range(1, N + 1)}
    for i, ap in enumerate(aps4):
        for x in ap:
            elem_to_aps4[x].append(i)
            
    return aps3, aps4, elem_to_aps3, elem_to_aps4

def _initial_greedy_set():
    """Greedy construction for no 4-AP."""
    selected = set()
    for x in range(1, N + 1):
        # Check if adding x creates a 4-AP
        valid = True
        # x could be the 1st, 2nd, 3rd, or 4th term
        # 1st: x, x+d, x+2d, x+3d
        for d in range(1, (N - x) // 3 + 1):
            if x + d in selected and x + 2 * d in selected and x + 3 * d in selected:
                valid = False
                break
        if not valid: continue
        # 2nd: x-d, x, x+d, x+2d
        for d in range(1, min(x - 1, (N - x) // 2) + 1):
            if x - d in selected and x + d in selected and x + 2 * d in selected:
                valid = False
                break
        if not valid: continue
        # 3rd: x-2d, x-d, x, x+d
        for d in range(1, min(x // 2, N - x) + 1):
            if x - 2 * d in selected and x - d in selected and x + d in selected:
                valid = False
                break
        if not valid: continue
        # 4th: x-3d, x-2d, x-d, x
        for d in range(1, x // 3 + 1):
            if x - 3 * d in selected and x - 2 * d in selected and x - d in selected:
                valid = False
                break
        if valid:
            selected.add(x)
    return selected

def construct():
    start_time = time.time()
    aps3, aps4, elem_to_aps3, elem_to_aps4 = _precompute_aps()
    
    # Initial set
    A = _initial_greedy_set()
    A_set = set(A)
    
    # Current score (number of 3-APs)
    current_score = sum(1 for ap in aps3 if ap[0] in A_set and ap[1] in A_set and ap[2] in A_set)
    
    best_score = current_score
    best_A = set(A_set)
    
    # Local Search
    # We will try swaps: remove x from A, add y not in A
    # To avoid getting stuck, we can do multiple passes or random restarts if time permits
    # But simple hill climbing on swaps is usually good for N=256
    
    # Precompute complement
    all_elements = set(range(1, N + 1))
    
    improved = True
    while improved and (time.time() - start_time) < DEADLINE - 5:
        improved = False
        # Shuffle to avoid bias, or just iterate
        # Iterating over all x in A and y not in A is O(|A| * (N-|A|)) ~ 100 * 150 = 15000 checks
        # Each check is O(1) if we use precomputed lists? 
        # Validity check for adding y: check 4-APs containing y.
        # Score delta: check 3-APs containing y and x.
        
        # To make it faster, we can sample or just do full pass
        # Full pass is fine
        
        candidates_x = list(A_set)
        random.shuffle(candidates_x)
        
        for x in candidates_x:
            if time.time() - start_time > DEADLINE - 5:
                break
            candidates_y = [y for y in all_elements if y not in A_set]
            random.shuffle(candidates_y)
            
            for y in candidates_y:
                # Check validity: Does A \ {x} U {y} have a 4-AP?
                # Only 4-APs containing y can be new.
                valid = True
                for ap_idx in elem_to_aps4[y]:
                    ap = aps4[ap_idx]
                    # Check if other 3 elements are in A \ {x}
                    other_elements = [z for z in ap if z != y]
                    if all(z in A_set and z != x for z in other_elements):
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # Calculate score delta
                # Loss: 3-APs containing x where other 2 are in A
                loss = 0
                for ap_idx in elem_to_aps3[x]:
                    ap = aps3[ap_idx]
                    other_elements = [z for z in ap if z != x]
                    if all(z in A_set for z in other_elements):
                        loss += 1
                
                # Gain: 3-APs containing y where other 2 are in A \ {x}
                gain = 0
                for ap_idx in elem_to_aps3[y]:
                    ap = aps3[ap_idx]
                    other_elements = [z for z in ap if z != y]
                    if all(z in A_set and z != x for z in other_elements):
                        gain += 1
                
                delta = gain - loss
                if delta > 0:
                    # Apply swap
                    A_set.remove(x)
                    A_set.add(y)
                    current_score += delta
                    improved = True
                    if current_score > best_score:
                        best_score = current_score
                        best_A = set(A_set)
                    break # Restart inner loop for this x with new state? Or just continue?
                    # Usually better to restart the x loop to reflect changes
                    # But breaking inner and continuing outer is okay.
                    # However, the list candidates_x is static.
                    # Let's just break and let the outer while loop handle it.
                    # Actually, breaking inner loop and continuing outer x loop is fine, 
                    # but A_set changed.
                    pass

    # If time allows, try a few random restarts or perturbations?
    # For now, return the best found
    
    return sorted(best_A)

# EVOLVE-BLOCK-START
# The construct function is defined above.
# We need to ensure the entry point is exactly 'construct'
# The code above defines it.
# EVOLVE-BLOCK-END
