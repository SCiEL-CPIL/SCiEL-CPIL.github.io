import time
import random
import numpy as np

# Precompute K4s
# Vertices 0..15
# Edges (i, j) with i < j
# Edge index mapping
edge_idx = {}
edges_list = []
for i in range(16):
    for j in range(i+1, 16):
        edge_idx[(i, j)] = len(edges_list)
        edges_list.append((i, j))

# List of K4s: each is a list of 6 edge indices
k4s = []
for a in range(16):
    for b in range(a+1, 16):
        for c in range(b+1, 16):
            for d in range(c+1, 16):
                # Vertices a, b, c, d
                # Edges: (a,b), (a,c), (a,d), (b,c), (b,d), (c,d)
                e1 = edge_idx[(a, b)]
                e2 = edge_idx[(a, c)]
                e3 = edge_idx[(a, d)]
                e4 = edge_idx[(b, c)]
                e5 = edge_idx[(b, d)]
                e6 = edge_idx[(c, d)]
                k4s.append([e1, e2, e3, e4, e5, e6])

# Map edge index to list of K4 indices containing it
edge_to_k4 = [[] for _ in range(120)]
for k_idx, k4 in enumerate(k4s):
    for e in k4:
        edge_to_k4[e].append(k_idx)

def get_colours(k4, edges):
    # k4 is list of 6 edge indices
    # edges is array of colours
    return set(edges[e] for e in k4)

def count_violations(edges):
    # Returns list of bools and count
    violations = [False] * len(k4s)
    count = 0
    for k_idx, k4 in enumerate(k4s):
        if len(get_colours(k4, edges)) < 5:
            violations[k_idx] = True
            count += 1
    return violations, count

def solve(K, time_limit, initial_state=None):
    start_time = time.time()
    
    if initial_state is not None:
        edges = np.array(initial_state, dtype=int)
        # Ensure colours are in range 0..K-1? 
        # If initial state used fewer colours, it's fine.
        # But we might want to use the full range K.
        # For now, just use it as is.
    else:
        edges = np.random.randint(0, K, 120)
        
    violations, violation_count = count_violations(edges)
    
    if violation_count == 0:
        return edges, True
        
    # Local search
    # To speed up, we can pick a random violating K4 and change one of its edges.
    # We need to update violations efficiently.
    
    # Precompute list of violating K4 indices for faster picking
    viol_indices = [i for i, v in enumerate(violations) if v]
    
    while time.time() - start_time < time_limit:
        if violation_count == 0:
            return edges, True
            
        # Pick a random violating K4
        k_idx = random.choice(viol_indices)
        k4 = k4s[k_idx]
        
        # Pick a random edge in this K4
        e_idx = random.choice(k4)
        
        # Change colour
        old_colour = edges[e_idx]
        new_colour = random.randint(0, K-1)
        if new_colour == old_colour:
            continue
            
        edges[e_idx] = new_colour
        
        # Update violations for all K4s containing e_idx
        # These are in edge_to_k4[e_idx]
        # We need to recalculate status for these K4s
        # And update viol_indices and violation_count
        
        # To avoid O(N) update of viol_indices, we can just recalculate 
        # the set of colours for the affected K4s and update the violations array.
        # Then if we need to pick a random violating one, we can 
        # either rebuild viol_indices (expensive) or pick from a list and check.
        
        # Strategy: Maintain a list of currently violating K4s.
        # When a K4 becomes valid, remove it.
        # When a K4 becomes invalid, add it.
        # Removal from list is O(N). 
        # Given 1820 K4s, O(N) is ~1820 ops. 
        # 100k moves * 1820 = 180M ops. Might be too slow in Python.
        
        # Alternative: Just pick a random K4 index from 0..1819.
        # Check if it's violating. If yes, fix it. If no, try again.
        # If most are valid, this is slow.
        # But initially many are violating.
        
        # Better: Rebuild viol_indices periodically?
        # Or just accept the O(N) cost of updating the list if we use a set or similar?
        
        # Let's try a simpler approach:
        # Just iterate through edge_to_k4[e_idx]
        # For each affected k4:
        #   new_status = len(get_colours(k4, edges)) < 5
        #   if new_status != violations[k_idx]:
        #       violations[k_idx] = new_status
        #       violation_count += 1 if new_status else -1
        #       # Update viol_indices?
        
        # To avoid complex list maintenance, I will just 
        # pick a random index from the full range of K4s, 
        # and check if it's violating. 
        # To bias towards violating ones, I can keep a list 
        # and rebuild it every X steps.
        
        # Let's rebuild viol_indices every 100 steps.
        if random.randint(0, 99) == 0:
            viol_indices = [i for i, v in enumerate(violations) if v]
            if not viol_indices:
                return edges, True
        
        # Pick from current viol_indices (might be stale)
        if not viol_indices:
            return edges, True
            
        k_idx = random.choice(viol_indices)
        
        # Re-check if it's still violating (in case stale)
        if not violations[k_idx]:
            continue
            
        k4 = k4s[k_idx]
        e_idx = random.choice(k4)
        old_colour = edges[e_idx]
        new_colour = random.randint(0, K-1)
        if new_colour == old_colour:
            continue
            
        edges[e_idx] = new_colour
        
        # Update affected K4s
        for aff_k_idx in edge_to_k4[e_idx]:
            new_status = len(get_colours(k4s[aff_k_idx], edges)) < 5
            if new_status != violations[aff_k_idx]:
                violations[aff_k_idx] = new_status
                if new_status:
                    violation_count += 1
                else:
                    violation_count -= 1
                    
    return edges, False

def construct():
    t_start = time.time()
    
    # Try K=13
    res13, valid13 = solve(13, 10.0)
    if valid13:
        return res13.tolist()
        
    # Try K=14
    res14, valid14 = solve(14, 30.0, initial_state=res13.tolist())
    if valid14:
        return res14.tolist()
        
    # Try K=15
    res15, valid15 = solve(15, 70.0, initial_state=res14.tolist())
    if valid15:
        return res15.tolist()
        
    # Fallback to seed
    return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
