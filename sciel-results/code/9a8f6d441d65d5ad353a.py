import itertools
import random
import time

N = 16
POINTS = list(itertools.product(range(N), repeat=2))
NUM_POINTS = len(POINTS)

# Precompute squared distances
dist = [[0]*NUM_POINTS for _ in range(NUM_POINTS)]
for i in range(NUM_POINTS):
    x1, y1 = POINTS[i]
    for j in range(i+1, NUM_POINTS):
        x2, y2 = POINTS[j]
        d = (x1-x2)**2 + (y1-y2)**2
        dist[i][j] = d
        dist[j][i] = d

def can_add(p_idx, set_indices):
    m = len(set_indices)
    if m < 2:
        return True
    
    for i in range(m):
        a = set_indices[i]
        da = dist[p_idx][a]
        for j in range(i+1, m):
            b = set_indices[j]
            db = dist[p_idx][b]
            dab = dist[a][b]
            
            if da == db or da == dab or db == dab:
                return False
    return True

def is_maximal(set_indices):
    set_set = set(set_indices)
    for p in range(NUM_POINTS):
        if p not in set_set:
            if can_add(p, set_indices):
                return False
    return True

def greedy_construction(order):
    chosen = []
    for p in order:
        if can_add(p, chosen):
            chosen.append(p)
    return chosen

def local_search(set_indices, deadline):
    current = list(set_indices)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Randomize the order in which we try to remove points
        indices = list(range(len(current)))
        random.shuffle(indices)
        for i in indices:
            if time.time() > deadline:
                break
            p = current[i]
            new_set = current[:i] + current[i+1:]
            if is_maximal(new_set):
                current = new_set
                improved = True
                break
    return current

def swap_search(set_indices, deadline):
    """Try to swap points to reduce size or maintain size while improving potential for further reduction."""
    current = list(set_indices)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        current_set = set(current)
        # Try removing a point and adding a different point to maintain maximality and potentially enable further removals
        # This is a 1-move local search: remove p, add q
        for i in range(len(current)):
            if time.time() > deadline:
                break
            p = current[i]
            # Try to remove p
            temp_set = [x for j, x in enumerate(current) if j != i]
            # Check if we can add any point q not in current to make it maximal again (or just keep it if already maximal)
            # Actually, if we remove p, it might not be maximal. We need to see if we can replace p with another point q
            # such that the new set is valid (no isosceles) and maximal.
            
            # Optimization: Only try to swap if removing p breaks maximality?
            # If removing p keeps it maximal, local_search would have caught it.
            # So we assume removing p breaks maximality.
            
            # Find points q that can be added to temp_set
            candidates = []
            for q in range(NUM_POINTS):
                if q in current_set:
                    continue
                if can_add(q, temp_set):
                    candidates.append(q)
            
            if not candidates:
                continue
            
            # Try adding a candidate q
            for q in candidates:
                new_set = temp_set + [q]
                # Check if new_set is maximal
                if is_maximal(new_set):
                    # We swapped p for q. Size is same.
                    # This might allow further removals?
                    # It's a lateral move. We accept it with some probability or if it leads to improvement.
                    # For now, let's just do it if it allows a subsequent removal?
                    # That's expensive. Let's just accept the swap to escape local optima.
                    # But we need to avoid cycles. 
                    # Simple heuristic: accept if random() < 0.5 or if it's "better" (not defined).
                    # Let's just do a few random swaps.
                    if random.random() < 0.1: # Low probability to avoid cycling
                        current = new_set
                        improved = True
                        break
            if improved:
                break
    return current

def construct():
    start_time = time.time()
    deadline = start_time + 320.0 # Safety margin for 330s limit
    
    best_set = None
    best_size = float('inf')
    
    # Phase 1: Randomized Greedy + Local Search
    restarts = 0
    while time.time() < deadline - 5.0:
        restarts += 1
        order = list(range(NUM_POINTS))
        random.shuffle(order)
        s = greedy_construction(order)
        s = local_search(s, deadline)
        
        if len(s) < best_size:
            best_size = len(s)
            best_set = s
        
        # If we find a very small set, we can be more aggressive with swap search
        if best_size <= 20:
             # Do a bit of swap search to escape local optima
             s = swap_search(s, deadline)
             if len(s) < best_size:
                 best_size = len(s)
                 best_set = s
             else:
                 # Try to reduce the swapped set further
                 s = local_search(s, deadline)
                 if len(s) < best_size:
                     best_size = len(s)
                     best_set = s

    if best_set is None:
        # Fallback: deterministic greedy
        s = greedy_construction(list(range(NUM_POINTS)))
        best_set = s

    # Final local search on the best set found
    best_set = local_search(best_set, deadline)

    # Convert indices to points
    return [POINTS[i] for i in best_set]

