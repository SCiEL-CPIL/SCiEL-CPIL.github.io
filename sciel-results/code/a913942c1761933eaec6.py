import numpy as np
import time
import random

N = 16
N2 = N * N
DEADLINE = time.time() + 290

# Precompute coordinates
coords = [(x, y) for y in range(N) for x in range(N)]

# Precompute squared distance matrix
# dist[i, j] = (x_i - x_j)^2 + (y_i - y_j)^2
xs = [c[0] for c in coords]
ys = [c[1] for c in coords]
xs_arr = np.array(xs)
ys_arr = np.array(ys)

# Compute distance matrix
# dist[i, j] = (xs[i] - xs[j])^2 + (ys[i] - ys[j])^2
dx = xs_arr[:, np.newaxis] - xs_arr[np.newaxis, :]
dy = ys_arr[:, np.newaxis] - ys_arr[np.newaxis, :]
dist_matrix = dx**2 + dy**2

def can_add(p, S_indices):
    """
    Check if point p can be added to set S_indices.
    S_indices is a list or array of indices.
    Returns True if no isosceles triangle is formed.
    """
    if not S_indices:
        return True
    
    # Get distances from p to all points in S
    dists = dist_matrix[p, S_indices]
    
    # 1. Check for duplicates (P is apex)
    # If any distance from P to S is repeated, P forms isosceles with two S points.
    if len(dists) != len(np.unique(dists)):
        return False
    
    # 2. Check if P forms isosceles with S points as apexes
    # For each S[i], d(P, S[i]) must not equal d(S[i], S[j]) for any j in S
    S_list = list(S_indices)
    k = len(S_list)
    
    # Extract submatrix of distances within S
    # sub_dist[i, j] = dist(S[i], S[j])
    sub_dist = dist_matrix[np.ix_(S_list, S_list)]
    
    for i in range(k):
        dp_i = dists[i]
        # Check if dp_i exists in the row i of sub_dist
        # This checks if d(P, S[i]) == d(S[i], S[j]) for any j
        if dp_i in sub_dist[i, :]:
            return False
            
    return True

def get_addable_points(S_set, S_list):
    """
    Return list of points not in S_set that can be added to S_list.
    """
    addable = []
    # Iterate over all points not in S_set
    for p in range(N2):
        if p in S_set:
            continue
        if can_add(p, S_list):
            addable.append(p)
    return addable

def make_maximal(S_list):
    """
    Given a no-isosceles set S_list, add points until it becomes maximal.
    Returns the new list.
    """
    S_set = set(S_list)
    S_list = list(S_list)
    
    while True:
        addable = get_addable_points(S_set, S_list)
        if not addable:
            break
        
        # Heuristic: Pick a point that "blocks" the most other addable points?
        # For simplicity and speed, pick random.
        # The local search will handle the optimization.
        p = random.choice(addable)
        S_list.append(p)
        S_set.add(p)
        
    return S_list

def greedy_construct(seed_val):
    """
    Construct a maximal no-isosceles set using greedy random order.
    """
    rng = random.Random(seed_val)
    order = list(range(N2))
    rng.shuffle(order)
    
    S_list = []
    S_set = set()
    
    for p in order:
        if can_add(p, S_list):
            S_list.append(p)
            S_set.add(p)
    
    # Ensure maximal
    S_list = make_maximal(S_list)
    return S_list

def local_search(S_initial, rng, max_iters=1000):
    """
    Local search to minimize the size of the maximal no-isosceles set.
    """
    S_list = list(S_initial)
    best_list = list(S_initial)
    best_len = len(S_initial)
    
    for _ in range(max_iters):
        if time.time() > DEADLINE:
            break
        
        if len(S_list) <= 1:
            break
            
        # Pick a random point to remove
        idx = rng.randint(0, len(S_list) - 1)
        p_remove = S_list[idx]
        
        # Create S without p_remove
        S_temp = [x for i, x in enumerate(S_list) if i != idx]
        
        # Repair to make maximal
        S_repaired = make_maximal(S_temp)
        
        if len(S_repaired) < best_len:
            best_list = S_repaired
            best_len = len(S_repaired)
            S_list = S_repaired
        elif len(S_repaired) == best_len:
            # Accept with some probability to escape local minima
            if rng.random() < 0.2:
                S_list = S_repaired
        else:
            # Revert
            S_list = list(best_list)
            
    return best_list

def construct():
    best_solution = None
    best_len = float('inf')
    
    # Try multiple random restarts
    seed_counter = 0
    while time.time() < DEADLINE - 10: # Safety margin
        seed_counter += 1
        rng = random.Random(seed_counter)
        
        # 1. Greedy construction
        S_greedy = greedy_construct(seed_counter)
        
        if len(S_greedy) < best_len:
            best_solution = S_greedy
            best_len = len(S_greedy)
            
        # 2. Local search improvement
        # Use a separate RNG for local search to ensure reproducibility if needed, 
        # but mixing is fine.
        S_improved = local_search(S_greedy, rng, max_iters=200)
        
        if len(S_improved) < best_len:
            best_solution = S_improved
            best_len = len(S_improved)
            
        # Check time
        if time.time() > DEADLINE - 5:
            break
            
    if best_solution is None:
        # Fallback
        best_solution = greedy_construct(0)
        
    # Convert indices to coordinates
    return [coords[i] for i in best_solution]
