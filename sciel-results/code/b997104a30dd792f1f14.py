import numpy as np
import time
import itertools
import random

N = 16
POINTS = list(itertools.product(range(N), repeat=2))
N_POINTS = len(POINTS)
coords = np.array(POINTS)

# Precompute squared distances
diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
D = np.sum(diff ** 2, axis=2)

# Precompute isosceles masks
# I_mask[x][u] is a bitmask of points v such that (x, u, v) form an isosceles triangle
# v must be distinct from x and u.
I_mask = [[0] * N_POINTS for _ in range(N_POINTS)]

for x in range(N_POINTS):
    for u in range(N_POINTS):
        if x == u:
            continue
        mask = 0
        dx_u = D[x, u]
        for v in range(N_POINTS):
            if v == x or v == u:
                continue
            dx_v = D[x, v]
            du_v = D[u, v]
            # Isosceles if dx_u == dx_v or dx_u == du_v or dx_v == du_v
            if dx_u == dx_v or dx_u == du_v or dx_v == du_v:
                mask |= (1 << v)
        I_mask[x][u] = mask

def is_blocked(x, S_mask):
    """
    Check if point x is blocked by set S (represented by S_mask).
    x is blocked if there exist u, v in S such that (x, u, v) is isosceles.
    """
    for u in range(N_POINTS):
        if (S_mask >> u) & 1:
            # u is in S
            # Check if any v in S is in I_mask[x][u]
            if S_mask & I_mask[x][u]:
                return True
    return False

def is_maximal(S_mask):
    """
    Check if set S is maximal.
    S is maximal if for all x not in S, x is blocked by S.
    """
    for x in range(N_POINTS):
        if not ((S_mask >> x) & 1):
            if not is_blocked(x, S_mask):
                return False
    return True

def greedy_construct():
    S_mask = 0
    perm = list(range(N_POINTS))
    random.shuffle(perm)
    for p in perm:
        if not is_blocked(p, S_mask):
            S_mask |= (1 << p)
    return S_mask

def local_search(S_mask):
    start_time = time.time()
    deadline = 280
    improved = True
    while improved and time.time() - start_time < deadline:
        improved = False
        # Get list of points in S
        points_in_S = [i for i in range(N_POINTS) if (S_mask >> i) & 1]
        random.shuffle(points_in_S)
        for p in points_in_S:
            if time.time() - start_time > deadline:
                break
            # Try removing p
            new_S_mask = S_mask & ~(1 << p)
            if is_maximal(new_S_mask):
                S_mask = new_S_mask
                improved = True
                # Restart the loop to re-evaluate with the new set
                break
    return S_mask

def construct():
    start_time = time.time()
    deadline = 280
    
    best_mask = None
    best_size = float('inf')
    
    # Run multiple restarts
    while time.time() - start_time < deadline:
        S_mask = greedy_construct()
        S_mask = local_search(S_mask)
        
        size = bin(S_mask).count('1')
        if size < best_size:
            best_size = size
            best_mask = S_mask
            
    if best_mask is None:
        # Fallback
        best_mask = greedy_construct()
        
    # Convert mask to list of points
    result = [POINTS[i] for i in range(N_POINTS) if (best_mask >> i) & 1]
    return result
