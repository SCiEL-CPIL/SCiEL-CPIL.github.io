import random
import time
import math

N = 16
num_points = N * N
coords = [(i // N, i % N) for i in range(num_points)]

# Precompute distances squared
dist = [[0] * num_points for _ in range(num_points)]
for i in range(num_points):
    x1, y1 = coords[i]
    for j in range(i + 1, num_points):
        x2, y2 = coords[j]
        d = (x1 - x2) ** 2 + (y1 - y2) ** 2
        dist[i][j] = d
        dist[j][i] = d

# Precompute conflict masks:
# conflict_mask[i][j] is a bitmask of points k such that (i, j, k) forms an isosceles triangle.
# This is used to check if adding k to S creates an isosceles triangle with existing pairs in S.
# Specifically, k conflicts with pair (i,j) if k is in conflict_mask[i][j].
# For a set S, it is isosceles-free iff for all i, j in S, (S_mask & conflict_mask[i][j]) == 0.

# Also, for maximality:
# A point v not in S is "blocked" if there exist i, j in S such that v is in conflict_mask[i][j].
# Equivalently, v is blocked if (S_mask & conflict_mask[i][j]) has v set? No.
# v is blocked if there is a pair (i,j) in S such that v in conflict_mask[i][j].
# Let's precompute for each v, the list of pairs (i,j) that block it.
# blocked_by[v] = list of (i, j) such that v in conflict_mask[i][j].
# Then v is blocked by S if any (i,j) in blocked_by[v] has both i and j in S.

conflict_mask = [[0] * num_points for _ in range(num_points)]
blocked_by = [[] for _ in range(num_points)]

for i in range(num_points):
    for j in range(i + 1, num_points):
        mask = 0
        for k in range(num_points):
            if k == i or k == j:
                continue
            d_ik = dist[i][k]
            d_jk = dist[j][k]
            d_ij = dist[i][j]
            if d_ik == d_jk or d_ik == d_ij or d_jk == d_ij:
                mask |= (1 << k)
                blocked_by[k].append((i, j))
        conflict_mask[i][j] = mask
        conflict_mask[j][i] = mask

def is_isosceles_free(S_mask, s_list):
    """Check if S is isosceles-free."""
    for idx_a, a in enumerate(s_list):
        for b in s_list[idx_a + 1:]:
            if (S_mask & conflict_mask[a][b]) != 0:
                return False
    return True

def is_maximal(S_mask, s_list):
    """Check if S is maximal (every point not in S is blocked)."""
    for v in range(num_points):
        if (S_mask >> v) & 1:
            continue
        # v is not in S. Check if it is blocked.
        blocked = False
        for (i, j) in blocked_by[v]:
            if ((S_mask >> i) & 1) and ((S_mask >> j) & 1):
                blocked = True
                break
        if not blocked:
            return False
    return True

def generate_initial_set(seed):
    """Generate a random maximal no-isosceles set using greedy."""
    random.seed(seed)
    order = list(range(num_points))
    random.shuffle(order)
    
    S_mask = 0
    s_list = []
    
    for p in order:
        # Check if p can be added
        valid = True
        for idx_a, a in enumerate(s_list):
            for b in s_list[idx_a + 1:]:
                if (conflict_mask[a][b] >> p) & 1:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            S_mask |= (1 << p)
            s_list.append(p)
            
    # Repair to maximal
    changed = True
    while changed:
        changed = False
        for p in range(num_points):
            if (S_mask >> p) & 1:
                continue
            valid = True
            for idx_a, a in enumerate(s_list):
                for b in s_list[idx_a + 1:]:
                    if (conflict_mask[a][b] >> p) & 1:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                S_mask |= (1 << p)
                s_list.append(p)
                changed = True
                
    return S_mask, s_list

def local_search(S_mask, s_list, deadline):
    """
    Local search to minimize |S|.
    1. Try to remove points.
    2. If stuck, try to swap points to change structure.
    """
    best_mask = S_mask
    best_size = len(s_list)
    
    while time.time() < deadline:
        improved = False
        
        # Phase 1: Try to remove points
        random.shuffle(s_list)
        for p in s_list:
            if not ((S_mask >> p) & 1):
                continue
            new_mask = S_mask ^ (1 << p)
            new_list = [x for x in s_list if x != p]
            
            # Check validity of new set
            if is_isosceles_free(new_mask, new_list) and is_maximal(new_mask, new_list):
                S_mask = new_mask
                s_list = new_list
                improved = True
                if len(s_list) < best_size:
                    best_size = len(s_list)
                    best_mask = S_mask
                break # Restart removal loop with new set
        
        if not improved:
            # Phase 2: Try to swap points
            if not s_list:
                break
                
            p = random.choice(s_list)
            q_found = False
            for _ in range(100):
                q = random.randint(0, num_points - 1)
                if not ((S_mask >> q) & 1):
                    q_found = True
                    break
            
            if not q_found:
                break
                
            new_mask = S_mask ^ (1 << p) ^ (1 << q)
            new_list = [x for x in s_list if x != p]
            new_list.append(q)
            
            # Check validity
            if is_isosceles_free(new_mask, new_list) and is_maximal(new_mask, new_list):
                S_mask = new_mask
                s_list = new_list
                continue
                
    return best_mask, best_size

def construct():
    start_time = time.time()
    deadline = start_time + 290
    
    best_size = num_points
    best_mask = 0
    
    seed = 0
    while time.time() < deadline:
        S_mask, s_list = generate_initial_set(seed)
        
        final_mask, final_size = local_search(S_mask, s_list, deadline)
        
        if final_size < best_size:
            best_size = final_size
            best_mask = final_mask
            
        seed += 1
        
    result = []
    for i in range(num_points):
        if (best_mask >> i) & 1:
            result.append(coords[i])
            
    return result
