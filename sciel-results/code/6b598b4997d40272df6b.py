import random
import itertools
import numpy as np
import time

N = 16
M = N * N

# Precompute conflict masks
# conflict_mask[i][j] is a bitset of points k such that i,j,k form an isosceles triangle
conflict_mask = [[0] * M for _ in range(M)]

def dist_sq(p, q):
    return (p[0] - q[0])**2 + (p[1] - q[1])**2

# Generate all points
points = [(x, y) for y in range(N) for x in range(N)]
point_to_idx = {p: i for i, p in enumerate(points)}

# Precompute distances for speed
# dist[i][j]
dist = [[0]*M for _ in range(M)]
for i in range(M):
    for j in range(i+1, M):
        d = dist_sq(points[i], points[j])
        dist[i][j] = d
        dist[j][i] = d

# Precompute conflict masks
# For each pair (i, j), find all k such that (i, j, k) is isosceles.
# Isosceles if dist[i][j] == dist[i][k] or dist[i][j] == dist[j][k] or dist[i][k] == dist[j][k]
for i in range(M):
    for j in range(i + 1, M):
        d_ij = dist[i][j]
        mask = 0
        for k in range(M):
            if k == i or k == j:
                continue
            d_ik = dist[i][k]
            d_jk = dist[j][k]
            if d_ik == d_jk or d_ik == d_ij or d_jk == d_ij:
                mask |= (1 << k)
        conflict_mask[i][j] = mask
        conflict_mask[j][i] = mask

# Heuristic order: points with high "conflict degree" first
# Degree = number of points k such that (i, j, k) is isosceles for some j?
# Or simply, sum of popcount(conflict_mask[i][j]) for all j.
scores = [0] * M
for i in range(M):
    for j in range(M):
        if i != j:
            scores[i] += bin(conflict_mask[i][j]).count('1')

order = list(range(M))
order.sort(key=lambda x: scores[x], reverse=True)

best_size = M + 1
best_S = 0
start_time = time.time()
time_limit = 280.0
nodes_visited = 0

def dfs(idx, S, Forbidden, Must_Cover, current_size):
    global best_size, best_S, nodes_visited
    
    if time.time() - start_time > time_limit:
        return
    
    nodes_visited += 1
    
    if current_size >= best_size:
        return
        
    if Must_Cover == 0:
        if current_size < best_size:
            best_size = current_size
            best_S = S
        return
        
    if idx == M:
        return
        
    u = order[idx]
    
    # Option 1: Add u to S
    # u can be added if it is not in Forbidden
    if not (Forbidden & (1 << u)):
        new_Forbidden = Forbidden | (1 << u) # u itself is now in S, so it's "covered" by being in S? 
        # Actually, Forbidden is for points that CANNOT be in S.
        # Points in S are not in Forbidden.
        # But we need to update Forbidden to include points that conflict with u and existing S.
        
        # Check if u conflicts with any pair in S?
        # If u is in Forbidden, it means some pair in S forms isosceles with u.
        # We already checked `not (Forbidden & (1 << u))`.
        
        # Update Forbidden with new conflicts introduced by u
        s_bits = S
        while s_bits:
            lsb = s_bits & -s_bits
            v = lsb.bit_length() - 1
            s_bits ^= lsb
            new_Forbidden |= conflict_mask[u][v]
            
        # Check validity: S_new must not intersect Forbidden_new
        # S_new = S | (1 << u)
        S_new = S | (1 << u)
        if S_new & new_Forbidden:
            # Invalid, cannot add u
            pass
        else:
            # Calculate new Must_Cover
            # Must_Cover = U \ (S_new U Forbidden_new)
            # U is (1 << M) - 1
            U = (1 << M) - 1
            new_Must_Cover = U & ~(S_new | new_Forbidden)
            
            dfs(idx + 1, S_new, new_Forbidden, new_Must_Cover, current_size + 1)
            
    # Option 2: Skip u
    # If we skip u, it must be covered by future additions.
    # It remains in Must_Cover (unless it was in Forbidden, but we only pick from Must_Cover? 
    # No, we iterate through all points in order.
    # If u is in Must_Cover, and we skip it, it stays in Must_Cover.
    # If u is NOT in Must_Cover (i.e., it is in Forbidden or S), we just skip it.
    # But we should only branch on points in Must_Cover?
    # If u is in Forbidden, we MUST skip it.
    # If u is in S, we MUST have added it.
    # So we only have a choice if u is in Must_Cover.
    
    if Must_Cover & (1 << u):
        # We can skip u, but it must be covered later.
        # Just recurse
        dfs(idx + 1, S, Forbidden, Must_Cover, current_size)
    else:
        # u is not in Must_Cover, so it's either in S or Forbidden.
        # We must skip it (can't add if in Forbidden, already in S).
        dfs(idx + 1, S, Forbidden, Must_Cover, current_size)

def construct():
    global best_size, best_S, start_time, time_limit, nodes_visited
    start_time = time.time()
    nodes_visited = 0
    best_size = M + 1
    best_S = 0
    
    # Initial state
    S = 0
    Forbidden = 0
    U = (1 << M) - 1
    Must_Cover = U
    
    dfs(0, S, Forbidden, Must_Cover, 0)
    
    if best_S == 0:
        # Fallback: Greedy
        S = 0
        Forbidden = 0
        Must_Cover = U
        for idx in order:
            if Must_Cover & (1 << idx):
                # Try to add
                new_Forbidden = Forbidden | (1 << idx)
                s_bits = S
                while s_bits:
                    lsb = s_bits & -s_bits
                    v = lsb.bit_length() - 1
                    s_bits ^= lsb
                    new_Forbidden |= conflict_mask[idx][v]
                
                S_new = S | (1 << idx)
                if not (S_new & new_Forbidden):
                    S = S_new
                    Forbidden = new_Forbidden
                    Must_Cover = U & ~(S | Forbidden)
                    
        best_S = S
        best_size = bin(S).count('1')
        
    result = []
    s = best_S
    while s:
        lsb = s & -s
        idx = lsb.bit_length() - 1
        s ^= lsb
        result.append(points[idx])
    return result
