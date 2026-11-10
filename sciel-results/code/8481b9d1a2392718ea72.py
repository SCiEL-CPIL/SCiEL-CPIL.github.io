import random
import time

N = 16
num_points = N * N
coords = [(i // N, i % N) for i in range(num_points)]

# Precompute isosceles relationships
# iso_pairs[i][j] is a set of k such that (i, j, k) forms an isosceles triangle
# Note: (i, j, k) is isosceles if dist(i,j)==dist(i,k) or dist(i,j)==dist(j,k) or dist(i,k)==dist(j,k)
# We only care about distinct points i, j, k.

dist = [[0] * num_points for _ in range(num_points)]
for i in range(num_points):
    x1, y1 = coords[i]
    for j in range(i + 1, num_points):
        x2, y2 = coords[j]
        d = (x1 - x2) ** 2 + (y1 - y2) ** 2
        dist[i][j] = d
        dist[j][i] = d

iso_pairs = [[set() for _ in range(num_points)] for __ in range(num_points)]

for i in range(num_points):
    for j in range(i + 1, num_points):
        for k in range(num_points):
            if k == i or k == j:
                continue
            # Check isosceles
            d1 = dist[i][j]
            d2 = dist[i][k]
            d3 = dist[j][k]
            if d1 == d2 or d1 == d3 or d2 == d3:
                iso_pairs[i][j].add(k)
                iso_pairs[j][i].add(k)

def is_maximal(S_set):
    """
    Check if S_set is a maximal no-isosceles set.
    S_set is a set of point indices.
    Maximal means:
    1. No point in S_set forms an isosceles triangle with any other two in S_set.
       (Assumed to be true by construction, but we should verify if we start with arbitrary set?
        In our flow, we start with a valid set and only remove points, so it remains valid.)
    2. For every point v NOT in S_set, there exist a, b in S_set such that {v, a, b} is isosceles.
    """
    if len(S_set) < 2:
        # If S is small, can it be maximal?
        # If S is empty, any v can be added. Not maximal.
        # If S has 1 point, any v can be added (no triangle). Not maximal.
        return False
        
    s_list = list(S_set)
    outside = [v for v in range(num_points) if v not in S_set]
    
    # Check each outside point
    for v in outside:
        found = False
        # Check if v forms isosceles with any pair in S_set
        for i in range(len(s_list)):
            a = s_list[i]
            # v in iso_pairs[a][b] means {v, a, b} is isosceles
            # Note: iso_pairs[a][b] contains k such that {a, b, k} is isosceles.
            # So we check if v is in iso_pairs[a][b].
            for j in range(i + 1, len(s_list)):
                b = s_list[j]
                if v in iso_pairs[a][b]:
                    found = True
                    break
            if found:
                break
        if not found:
            return False
            
    return True

def generate_maximal_set(seed=None):
    if seed is not None:
        random.seed(seed)
        
    order = list(range(num_points))
    random.shuffle(order)
    
    S_set = set()
    
    # Greedy add
    for p in order:
        valid = True
        s_list = list(S_set)
        for i in range(len(s_list)):
            a = s_list[i]
            for j in range(i + 1, len(s_list)):
                b = s_list[j]
                if p in iso_pairs[a][b]:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            S_set.add(p)
            
    # Repair to maximal
    # We need to add any point that can be added.
    # Since we only add, we can just iterate until no more points can be added.
    changed = True
    while changed:
        changed = False
        for p in range(num_points):
            if p in S_set:
                continue
            valid = True
            s_list = list(S_set)
            for i in range(len(s_list)):
                a = s_list[i]
                for j in range(i + 1, len(s_list)):
                    b = s_list[j]
                    if p in iso_pairs[a][b]:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                S_set.add(p)
                changed = True
                
    return S_set

def minimize_set(S_set, deadline):
    """
    Try to remove points from S_set while keeping it maximal.
    """
    s_list = list(S_set)
    changed = True
    while changed and time.time() < deadline:
        changed = False
        random.shuffle(s_list)
        for p in s_list:
            if p not in S_set:
                continue
            # Try removing p
            new_S = S_set - {p}
            if is_maximal(new_S):
                S_set = new_S
                changed = True
                # Restart loop with updated S_set and s_list
                s_list = list(S_set)
                break
    return S_set

def construct():
    start_time = time.time()
    deadline = start_time + 290
    
    best_size = num_points
    best_set = None
    
    seed = 0
    while time.time() < deadline:
        S = generate_maximal_set(seed)
        S = minimize_set(S, deadline)
        
        if len(S) < best_size:
            best_size = len(S)
            best_set = S
            # If we find a very small set, we might want to polish it more?
            # But minimize_set already does local search.
        
        seed += 1
        
    if best_set is None:
        # Fallback
        return [coords[i] for i in range(num_points)]
        
    return [coords[i] for i in best_set]
