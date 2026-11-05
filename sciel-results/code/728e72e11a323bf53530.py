"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16
N2 = N * N

# EVOLVE-BLOCK-START
import itertools
import random
import time

# Precompute point coordinates and index mapping
points = list(itertools.product(range(N), repeat=2))
point_to_idx = {p: i for i, p in enumerate(points)}

# Precompute squared distances
dist2 = [[0] * N2 for _ in range(N2)]
for i in range(N2):
    xi, yi = points[i]
    for j in range(i + 1, N2):
        xj, yj = points[j]
        dx = xi - xj
        dy = yi - yj
        d = dx * dx + dy * dy
        dist2[i][j] = d
        dist2[j][i] = d

def _is_iso_idx(i, j, k):
    d1 = dist2[i][j]
    d2 = dist2[i][k]
    d3 = dist2[j][k]
    return d1 == d2 or d1 == d3 or d2 == d3

# Precompute iso_with[i][j] as a bitset of k such that i, j, k form an isosceles triangle
# iso_with[i][j] is an integer where bit k is set if _is_iso_idx(i, j, k) is True
iso_with = [[0] * N2 for _ in range(N2)]
for i in range(N2):
    for j in range(N2):
        if i == j:
            continue
        mask = 0
        for k in range(N2):
            if k == i or k == j:
                continue
            if _is_iso_idx(i, j, k):
                mask |= (1 << k)
        iso_with[i][j] = mask

def is_maximal(S_list):
    """
    S_list: list of indices in the set
    Returns True if S_list is a maximal no-isosceles set
    """
    S_bits = 0
    for idx in S_list:
        S_bits |= (1 << idx)
    
    # Check each point not in S
    for q in range(N2):
        if (S_bits >> q) & 1:
            continue
        
        blocked = False
        # Check if q is blocked by any pair in S
        # q is blocked if there exists a, b in S such that q, a, b is isosceles
        # This is equivalent to: for some a in S, there exists b in S such that b is in iso_with[q][a]
        for a in S_list:
            if (iso_with[q][a] & S_bits):
                blocked = True
                break
        
        if not blocked:
            return False
            
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 290
    
    best_size = float('inf')
    best_set = []
    
    def greedy_construct(order):
        chosen = []
        S_bits = 0
        for p in order:
            valid = True
            # Check if p can be added to chosen
            # p is valid if for all a in chosen, iso_with[p][a] & S_bits is 0
            for a in chosen:
                if (iso_with[p][a] & S_bits):
                    valid = False
                    break
            if valid:
                chosen.append(p)
                S_bits |= (1 << p)
        return chosen

    def shrink(S_list):
        S_list = list(S_list)
        changed = True
        while changed:
            changed = False
            indices = list(range(len(S_list)))
            random.shuffle(indices)
            for idx in indices:
                p = S_list[idx]
                candidate = S_list[:idx] + S_list[idx + 1:]
                if is_maximal(candidate):
                    S_list = candidate
                    changed = True
                    break
        return S_list

    def shrink_pairs(S_list):
        """Try removing pairs of points to escape local minima."""
        S_list = list(S_list)
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            n = len(S_list)
            if n < 2:
                break
            # Try random pairs
            for _ in range(min(20, n * (n - 1) // 2)):
                i, j = random.sample(range(n), 2)
                candidate = [S_list[k] for k in range(n) if k != i and k != j]
                if is_maximal(candidate):
                    S_list = candidate
                    improved = True
                    break
        return S_list

    def local_search(S_list):
        S_list = list(S_list)
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            
            # Try to remove single points (shrink)
            new_S = shrink(S_list)
            if len(new_S) < len(S_list):
                S_list = new_S
                improved = True
                continue
            
            # Try to remove pairs of points
            new_S = shrink_pairs(S_list)
            if len(new_S) < len(S_list):
                S_list = new_S
                improved = True
                continue
            
            # Try swaps
            S_bits = 0
            for idx in S_list:
                S_bits |= (1 << idx)
            
            # Pick a random point in S and a random point not in S
            if len(S_list) == 0:
                break
                
            p = random.choice(S_list)
            # Find a q not in S
            q = random.randint(0, N2 - 1)
            while (S_bits >> q) & 1:
                q = random.randint(0, N2 - 1)
                
            # Check if S \ {p} U {q} is valid
            S_minus_p_bits = S_bits ^ (1 << p)
            S_minus_p_list = [x for x in S_list if x != p]
            
            valid_swap = True
            for a in S_minus_p_list:
                if (iso_with[q][a] & S_minus_p_bits):
                    valid_swap = False
                    break
                    
            if valid_swap:
                new_S = S_minus_p_list + [q]
                # Try to shrink the new set
                new_S = shrink(new_S)
                new_S = shrink_pairs(new_S)
                if len(new_S) < len(S_list):
                    S_list = new_S
                    improved = True
                    break
        return S_list

    # Run restarts with different strategies
    while time.time() < deadline - 20:
        strategy = random.random()
        
        if strategy < 0.3:
            # Random order
            order = list(range(N2))
            random.shuffle(order)
        elif strategy < 0.6:
            # Prefer points with fewer conflicts (greedy heuristic)
            # Count how many points each point conflicts with
            conflict_count = [0] * N2
            for i in range(N2):
                for j in range(N2):
                    if i != j:
                        # i and j conflict if there exists k such that i,j,k is isosceles
                        # Actually, we want points that are "easy" to include
                        # A point is easy if it has few isosceles relationships
                        pass
            # Use a simpler heuristic: prefer corners and edges
            order = list(range(N2))
            random.shuffle(order)
            # Move points with fewer neighbors to the front
            # For now, just use random
        else:
            # Spiral order from center
            cx, cy = N // 2, N // 2
            order = sorted(range(N2), key=lambda i: (abs(points[i][0] - cx), abs(points[i][1] - cy), random.random()))
        
        S = greedy_construct(order)
        S = local_search(S)
        
        if len(S) < best_size:
            best_size = len(S)
            best_set = S[:]
            
    # Final polish: try to further shrink the best set found
    if best_set and time.time() < deadline - 5:
        best_set = shrink(best_set)
        best_set = shrink_pairs(best_set)
        if len(best_set) < best_size:
            best_size = len(best_set)
            
    # Convert indices back to points
    return [points[i] for i in best_set]

# EVOLVE-BLOCK-END
