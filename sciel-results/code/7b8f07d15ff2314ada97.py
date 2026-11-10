import random
import time

N = 16
POINTS = [(x, y) for x in range(N) for y in range(N)]
NUM_POINTS = N * N

# Precompute squared distances
dist = [[0] * NUM_POINTS for _ in range(NUM_POINTS)]
for i in range(NUM_POINTS):
    xi, yi = POINTS[i]
    for j in range(i + 1, NUM_POINTS):
        xj, yj = POINTS[j]
        d = (xi - xj) ** 2 + (yi - yj) ** 2
        dist[i][j] = d
        dist[j][i] = d

# Precompute: iso[k][a] = bitset of b such that (k,a,b) forms isosceles triangle
# where k,a,b are distinct.
iso = [[0] * NUM_POINTS for _ in range(NUM_POINTS)]
for k in range(NUM_POINTS):
    for a in range(NUM_POINTS):
        if k == a:
            continue
        dk = dist[k][a]
        mask = 0
        for b in range(NUM_POINTS):
            if b == k or b == a:
                continue
            da = dist[k][b]
            dab = dist[a][b]
            if dk == da or dk == dab or da == dab:
                mask |= (1 << b)
        iso[k][a] = mask
        iso[a][k] = mask

def can_add_bitset(k, S_mask):
    """Check if point k can be added to set represented by S_mask (bitset)."""
    for a_bit in range(NUM_POINTS):
        if not (S_mask & (1 << a_bit)):
            continue
        a = a_bit
        if iso[k][a] & S_mask:
            return False
    return True

def is_maximal_bitset(S_mask):
    """Check if set S_mask is maximal (no point can be added)."""
    for k in range(NUM_POINTS):
        if S_mask & (1 << k):
            continue
        if can_add_bitset(k, S_mask):
            return False
    return True

def greedy_maximal_bitset(order):
    """Construct a maximal set using the given order."""
    S_mask = 0
    for k in order:
        if can_add_bitset(k, S_mask):
            S_mask |= (1 << k)
    return S_mask

def mask_to_list(mask):
    return [i for i in range(NUM_POINTS) if mask & (1 << i)]

def shrink_bitset(S_mask):
    """Try to remove points from S_mask while maintaining maximality."""
    improved = True
    while improved:
        improved = False
        for p in range(NUM_POINTS):
            if not (S_mask & (1 << p)):
                continue
            S_trial = S_mask & ~(1 << p)
            if is_maximal_bitset(S_trial):
                S_mask = S_trial
                improved = True
                break
    return S_mask

def local_search_swap(S_mask, rng, max_iters=50):
    """Try to improve S_mask by removing a point and adding another."""
    current = S_mask
    current_size = bin(current).count('1')
    
    for _ in range(max_iters):
        # Try removing a random point from the set
        in_set = [i for i in range(NUM_POINTS) if current & (1 << i)]
        if not in_set:
            break
        
        p_to_remove = rng.choice(in_set)
        S_trial = current & ~(1 << p_to_remove)
        
        # Try to find a point that can be added
        candidates = [k for k in range(NUM_POINTS) if not (S_trial & (1 << k))]
        rng.shuffle(candidates)
        
        found = False
        for k in candidates:
            if can_add_bitset(k, S_trial):
                S_trial |= (1 << k)
                # Check if still maximal
                if is_maximal_bitset(S_trial):
                    sz = bin(S_trial).count('1')
                    if sz < current_size:
                        current = S_trial
                        current_size = sz
                    found = True
                    break
        
        if not found:
            continue
    
    return current

def construct():
    start_time = time.time()
    deadline = start_time + 280
    
    best_mask = None
    best_size = NUM_POINTS + 1
    
    rng = random.Random(42)
    
    # Initial greedy with row-major
    base_order = list(range(NUM_POINTS))
    S_base = greedy_maximal_bitset(base_order)
    S_base_shrunk = shrink_bitset(S_base)
    best_mask = S_base_shrunk
    best_size = bin(S_base_shrunk).count('1')
    
    # Randomized greedy attempts with local search
    while time.time() < deadline:
        order = list(range(NUM_POINTS))
        rng.shuffle(order)
        
        S = greedy_maximal_bitset(order)
        S_shrunk = shrink_bitset(S)
        
        # Apply local search swap
        S_local = local_search_swap(S_shrunk, rng, max_iters=30)
        
        sz = bin(S_local).count('1')
        if sz < best_size:
            best_mask = S_local
            best_size = sz
    
    return [POINTS[i] for i in mask_to_list(best_mask)]
