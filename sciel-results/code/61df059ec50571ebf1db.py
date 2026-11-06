import numpy as np
import random
import time

N = 16
GRID_POINTS = [(x, y) for x in range(N) for y in range(N)]
NUM_POINTS = len(GRID_POINTS)
POINT_TO_INDEX = {p: i for i, p in enumerate(GRID_POINTS)}

# Precompute squared distances
coords = np.array(GRID_POINTS, dtype=np.int32)
diffs = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
dist_sq = np.sum(diffs * diffs, axis=-1)

def is_no_isosceles(S_indices):
    if len(S_indices) < 3:
        return True
    
    S_arr = np.array(S_indices)
    D = dist_sq[S_arr, S_arr]
    
    # For each point a in S, check if distances to other points in S are unique
    # D[i, j] is dist between S[i] and S[j]
    # We need to check rows of D, excluding diagonal
    
    # Mask diagonal
    D_masked = np.ma.masked_array(D, mask=np.eye(len(S_arr), dtype=bool))
    
    # Check uniqueness in each row
    for i in range(len(S_arr)):
        row = D_masked[i, :].compressed()
        if len(np.unique(row)) != len(row):
            return False
            
    return True

def is_maximal(S_indices):
    S_set = set(S_indices)
    if not S_indices:
        return True
        
    # dist_mask[a] is a bitmask of distances from a to other points in S
    dist_masks = {}
    for a in S_indices:
        mask = 0
        for b in S_indices:
            if a != b:
                d = dist_sq[a, b]
                mask |= (1 << d)
        dist_masks[a] = mask
    
    for q in range(NUM_POINTS):
        if q in S_set:
            continue
        
        # Check 1: Distances from q to S are unique
        dists = dist_sq[q, S_indices]
        if len(np.unique(dists)) != len(dists):
            continue
            
        # Check 2: For each a in S, d(q,a) must not be in dist_masks[a]
        valid = True
        for i, a in enumerate(S_indices):
            d = dists[i]
            if (1 << d) & dist_masks[a]:
                valid = False
                break
                
        if valid:
            # q can be added, so S is NOT maximal
            return False
            
    return True

def is_valid(S_indices):
    return is_no_isosceles(S_indices) and is_maximal(S_indices)

def greedy_construct(random_order):
    S = []
    S_set = set()
    dist_sets = {}
    
    for p in random_order:
        if p in S_set:
            continue
        
        if not S:
            S.append(p)
            S_set.add(p)
            dist_sets[p] = set()
            continue
            
        dists_from_p = dist_sq[p, S]
        if len(np.unique(dists_from_p)) != len(dists_from_p):
            continue
        
        valid = True
        for a in S:
            if dist_sq[p, a] in dist_sets[a]:
                valid = False
                break
        
        if valid:
            S.append(p)
            S_set.add(p)
            
            for a in S[:-1]:
                d = dist_sq[p, a]
                dist_sets[a].add(d)
            
            dist_sets[p] = set(dist_sq[p, S[:-1]])
    
    return S

def construct():
    start_time = time.time()
    deadline = start_time + 280
    
    # Initial solution
    random.seed(42)
    random_order = list(range(NUM_POINTS))
    random.shuffle(random_order)
    S = greedy_construct(random_order)
    
    if not is_valid(S):
        for _ in range(10):
            random.shuffle(random_order)
            S = greedy_construct(random_order)
            if is_valid(S):
                break
    
    best_S = S.copy()
    best_size = len(S)
    
    T = 5.0
    cooling = 0.995
    
    while time.time() < deadline:
        if T < 0.1:
            break
            
        current_size = len(S)
        
        # Generate a move
        move_type = random.randint(0, 2)
        S_new = None
        
        if move_type == 0: # Remove
            if S:
                p = random.choice(S)
                S_new = [x for x in S if x != p]
        elif move_type == 1: # Swap
            if S:
                p = random.choice(S)
                candidates = [q for q in range(NUM_POINTS) if q not in set(S)]
                if candidates:
                    q = random.choice(candidates)
                    S_new = [x for x in S if x != p] + [q]
        else: # Replace (2 out, 1 in)
            if len(S) >= 2:
                p, r = random.sample(S, 2)
                candidates = [q for q in range(NUM_POINTS) if q not in set(S)]
                if candidates:
                    q = random.choice(candidates)
                    S_new = [x for x in S if x != p and x != r] + [q]
        
        if S_new is None:
            T *= cooling
            continue
            
        if is_valid(S_new):
            new_size = len(S_new)
            delta = new_size - current_size
            
            if delta < 0 or random.random() < np.exp(-delta / T):
                S = S_new
                if new_size < best_size:
                    best_size = new_size
                    best_S = S_new.copy()
        
        T *= cooling
        
    return [GRID_POINTS[i] for i in best_S]

if __name__ == "__main__":
    res = construct()
    print(f"Found set of size {len(res)}")
