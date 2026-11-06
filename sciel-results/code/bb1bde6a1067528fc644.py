import random
import time
import itertools
import numpy as np

N = 16
ALL_POINTS = list(itertools.product(range(N), repeat=2))
NUM_POINTS = len(ALL_POINTS)
ALL_MASK = (1 << NUM_POINTS) - 1

# Pre-compute conflict masks
# conflict[i][j] is a bitmask of points k such that {i, j, k} is isosceles.
# i, j, k distinct.
def _dist2(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2

# conflict_masks[i][j]
conflict_masks = [[0] * NUM_POINTS for _ in range(NUM_POINTS)]

# We only need i < j for symmetry, but let's fill all i != j.
# For efficiency, we can compute for i < j and mirror.
for i in range(NUM_POINTS):
    for j in range(i + 1, NUM_POINTS):
        p_i = ALL_POINTS[i]
        p_j = ALL_POINTS[j]
        d_ij = _dist2(p_i, p_j)
        mask = 0
        for k in range(NUM_POINTS):
            if k == i or k == j:
                continue
            p_k = ALL_POINTS[k]
            d_ik = _dist2(p_i, p_k)
            d_jk = _dist2(p_j, p_k)
            if d_ij == d_ik or d_ij == d_jk or d_ik == d_jk:
                mask |= (1 << k)
        conflict_masks[i][j] = mask
        conflict_masks[j][i] = mask

def _compute_union_conflicts(S_indices):
    """Compute U(S) as a bitmask."""
    union_mask = 0
    n = len(S_indices)
    for a in range(n):
        i = S_indices[a]
        for b in range(a + 1, n):
            j = S_indices[b]
            union_mask |= conflict_masks[i][j]
    return union_mask

def _get_fixed_point(S_indices):
    """Compute f(S) = All \ U(S)."""
    union_mask = _compute_union_conflicts(S_indices)
    return ALL_MASK & ~union_mask

def construct():
    best_set = None
    best_size = float('inf')
    
    start_time = time.time()
    deadline = start_time + 290
    
    rng = random.Random(42)
    
    # Simulated Annealing
    # State: S (list of indices)
    # We want to find S such that S == f(S) and |S| is small.
    
    # Initialize with a random small set
    # Try to find a fixed point close to a small set.
    
    # Strategy: 
    # 1. Start with random S of size k.
    # 2. Compute S_fp = f(S).
    # 3. If S_fp is valid (it always is) and |S_fp| < best_size, check if S_fp == f(S_fp).
    #    If yes, update best.
    # 4. Use SA to navigate.
    
    # Actually, simpler: Just run many iterations of:
    # 1. Pick random S.
    # 2. Compute S_fp = f(S).
    # 3. If S_fp is a fixed point (S_fp == f(S_fp)), record it.
    # 4. Try to perturb S and repeat.
    
    # Let's try to minimize |S| directly with a penalty for not being a fixed point.
    # E(S) = |S| + 10 * |S ^ f(S)|
    
    # We will run SA on bitmasks.
    
    best_mask = None
    best_energy = float('inf')
    
    # Initial state: random small set
    # Let's start with size 10-15
    
    for restart in range(20):
        if time.time() > deadline:
            break
            
        # Random initial set
        init_size = rng.randint(8, 15)
        S_mask = 0
        S_indices = set(rng.sample(range(NUM_POINTS), init_size))
        for i in S_indices:
            S_mask |= (1 << i)
            
        # SA Parameters
        T = 1.0
        cooling = 0.999
        steps = 5000
        
        current_mask = S_mask
        current_indices = list(S_indices)
        
        # Compute initial energy
        # f(S)
        fp_mask = _get_fixed_point(current_indices)
        # Energy = popcount(S) + penalty * popcount(S ^ fp)
        # We want S to be small AND S == fp.
        # If S == fp, penalty is 0.
        # Then we just minimize popcount(S).
        
        # Helper to get indices from mask
        def mask_to_indices(m):
            return [i for i in range(NUM_POINTS) if (m >> i) & 1]
            
        def get_energy(mask):
            inds = mask_to_indices(mask)
            fp = _get_fixed_point(inds)
            # Check if fixed point
            # If fp == mask, energy is len(inds)
            # Else, energy is len(inds) + penalty * hamming_distance(mask, fp)
            # We want to find ANY fixed point with small size.
            # So if it's a fixed point, reward it heavily.
            if fp == mask:
                return len(inds)
            else:
                # Hamming distance
                diff = mask ^ fp
                hamming = bin(diff).count('1')
                return len(inds) + 5 * hamming
        
        current_energy = get_energy(current_mask)
        best_local_mask = current_mask
        best_local_energy = current_energy
        
        for step in range(steps):
            if time.time() > deadline:
                break
                
            # Pick a random point to flip
            p = rng.randint(0, NUM_POINTS - 1)
            new_mask = current_mask ^ (1 << p)
            
            # Quick check: if new_mask is 0, skip?
            # Compute new energy
            new_energy = get_energy(new_mask)
            
            delta = new_energy - current_energy
            
            if delta < 0 or rng.random() < pow(2.718, -delta / T):
                current_mask = new_mask
                current_energy = new_energy
                
                if current_energy < best_local_energy:
                    best_local_energy = current_energy
                    best_local_mask = current_mask
                    
            T *= cooling
            
        # End of SA run
        # Check best_local_mask
        if best_local_energy < best_energy:
            # Verify it's a valid fixed point
            inds = mask_to_indices(best_local_mask)
            fp = _get_fixed_point(inds)
            if fp == best_local_mask:
                # It is a valid maximal no-isosceles set
                if len(inds) < best_size:
                    best_size = len(inds)
                    best_set = [ALL_POINTS[i] for i in inds]
                    best_energy = best_local_energy
                    # print(f"Found size {best_size} at restart {restart}")
                    
    # Fallback
    if best_set is None:
        # Greedy fallback
        S = []
        valid = set(range(NUM_POINTS))
        while valid:
            # Pick point that blocks most others
            best_p = None
            max_blocked = -1
            for p in valid:
                # Check if p can be added
                # p must not be in U(S)
                # Compute U(S)
                if not S:
                    union_mask = 0
                else:
                    union_mask = _compute_union_conflicts(S)
                if (union_mask >> p) & 1:
                    continue
                # Calculate new blocked
                new_S = S + [p]
                new_union = _compute_union_conflicts(new_S)
                blocked_count = bin(new_union).count('1')
                if blocked_count > max_blocked:
                    max_blocked = blocked_count
                    best_p = p
            if best_p is None:
                break
            S.append(best_p)
        best_set = [ALL_POINTS[i] for i in S]
        best_size = len(S)

    return best_set
