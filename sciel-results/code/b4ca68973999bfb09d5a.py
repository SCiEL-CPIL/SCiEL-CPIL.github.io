"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

N_MAX = 4096
MAX_SUM = 3 * N_MAX  # 12288

def construct():
    deadline = time.time() + 85.0
    best_set = []
    best_len = 0
    
    for seed in range(200):
        if time.time() > deadline - 1.0:
            break
        
        random.seed(seed)
        
        # Build a valid sparse skeleton using geometric progression
        # Use base b where b^k * 3 <= 3*N_MAX for uniqueness via digit separation
        # For triples (a,b,c) with a<=b<=c, sum = a+b+c
        # If elements are powers of B >= 3, then each triple sum has a unique
        # base-B representation (since max coefficient per digit is 3 < B for B>=4)
        
        # Strategy: pick a base, generate skeleton, then greedily insert more elements
        # Try different bases and starting points
        
        base = 4  # base 4 ensures digit separation for triples (coeff up to 3)
        
        # Generate skeleton: powers of base
        skeleton = []
        val = 1
        while val <= N_MAX:
            skeleton.append(val)
            val *= base
        
        # Add some random perturbations to the skeleton for diversity
        if seed % 3 == 0:
            # Pure powers of 4
            pass
        elif seed % 3 == 1:
            # Powers of 4 with small offsets
            skeleton = []
            val = 1
            while val <= N_MAX:
                offset = random.randint(0, max(1, val // 10))
                skeleton.append(val + offset)
                val = val * base + random.randint(0, 2)
            skeleton = [x for x in skeleton if x <= N_MAX]
        else:
            # Mixed: some powers of 4, some random sparse
            skeleton = []
            val = 1
            while val <= N_MAX and len(skeleton) < 10:
                if random.random() < 0.7:
                    skeleton.append(val)
                else:
                    # Insert a random value in the gap
                    gap_start = val
                    gap_end = min(val * base, N_MAX)
                    if gap_end > gap_start:
                        skeleton.append(random.randint(gap_start + 1, gap_end))
                val *= base
            skeleton = sorted(set(x for x in skeleton if x <= N_MAX))
        
        # Use numpy boolean array for fast conflict checking
        sums_used = np.zeros(MAX_SUM + 1, dtype=bool)
        selected = []
        selected_set = set()
        
        # Add skeleton elements
        for x in sorted(skeleton):
            if x > N_MAX or x in selected_set:
                continue
            if _try_add_np(selected, selected_set, sums_used, x):
                pass
        
        # Now greedily try to add more elements
        # Try candidates in increasing order (smaller numbers are "cheaper" in terms of sum space)
        # But also try random order for diversity
        if seed % 2 == 0:
            # Increasing order - pack small numbers
            candidates = list(range(1, N_MAX + 1))
        else:
            # Random order
            candidates = list(range(1, N_MAX + 1))
            random.shuffle(candidates)
        
        for x in candidates:
            if time.time() > deadline - 0.5:
                break
            if x in selected_set:
                continue
            if _try_add_np(selected, selected_set, sums_used, x):
                pass
        
        if len(selected) > best_len:
            best_len = len(selected)
            best_set = list(selected)
        
        # Early stop
        if best_len >= 30:
            break
    
    return best_set

def _try_add_np(selected, selected_set, sums_used, x):
    """Try to add x to the set using numpy boolean array for conflict checking.
    Returns True if x was added successfully."""
    n = len(selected)
    
    # Check all triple sums involving x
    # Type 1: (x, x, x) -> 3x
    s = 3 * x
    if sums_used[s]:
        return False
    
    # Type 2: (y, x, x) for each y in selected -> y + 2x
    # Type 3: (y, y, x) for each y in selected -> 2y + x
    # Type 4: (y, z, x) for y < z in selected -> y + z + x
    
    # Collect all new sums and check for conflicts
    new_sums = [3 * x]
    
    for y in selected:
        new_sums.append(y + 2 * x)
        new_sums.append(2 * y + x)
    
    for i in range(n):
        for j in range(i + 1, n):
            new_sums.append(selected[i] + selected[j] + x)
    
    # Check all new sums against existing
    # Convert to numpy array for vectorized check
    arr = np.array(new_sums, dtype=np.int32)
    
    # Check if any new sum conflicts with existing
    if np.any(sums_used[arr]):
        return False
    
    # Check for duplicates within new_sums
    # Sort and check for consecutive duplicates
    arr_sorted = np.sort(arr)
    if np.any(np.diff(arr_sorted) == 0):
        return False
    
    # All good - add to set
    sums_used[arr] = True
    selected.append(x)
    selected_set.add(x)
    return True
# EVOLVE-BLOCK-END
