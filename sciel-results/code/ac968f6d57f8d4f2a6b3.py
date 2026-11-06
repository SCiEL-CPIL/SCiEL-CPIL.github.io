import time
import random
from itertools import combinations
from math import lcm

# Precomputation constants
N = 128
# lcm_matrix[i][j] = lcm(i+1, j+1)
# We use 1-based indexing for the problem, 0-based for arrays.
# So element i in problem corresponds to index i-1.
# Let's stick to 0-based indices for internal logic, 1-based for values.
# Value v corresponds to index v-1.

def construct():
    start_time = time.time()
    time_limit = 85.0  # Safety margin
    
    # 1. Precompute LCMs
    # lcm_val[i][j] for 0 <= i,j < N
    lcm_val = [[0]*N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            lcm_val[i][j] = lcm(i+1, j+1)
            
    # 2. Precompute bad_with[x][a]
    # bad_with[x][a] is a bitmask of indices b such that (x, a, b) is a bad triple.
    # Condition: b != x, b != a, and lcm(x,a) == lcm(x,b) == lcm(a,b)
    # Indices are 0-based.
    bad_with = [[0]*N for _ in range(N)]
    for x in range(N):
        for a in range(N):
            if x == a:
                continue
            l_xa = lcm_val[x][a]
            mask = 0
            for b in range(N):
                if b == x or b == a:
                    continue
                # Check if (x, a, b) forms equal LCMs
                if l_xa == lcm_val[x][b] == lcm_val[a][b]:
                    mask |= (1 << b)
            bad_with[x][a] = mask

    def can_add(x, S_mask):
        # Check if x can be added to set represented by S_mask
        # For each a in S_mask, check if bad_with[x][a] intersects S_mask
        # If yes, there is a b in S_mask such that (x,a,b) is bad.
        # Note: S_mask does not contain x.
        # We iterate through set bits in S_mask.
        m = S_mask
        while m:
            # Get lowest set bit
            lsb = m & (-m)
            a = lsb.bit_length() - 1
            if bad_with[x][a] & S_mask:
                return False
            m ^= lsb
        return True

    def get_size(mask):
        return bin(mask).count("1")
        
    def greedy_construction(order):
        S_mask = 0
        for x in order:
            if can_add(x, S_mask):
                S_mask |= (1 << x)
        return S_mask

    def local_search(S_mask):
        # Try to improve S_mask
        # 1. Maximize by adding any possible element
        changed = True
        while changed:
            changed = False
            for x in range(N):
                if not (S_mask & (1 << x)):
                    if can_add(x, S_mask):
                        S_mask |= (1 << x)
                        changed = True
        # 2. Try removing one element and re-maximizing
        # This allows escaping local maxima where removing one element allows adding two or more.
        improved = True
        while improved:
            improved = False
            # Iterate through elements in S_mask
            m = S_mask
            while m:
                lsb = m & (-m)
                u = lsb.bit_length() - 1
                S_temp = S_mask & ~(1 << u)
                # Re-maximize S_temp
                changed = True
                while changed:
                    changed = False
                    for x in range(N):
                        if not (S_temp & (1 << x)):
                            if can_add(x, S_temp):
                                S_temp |= (1 << x)
                                changed = True
                if get_size(S_temp) > get_size(S_mask):
                    S_mask = S_temp
                    improved = True
                    break # Restart the removal loop with the new larger set
                m ^= lsb
        return S_mask

    best_mask = 0
    best_size = 0
    
    # Run multiple random trials
    # Estimate time per trial:
    # Greedy: ~128 * 30 (avg size) * 1 (bitwise) = 3840 ops.
    # Local Search: 128 removals * (128 adds * 30 checks) = 128 * 3840 = 500k ops per pass.
    # If we do 10 passes, 5M ops. Python can do ~10M-50M simple ops/sec.
    # So 1 trial ~ 0.1s - 0.5s?
    # We can probably do 100-500 trials in 80s.
    
    random.seed(42)
    
    while time.time() - start_time < time_limit:
        order = list(range(N))
        random.shuffle(order)
        S_mask = greedy_construction(order)
        S_mask = local_search(S_mask)
        
        size = get_size(S_mask)
        if size > best_size:
            best_size = size
            best_mask = S_mask
            
        # Early stop if we found the theoretical max? 
        # Max possible is 128. Unlikely.
        # If best_size is very high, we might stop.
        
    # Convert mask to list of values (1-based)
    result = []
    for i in range(N):
        if best_mask & (1 << i):
            result.append(i + 1)
            
    if not result:
        # Fallback to seed behavior if something went wrong
        result = [1]
        
    return result
