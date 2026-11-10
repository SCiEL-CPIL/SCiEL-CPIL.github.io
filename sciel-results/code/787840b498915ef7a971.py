import random
import time
from math import gcd

N = 128
MAX_T = 85.0  # wall-clock budget for construct()

# Precompute gcd matrix
gcd_mat = [[0] * (N + 1) for _ in range(N + 1)]
for i in range(1, N + 1):
    for j in range(i, N + 1):
        g = gcd(i, j)
        gcd_mat[i][j] = g
        gcd_mat[j][i] = g

# Precompute Forbidden[x][y] bitmask
# Forbidden[x][y] contains bits for z such that {x, y, z} is a bad triple.
# A triple {a, b, c} is bad if gcd(a,b) == gcd(a,c) == gcd(b,c).
# We assume x, y, z are distinct.
Forbidden = [[0] * (N + 1) for _ in range(N + 1)]

for x in range(1, N + 1):
    for y in range(1, N + 1):
        if x == y:
            continue
        d_xy = gcd_mat[x][y]
        # Find all z != x, z != y such that gcd(x,z)==d_xy and gcd(y,z)==d_xy
        mask = 0
        for z in range(1, N + 1):
            if z == x or z == y:
                continue
            if gcd_mat[x][z] == d_xy and gcd_mat[y][z] == d_xy:
                mask |= (1 << z)
        Forbidden[x][y] = mask
        # Symmetry: Forbidden[y][x] is the same set of z's
        Forbidden[y][x] = mask

def is_valid_add(x, S_mask):
    """
    Check if x can be added to the set represented by S_mask.
    S_mask is an int where bit i is 1 if i is in the set.
    """
    # For each y in S, check if S contains any z in Forbidden[x][y]
    # Iterate through bits set in S_mask
    temp = S_mask
    while temp:
        # Get lowest set bit index
        lsb = temp & -temp
        y = lsb.bit_length() - 1
        temp ^= lsb
        
        # Check if S_mask intersects with Forbidden[x][y]
        # If it does, there is a z in S such that {x,y,z} is bad.
        if S_mask & Forbidden[x][y]:
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + MAX_T - 5.0  # safety margin
    
    best_S_mask = 0
    best_size = 0
    seed_counter = 0
    
    while time.time() < deadline:
        seed_counter += 1
        random.seed(seed_counter * 7919)
        
        # 1. Greedy Construction
        S_mask = 0
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        
        for x in candidates:
            if is_valid_add(x, S_mask):
                S_mask |= (1 << x)
        
        size = S_mask.bit_count()
        if size > best_size:
            best_size = size
            best_S_mask = S_mask
        
        # 2. Local Search: Try to add any remaining element
        # This is a cheap improvement step
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for x in range(1, N + 1):
                if not (S_mask & (1 << x)):
                    if is_valid_add(x, S_mask):
                        S_mask |= (1 << x)
                        size += 1
                        changed = True
                        if size > best_size:
                            best_size = size
                            best_S_mask = S_mask
        
        # 3. Local Search: Swaps
        # Try to replace one element with another to potentially allow more additions later
        # Or just random restarts are often sufficient for N=128 with good greedy.
        # Let's add a simple swap heuristic: 
        # Pick a random x not in S. Pick a random y in S.
        # If removing y allows adding x AND the new set is valid, do it?
        # Actually, removing y doesn't automatically allow adding x.
        # We need to check if x can be added to (S \ {y}).
        
        if time.time() < deadline - 2.0:
            # Perform some swap attempts
            S_list = [i for i in range(1, N + 1) if S_mask & (1 << i)]
            if S_list:
                for _ in range(10): # 10 swap attempts per iteration
                    if time.time() > deadline: break
                    x = random.randint(1, N)
                    if S_mask & (1 << x):
                        continue
                    y = random.choice(S_list)
                    
                    # Check if we can swap y for x
                    # New set would be (S_mask without y) with x added
                    S_temp = S_mask ^ (1 << y)
                    if is_valid_add(x, S_temp):
                        # The swap is valid. But does it help?
                        # We don't know if it helps until we try to fill it.
                        # Let's just accept it as a perturbation and then re-greedy fill.
                        S_mask = S_temp | (1 << x)
                        S_list.remove(y)
                        S_list.append(x)
                        # Re-greedy fill
                        changed = True
                        while changed and time.time() < deadline:
                            changed = False
                            for z in range(1, N + 1):
                                if not (S_mask & (1 << z)):
                                    if is_valid_add(z, S_mask):
                                        S_mask |= (1 << z)
                                        size += 1
                                        changed = True
                        if size > best_size:
                            best_size = size
                            best_S_mask = S_mask
    
    # Convert best_S_mask to list
    result = []
    for i in range(1, N + 1):
        if best_S_mask & (1 << i):
            result.append(i)
            
    if not result:
        return [1]
        
    return result
