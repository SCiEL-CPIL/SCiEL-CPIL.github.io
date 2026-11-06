import itertools
import random
import time
import sys

N = 16
POINTS = [(x, y) for x in range(N) for y in range(N)]
P = len(POINTS)
POINT_INDEX = {p: i for i, p in enumerate(POINTS)}

# Precompute isosceles relationships
# iso_mask[i] is a list of masks? Or just a set of pairs?
# For a point i, and a set S, i is blocked if there exist a, b in S such that {i, a, b} is isosceles.
# Let's precompute for each i, a list of (a, b) pairs that form isosceles with i.
# Actually, for speed, for each i, and each a, let's store the set of b's such that {i, a, b} is isosceles.
# iso_partners[i][a] = set of b such that dist(i,a) == dist(i,b) or dist(i,a) == dist(a,b) or dist(i,b) == dist(a,b)
# Note: The condition is that the triangle is isosceles.
# d(i,a) == d(i,b) OR d(i,a) == d(a,b) OR d(i,b) == d(a,b).

# To speed up maximality check:
# We want to know if for all x not in S, x is blocked.
# x is blocked if exists a in S such that (S intersects iso_partners[x][a]) is non-empty.
# iso_partners[x][a] can be precomputed as a bitset or a set of indices.

# Let's use bitsets (integers) for S and for the partner sets.
# P = 256. Python integers can handle 256 bits easily.

iso_partners = [[0] * P for _ in range(P)] # iso_partners[x][a] is a bitmask of b's

for x in range(P):
    px, py = POINTS[x]
    for a in range(P):
        if a == x: continue
        pa, qa = POINTS[a]
        dax = (px - pa)**2 + (py - qa)**2
        for b in range(P):
            if b == x or b == a: continue
            pb, qb = POINTS[b]
            dbx = (px - pb)**2 + (py - qb)**2
            dab = (pa - pb)**2 + (qa - qb)**2
            if dax == dbx or dax == dab or dbx == dab:
                iso_partners[x][a] |= (1 << b)

def is_blocked(x, S_mask):
    """Check if point x is blocked by set S_mask."""
    for a in range(P):
        if S_mask & (1 << a):
            if S_mask & iso_partners[x][a]:
                return True
    return False

def is_maximal(S_mask):
    """Check if S_mask is a maximal no-isosceles set."""
    # For all x not in S, x must be blocked.
    for x in range(P):
        if not (S_mask & (1 << x)):
            if not is_blocked(x, S_mask):
                return False
    return True

def is_no_isosceles(S_mask):
    """Check if S_mask is a no-isosceles set."""
    # This is implicitly maintained by construction, but useful for verification.
    # For any a, b in S, no c in S such that {a, b, c} is isosceles.
    # We can check: for each a in S, for each b in S, S_mask & iso_partners[a][b] should be 0?
    # Wait, iso_partners[a][b] contains c such that {a, b, c} is isosceles.
    # So if S_mask contains a and b, it must not contain any c in iso_partners[a][b].
    for a in range(P):
        if S_mask & (1 << a):
            for b in range(P):
                if b > a and (S_mask & (1 << b)):
                    if S_mask & iso_partners[a][b]:
                        return False
    return True

def construct():
    start_time = time.time()
    time_limit = 280  # Leave margin for overhead
    
    best_size = P
    best_set = 0
    
    # Precompute a list of points for random shuffling
    # We will use random restarts
    
    for restart in range(1000):
        if time.time() - start_time > time_limit:
            break
            
        # 1. Greedy Construction
        # Random order
        order = list(range(P))
        random.shuffle(order)
        
        S_mask = 0
        for p in order:
            # Check if p can be added to S_mask
            # p can be added if for all a, b in S_mask, {p, a, b} is NOT isosceles.
            # This is equivalent to: for all a in S_mask, S_mask & iso_partners[p][a] == 0.
            # Wait, iso_partners[p][a] contains b such that {p, a, b} is isosceles.
            # So if any b in S_mask is in iso_partners[p][a], we have a conflict.
            can_add = True
            for a in range(P):
                if S_mask & (1 << a):
                    if S_mask & iso_partners[p][a]:
                        can_add = False
                        break
            if can_add:
                S_mask |= (1 << p)
        
        # Now S_mask is a maximal no-isosceles set (greedy ensures maximality if we process all points)
        # Actually, greedy processing all points ensures that any point not added was blocked at the time.
        # Since we only add points, the set of blockers only grows. So if it was blocked then, it is blocked now.
        # Thus, S_mask is maximal.
        
        current_size = bin(S_mask).count('1')
        
        # 2. Local Search to minimize size
        # Try to remove points
        improved = True
        while improved and time.time() - start_time < time_limit:
            improved = False
            # Random order for removal attempts
            removal_order = list(range(P))
            random.shuffle(removal_order)
            
            for v in removal_order:
                if not (S_mask & (1 << v)):
                    continue
                
                # Try removing v
                new_S_mask = S_mask & ~(1 << v)
                
                # Check if new_S_mask is maximal
                if is_maximal(new_S_mask):
                    S_mask = new_S_mask
                    current_size -= 1
                    improved = True
                    if current_size < best_size:
                        best_size = current_size
                        best_set = S_mask
                else:
                    # Optional: Try swaps?
                    # If not maximal, find points that can be added.
                    # If we add one, size stays same.
                    # Let's stick to removal for now to keep it simple and fast.
                    pass
                    
        # Update best if current is better
        if current_size < best_size:
            best_size = current_size
            best_set = S_mask
            
        # Early stop if we found a very small set?
        # Theoretical lower bound is unknown, but for N=16, maybe ~10-20?
        # If best_size is small enough, we can stop.
        if best_size <= 10: 
             # Heuristic threshold
             pass

    # Convert best_set bitmask to list of points
    result = []
    for i in range(P):
        if best_set & (1 << i):
            result.append(POINTS[i])
            
    return result

# Ensure random seed is not fixed to allow exploration, or fix for reproducibility?
# The task says "score is seed/value". It implies we run the program.
# If we use random without seed, it's non-deterministic.
# Usually, for these tasks, non-determinism is allowed unless specified.
# However, to be safe, we might want to try a few different strategies.
# The prompt says "Keep the function names...".

# Let's add a bit more sophistication to the local search:
# If removal fails, try a swap.
# If S \ {v} is not maximal, there exists u not in S \ {v} such that S \ {v} \cup {u} is no-isosceles.
# If we swap v with u, size is constant.
# This might allow further removals later.

def construct_with_swaps():
    start_time = time.time()
    time_limit = 280
    
    best_size = P
    best_set = 0
    
    for restart in range(500):
        if time.time() - start_time > time_limit:
            break
            
        order = list(range(P))
        random.shuffle(order)
        
        S_mask = 0
        for p in order:
            can_add = True
            for a in range(P):
                if S_mask & (1 << a):
                    if S_mask & iso_partners[p][a]:
                        can_add = False
                        break
            if can_add:
                S_mask |= (1 << p)
        
        current_size = bin(S_mask).count('1')
        
        improved = True
        while improved and time.time() - start_time < time_limit:
            improved = False
            removal_order = list(range(P))
            random.shuffle(removal_order)
            
            for v in removal_order:
                if not (S_mask & (1 << v)):
                    continue
                
                new_S_mask = S_mask & ~(1 << v)
                
                if is_maximal(new_S_mask):
                    S_mask = new_S_mask
                    current_size -= 1
                    improved = True
                    if current_size < best_size:
                        best_size = current_size
                        best_set = S_mask
                else:
                    # Try swaps
                    # Find candidates u that can be added to new_S_mask
                    # u must not be in new_S_mask
                    # and u must not form isosceles with any two in new_S_mask
                    # This is the same check as "can_add" but for new_S_mask.
                    # However, we only want to swap if it leads to a "better" state?
                    # Since size is same, we accept swap with some probability or if it unlocks removals.
                    # To keep it simple, we just try one random swap if available.
                    
                    # Identify u's that can be added
                    # This is O(P * |S| * ...) which is expensive if done for every v.
                    # We'll skip swaps for now to ensure speed, as removal is the primary goal.
                    pass
                    
        if current_size < best_size:
            best_size = current_size
            best_set = S_mask

    result = []
    for i in range(P):
        if best_set & (1 << i):
            result.append(POINTS[i])
            
    return result

# Use the swap version? The removal-only version is safer for time.
# Let's stick to removal-only but with many restarts.

def construct():
    start_time = time.time()
    time_limit = 280  # 20s margin
    
    best_size = P
    best_set = 0
    
    # Increase restarts if possible
    max_restarts = 2000
    
    for restart in range(max_restarts):
        if time.time() - start_time > time_limit:
            break
            
        # 1. Greedy Construction
        order = list(range(P))
        random.shuffle(order)
        
        S_mask = 0
        for p in order:
            can_add = True
            for a in range(P):
                if S_mask & (1 << a):
                    if S_mask & iso_partners[p][a]:
                        can_add = False
                        break
            if can_add:
                S_mask |= (1 << p)
        
        current_size = bin(S_mask).count('1')
        
        # 2. Local Search: Remove points
        improved = True
        while improved and time.time() - start_time < time_limit:
            improved = False
            # Randomize removal order
            removal_order = list(range(P))
            random.shuffle(removal_order)
            
            for v in removal_order:
                if not (S_mask & (1 << v)):
                    continue
                
                new_S_mask = S_mask & ~(1 << v)
                
                if is_maximal(new_S_mask):
                    S_mask = new_S_mask
                    current_size -= 1
                    improved = True
                    if current_size < best_size:
                        best_size = current_size
                        best_set = S_mask
                else:
                    # Optional: If not maximal, we could try to "fix" it by adding a point?
                    # No, that increases size.
                    # We just skip.
                    pass
                    
        if current_size < best_size:
            best_size = current_size
            best_set = S_mask
            
    result = []
    for i in range(P):
        if best_set & (1 << i):
            result.append(POINTS[i])
            
    return result
