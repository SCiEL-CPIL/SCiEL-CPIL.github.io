import itertools
import random
import numpy as np

N = 16
POINTS = list(itertools.product(range(N), repeat=2))
NUM_POINTS = len(POINTS)

# Precomputation
# iso_adj[i][j] = bitmask of k such that {i, j, k} is isosceles
# We need this to check if a point p can be added to a set S.
# p can be added to S iff for all a in S, (iso_adj[p][a] & mask_S) == 0.

print("Precomputing...")
iso_adj = [[0] * NUM_POINTS for _ in range(NUM_POINTS)]

def dist_sq(p1, p2):
    return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2

# Build iso_adj
# For each triple (i, j, k), if isosceles, set bits.
# We can optimize by iterating pairs (i, j) and finding k.
# Or just iterate all triples. 256^3 is ~16M. 
# In Python, 16M iterations is ~1-2 seconds. Acceptable.

for i in range(NUM_POINTS):
    pi = POINTS[i]
    for j in range(i + 1, NUM_POINTS):
        pj = POINTS[j]
        for k in range(j + 1, NUM_POINTS):
            pk = POINTS[k]
            d_ij = dist_sq(pi, pj)
            d_ik = dist_sq(pi, pk)
            d_jk = dist_sq(pj, pk)
            if d_ij == d_ik or d_ij == d_jk or d_ik == d_jk:
                # {i, j, k} is isosceles
                # Update iso_adj
                iso_adj[i][j] |= (1 << k)
                iso_adj[j][i] |= (1 << k)
                
                iso_adj[i][k] |= (1 << j)
                iso_adj[k][i] |= (1 << j)
                
                iso_adj[j][k] |= (1 << i)
                iso_adj[k][j] |= (1 << i)

print("Precomputation done.")

def can_add(p, mask_S):
    """
    Check if point p can be added to the set represented by mask_S.
    p is not in mask_S.
    """
    # For each a in S, check if any b in S forms isosceles with p and a.
    # If so, p cannot be added.
    # We check: for all a in S, (iso_adj[p][a] & mask_S) == 0.
    # Note: iso_adj[p][a] contains b such that {p, a, b} is isosceles.
    # If b is in S, then {p, a, b} is an isosceles triple in S U {p}.
    
    # Iterate through set bits in mask_S
    m = mask_S
    while m:
        # Get lowest set bit
        lsb = m & -m
        a = lsb.bit_length() - 1
        if (iso_adj[p][a] & mask_S) != 0:
            return False
        m ^= lsb
    return True

def construct():
    best_set_mask = 0
    best_size = float('inf')
    
    # Greedy construction function
    def greedy_complete(mask_S, order):
        """
        Greedily add points from 'order' to mask_S until maximal.
        Returns the new mask.
        """
        current_mask = mask_S
        for p in order:
            if (current_mask >> p) & 1:
                continue
            if can_add(p, current_mask):
                current_mask |= (1 << p)
        return current_mask

    # Local Search / ILS
    for restart in range(20):
        # Random order for initial greedy
        order = list(range(NUM_POINTS))
        random.shuffle(order)
        
        # Initial greedy
        current_mask = 0
        for p in order:
            if can_add(p, current_mask):
                current_mask |= (1 << p)
        
        current_size = current_mask.bit_count()
        
        if current_size < best_size:
            best_size = current_size
            best_set_mask = current_mask
            
        # Local search: Try to improve
        # We try to remove a point and complete greedily.
        # If the new size is smaller, we accept.
        
        for _ in range(100): # 100 local search steps
            # Pick a random point in current set to remove
            # Get list of points in set
            points_in_set = []
            m = current_mask
            while m:
                lsb = m & -m
                points_in_set.append(lsb.bit_length() - 1)
                m ^= lsb
            
            if not points_in_set:
                break
                
            p_remove = random.choice(points_in_set)
            new_mask = current_mask & ~(1 << p_remove)
            
            # Complete greedily with random order
            order = list(range(NUM_POINTS))
            random.shuffle(order)
            completed_mask = greedy_complete(new_mask, order)
            
            new_size = completed_mask.bit_count()
            
            # Accept if smaller or equal (with probability)
            if new_size < current_size:
                current_mask = completed_mask
                current_size = new_size
                if current_size < best_size:
                    best_size = current_size
                    best_set_mask = current_mask
            elif new_size == current_size and random.random() < 0.1:
                # Accept equal size to escape local minima
                current_mask = completed_mask
                # current_size unchanged
                
    # Convert best_set_mask to list of points
    result = []
    m = best_set_mask
    while m:
        lsb = m & -m
        idx = lsb.bit_length() - 1
        result.append(POINTS[idx])
        m ^= lsb
        
    return result
