"""Minimal maximal no-isosceles set via Branch and Bound with bitset pruning."""
import itertools
import numpy as np

N = 16
ALL_POINTS = [(x, y) for x in range(N) for y in range(N)]
N_POINTS = len(ALL_POINTS)
FULL_MASK = (1 << N_POINTS) - 1

# Precompute isosceles masks
# iso_mask[i][j] is a bitmask of points k such that (i, j, k) is isosceles
# Note: i, j, k must be distinct.
iso_mask = [[0] * N_POINTS for _ in range(N_POINTS)]

for i in range(N_POINTS):
    pi = ALL_POINTS[i]
    for j in range(i + 1, N_POINTS):
        pj = ALL_POINTS[j]
        mask = 0
        for k in range(N_POINTS):
            if k == i or k == j:
                continue
            pk = ALL_POINTS[k]
            d1 = (pi[0] - pj[0])**2 + (pi[1] - pj[1])**2
            d2 = (pi[0] - pk[0])**2 + (pi[1] - pk[1])**2
            d3 = (pj[0] - pk[0])**2 + (pj[1] - pk[1])**2
            if d1 == d2 or d1 == d3 or d2 == d3:
                mask |= (1 << k)
        iso_mask[i][j] = mask
        iso_mask[j][i] = mask

best_solution = None
best_size = N_POINTS + 1

def construct():
    global best_solution, best_size
    
    # Reset
    best_solution = None
    best_size = N_POINTS + 1
    
    # We use a recursive branch and bound
    # State: chosen_mask, blocked_mask
    # chosen_mask: bitmask of points in the current set S
    # blocked_mask: bitmask of points that are blocked by S (cannot be added)
    # Unblocked points: ~chosen_mask & ~blocked_mask & FULL_MASK
    
    def search(chosen_mask, blocked_mask, depth):
        global best_solution, best_size
        
        # Check if all non-chosen points are blocked
        unblocked = FULL_MASK & ~chosen_mask & ~blocked_mask
        if unblocked == 0:
            # All non-chosen points are blocked, so S is maximal
            if depth < best_size:
                best_size = depth
                best_solution = chosen_mask
            return
        
        # Lower bound pruning
        # Number of unblocked points
        num_unblocked = bin(unblocked).count('1')
        # Max points a single new point can block
        # This is an upper bound on efficiency, so lower bound on steps
        # A rough estimate: max possible blocks per point is ~N_POINTS
        # Better: calculate max blocks for any candidate in current unblocked set
        # But that's expensive. Let's use a static upper bound or a heuristic.
        # Let's just use depth + 1 < best_size as a basic prune
        if depth + 1 >= best_size:
            return
            
        # Heuristic Lower Bound:
        # Each new point can block at most `max_block_capacity` points.
        # We can precompute for each point c, the max number of points it can block 
        # given ANY set of chosen points? No, it depends on chosen.
        # However, `iso_mask[a][c]` is fixed.
        # The new blocked points when adding c are union of iso_mask[a][c] for a in chosen.
        # Plus c itself is chosen, so it's not "blocked" in the sense of being unblocked, 
        # but it removes c from the unblocked set.
        
        # Let's pick candidates from unblocked.
        # We want to pick a point that blocks the most *currently unblocked* points.
        
        candidates = []
        # Iterate through unblocked bits
        u = unblocked
        while u:
            lsb = u & -u
            c = lsb.bit_length() - 1
            u ^= lsb
            
            # Calculate new blocked points if we add c
            # new_blocked = union of iso_mask[a][c] for all a in chosen_mask
            # We can compute this efficiently if we maintain some structure, 
            # but for now, let's just loop through chosen bits.
            new_blocked = 0
            cm = chosen_mask
            while cm:
                lsb_a = cm & -cm
                a = lsb_a.bit_length() - 1
                cm ^= lsb_a
                new_blocked |= iso_mask[a][c]
                
            # Points that become blocked (were unblocked, will be blocked)
            # Note: c itself is now chosen, so it's not in unblocked anymore effectively.
            # The points that are "covered" by adding c are:
            # 1. c itself (removed from unblocked)
            # 2. Points in (new_blocked & unblocked)
            
            covered = new_blocked & unblocked
            # c is also covered (removed from unblocked)
            # So total reduction in unblocked count is 1 + popcount(covered)
            
            score = bin(covered).count('1') + 1
            
            candidates.append((score, c))
            
        # Sort candidates by score descending (greedy choice)
        candidates.sort(key=lambda x: x[0], reverse=True)
        
        for score, c in candidates:
            # Add c to chosen
            new_chosen = chosen_mask | (1 << c)
            
            # Update blocked mask
            new_blocked_mask = blocked_mask
            cm = chosen_mask
            while cm:
                lsb_a = cm & -cm
                a = lsb_a.bit_length() - 1
                cm ^= lsb_a
                new_blocked_mask |= iso_mask[a][c]
                
            # Ensure chosen points are not in blocked_mask (they are chosen)
            new_blocked_mask &= ~new_chosen
            
            # Recurse
            search(new_chosen, new_blocked_mask, depth + 1)
            
            # Prune if we found a solution of size depth+1
            if best_size == depth + 1:
                return

    # Start search
    search(0, 0, 0)
    
    if best_solution is None:
        # Fallback: return a valid maximal set using greedy
        # This should not happen if the grid is small enough
        chosen_mask = 0
        blocked_mask = 0
        while True:
            unblocked = FULL_MASK & ~chosen_mask & ~blocked_mask
            if unblocked == 0:
                break
            # Pick first unblocked
            lsb = unblocked & -unblocked
            c = lsb.bit_length() - 1
            chosen_mask |= (1 << c)
            cm = chosen_mask & ~(1 << c) # previously chosen
            while cm:
                lsb_a = cm & -cm
                a = lsb_a.bit_length() - 1
                cm ^= lsb_a
                blocked_mask |= iso_mask[a][c]
            blocked_mask &= ~chosen_mask
        best_solution = chosen_mask
        
    # Convert bitmask to list of points
    result = []
    m = best_solution
    while m:
        lsb = m & -m
        i = lsb.bit_length() - 1
        m ^= lsb
        result.append(ALL_POINTS[i])
        
    return result

