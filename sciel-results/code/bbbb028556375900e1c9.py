"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16
TOTAL_POINTS = N * N

# EVOLVE-BLOCK-START
import time
import random

# Precomputation
# dist2[i][j]
# iso_with[p][i]: bitset of j such that {p, i, j} is isosceles

def _precompute():
    global iso_with
    points = [(x, y) for x in range(N) for y in range(N)]
    dist2 = [[0] * TOTAL_POINTS for _ in range(TOTAL_POINTS)]
    for i in range(TOTAL_POINTS):
        for j in range(TOTAL_POINTS):
            dx = points[i][0] - points[j][0]
            dy = points[i][1] - points[j][1]
            dist2[i][j] = dx*dx + dy*dy
            
    iso_with = [[0] * TOTAL_POINTS for _ in range(TOTAL_POINTS)]
    for p in range(TOTAL_POINTS):
        for i in range(TOTAL_POINTS):
            if p == i:
                continue
            mask = 0
            for j in range(TOTAL_POINTS):
                if j == p or j == i:
                    continue
                d_pi = dist2[p][i]
                d_pj = dist2[p][j]
                d_ij = dist2[i][j]
                if d_pi == d_pj or d_pi == d_ij or d_pj == d_ij:
                    mask |= (1 << j)
            iso_with[p][i] = mask

_precompute()

ALL_MASK = (1 << TOTAL_POINTS) - 1

def _is_maximal(S_mask):
    """
    Check if set S (represented by S_mask) is maximal.
    S is maximal if no point p not in S can be added.
    p can be added if for all i, j in S, {p, i, j} is NOT isosceles.
    p is blocked if exists i, j in S such that p is in iso_with[i][j]? 
    No, iso_with[p][i] contains j such that {p, i, j} is isosceles.
    So p is blocked if exists i in S such that (S_mask & iso_with[p][i]) != 0.
    """
    not_S = ALL_MASK ^ S_mask
    p_bit = not_S
    while p_bit:
        lsb = p_bit & -p_bit
        p = lsb.bit_length() - 1
        p_bit ^= lsb
        
        # Check if p is blocked by S
        # Iterate i in S
        S_temp = S_mask
        blocked = False
        while S_temp:
            i_lsb = S_temp & -S_temp
            i = i_lsb.bit_length() - 1
            S_temp ^= i_lsb
            
            # If any j in S is in iso_with[p][i], then p is blocked
            if S_mask & iso_with[p][i]:
                blocked = True
                break
        
        if not blocked:
            return False
    return True

def _greedy_maximal():
    """Construct a maximal no-isosceles set greedily."""
    S_mask = 0
    for p in range(TOTAL_POINTS):
        # Check if p can be added to S
        # p can be added if for all i, j in S, {p, i, j} is not isosceles
        # This is equivalent to: for all i in S, (S_mask & iso_with[p][i]) == 0
        # Wait, if we add p, we need to check against existing pairs in S.
        # The condition for p to be addable to S is:
        # For all i in S, there is no j in S such that {p, i, j} is isosceles.
        # i.e., for all i in S, (S_mask & iso_with[p][i]) == 0.
        
        can_add = True
        S_temp = S_mask
        while S_temp:
            i_lsb = S_temp & -S_temp
            i = i_lsb.bit_length() - 1
            S_temp ^= i_lsb
            if S_mask & iso_with[p][i]:
                can_add = False
                break
        
        if can_add:
            S_mask |= (1 << p)
    return S_mask

def _mask_to_list(S_mask):
    points = []
    m = S_mask
    while m:
        lsb = m & -m
        p = lsb.bit_length() - 1
        points.append((p // N, p % N))
        m ^= lsb
    return points

def construct():
    start_time = time.time()
    deadline = start_time + 280.0 # Safety margin
    
    # Start with greedy
    S_mask = _greedy_maximal()
    best_mask = S_mask
    best_size = bin(S_mask).count('1')
    
    # Local Search
    # Try to remove points
    while time.time() < deadline:
        # Find a removable point
        # We scan all points in S
        S_temp = S_mask
        found = False
        # Randomize order to escape local minima?
        # For now, simple scan
        while S_temp:
            i_lsb = S_temp & -S_temp
            i = i_lsb.bit_length() - 1
            S_temp ^= i_lsb
            
            new_S = S_mask ^ i_lsb
            if _is_maximal(new_S):
                S_mask = new_S
                found = True
                break # Restart scan with new set
        
        if not found:
            # Try swaps?
            # Or just break if we can't improve size easily
            # Let's try a random restart or different initialization
            # For now, just break to avoid infinite loop on stuck state
            # Or try adding a point? No, S is maximal.
            
            # Try a "kick": Remove a point, add a random addable point to new set
            # This changes the set structure
            S_temp = S_mask
            moved = False
            while S_temp and not moved:
                i_lsb = S_temp & -S_temp
                i = i_lsb.bit_length() - 1
                S_temp ^= i_lsb
                
                S_temp_new = S_mask ^ i_lsb
                # Find addable points to S_temp_new
                # p is addable if for all k in S_temp_new, (S_temp_new & iso_with[p][k]) == 0
                candidates = []
                not_S = ALL_MASK ^ S_temp_new
                p_bit = not_S
                while p_bit:
                    p_lsb = p_bit & -p_bit
                    p = p_lsb.bit_length() - 1
                    p_bit ^= p_lsb
                    
                    can_add = True
                    S_check = S_temp_new
                    while S_check:
                        k_lsb = S_check & -S_check
                        k = k_lsb.bit_length() - 1
                        S_check ^= k_lsb
                        if S_temp_new & iso_with[p][k]:
                            can_add = False
                            break
                    if can_add:
                        candidates.append(p)
                
                if candidates:
                    # Pick random candidate
                    p = random.choice(candidates)
                    new_S = S_temp_new | (1 << p)
                    # Check if new_S is maximal
                    if _is_maximal(new_S):
                        S_mask = new_S
                        moved = True
                        break
            
            if not moved:
                # Stuck, try random restart with different greedy order?
                # Or just break
                break

        if bin(S_mask).count('1') < best_size:
            best_size = bin(S_mask).count('1')
            best_mask = S_mask
            
    return _mask_to_list(best_mask)
# EVOLVE-BLOCK-END
