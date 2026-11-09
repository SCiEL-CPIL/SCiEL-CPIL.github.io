import time
import random
import itertools

N = 16
NUM_PTS = N * N

# Precompute geometry
POINTS = [(x, y) for y in range(N) for x in range(N)]

# Map point index to (x,y)
# p_idx = y * N + x

# Precompute squared distances
dist = [[0] * NUM_PTS for _ in range(NUM_PTS)]
for i in range(NUM_PTS):
    x1, y1 = POINTS[i]
    for j in range(i + 1, NUM_PTS):
        x2, y2 = POINTS[j]
        d = (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)
        dist[i][j] = d
        dist[j][i] = d

# Precompute blocked masks for each pair
# blocked_mask[i][j] is a bitmask of points k such that triangle i,j,k is isosceles
blocked_masks = [[0] * NUM_PTS for _ in range(NUM_PTS)]

for i in range(NUM_PTS):
    for j in range(i + 1, NUM_PTS):
        d_ij = dist[i][j]
        mask = 0
        for k in range(NUM_PTS):
            if k == i or k == j:
                continue
            d_ik = dist[i][k]
            d_jk = dist[j][k]
            if d_ij == d_ik or d_ij == d_jk or d_ik == d_jk:
                mask |= (1 << k)
        blocked_masks[i][j] = mask
        blocked_masks[j][i] = mask

def is_valid(set_list):
    """Check if no three points in set_list form an isosceles triangle."""
    n = len(set_list)
    for i in range(n):
        pi = set_list[i]
        for j in range(i + 1, n):
            pj = set_list[j]
            mask_ij = blocked_masks[pi][pj]
            for k in range(j + 1, n):
                if (mask_ij >> set_list[k]) & 1:
                    return False
    return True

def get_blocked_mask(set_list):
    """Return the union of blocked masks for all pairs in set_list."""
    mask = 0
    n = len(set_list)
    for i in range(n):
        pi = set_list[i]
        for j in range(i + 1, n):
            pj = set_list[j]
            mask |= blocked_masks[pi][pj]
    return mask

def is_maximal(set_list):
    """Check if no point outside set_list can be added."""
    if not is_valid(set_list):
        return False
    
    set_mask = 0
    for p in set_list:
        set_mask |= (1 << p)
    
    blocked = get_blocked_mask(set_list)
    
    coverage = set_mask | blocked
    full_mask = (1 << NUM_PTS) - 1
    return (coverage & full_mask) == full_mask

def construct():
    import time
    import random
    
    DEADLINE = time.time() + 290
    
    best_set = None
    best_len = float('inf')
    
    # Greedy construction with randomized ordering
    def greedy_construct(seed):
        rng = random.Random(seed)
        order = list(range(NUM_PTS))
        rng.shuffle(order)
        
        S = []
        S_mask = 0
        
        for p in order:
            can_add = True
            for i in range(len(S)):
                for j in range(i + 1, len(S)):
                    if (blocked_masks[S[i]][S[j]] >> p) & 1:
                        can_add = False
                        break
                if not can_add:
                    break
            
            if can_add:
                S.append(p)
                S_mask |= (1 << p)
                
        return S

    # Run multiple greedy restarts
    num_restarts = 30
    for i in range(num_restarts):
        if time.time() > DEADLINE:
            break
        S = greedy_construct(i)
        
        # Try to make it maximal by adding points greedily
        if not is_maximal(S):
            S_mask = 0
            for p in S:
                S_mask |= (1 << p)
            
            while True:
                if time.time() > DEADLINE:
                    break
                blocked = get_blocked_mask(S)
                unblocked = []
                for p in range(NUM_PTS):
                    if not (S_mask >> p) & 1 and not (blocked >> p) & 1:
                        unblocked.append(p)
                
                if not unblocked:
                    break
                
                # Pick a point that blocks the most other unblocked points
                best_p = None
                best_blocks = -1
                for p in unblocked:
                    # Count how many unblocked points p would block
                    # This is expensive, so just pick randomly or first
                    # Let's pick the one that blocks the most
                    # Actually, let's just pick the first one for speed
                    best_p = p
                    break
                
                # Check validity
                can_add = True
                for i in range(len(S)):
                    for j in range(i + 1, len(S)):
                        if (blocked_masks[S[i]][S[j]] >> best_p) & 1:
                            can_add = False
                            break
                    if not can_add:
                        break
                
                if can_add:
                    S.append(best_p)
                    S_mask |= (1 << best_p)
                else:
                    # Should not happen if blocked is correct
                    break
        
        if is_maximal(S):
            if len(S) < best_len:
                best_len = len(S)
                best_set = S.copy()
    
    # Local Search: Remove points
    if best_set is not None:
        S = best_set
        improved = True
        while improved and time.time() < DEADLINE:
            improved = False
            n = len(S)
            for i in range(n):
                S_trial = S[:i] + S[i+1:]
                if is_maximal(S_trial):
                    S = S_trial
                    improved = True
                    break
    
    # Local Search: Swaps
    if best_set is not None:
        S = best_set
        for it in range(50):
            if time.time() > DEADLINE:
                break
            improved = False
            n = len(S)
            S_mask = 0
            for p in S:
                S_mask |= (1 << p)
            
            for i in range(n):
                x = S[i]
                S_without_x = S[:i] + S[i+1:]
                
                candidates = [y for y in range(NUM_PTS) if not (S_mask >> y) & 1]
                random.shuffle(candidates)
                
                for y in candidates[:10]:
                    S_swap = S_without_x + [y]
                    if is_valid(S_swap):
                        S = S_swap
                        improved = True
                        break
                if improved:
                    break
            
            if not improved:
                break
            
            # After swaps, try to remove points again
            improved2 = True
            while improved2 and time.time() < DEADLINE:
                improved2 = False
                n = len(S)
                for i in range(n):
                    S_trial = S[:i] + S[i+1:]
                    if is_maximal(S_trial):
                        S = S_trial
                        improved2 = True
                        break
            
            if len(S) < best_len:
                best_len = len(S)
                best_set = S.copy()
    
    if best_set is None:
        # Fallback
        best_set = greedy_construct(0)
        # Ensure maximal
        S_mask = 0
        for p in best_set:
            S_mask |= (1 << p)
        
        while True:
            if time.time() > DEADLINE:
                break
            blocked = get_blocked_mask(best_set)
            unblocked = None
            for p in range(NUM_PTS):
                if not (S_mask >> p) & 1 and not (blocked >> p) & 1:
                    unblocked = p
                    break
            
            if unblocked is None:
                break
            
            can_add = True
            for i in range(len(best_set)):
                for j in range(i + 1, len(best_set)):
                    if (blocked_masks[best_set[i]][best_set[j]] >> unblocked) & 1:
                        can_add = False
                        break
                if not can_add:
                    break
            
            if can_add:
                best_set.append(unblocked)
                S_mask |= (1 << unblocked)
    
    return [POINTS[i] for i in best_set]

# EVOLVE-BLOCK-END
