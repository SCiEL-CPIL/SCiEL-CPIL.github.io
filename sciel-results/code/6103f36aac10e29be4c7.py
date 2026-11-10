# EVOLVE-BLOCK-START
import numpy as np
import time


def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2.
    Uses a greedy algorithm with multiple random restarts to maximize the size of the complement S.
    """
    p_prime = 7
    q = 49
    
    def idx_to_coords(i):
        yb = i % 7
        ya = (i // 7) % 7
        xb = (i // 49) % 7
        xa = (i // 343) % 7
        return (xa, xb, ya, yb)

    def coords_to_idx(xa, xb, ya, yb):
        return ((xa * 7 + xb) * 7 + ya) * 7 + yb

    def greedy_once(seed):
        valid_count = np.full(2401, 50, dtype=np.int32)
        in_S = np.zeros(2401, dtype=bool)
        
        rng = np.random.RandomState(seed)
        candidates = np.arange(2401)
        rng.shuffle(candidates)
        
        for p_idx in candidates:
            # Tentatively add p_idx to S
            # This decreases valid_count[x] by 1 for all x != p_idx
            valid_count -= 1
            valid_count[p_idx] += 1
            
            # Check if any point is blocked (valid_count == 0)
            if np.min(valid_count) > 0:
                in_S[p_idx] = True
            else:
                # Revert
                valid_count += 1
                valid_count[p_idx] -= 1
                
        return np.sum(in_S), in_S

    start_time = time.time()
    deadline = start_time + 580  # Leave 20s for safety
    
    best_size = 0
    best_in_S = None
    seed_id = 0
    
    # Run multiple restarts
    while time.time() < deadline:
        size, in_S = greedy_once(seed_id)
        if size > best_size:
            best_size = size
            best_in_S = in_S
            # If we found a very large set, we might stop early? 
            # Max possible is 2401. But realistically ~200.
            # If best_size > 200, it's already good.
            # Let's just keep going to find the best.
        seed_id += 1
        
        # Optional: If we have time, try some structured orders?
        # For now, random is robust.
        
    if best_in_S is None:
        # Fallback: return a single point? Or empty?
        # N must have k >= 1.
        # If S is empty, N is full. |S|=0. Score 0.
        # We need at least one point in N.
        # If greedy failed completely (unlikely), we return all points?
        # But we want to maximize S.
        # If best_size is 0, we return a set with 1 point in N?
        # Actually, if S is empty, N is F49^2.
        # Is F49^2 a valid Nikodym set?
        # For any x, is there a line L through x such that L \ {x} subset N?
        # Yes, any line works because N is everything.
        # So S=empty is valid.
        # But score is 0.
        # We need to maximize S.
        
        # If we didn't find anything, maybe return a small valid S?
        # A single point p in S.
        # N = F49^2 \ {p}.
        # For x != p, line L through x and p is invalid.
        # But there are 49 other lines through x.
        # They are disjoint from S={p} (since p is not on them).
        # So x is satisfied.
        # For x=p, line L through p. L \ {p} disjoint from S={p}?
        # L \ {p} does not contain p. So yes.
        # So S={p} is valid.
        
        best_in_S = np.zeros(2401, dtype=bool)
        best_in_S[0] = True
        best_size = 1

    # Construct the result N
    # N is the set of points NOT in S.
    
    result_list = []
    for i in range(2401):
        if not best_in_S[i]:
            xa, xb, ya, yb = idx_to_coords(i)
            result_list.append([
                [xa, xb],
                [ya, yb]
            ])
            
    if len(result_list) == 0:
        return np.empty((0, 2, 2), dtype=np.int64)
        
    result = np.array(result_list, dtype=np.int64)
    
    return result

# EVOLVE-BLOCK-END
