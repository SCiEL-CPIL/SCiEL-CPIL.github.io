import time
import random

N = 9
V = 1 << N  # 512

# Precompute neighbors and neighbor masks
NEIGHBORS = [[] for _ in range(V)]
NEIGHBOR_MASKS = [0] * V
for v in range(V):
    mask = 0
    for b in range(N):
        u = v ^ (1 << b)
        NEIGHBORS[v].append(u)
        mask |= (1 << u)
    NEIGHBOR_MASKS[v] = mask

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s kill
    
    # Get baseline using simple greedy
    best_path = [0]
    best_len = 1
    best_used_mask = 1  # bit 0 set
    best_forbidden_mask = 0
    
    cur = 0
    used_mask = 1
    forbidden_mask = 0
    while True:
        cur = best_path[-1]
        cur_neigh_mask = NEIGHBOR_MASKS[cur]
        valid_mask = cur_neigh_mask & ~used_mask & ~forbidden_mask
        if valid_mask == 0:
            break
        # Pick first available neighbor
        nxt = (valid_mask & -valid_mask).bit_length() - 1
        best_path.append(nxt)
        used_mask |= (1 << nxt)
        forbidden_mask |= NEIGHBOR_MASKS[cur]
        best_len = len(best_path)
    
    best_used_mask = used_mask
    best_forbidden_mask = forbidden_mask
    
    # Randomized Warnsdorff with bitmask operations
    while time.time() < deadline:
        # Random start
        start_v = random.randint(0, V - 1)
        path = [start_v]
        used_mask = 1 << start_v
        forbidden_mask = 0
        
        while time.time() < deadline:
            cur = path[-1]
            cur_neigh_mask = NEIGHBOR_MASKS[cur]
            
            # Find valid candidates: neighbors of cur that are not used and not forbidden
            valid_mask = cur_neigh_mask & ~used_mask & ~forbidden_mask
            if valid_mask == 0:
                break
            
            # Score each candidate using Warnsdorff's rule
            # For each candidate v, count how many neighbors u of v are available
            # (not used, not forbidden, and not neighbor of cur)
            min_score = N + 1
            best_cands_mask = 0
            
            # Precompute: neighbors of cur to exclude from scoring
            cur_n_mask = cur_neigh_mask
            
            v_mask = valid_mask
            while v_mask:
                v_bit = v_mask & -v_mask
                v = v_bit.bit_length() - 1
                v_mask ^= v_bit
                
                # Count available extensions from v
                # Extensions are neighbors of v that are:
                # - not in used_mask
                # - not in forbidden_mask  
                # - not in cur_neigh_mask (since cur becomes internal)
                available = NEIGHBOR_MASKS[v] & ~used_mask & ~forbidden_mask & ~cur_n_mask
                score = available.bit_count()
                
                if score < min_score:
                    min_score = score
                    best_cands_mask = v_bit
                elif score == min_score:
                    best_cands_mask |= v_bit
            
            # Random choice among best candidates
            # Convert best_cands_mask to list for random choice
            cands_list = []
            bm = best_cands_mask
            while bm:
                bit = bm & -bm
                cands_list.append(bit.bit_length() - 1)
                bm ^= bit
            
            nxt = random.choice(cands_list)
            
            path.append(nxt)
            used_mask |= (1 << nxt)
            forbidden_mask |= cur_neigh_mask  # cur becomes internal
            
        if len(path) > best_len:
            best_len = len(path)
            best_path = path.copy()
            best_used_mask = used_mask
            best_forbidden_mask = forbidden_mask
            
    return best_path
