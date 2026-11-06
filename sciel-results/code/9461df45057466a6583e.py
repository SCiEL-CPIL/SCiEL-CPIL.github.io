import random
import time
import sys

N = 16
NUM_POINTS = N * N
POINTS = [(x, y) for y in range(N) for x in range(N)]

def precompute():
    dist2 = [[0] * NUM_POINTS for _ in range(NUM_POINTS)]
    for i in range(NUM_POINTS):
        xi, yi = POINTS[i]
        for j in range(NUM_POINTS):
            xj, yj = POINTS[j]
            dist2[i][j] = (xi - xj) ** 2 + (yi - yj) ** 2

    block_mask = [[0] * NUM_POINTS for _ in range(NUM_POINTS)]
    for i in range(NUM_POINTS):
        for j in range(i + 1, NUM_POINTS):
            mask = 0
            d_ij = dist2[i][j]
            for x in range(NUM_POINTS):
                if x == i or x == j:
                    continue
                d_ix = dist2[i][x]
                d_jx = dist2[j][x]
                if d_ix == d_jx or d_ix == d_ij or d_jx == d_ij:
                    mask |= (1 << x)
            block_mask[i][j] = mask
            block_mask[j][i] = mask
    return dist2, block_mask

DIST2, BLOCK_MASK = precompute()
FULL_MASK = (1 << NUM_POINTS) - 1

def construct():
    start_time = time.time()
    deadline = start_time + 290
    
    best_set = None
    best_size = float('inf')

    def greedy_initial(seed_val):
        random.seed(seed_val)
        S = set()
        counts = [0] * NUM_POINTS
        indices = list(range(NUM_POINTS))
        random.shuffle(indices)
        
        for p_idx in indices:
            if counts[p_idx] == 0:
                S.add(p_idx)
                for q_idx in S:
                    if q_idx == p_idx:
                        continue
                    mask = BLOCK_MASK[p_idx][q_idx]
                    m = mask
                    while m:
                        lsb = m & -m
                        x_idx = lsb.bit_length() - 1
                        counts[x_idx] += 1
                        m ^= lsb
        return S

    def build_state(S):
        blocked_mask = 0
        pair_counts = [0] * NUM_POINTS
        s_list = list(S)
        for i in range(len(s_list)):
            a = s_list[i]
            for j in range(i + 1, len(s_list)):
                b = s_list[j]
                mask = BLOCK_MASK[a][b]
                blocked_mask |= mask
                m = mask
                while m:
                    lsb = m & -m
                    x = lsb.bit_length() - 1
                    pair_counts[x] += 1
                    m ^= lsb
        return blocked_mask, pair_counts

    def check_maximal(S, blocked_mask):
        s_mask = 0
        for i in S:
            s_mask |= (1 << i)
        not_in_S = FULL_MASK ^ s_mask
        return (not_in_S & ~blocked_mask) == 0

    def try_remove(S, p_idx, blocked_mask, pair_counts):
        new_blocked_mask = 0
        new_pair_counts = [0] * NUM_POINTS
        s_list = list(S)
        for i in range(len(s_list)):
            a = s_list[i]
            if a == p_idx:
                continue
            for j in range(i + 1, len(s_list)):
                b = s_list[j]
                if b == p_idx:
                    continue
                mask = BLOCK_MASK[a][b]
                new_blocked_mask |= mask
                m = mask
                while m:
                    lsb = m & -m
                    x = lsb.bit_length() - 1
                    new_pair_counts[x] += 1
                    m ^= lsb
        
        if (new_blocked_mask >> p_idx) & 1 == 0:
            return None, None, None, False
        
        new_s_mask = 0
        for i in S:
            if i != p_idx:
                new_s_mask |= (1 << i)
        not_in_new = FULL_MASK ^ new_s_mask
        if (not_in_new & ~new_blocked_mask) != 0:
            return None, None, None, False
        
        new_S = set(S)
        new_S.discard(p_idx)
        return new_S, new_blocked_mask, new_pair_counts, True

    def local_search(S_init, seed_val, time_limit):
        random.seed(seed_val)
        S = set(S_init)
        blocked_mask, pair_counts = build_state(S)
        
        if not check_maximal(S, blocked_mask):
            return len(S), S
        
        current_size = len(S)
        best_size_local = current_size
        best_set_local = set(S)
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            s_list = list(S)
            random.shuffle(s_list)
            
            for p_idx in s_list:
                if p_idx not in S:
                    continue
                
                new_S, new_blocked_mask, new_pair_counts, success = try_remove(S, p_idx, blocked_mask, pair_counts)
                if success:
                    S = new_S
                    blocked_mask = new_blocked_mask
                    pair_counts = new_pair_counts
                    current_size = len(S)
                    improved = True
                    
                    if current_size < best_size_local:
                        best_size_local = current_size
                        best_set_local = set(S)
                    
                    break
        
        # Simulated Annealing Phase
        # Temperature schedule
        T = 10.0
        T_min = 0.01
        cooling = 0.995
        
        while T > T_min and time.time() < time_limit:
            s_list = list(S)
            random.shuffle(s_list)
            p_idx = random.choice(s_list)
            
            # Compute state without p_idx
            temp_blocked_mask = 0
            temp_pair_counts = [0] * NUM_POINTS
            s_list2 = list(S)
            for i in range(len(s_list2)):
                a = s_list2[i]
                if a == p_idx:
                    continue
                for j in range(i + 1, len(s_list2)):
                    b = s_list2[j]
                    if b == p_idx:
                        continue
                    mask = BLOCK_MASK[a][b]
                    temp_blocked_mask |= mask
                    m = mask
                    while m:
                        lsb = m & -m
                        x = lsb.bit_length() - 1
                        temp_pair_counts[x] += 1
                        m ^= lsb
            
            s_mask_without_p = 0
            for i in S:
                if i != p_idx:
                    s_mask_without_p |= (1 << i)
            
            unblocked = (~temp_blocked_mask) & FULL_MASK & (~s_mask_without_p)
            
            if unblocked == 0:
                continue
            
            m = unblocked
            # Pick a random candidate from unblocked
            # Count bits to pick random one
            num_unblocked = bin(m).count('1')
            if num_unblocked == 0:
                continue
            rand_idx = random.randint(0, num_unblocked - 1)
            target_bit = m
            for _ in range(rand_idx + 1):
                lsb = target_bit & -target_bit
                target_bit ^= lsb
            q_idx = lsb.bit_length() - 1
            
            # Check validity of S\{p} U {q}
            valid_no_iso = True
            for i in range(len(s_list2)):
                a = s_list2[i]
                if a == p_idx:
                    continue
                for j in range(i + 1, len(s_list2)):
                    b = s_list2[j]
                    if b == p_idx:
                        continue
                    if (BLOCK_MASK[a][b] >> q_idx) & 1:
                        valid_no_iso = False
                        break
                if not valid_no_iso:
                    break
            
            if not valid_no_iso:
                continue
            
            # Compute new blocked mask and check maximality
            new_blocked_mask = 0
            s_list_new = list(S)
            s_list_new.remove(p_idx)
            s_list_new.append(q_idx)
            for i in range(len(s_list_new)):
                a = s_list_new[i]
                for j in range(i + 1, len(s_list_new)):
                    b = s_list_new[j]
                    new_blocked_mask |= BLOCK_MASK[a][b]
            
            new_s_mask = s_mask_without_p | (1 << q_idx)
            not_in_new = FULL_MASK ^ new_s_mask
            if (not_in_new & ~new_blocked_mask) != 0:
                continue
            
            # Valid move: swap p for q
            # Size remains same, but structure changes.
            # We accept based on SA criteria?
            # Since size is constant, we might want to accept if it "looks" better or randomly.
            # Actually, the goal is to minimize size. Swaps don't change size.
            # However, a different structure might allow future removals.
            # So we treat this as a "neutral" move in terms of objective, but it explores the space.
            # We can use a simple acceptance: always accept if it leads to a state that allows a removal?
            # Or just random walk with backtracking?
            # Let's just accept it with probability exp(-delta/T) where delta=0, so always accept.
            # But we need to avoid cycles.
            # For now, let's just perform the swap and then try to remove again.
            
            S = set(S)
            S.discard(p_idx)
            S.add(q_idx)
            blocked_mask = new_blocked_mask
            # pair_counts needs update, but we rebuild in try_remove or build_state if needed.
            # Let's rebuild pair_counts for consistency if we do removals.
            # Actually, try_remove rebuilds from scratch anyway.
            
            # Try immediate removals after swap
            s_list_after_swap = list(S)
            random.shuffle(s_list_after_swap)
            for rem_idx in s_list_after_swap:
                if rem_idx not in S:
                    continue
                new_S, new_blocked_mask, new_pair_counts, success = try_remove(S, rem_idx, blocked_mask, pair_counts)
                if success:
                    S = new_S
                    blocked_mask = new_blocked_mask
                    pair_counts = new_pair_counts
                    current_size = len(S)
                    if current_size < best_size_local:
                        best_size_local = current_size
                        best_set_local = set(S)
                    break # Restart removal loop
            
            T *= cooling
        
        return best_size_local, best_set_local

    restarts = 0
    while time.time() < deadline:
        S_init = greedy_initial(restarts)
        time_limit = min(deadline, start_time + 290)
        size, S_final = local_search(S_init, restarts * 7919 + 1, time_limit)
        
        if size < best_size:
            best_size = size
            best_set = S_final
        
        restarts += 1

    if best_set is None:
        S = set()
        for p in range(NUM_POINTS):
            valid = True
            for q in S:
                for r in S:
                    if r == q:
                        continue
                    if (BLOCK_MASK[q][r] >> p) & 1:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                S.add(p)
        best_set = S
    
    return [POINTS[i] for i in best_set]

if __name__ == "__main__":
    s = construct()
    print(len(s))
