import random
import time
import sys
import math

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

    # For each pair (i,j), compute the set of points x that form an isosceles triangle with i,j
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
    
    # For each point i, compute the set of pairs (j,k) such that i is blocked by pair (j,k)
    # i.e., i is in BLOCK_MASK[j][k]
    # This helps us understand which points are "critical" for maximality
    return dist2, block_mask

DIST2, BLOCK_MASK = precompute()

def construct():
    start_time = time.time()
    deadline = start_time + 290  # Safety margin before 300s limit
    FULL_MASK = (1 << NUM_POINTS) - 1
    
    best_set = None
    best_size = float('inf')
    
    def build_state(S):
        """Compute blocked_mask and pair_counts for set S"""
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
        """Try to remove p_idx from S. Returns (new_S, new_blocked_mask, new_pair_counts, success)"""
        s_list = list(S)
        new_blocked_mask = 0
        new_pair_counts = [0] * NUM_POINTS
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
    
    def greedy_construct(seed_val):
        """Greedy construction with conflict-aware ordering"""
        random.seed(seed_val)
        S = set()
        blocked_mask = 0
        pair_counts = [0] * NUM_POINTS
        
        # Start with all points as candidates
        candidates = list(range(NUM_POINTS))
        random.shuffle(candidates)
        
        for p_idx in candidates:
            # Check if p_idx can be added (no conflict with existing pairs)
            valid = True
            s_list = list(S)
            for i in range(len(s_list)):
                a = s_list[i]
                for j in range(i + 1, len(s_list)):
                    b = s_list[j]
                    if (BLOCK_MASK[a][b] >> p_idx) & 1:
                        valid = False
                        break
                if not valid:
                    break
            
            if not valid:
                continue
            
            # Add p_idx
            S.add(p_idx)
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
                    blocked_mask |= mask
                    m = mask
                    while m:
                        lsb = m & -m
                        x = lsb.bit_length() - 1
                        pair_counts[x] += 1
                        m ^= lsb
        
        return S
    
    def simulated_annealing(S_init, seed_val, time_limit):
        """Simulated annealing to minimize set size while maintaining no-isosceles and maximal properties"""
        random.seed(seed_val)
        S = set(S_init)
        blocked_mask, pair_counts = build_state(S)
        
        if not check_maximal(S, blocked_mask):
            return len(S), S
        
        current_size = len(S)
        best_size_local = current_size
        best_set_local = set(S)
        
        # Temperature schedule
        T = 1.0
        T_min = 0.001
        cooling = 0.995
        
        no_improve_count = 0
        max_no_improve = 200
        
        while time.time() < time_limit:
            if T < T_min or no_improve_count > max_no_improve:
                # Restart with higher temperature
                T = 0.5
                no_improve_count = 0
                # Random restart: perturb current solution
                if len(S) > 3:
                    s_list = list(S)
                    random.shuffle(s_list)
                    # Try removing a random point and see if we can re-add something
                    for p_idx in s_list[:3]:
                        new_S, new_blocked_mask, new_pair_counts, success = try_remove(S, p_idx, blocked_mask, pair_counts)
                        if success:
                            S = new_S
                            blocked_mask = new_blocked_mask
                            pair_counts = new_pair_counts
                            current_size = len(S)
                            if current_size < best_size_local:
                                best_size_local = current_size
                                best_set_local = set(S)
                            break
                continue
            
            # Generate a move
            move_type = random.random()
            s_list = list(S)
            random.shuffle(s_list)
            
            if move_type < 0.5:
                # Try removing a point
                p_idx = random.choice(s_list)
                new_S, new_blocked_mask, new_pair_counts, success = try_remove(S, p_idx, blocked_mask, pair_counts)
                if success:
                    delta = -1  # Size decreased by 1
                    # Always accept improvements
                    S = new_S
                    blocked_mask = new_blocked_mask
                    pair_counts = new_pair_counts
                    current_size = len(S)
                    no_improve_count = 0
                    if current_size < best_size_local:
                        best_size_local = current_size
                        best_set_local = set(S)
                else:
                    no_improve_count += 1
                    
            else:
                # Try swapping: remove p_idx, add q_idx
                p_idx = random.choice(s_list)
                
                # Compute state without p_idx
                temp_blocked_mask = 0
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
                
                s_mask_without_p = 0
                for i in S:
                    if i != p_idx:
                        s_mask_without_p |= (1 << i)
                
                # Unblocked points not in S\{p_idx}
                unblocked = (~temp_blocked_mask) & FULL_MASK & (~s_mask_without_p)
                
                if unblocked == 0:
                    no_improve_count += 1
                    continue
                
                # Pick a random unblocked point
                m = unblocked
                count = bin(m).count('1')
                r = random.randint(0, count - 1)
                q_idx = None
                temp_m = m
                for _ in range(r):
                    lsb = temp_m & -temp_m
                    temp_m ^= lsb
                lsb = temp_m & -temp_m
                q_idx = lsb.bit_length() - 1
                
                # Check if S\{p_idx} ∪ {q_idx} is no-isosceles
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
                    no_improve_count += 1
                    continue
                
                # Check maximality
                new_blocked_mask = 0
                s_list_new = list(S)
                s_list_new.remove(p_idx)
                s_list_new.append(q_idx)
                for i in range(len(s_list_new)):
                    a = s_list_new[i]
                    for j in range(i + 1, len(s_list_new)):
                        b = s_list_new[j]
                        new_blocked_mask |= BLOCK_MASK[a][b]
                
                new_s_mask = 0
                for i in S:
                    if i != p_idx:
                        new_s_mask |= (1 << i)
                new_s_mask |= (1 << q_idx)
                not_in_new = FULL_MASK ^ new_s_mask
                if (not_in_new & ~new_blocked_mask) != 0:
                    no_improve_count += 1
                    continue
                
                # Valid swap - size stays the same
                # Accept with some probability to escape local optima
                if random.random() < 0.3:  # Accept non-improving swaps with some probability
                    S = set(S)
                    S.discard(p_idx)
                    S.add(q_idx)
                    blocked_mask = new_blocked_mask
                    no_improve_count += 1
                else:
                    no_improve_count += 1
            
            T *= cooling
        
        return best_size_local, best_set_local
    
    # Main loop
    restarts = 0
    while time.time() < deadline:
        S_init = greedy_construct(restarts * 104729 + 7)
        time_limit = min(deadline, time.time() + 250)
        size, S_final = simulated_annealing(S_init, restarts * 7919 + 1, time_limit)
        
        if size < best_size:
            best_size = size
            best_set = S_final
        
        restarts += 1
        
        # Early termination if we've found a very small set
        if best_size <= 8:
            break
    
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
