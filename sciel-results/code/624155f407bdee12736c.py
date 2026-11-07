import time
import random

def construct():
    N = 12
    M = 40
    deadline = time.time() + 55  # Safety margin
    
    # 1. Greedy Initialization
    F = [0]  # Start with empty set
    F_set = {0}
    
    # Precompute popcount or just use bit_count()
    for _ in range(M - 1):
        best_mask = -1
        best_score = -1
        # To make it faster, we might not iterate all 4096 if we have a heuristic, but 4096 is small.
        # Randomize iteration order to avoid bias towards lower indices?
        candidates = list(range(1, 1 << N))
        random.shuffle(candidates)
        
        for mask in candidates:
            if mask in F_set:
                continue
            # Calculate comparability with current F
            score = 0
            for m in F:
                # Check if m is subset of mask or mask is subset of m
                # (m & mask) == m means m subset mask
                # (m & mask) == mask means mask subset m
                if (m & mask) == m or (m & mask) == mask:
                    if m != mask: # Strict subset
                        score += 1
            
            if score > best_score:
                best_score = score
                best_mask = mask
                # Early break if we found a perfect extension? Max possible score is len(F)
                if best_score == len(F):
                    break
        
        if best_mask != -1:
            F.append(best_mask)
            F_set.add(best_mask)
        else:
            # Fallback: pick a random unused mask
            while True:
                r = random.randint(0, (1 << N) - 1)
                if r not in F_set:
                    F.append(r)
                    F_set.add(r)
                    break

    # 2. Simulated Annealing
    def get_score_delta(mask, F_list, F_set, index_to_remove):
        # Calculate change in score if we replace F_list[index_to_remove] with mask
        # Old score contribution of F_list[index_to_remove]
        old_m = F_list[index_to_remove]
        delta = 0
        for i, m in enumerate(F_list):
            if i == index_to_remove:
                continue
            # Check old_m vs m
            if (old_m & m) == old_m or (old_m & m) == m:
                if old_m != m:
                    delta -= 1
            # Check new mask vs m
            if (mask & m) == mask or (mask & m) == m:
                if mask != m:
                    delta += 1
        return delta

    # Initial full score calculation for reference (optional, we only need deltas)
    # But we need to know if we are improving. We track current_score.
    
    def calc_full_score(F_list):
        score = 0
        for i in range(len(F_list)):
            for j in range(i + 1, len(F_list)):
                mi = F_list[i]
                mj = F_list[j]
                if (mi & mj) == mi or (mi & mj) == mj:
                    if mi != mj:
                        score += 1
        return score

    current_score = calc_full_score(F)
    best_score = current_score
    best_F = F[:]
    
    T = 10.0
    T_min = 0.01
    alpha = 0.9995
    
    start_time = time.time()
    
    while time.time() < deadline and T > T_min:
        # Generate move
        idx = random.randint(0, M - 1)
        old_m = F[idx]
        
        # Generate new mask by flipping bits
        # Probability of flipping k bits
        # Usually 1 bit is good for local search, 2-3 for jumps
        r = random.random()
        if r < 0.7:
            flips = 1
        elif r < 0.9:
            flips = 2
        else:
            flips = 3
            
        new_m = old_m
        for _ in range(flips):
            bit = random.randint(0, N - 1)
            new_m ^= (1 << bit)
            
        if new_m in F_set:
            # Collision, try a different move or skip
            # Simple strategy: skip this move
            T *= alpha
            continue
            
        # Calculate delta
        delta = get_score_delta(new_m, F, F_set, idx)
        
        if delta > 0 or random.random() < pow(2.718281828, delta / T):
            # Accept move
            F_set.remove(old_m)
            F_set.add(new_m)
            F[idx] = new_m
            current_score += delta
            
            if current_score > best_score:
                best_score = current_score
                best_F = F[:]
                
        T *= alpha
        
    # Convert best_F (bitmasks) to list of lists
    result = []
    for mask in best_F:
        subset = [i for i in range(N) if (mask >> i) & 1]
        result.append(subset)
        
    return result
