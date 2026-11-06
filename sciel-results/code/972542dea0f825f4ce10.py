import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    # Precompute shift masks for fast scoring
    # For a given bitset b, the score is the number of d in 1..255
    # such that (b & (b >> d) & (b >> 2*d)) != 0
    
    # Precompute bit masks for each value 1..512
    # masks[x] = 1 << (x-1)
    masks = [0] * 513
    for i in range(1, 513):
        masks[i] = 1 << (i - 1)
        
    def score_bitset(b):
        count = 0
        # d can be at most 255 because a+2d <= 512 => 2d <= 511 => d <= 255
        for d in range(1, 256):
            if (b & (b >> d) & (b >> (2 * d))):
                count += 1
        return count
    
    def get_bitset(S):
        b = 0
        for x in S:
            b |= masks[x]
        return b
    
    def random_set():
        return random.sample(range(1, 513), 32)
        
    def local_search(current_set, current_bit, current_score):
        """
        Hill climbing: Try replacing each element with any other valid value.
        Returns improved set, bit, score.
        """
        S = list(current_set)
        bit = current_bit
        score = current_score
        S_set = set(S)
        
        improved = True
        # Limit passes to avoid infinite loops in case of bugs, though convergence is expected
        max_passes = 50
        
        while improved and max_passes > 0 and time.time() < deadline - 5:
            improved = False
            max_passes -= 1
            for i in range(32):
                old_val = S[i]
                best_new_val = None
                best_new_score = score
                best_new_bit = bit
                
                # Try all possible replacements
                for v in range(1, 513):
                    if v == old_val or v in S_set:
                        continue
                    
                    # Update bit
                    new_bit = (bit & ~masks[old_val]) | masks[v]
                    
                    # Score
                    new_score = score_bitset(new_bit)
                    
                    if new_score > best_new_score:
                        best_new_score = new_score
                        best_new_val = v
                        best_new_bit = new_bit
                
                if best_new_val is not None:
                    # Apply best move
                    S[i] = best_new_val
                    S_set.remove(old_val)
                    S_set.add(best_new_val)
                    bit = best_new_bit
                    score = best_new_score
                    improved = True
        
        return S, bit, score

    def greedy_init():
        """
        Greedy construction: Add elements one by one, choosing the one
        that maximizes the score contribution.
        """
        S = []
        S_set = set()
        bit = 0
        current_score = 0
        
        for _ in range(32):
            best_v = -1
            best_s = -1
            # Randomize order of candidates to break ties and explore different paths
            candidates = list(range(1, 513))
            random.shuffle(candidates)
            
            for v in candidates:
                if v in S_set:
                    continue
                new_bit = bit | masks[v]
                new_score = score_bitset(new_bit)
                if new_score > best_s:
                    best_s = new_score
                    best_v = v
            if best_v != -1:
                S.append(best_v)
                S_set.add(best_v)
                bit = bit | masks[best_v]
                current_score = best_s
        return S, bit, current_score

    best_set = None
    best_bit = 0
    best_score = -1

    # 1. Start with Greedy Init + Local Search
    if time.time() < deadline - 10:
        g_set, g_bit, g_score = greedy_init()
        g_set, g_bit, g_score = local_search(g_set, g_bit, g_score)
        
        if g_score > best_score:
            best_score = g_score
            best_set = list(g_set)
            best_bit = g_bit

    # 2. Multi-start Simulated Annealing + Local Search
    # Determine number of restarts based on remaining time
    # Estimate ~10s per restart
    remaining_time = deadline - time.time()
    num_restarts = max(1, int(remaining_time / 12))
    
    for restart in range(num_restarts):
        if time.time() >= deadline - 15:
            break
            
        # Random Init
        current_set = random_set()
        current_bit = get_bitset(current_set)
        current_score = score_bitset(current_bit)
        
        # Simulated Annealing
        T = 10.0
        T_min = 0.01
        stall_count = 0
        
        # Run SA for ~5 seconds
        sa_start = time.time()
        sa_duration = 4.0 + random.random() * 2.0
        
        while time.time() < sa_start + sa_duration and T > T_min:
            idx = random.randint(0, 31)
            old_val = current_set[idx]
            new_val = random.randint(1, 512)
            
            if new_val == old_val or new_val in current_set:
                continue
            
            new_bit = (current_bit & ~masks[old_val]) | masks[new_val]
            new_score = score_bitset(new_bit)
            delta = new_score - current_score
            
            # Metropolis criterion
            if delta >= 0 or random.random() < (2.718281828 ** (delta / T)):
                current_set[idx] = new_val
                current_bit = new_bit
                current_score = new_score
                stall_count = 0
            else:
                stall_count += 1
                
            if stall_count > 2000:
                T *= 0.9
                stall_count = 0
                
            T *= 0.99995
            
        # Local Search to converge
        current_set, current_bit, current_score = local_search(current_set, current_bit, current_score)
        
        # Update Global Best
        if current_score > best_score:
            best_score = current_score
            best_set = list(current_set)
            best_bit = current_bit

    # Final Polish: Ensure the best set is locally optimal
    if best_set is not None and time.time() < deadline - 5:
        best_set, best_bit, best_score = local_search(best_set, best_bit, best_score)

    if best_set is None:
        best_set = list(range(1, 33))
        
    return best_set
