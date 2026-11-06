import time
import random

def construct():
    start_time = time.time()
    time_limit = 110  # Leave 10s margin
    
    N = 512
    K = 32
    
    def set_to_bits(s):
        bits = 0
        for x in s:
            bits |= (1 << (x - 1))
        return bits
            
    def bits_to_set(bits):
        return [i + 1 for i in range(N) if (bits >> i) & 1]

    def score_bits(bits):
        # Count distinct d > 0 such that a, a+d, a+2d in set
        count = 0
        for d in range(1, 256):
            if (bits & (bits >> d) & (bits >> (2 * d))) != 0:
                count += 1
        return count

    # 1. Greedy Initialization
    S = set()
    current_bits = 0
    
    for _ in range(K):
        best_x = -1
        best_score = -1
        for x in range(1, N + 1):
            if x in S:
                continue
            new_bits = current_bits | (1 << (x - 1))
            sc = score_bits(new_bits)
            if sc > best_score:
                best_score = sc
                best_x = x
        
        if best_x != -1:
            S.add(best_x)
            current_bits |= (1 << (best_x - 1))
            
    current_score = best_score
    best_score = current_score
    best_bits = current_bits
    current_set = S
    
    # 2. Simulated Annealing Polish
    if time.time() - start_time > time_limit - 5:
        return bits_to_set(best_bits)
        
    T_start = 5.0
    T_end = 0.01
    
    # Precompute list of set elements for faster sampling? 
    # Actually, random.choice(list(current_set)) is O(K) which is fine for K=32.
    
    # To speed up the inner loop, we can try to avoid full list conversion if possible, 
    # but K is small. The bottleneck is score_bits which is O(N) bit ops.
    # score_bits is extremely fast in Python (big int ops are C-speed).
    
    while time.time() - start_time < time_limit:
        # Pick random x in S, y not in S
        x = random.choice(list(current_set))
        y = random.randint(1, N)
        
        if y in current_set:
            continue
            
        # Compute new bits
        new_bits = current_bits
        new_bits &= ~(1 << (x - 1))
        new_bits |= (1 << (y - 1))
        
        new_score = score_bits(new_bits)
        delta = new_score - current_score
        
        # Calculate T based on time elapsed in SA phase
        t_elapsed = time.time() - start_time
        progress = t_elapsed / time_limit
        if progress > 1.0:
            progress = 1.0
            
        T = T_start * pow(T_end / T_start, progress)
        
        if delta > 0 or random.random() < pow(2.718281828, delta / T):
            # Accept
            current_bits = new_bits
            current_set.remove(x)
            current_set.add(y)
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_bits = current_bits

    return bits_to_set(best_bits)
