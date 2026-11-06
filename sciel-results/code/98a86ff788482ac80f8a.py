import time
import random
import math
import bisect

def _score(S_bits):
    """
    S_bits is a bitset integer where bit i is 1 if i is in S.
    Returns count of distinct d > 0 such that a, a+d, a+2d in S.
    """
    count = 0
    # Max possible d is (512-1)//2 = 255
    for d in range(1, 256):
        if (S_bits & (S_bits >> d) & (S_bits >> (2 * d))) != 0:
            count += 1
    return count

def _score_delta(S_bits, x, y):
    """
    Compute the change in score when removing x and adding y.
    Returns (delta, new_score) where delta = new_score - old_score.
    """
    old_score = _score(S_bits)
    new_bits = S_bits ^ (1 << x) ^ (1 << y)
    new_score = _score(new_bits)
    return new_score - old_score, new_score

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    N = 32
    MAXV = 512
    
    best_score = -1
    best_list = []
    
    # Helper to update bits from list
    def get_bits(lst):
        b = 0
        for x in lst:
            b |= (1 << x)
        return b
    
    # Helper to check if a value is in the set
    def in_set(S_bits, val):
        return (S_bits >> val) & 1
    
    # Seeds - more diverse starting points
    seeds = []
    # Seed 1: Interval [1..32]
    seeds.append(list(range(1, 33)))
    
    # Seed 2: Interval [1..31] + [33]
    seeds.append(list(range(1, 32)) + [33])
    
    # Seed 3: Random
    seeds.append(random.sample(range(1, MAXV + 1), N))
    
    # Seed 4: Two intervals
    seeds.append(list(range(1, 17)) + list(range(256, 272)))
    
    # Seed 5: Spread out
    seeds.append([i * 16 + 1 for i in range(32)])
    
    # Seed 6: Dense cluster + sparse
    seeds.append(list(range(1, 25)) + [25 + i * 12 for i in range(8)])
    
    # Seed 7: Geometric-ish
    seeds.append([1, 2, 4, 8, 16, 32, 64, 128, 256, 3, 6, 12, 24, 48, 96, 192, 5, 10, 20, 40, 80, 160, 320, 7, 14, 28, 56, 112, 224, 9, 18, 36])
    # Deduplicate and ensure 32 distinct
    seed7 = list(dict.fromkeys(seeds[-1]))
    while len(seed7) < 32:
        seed7.append(random.randint(1, MAXV))
    seeds[-1] = seed7[:32]
    
    for seed in seeds:
        if time.time() > deadline:
            break
            
        current_list = seed[:]
        current_list.sort()
        current_bits = get_bits(current_list)
        current_score = _score(current_bits)
        
        if current_score > best_score:
            best_score = current_score
            best_list = current_list.copy()
        
        # Simulated Annealing with better move generation
        T = 15.0
        T_min = 0.01
        cooling = 0.9997
        local_deadline = min(start_time + 15.0, deadline - 5.0)
        
        while time.time() < local_deadline:
            # Pick x to remove
            x_idx = random.randint(0, N - 1)
            x = current_list[x_idx]
            
            # Generate candidate y values with bias toward creating progressions
            # Try several candidates and pick the best
            best_y = None
            best_delta = -1000
            best_new_score = -1
            
            # Strategy 1: Random candidates
            for _ in range(5):
                y = random.randint(1, MAXV)
                if y == x or in_set(current_bits, y):
                    continue
                
                new_bits = current_bits ^ (1 << x) ^ (1 << y)
                new_score = _score(new_bits)
                delta = new_score - current_score
                
                if delta > best_delta:
                    best_delta = delta
                    best_y = y
                    best_new_score = new_score
            
            # Strategy 2: Targeted moves - find values that would create new progressions
            # For each d, check if adding y would create a progression with existing elements
            # This is expensive, so do it sparsely
            if random.random() < 0.3:
                # Try values near existing elements to form progressions
                for x2 in random.sample(current_list, min(8, N)):
                    for d in random.sample(range(1, 128), 4):
                        for offset in [-2*d, -d, d, 2*d]:
                            y = x2 + offset
                            if 1 <= y <= MAXV and y != x and not in_set(current_bits, y):
                                new_bits = current_bits ^ (1 << x) ^ (1 << y)
                                new_score = _score(new_bits)
                                delta = new_score - current_score
                                if delta > best_delta:
                                    best_delta = delta
                                    best_y = y
                                    best_new_score = new_score
            
            # If no good move found, accept random or skip
            if best_y is None:
                y = random.randint(1, MAXV)
                if y == x or in_set(current_bits, y):
                    continue
                new_bits = current_bits ^ (1 << x) ^ (1 << y)
                best_new_score = _score(new_bits)
                best_delta = best_new_score - current_score
            
            # Acceptance criterion
            if best_delta > 0 or (T > 0 and random.random() < math.exp(best_delta / T)):
                # Update list
                new_list = current_list[:x_idx] + current_list[x_idx+1:]
                ins_idx = bisect.bisect_left(new_list, best_y)
                new_list.insert(ins_idx, best_y)
                
                current_list = new_list
                current_bits = current_bits ^ (1 << x) ^ (1 << best_y)
                current_score = best_new_score
                
                if current_score > best_score:
                    best_score = current_score
                    best_list = current_list.copy()
            
            T *= cooling
            if T < T_min:
                T = T_min
    
    # Final polish: local search with only improving moves
    if time.time() < deadline - 2.0:
        polish_deadline = deadline - 1.0
        current_list = best_list[:]
        current_list.sort()
        current_bits = get_bits(current_list)
        current_score = _score(current_bits)
        
        improved = True
        while improved and time.time() < polish_deadline:
            improved = False
            for x_idx in range(N):
                x = current_list[x_idx]
                # Try replacing x with each y not in set
                for y in range(1, MAXV + 1):
                    if y == x or in_set(current_bits, y):
                        continue
                    new_bits = current_bits ^ (1 << x) ^ (1 << y)
                    new_score = _score(new_bits)
                    if new_score > current_score:
                        new_list = current_list[:x_idx] + current_list[x_idx+1:]
                        ins_idx = bisect.bisect_left(new_list, y)
                        new_list.insert(ins_idx, y)
                        current_list = new_list
                        current_bits = new_bits
                        current_score = new_score
                        best_score = new_score
                        best_list = current_list.copy()
                        improved = True
                        break
                if time.time() > polish_deadline:
                    break
    
    return best_list
