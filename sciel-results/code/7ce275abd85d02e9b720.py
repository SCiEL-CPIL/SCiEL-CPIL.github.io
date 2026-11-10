import random
import time

def construct():
    n = 256
    
    # Precompute all 3-APs and 4-APs
    # 3-AP: (a, a+d, a+2d)
    three_aps = []
    for d in range(1, n + 1):
        for a in range(1, n + 1):
            c = a + 2 * d
            if c <= n:
                three_aps.append((a, a + d, c))
                
    # 4-AP: (a, a+d, a+2d, a+3d)
    four_aps = []
    for d in range(1, n + 1):
        for a in range(1, n + 1):
            e = a + 3 * d
            if e <= n:
                four_aps.append((a, a + d, a + 2 * d, e))

    def score(S):
        c3 = 0
        for ap in three_aps:
            if all(x in S for x in ap):
                c3 += 1
        c4 = 0
        for ap in four_aps:
            if all(x in S for x in ap):
                c4 += 1
        return c3 - 10000 * c4

    # Simulated Annealing
    # Initial random set
    S = set(random.sample(range(1, n + 1), n // 2))
    current_score = score(S)
    best_S = set(S)
    best_score = current_score
    
    T = 10.0
    cooling = 0.999
    min_T = 0.1
    
    start_time = time.time()
    time_limit = 55.0 # Leave 5s buffer
    
    while time.time() - start_time < time_limit:
        # Pick a random element to toggle
        x = random.randint(1, n)
        
        # Toggle
        if x in S:
            S.remove(x)
        else:
            S.add(x)
            
        new_score = score(S)
        
        if new_score > current_score:
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_S = set(S)
        else:
            # Accept with probability exp((new - current) / T)
            delta = new_score - current_score
            if delta < 0:
                prob = 2.718281828 ** (delta / T)
                if random.random() < prob:
                    current_score = new_score
                else:
                    # Revert
                    if x in S:
                        S.remove(x)
                    else:
                        S.add(x)
            else:
                # Should not happen if logic is correct, but just in case
                current_score = new_score
                
        T *= cooling
        if T < min_T:
            T = min_T

    return sorted(best_S)
