import random
import time

def construct():
    n = 512
    
    # Precompute 4-APs
    # ap_list[i] = (a, b, c, d)
    # elem_to_aps[x] = list of indices i where x is in ap_list[i]
    # For fast checking: for each x, we need to know the other 3 elements in each AP containing x.
    # conflict_map[x] = list of tuples (y, z, w) such that {x, y, z, w} is a 4-AP
    
    conflict_map = [[] for _ in range(n + 1)]
    
    for a in range(1, n + 1):
        max_d = (n - a) // 3
        for d in range(1, max_d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            # The AP is (a, b, c, e)
            
            # For each element in the AP, store the other 3
            conflict_map[a].append((b, c, e))
            conflict_map[b].append((a, c, e))
            conflict_map[c].append((a, b, e))
            conflict_map[e].append((a, b, c))

    def greedy_run(seed):
        random.seed(seed)
        S = set()
        # We process elements in a random order
        # But simply shuffling 1..n and adding if possible is O(n^2 * k) roughly.
        # n=512, k~100 APs per element. 512 * 512 * 100 = 26M ops. 
        # This is fast enough for a few hundred runs in 60s.
        
        candidates = list(range(1, n + 1))
        random.shuffle(candidates)
        
        for x in candidates:
            # Check if x can be added to S
            can_add = True
            for y, z, w in conflict_map[x]:
                if y in S and z in S and w in S:
                    can_add = False
                    break
            if can_add:
                S.add(x)
                
        return S

    best_set = set()
    start_time = time.time()
    seed = 0
    
    # Run as many trials as possible within time limit
    while time.time() - start_time < 50: # Leave 10s buffer
        s = greedy_run(seed)
        if len(s) > len(best_set):
            best_set = s
        seed += 1
        
    return sorted(best_set)
