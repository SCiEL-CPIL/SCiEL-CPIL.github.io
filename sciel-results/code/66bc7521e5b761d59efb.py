import random

def construct():
    n = 512
    
    # Precompute for each x, the list of triples (y, z, w) such that {x, y, z, w} is a 4-AP
    # We need to ensure y, z, w are distinct and in [1, n]
    aps_for_x = [[] for _ in range(n + 1)]
    
    for a in range(1, n + 1):
        max_d = (n - a) // 3
        for d in range(1, max_d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            
            # The AP is (a, b, c, e)
            # For each element, the other three are the rest
            aps_for_x[a].append((b, c, e))
            aps_for_x[b].append((a, c, e))
            aps_for_x[c].append((a, b, e))
            aps_for_x[e].append((a, b, c))
            
    def run_greedy(seed):
        random.seed(seed)
        order = list(range(1, n + 1))
        random.shuffle(order)
        
        S = set()
        
        for x in order:
            # Check if x can be added
            can_add = True
            for triple in aps_for_x[x]:
                if all(y in S for y in triple):
                    can_add = False
                    break
            if can_add:
                S.add(x)
                
        return S

    best_set = set()
    
    # Run multiple trials
    # 200 trials should be fast enough
    for trial in range(200):
        s = run_greedy(trial)
        if len(s) > len(best_set):
            best_set = s
            
    return sorted(best_set)
