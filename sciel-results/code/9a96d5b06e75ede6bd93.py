import random

def construct():
    candidates = list(range(1, 23))
    
    def try_greedy(order):
        S = []
        sums_set = {0.0}
        sums_list = [0.0] # To generate next sums
        
        for x in order:
            r = 1.0 / x
            new_sums = [s + r for s in sums_list]
            
            collision = False
            for ns in new_sums:
                if ns in sums_set:
                    collision = True
                    break
            
            if not collision:
                S.append(x)
                sums_set.update(new_sums)
                sums_list.extend(new_sums)
        
        return S

    best_set = []
    best_size = 0
    
    # Try deterministic orders
    orders = [
        list(range(1, 23)),
        list(range(22, 0, -1)),
    ]
    
    for order in orders:
        s = try_greedy(order)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s
            
    # Randomized search
    for _ in range(2000):
        random.shuffle(candidates)
        s = try_greedy(candidates)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s
            
    return best_set
