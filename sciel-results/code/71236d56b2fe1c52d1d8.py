import time
import random
import itertools

def construct(n):
    start_time = time.time()
    # Use a safety margin, e.g., 110s
    deadline = start_time + 110.0
    
    if n == 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "1213121"
    
    symbols = list(range(1, n))
    perms = list(itertools.permutations(symbols))
    N = len(perms)
    
    # Precompute cost matrix
    # cost[i][j] = (n-1) - max_overlap(perms[i], perms[j])
    cost = [[0] * N for _ in range(N)]
    for i in range(N):
        u = perms[i]
        for j in range(N):
            v = perms[j]
            max_k = 0
            # Max possible overlap is n-1, but since they are distinct perms, 
            # overlap < n-1 unless u==v (which is 0 cost, but we don't visit same node twice in path)
            # Actually if u==v, overlap is n-1, cost 0.
            for k in range(1, n):
                if u[-k:] == v[:k]:
                    max_k = k
            cost[i][j] = (n - 1) - max_k
            
    # Initialize path
    path = list(range(N))
    random.shuffle(path)
    
    def calc_cost(p):
        c = 0
        for i in range(len(p) - 1):
            c += cost[p[i]][p[i+1]]
        return c
        
    current_cost = calc_cost(path)
    best_path = path[:]
    best_cost = current_cost
    
    T = 10.0
    cooling_rate = 0.999
    min_T = 0.1
    
    # Simulated Annealing
    while T > min_T:
        if time.time() > deadline:
            break
            
        # Inner loop: try a few moves at this temperature
        # Adjust inner loop size based on time?
        # Just run until deadline or T drops.
        # To avoid long stalls, limit iterations per T.
        max_iters_per_T = 1000
        iters = 0
        while iters < max_iters_per_T:
            if time.time() > deadline:
                break
            iters += 1
            
            # 2-opt move
            i = random.randint(0, N - 2)
            j = random.randint(i + 1, N - 1)
            
            old_c = 0
            new_c = 0
            
            if i > 0:
                old_c += cost[path[i-1]][path[i]]
                new_c += cost[path[i-1]][path[j]]
            if j < N - 1:
                old_c += cost[path[j]][path[j+1]]
                new_c += cost[path[i]][path[j+1]]
                
            delta = new_c - old_c
            
            if delta < 0 or random.random() < pow(2.718281828, -delta / T):
                path[i:j+1] = reversed(path[i:j+1])
                current_cost += delta
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_path = path[:]
        
        T *= cooling_rate
        
    path = best_path
    
    # Construct string
    result = list(perms[path[0]])
    for k in range(1, N):
        prev_idx = path[k-1]
        curr_idx = path[k]
        u = perms[prev_idx]
        v = perms[curr_idx]
        max_k = 0
        for k_len in range(1, n):
            if u[-k_len:] == v[:k_len]:
                max_k = k_len
        result.extend(v[max_k:])
        
    return "".join(map(str, result))
