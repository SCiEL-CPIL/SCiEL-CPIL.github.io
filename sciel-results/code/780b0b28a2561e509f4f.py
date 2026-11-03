"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    # 1. Initialization: Rank Matching
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    
    # Helper to calculate total cost
    def total_cost(p):
        return np.sum(a * b[np.ix_(p, p)])
    
    # Helper to calculate delta for swapping facilities x and y
    # x, y are facility indices
    # p[x] is location of x, p[y] is location of y
    def calc_delta(x, y, p, a, b):
        Lx = p[x]
        Ly = p[y]
        if Lx == Ly:
            return 0
        
        # Direct term
        direct = a[x, y] * (b[Ly, Lx] - b[Lx, Ly]) + a[y, x] * (b[Lx, Ly] - b[Ly, Lx])
        
        # Sum over k != x, y
        # Vectorized computation
        # Mask to exclude x and y
        mask = np.ones(n, dtype=bool)
        mask[x] = False
        mask[y] = False
        ks = np.where(mask)[0]
        
        # a[x, ks] - a[y, ks]
        ux = a[x, ks] - a[y, ks]
        # b[Ly, p[ks]] - b[Lx, p[ks]]
        dx = b[Ly, p[ks]] - b[Lx, p[ks]]
        
        # a[ks, x] - a[ks, y]
        vx = a[ks, x] - a[ks, y]
        # b[p[ks], Ly] - b[p[ks], Lx]
        ex = b[p[ks], Ly] - b[p[ks], Lx]
        
        sum_term = np.dot(ux, dx) + np.dot(vx, ex)
        
        return direct + sum_term

    # 2. Local Search Polish (2-opt)
    def local_search(p, max_iter=1000):
        current_cost = total_cost(p)
        improved = True
        iters = 0
        while improved and iters < max_iter:
            improved = False
            iters += 1
            best_delta = 0
            best_x = -1
            best_y = -1
            
            # To speed up, we can just loop. n=100 is small.
            for x in range(n):
                for y in range(x + 1, n):
                    delta = calc_delta(x, y, p, a, b)
                    if delta < best_delta:
                        best_delta = delta
                        best_x = x
                        best_y = y
            
            if best_delta < 0:
                # Swap
                p[best_x], p[best_y] = p[best_y], p[best_x]
                current_cost += best_delta
                improved = True
                
        return p, current_cost

    p, current_cost = local_search(p)
    best_p = p.copy()
    best_cost = current_cost
    
    # 3. Iterated Tabu Search
    tabu_len = 15
    tabu_list = []
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() < deadline:
        # Local Search Phase
        p, current_cost = local_search(p, max_iter=200)
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        # Perturbation
        if no_improve_count > max_no_improve:
            # Random kick
            for _ in range(5):
                x = np.random.randint(n)
                y = np.random.randint(n)
                if x != y:
                    p[x], p[y] = p[y], p[x]
            no_improve_count = 0
            tabu_list = []
            
        # Tabu Move (if local search didn't move, pick a tabu move or random)
        # For simplicity, we rely on the kick.
        
    return best_p.tolist()

# EVOLVE-BLOCK-END
