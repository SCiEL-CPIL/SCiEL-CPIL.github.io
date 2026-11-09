def construct():
    import time
    import math
    import random
    
    start_time = time.time()
    deadline = start_time + 80
    
    N = 256
    n = N - 1  # 255 vertices, index i -> value i+2
    
    conflict_neighbors = [set() for _ in range(n)]
    for i in range(n):
        a = i + 2
        for j in range(i + 1, n):
            b = j + 2
            if math.lcm(a, b) <= 256:
                conflict_neighbors[i].add(j)
                conflict_neighbors[j].add(i)
    
    weights = [1.0 / (i + 2) for i in range(n)]
    
    def objective(s):
        return sum(weights[i] for i in s)
    
    def greedy(order):
        sel = set()
        for idx in order:
            if not (conflict_neighbors[idx] & sel):
                sel.add(idx)
        return sel
    
    best = set()
    best_val = 0.0
    
    orders = [
        list(range(n)),
        list(range(n - 1, -1, -1)),
        sorted(range(n), key=lambda i: len(conflict_neighbors[i])),
        sorted(range(n), key=lambda i: -weights[i]),
    ]
    
    for order in orders:
        sol = greedy(order)
        val = objective(sol)
        if val > best_val:
            best_val = val
            best = set(sol)
    
    def local_search(cur, max_iters=20000):
        cur = set(cur)
        cv = objective(cur)
        no_imp = 0
        it = 0
        while it < max_iters:
            it += 1
            imp = False
            for rem in list(cur):
                for add in range(n):
                    if add in cur:
                        continue
                    ns = cur - {rem}
                    if not (conflict_neighbors[add] & ns):
                        nv = cv - weights[rem] + weights[add]
                        if nv > cv + 1e-15:
                            cur = ns | {add}
                            cv = nv
                            imp = True
                            break
                if imp:
                    break
            if not imp:
                no_imp += 1
                if no_imp > 5:
                    break
            else:
                no_imp = 0
            if time.time() > deadline:
                break
        return cur, cv
    
    best, best_val = local_search(best, 30000)
    
    for order in orders:
        if time.time() > deadline:
            break
        sol = greedy(order)
        sol_ls, val_ls = local_search(sol, 20000)
        if val_ls > best_val:
            best_val = val_ls
            best = sol_ls
    
    random.seed(42)
    for _ in range(20):
        if time.time() > deadline:
            break
        pert = set(best)
        if len(pert) > 3:
            for r in random.sample(list(pert), random.randint(1, 3)):
                pert.discard(r)
        for idx in sorted(range(n), key=lambda i: -weights[i]):
            if idx not in pert and not (conflict_neighbors[idx] & pert):
                pert.add(idx)
        sol_ls, val_ls = local_search(pert, 10000)
        if val_ls > best_val:
            best_val = val_ls
            best = sol_ls
    
    result = sorted([i + 2 for i in best])
    if not result:
        result = [2]
    return result
