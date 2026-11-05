"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 78  # safety margin
    
    N = 256
    squares = set()
    k = 2
    while k * k <= 2 * N:
        squares.add(k * k)
        k += 1
    
    # Exclude numbers where 2*x is a perfect square
    excluded = set()
    m = 1
    while 2 * m * m <= N:
        excluded.add(2 * m * m)
        m += 1
    
    candidates = [x for x in range(1, N + 1) if x not in excluded]
    n_cand = len(candidates)
    val_to_idx = {v: i for i, v in enumerate(candidates)}
    
    # Build adjacency
    adj = [set() for _ in range(n_cand)]
    for i in range(n_cand):
        a = candidates[i]
        for j in range(i + 1, n_cand):
            b = candidates[j]
            if a + b in squares:
                adj[i].add(j)
                adj[j].add(i)
    
    degree = [len(adj[i]) for i in range(n_cand)]
    
    def greedy_construct(order):
        selected = set()
        for i in order:
            if all(j not in selected for j in adj[i]):
                selected.add(i)
        return selected
    
    def local_improve(sel):
        improved = True
        while improved:
            improved = False
            for i in range(n_cand):
                if i not in sel and all(j not in sel for j in adj[i]):
                    sel.add(i)
                    improved = True
        return sel
    
    # Initial greedy
    order = sorted(range(n_cand), key=lambda i: degree[i])
    current = greedy_construct(order)
    current = local_improve(current)
    best = set(current)
    best_size = len(current)
    
    random.seed(123)
    
    # Local search with random restarts
    while time.time() < deadline:
        # Random perturbation
        if random.random() < 0.5 and len(best) > 5:
            current = set(best)
            k_remove = random.randint(1, min(5, len(current)))
            for _ in range(k_remove):
                if current:
                    current.discard(random.choice(list(current)))
            # Re-greedy fill
            for i in sorted(range(n_cand), key=lambda x: degree[x]):
                if i not in current and all(j not in current for j in adj[i]):
                    current.add(i)
            current = local_improve(current)
        else:
            # Random order greedy
            order_rand = list(range(n_cand))
            random.shuffle(order_rand)
            order_rand.sort(key=lambda i: degree[i] + random.random())
            current = greedy_construct(order_rand)
            current = local_improve(current)
        
        # 1-swap local search
        for _ in range(50):
            if time.time() > deadline:
                break
            improved = False
            for i in list(current):
                for j in range(n_cand):
                    if j not in current:
                        # Try: remove i, add j, then fill
                        test = set(current)
                        test.discard(i)
                        if all(k not in test for k in adj[j]):
                            test.add(j)
                            # Fill
                            for k in range(n_cand):
                                if k not in test and all(m not in test for m in adj[k]):
                                    test.add(k)
                            if len(test) > best_size:
                                best = test
                                best_size = len(test)
                                current = set(test)
                                improved = True
                                break
                if improved:
                    break
            if not improved:
                break
        
        if len(best) == n_cand:
            break
    
    result = sorted(candidates[i] for i in best)
    if not result:
        return [1]
    return result
# EVOLVE-BLOCK-END
