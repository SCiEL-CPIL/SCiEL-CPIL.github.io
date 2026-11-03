import numpy as np
import random
import itertools
import time

def construct():
    D = 11
    supports = list(itertools.combinations(range(D), 4))
    n_sup = len(supports)
    
    # Build conflict graph
    # Conflict if intersection >= 3
    adj = [set() for _ in range(n_sup)]
    for i in range(n_sup):
        s_i = set(supports[i])
        for j in range(i+1, n_sup):
            s_j = set(supports[j])
            if len(s_i.intersection(s_j)) >= 3:
                adj[i].add(j)
                adj[j].add(i)
                
    # Objective: Maximize size of Independent Set
    # QUBO Energy: E = -sum(x) + lambda * sum(x_i x_j for edges)
    # We want to minimize E.
    
    def get_energy(S):
        # S is a set of indices
        # Energy = -|S| + lambda * (number of conflicts in S)
        # If S is independent, conflicts = 0.
        # We maintain S as an independent set in SA, so conflicts=0 always.
        # So we just maximize |S|.
        return -len(S)

    def greedy_init():
        remaining = set(range(n_sup))
        current_set = set()
        while remaining:
            min_deg = float('inf')
            best_v = None
            for v in remaining:
                deg = len(adj[v].intersection(remaining))
                if deg < min_deg:
                    min_deg = deg
                    best_v = v
                    if deg == 0: break
            current_set.add(best_v)
            to_remove = {best_v}.union(adj[best_v].intersection(remaining))
            remaining.difference_update(to_remove)
        return current_set

    best_set = set()
    best_size = 0
    
    # Run multiple SA instances
    start_time = time.time()
    time_limit = 110 # leave 10s for overhead
    
    num_runs = 5
    iters_per_run = 100000
    
    for run in range(num_runs):
        S = greedy_init()
        current_size = len(S)
        if current_size > best_size:
            best_size = current_size
            best_set = S.copy()
            
        # Local Search: 1-swap
        improved = True
        while improved:
            improved = False
            for v in S:
                # Try removing v? No, we want to grow.
                # Try swapping v with u not in S?
                # Or just check if we can add u?
                # If S is maximal, we can't add.
                # So we do swaps.
                pass # The SA below handles this better.
        
        # Simulated Annealing with Quantum Tunneling
        T = 1.0
        cooling = 0.9995 # slower cooling
        # Quantum tunneling probability
        p_tunnel = 0.05
        
        for step in range(iters_per_run):
            if time.time() - start_time > time_limit:
                break
                
            if random.random() < p_tunnel:
                # Quantum Tunneling Move: Flip a small random set of bits
                # Pick 2-3 nodes to flip
                k = random.choice([2, 3])
                candidates = random.sample(range(n_sup), k)
                # Check if flipping these bits improves or maintains independence
                # This is complex to check efficiently for arbitrary k.
                # Simpler tunneling: Pick a random node v not in S, and a random node u in S.
                # Swap them if valid?
                # Let's stick to standard SA moves but with a "boost" in temperature occasionally.
                pass # We will implement a simpler tunneling: Random Restart if stuck.
                
            # Standard SA Move
            v = random.randint(0, n_sup - 1)
            if v in S:
                # Try to remove v?
                # Or swap v with some u?
                # Let's allow removal with small prob to escape local optima
                if random.random() < 0.1:
                    S.remove(v)
                    current_size -= 1
                else:
                    # Try to swap v with a random neighbor u of v not in S?
                    # Or just skip.
                    continue
            
            # v not in S
            N = adj[v].intersection(S)
            if len(N) == 0:
                S.add(v)
                current_size += 1
                if current_size > best_size:
                    best_size = current_size
                    best_set = S.copy()
            else:
                # Swap: Remove N, Add v
                # Delta = 1 - len(N)
                delta = 1 - len(N)
                if delta > 0:
                    S.difference_update(N)
                    S.add(v)
                    current_size += delta
                    if current_size > best_size:
                        best_size = current_size
                        best_set = S.copy()
                else:
                    # delta <= 0
                    if delta == 0:
                        if random.random() < 0.5:
                            S.difference_update(N)
                            S.add(v)
                    else:
                        p = np.exp(delta / T)
                        if random.random() < p:
                            S.difference_update(N)
                            S.add(v)
                            current_size += delta
                            
            T *= cooling
            if T < 0.01:
                T = 0.01 # floor

    # Construct vectors
    vectors = []
    for idx in best_set:
        sup = supports[idx]
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for k, s in enumerate(sup):
                v[s] = signs[k]
            vectors.append(v)
            
    return np.array(vectors, dtype=np.int64)
