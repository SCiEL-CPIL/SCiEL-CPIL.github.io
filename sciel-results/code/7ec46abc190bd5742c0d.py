import numpy as np
import random
import itertools
import time

def construct():
    D = 11
    supports = list(itertools.combinations(range(D), 4))
    n_sup = len(supports)
    
    # Build adjacency list: two supports conflict if they share exactly 3 coordinates
    adj = [set() for _ in range(n_sup)]
    for i in range(n_sup):
        s_i = set(supports[i])
        for j in range(i+1, n_sup):
            s_j = set(supports[j])
            if len(s_i.intersection(s_j)) == 3:
                adj[i].add(j)
                adj[j].add(i)
    
    # Greedy initialization: find a maximal independent set
    def greedy_is():
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
                    if deg == 0:
                        break
            current_set.add(best_v)
            to_remove = {best_v}.union(adj[best_v].intersection(remaining))
            remaining.difference_update(to_remove)
        return current_set
    
    best_set = greedy_is()
    best_size = len(best_set)
    
    # Simulated quantum annealing inspired approach
    # State: S (set of support indices)
    # We use a hybrid of standard SA moves and "quantum tunneling" moves
    # that allow temporary violations to escape local optima
    
    S = best_set.copy()
    current_size = len(S)
    
    # Precompute for faster neighbor access
    adj_list = [list(s) for s in adj]
    
    # Track time
    start_time = time.time()
    time_budget = 115.0  # Leave 5s margin
    
    # Quantum-inspired annealing with multiple restarts
    n_restarts = 3
    for restart in range(n_restarts):
        if restart > 0:
            # Restart from a perturbed version of best_set
            S = best_set.copy()
            # Remove 10% of vertices randomly
            to_remove = list(S)
            random.shuffle(to_remove)
            n_remove = max(1, len(S) // 10)
            for v in to_remove[:n_remove]:
                S.remove(v)
            current_size = len(S)
        
        T = 2.0
        cooling = 0.9995
        T_min = 0.01
        
        steps_in_restart = 0
        max_steps = 80000
        
        while steps_in_restart < max_steps and T > T_min:
            elapsed = time.time() - start_time
            if elapsed > time_budget:
                break
            
            # Choose move type
            move_type = random.random()
            
            if move_type < 0.7:
                # Standard swap move: pick v not in S, replace its neighbors in S with v
                v = random.randint(0, n_sup - 1)
                if v in S:
                    continue
                
                N = adj[v].intersection(S)
                if len(N) == 0:
                    S.add(v)
                    current_size += 1
                    if current_size > best_size:
                        best_size = current_size
                        best_set = S.copy()
                else:
                    delta = 1 - len(N)
                    if delta >= 0:
                        S.difference_update(N)
                        S.add(v)
                        current_size += delta
                        if current_size > best_size:
                            best_size = current_size
                            best_set = S.copy()
                    else:
                        p = np.exp(delta / T)
                        if random.random() < p:
                            S.difference_update(N)
                            S.add(v)
                            current_size += delta
            
            elif move_type < 0.85:
                # Removal move: remove a vertex from S (allows recovery)
                if len(S) > 5:
                    v = random.choice(list(S))
                    S.remove(v)
                    current_size -= 1
            
            else:
                # Quantum tunneling move: add v even if it conflicts,
                # then immediately resolve by removing one conflicting neighbor
                v = random.randint(0, n_sup - 1)
                if v not in S:
                    N = adj[v].intersection(S)
                    if len(N) > 0:
                        # Remove a random neighbor to make room
                        v_remove = random.choice(list(N))
                        S.remove(v_remove)
                        S.add(v)
                        # Size unchanged, but configuration changed
                        # This allows escaping local optima
                        if current_size > best_size:
                            best_size = current_size
                            best_set = S.copy()
            
            T *= cooling
            steps_in_restart += 1
    
    # Construct vectors from best_set
    vectors = []
    for idx in best_set:
        sup = supports[idx]
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for k, s in enumerate(sup):
                v[s] = signs[k]
            vectors.append(v)
    
    return np.array(vectors, dtype=np.int64)
