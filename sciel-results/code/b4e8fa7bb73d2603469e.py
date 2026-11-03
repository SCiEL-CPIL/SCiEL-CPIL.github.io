import numpy as np
import random
import itertools
import time

def construct():
    D = 11
    supports = list(itertools.combinations(range(D), 4))
    n_sup = len(supports)
    
    # Build adjacency list
    adj = [set() for _ in range(n_sup)]
    for i in range(n_sup):
        s_i = set(supports[i])
        for j in range(i+1, n_sup):
            s_j = set(supports[j])
            if len(s_i.intersection(s_j)) == 3:
                adj[i].add(j)
                adj[j].add(i)
                
    # Convert to lists for faster iteration if needed, but sets are fine for intersection
    # Precompute degree
    degrees = [len(adj[i]) for i in range(n_sup)]
    
    def greedy_is():
        remaining = list(range(n_sup))
        # Sort by degree descending? No, for MIS, we usually pick min degree in remaining graph.
        # Greedy: pick node with min degree in remaining, remove it and its neighbors.
        # This is a heuristic for MIS.
        current_set = set()
        # To make it faster, we can just iterate
        # But the original greedy was okay.
        # Let's use a slightly better greedy: 
        # 1. Sort all nodes by degree ascending.
        # 2. Iterate through sorted list. If node is still in 'available', add to set, remove neighbors.
        
        # This specific greedy (Kahn's algorithm variant for MIS?)
        # Actually, simple greedy: pick min degree in remaining.
        remaining_set = set(range(n_sup))
        while remaining_set:
            # Find min degree in remaining_set
            min_deg = float('inf')
            best_v = None
            # Optimization: If we maintain degrees dynamically, it's faster.
            # But for 330 nodes, O(N^2) is fine.
            for v in remaining_set:
                deg = len(adj[v].intersection(remaining_set))
                if deg < min_deg:
                    min_deg = deg
                    best_v = v
            current_set.add(best_v)
            to_remove = {best_v}.union(adj[best_v].intersection(remaining_set))
            remaining_set.difference_update(to_remove)
        return current_set

    best_set = greedy_is()
    best_size = len(best_set)
    
    # Simulated Annealing
    # We will run multiple restarts or a long single run.
    # Time limit 120s.
    start_time = time.time()
    time_limit = 110.0 # Leave 10s for construction
    
    # Parameters
    T_start = 1.0
    T_end = 0.01
    # Number of steps per restart? Or just one long run?
    # Let's do a long run with occasional reheating.
    
    S = best_set.copy()
    current_size = len(S)
    
    # Precompute neighbors as lists for faster `in` check? No, set intersection is O(min(n1, n2)).
    
    # To speed up SA, we can maintain a list of "candidates" for moves.
    # But random selection is simple.
    
    # Let's use a "Worst-First" or "Random" selection.
    # Random is standard.
    
    # To improve, we can use "Simulated Quantum Annealing" style:
    # Acceptance prob P = 1 / (1 + exp(-delta / T)) ? No, that's sigmoid.
    # Standard SA: P = exp(delta / T).
    # SQA often uses a different schedule or tunneling.
    # Let's just increase the number of steps significantly.
    
    total_steps = 0
    T = T_start
    cooling_factor = 0.99999 # Very slow cooling
    # Or fixed number of steps per temperature?
    
    # Let's try a "Restarting SA" strategy.
    # Run SA for X steps, if no improvement for Y steps, reheat or restart from best.
    
    no_improve_steps = 0
    max_no_improve = 50000
    
    while time.time() - start_time < time_limit:
        # Pick random move
        v = random.randint(0, n_sup - 1)
        
        if v in S:
            # Try to remove v?
            # If we remove v, we lose 1. But we might gain more later.
            # Let's allow removal with low probability.
            if random.random() < 0.01:
                S.remove(v)
                current_size -= 1
                no_improve_steps += 1
            else:
                # Skip
                pass
            T *= cooling_factor
            continue
            
        # v not in S
        N = adj[v].intersection(S)
        n_conflicts = len(N)
        
        if n_conflicts == 0:
            S.add(v)
            current_size += 1
            if current_size > best_size:
                best_size = current_size
                best_set = S.copy()
                no_improve_steps = 0
            else:
                no_improve_steps += 1
        else:
            delta = 1 - n_conflicts
            # delta <= 0
            if delta == 0:
                if random.random() < 0.5:
                    S.difference_update(N)
                    S.add(v)
                    no_improve_steps += 1
            else:
                # delta < 0
                # Avoid underflow/overflow in exp
                if T < 1e-6:
                    p = 0.0
                else:
                    p = np.exp(delta / T)
                if random.random() < p:
                    S.difference_update(N)
                    S.add(v)
                    current_size += delta
                    if current_size > best_size:
                        best_size = current_size
                        best_set = S.copy()
                        no_improve_steps = 0
                    else:
                        no_improve_steps += 1
                        # Reheat if stuck?
                        if no_improve_steps > max_no_improve:
                            T = T_start * 0.5
                            no_improve_steps = 0
                            # Optionally restart from best
                            if random.random() < 0.1:
                                S = best_set.copy()
                                current_size = best_size
        T *= cooling_factor
        total_steps += 1
        
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
