"""Kissing number in dimension 11: support-level MIS + sign pattern optimization."""
import numpy as np
import itertools
import time
import random

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    # Generate all 4-subsets (supports)
    supports = list(itertools.combinations(range(D), 4))
    n_sup = len(supports)  # 330
    
    # Build conflict graph: edge if |intersection| == 3
    adj = [set() for _ in range(n_sup)]
    support_sets = [set(s) for s in supports]
    for i in range(n_sup):
        for j in range(i + 1, n_sup):
            if len(support_sets[i] & support_sets[j]) == 3:
                adj[i].add(j)
                adj[j].add(i)
    
    adj_list = [list(s) for s in adj]
    
    # Greedy independent set construction
    def greedy_is():
        remaining = set(range(n_sup))
        current_set = set()
        while remaining:
            min_deg = float('inf')
            best_v = None
            for v in remaining:
                deg = len(adj[v] & remaining)
                if deg < min_deg:
                    min_deg = deg
                    best_v = v
                    if deg == 0:
                        break
            current_set.add(best_v)
            to_remove = {best_v} | (adj[best_v] & remaining)
            remaining -= to_remove
        return current_set
    
    # Simulated annealing on support-level MIS
    def sa_mis(initial_set, max_steps, deadline):
        S = initial_set.copy()
        current_size = len(S)
        best_set = S.copy()
        best_size = current_size
        T = 1.0
        cooling = 0.9999
        
        step = 0
        while step < max_steps and time.time() < deadline:
            step += 1
            v = random.randint(0, n_sup - 1)
            
            if v in S:
                if random.random() < 0.05:
                    S.discard(v)
                    current_size -= 1
            else:
                N = adj[v] & S
                n_N = len(N)
                delta = 1 - n_N
                
                if delta >= 0:
                    S -= N
                    S.add(v)
                    current_size += delta
                    if current_size > best_size:
                        best_size = current_size
                        best_set = S.copy()
                else:
                    if T > 0.01:
                        p = np.exp(delta / T)
                        if random.random() < p:
                            S -= N
                            S.add(v)
                            current_size += delta
                            if current_size > best_size:
                                best_size = current_size
                                best_set = S.copy()
            
            T = max(0.01, T * cooling)
        
        return best_set, best_size
    
    # Multiple restarts of greedy + SA
    rng = random.Random(42)
    np.random.seed(42)
    
    best_support_set = set()
    best_support_size = 0
    
    # Initial greedy
    initial = greedy_is()
    best_support_set = initial.copy()
    best_support_size = len(initial)
    
    # SA refinement with multiple restarts
    max_sa_time = deadline - 30.0  # Leave time for vector construction
    
    for restart in range(50):
        if time.time() > max_sa_time:
            break
        
        # New greedy with random tie-breaking
        remaining = set(range(n_sup))
        current_set = set()
        while remaining:
            # Randomly perturb degree calculation for diversity
            min_deg = float('inf')
            best_v = None
            for v in remaining:
                deg = len(adj[v] & remaining) + rng.random() * 0.1
                if deg < min_deg:
                    min_deg = deg
                    best_v = v
            current_set.add(best_v)
            to_remove = {best_v} | (adj[best_v] & remaining)
            remaining -= to_remove
        
        # SA from this start
        sa_set, sa_size = sa_mis(current_set, 100000, max_sa_time)
        
        if sa_size > best_support_size:
            best_support_size = sa_size
            best_support_set = sa_set.copy()
    
    # Construct vectors from best support set
    # For each selected support, include all 16 sign patterns
    vectors = []
    for sup_idx in best_support_set:
        support = supports[sup_idx]
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(support, signs):
                v[pos] = s
            vectors.append(v)
    
    # Convert to numpy array
    if len(vectors) == 0:
        # Fallback
        v = np.zeros(D, dtype=np.int64)
        v[0] = 1
        v[1] = 1
        v[2] = 1
        v[3] = 1
        return v.reshape(1, -1)
    
    result = np.array(vectors, dtype=np.int64)
    
    # Try to add more vectors from non-selected supports using remaining time
    if time.time() < deadline - 10.0:
        # Build conflict matrix for current vectors
        n_vec = len(result)
        c_int = result.astype(np.int32)
        dot = c_int @ c_int.T
        # Conflict if dot > 2
        conflict = (dot > 2).astype(np.uint8)
        np.fill_diagonal(conflict, 0)
        
        # Generate candidate vectors from non-selected supports
        selected_supports = set(best_support_set)
        candidate_vectors = []
        for sup_idx in range(n_sup):
            if sup_idx not in selected_supports:
                support = supports[sup_idx]
                for signs in itertools.product([1, -1], repeat=4):
                    v = [0] * D
                    for pos, s in zip(support, signs):
                        v[pos] = s
                    candidate_vectors.append((sup_idx, v))
        
        # Also try weight-3 vectors
        for positions in itertools.combinations(range(D), 3):
            for signs in itertools.product([1, -1], repeat=3):
                v = [0] * D
                for pos, s in zip(positions, signs):
                    v[pos] = s
                candidate_vectors.append((-1, v))  # -1 indicates non-weight-4
        
        # Try to add candidates greedily
        added = 0
        for sup_idx, cv in candidate_vectors:
            if time.time() > deadline - 5.0:
                break
            
            cv_arr = np.array(cv, dtype=np.int32)
            # Check if compatible with all current vectors
            dots = cv_arr @ c_int.T
            if (dots > 2).any():
                continue
            
            # Add this vector
            result = np.vstack([result, cv_arr.reshape(1, -1)])
            c_int = result.astype(np.int32)
            added += 1
    
    return result

# EVOLVE-BLOCK-END
