import numpy as np
import time
import random
import heapq

def priority(adjacency):
    start_time = time.time()
    adj = np.asarray(adjacency, dtype=bool)
    n = adj.shape[0]
    
    # Precompute neighbor lists for fast access
    # adj is boolean matrix
    neighbor_lists = []
    for i in range(n):
        neighbors = np.nonzero(adj[i])[0]
        neighbor_lists.append(neighbors)
    
    # Helper to run greedy decoder on a given permutation (list of vertices)
    # Returns the set of chosen vertices and the count
    def greedy_decode(permutation):
        chosen = []
        blocked = np.zeros(n, dtype=bool)
        # We process vertices in the order given by permutation
        # If a vertex is not blocked, we pick it and block it + neighbors
        for v in permutation:
            if not blocked[v]:
                chosen.append(v)
                blocked[v] = True
                for u in neighbor_lists[v]:
                    blocked[u] = True
        return chosen, len(chosen)
    
    # Initial heuristic: sort by degree ascending (prefer low degree)
    # This is a good starting point for Maximum Independent Set
    degrees = adj.sum(axis=1)
    # Stable sort by degree, then by index for determinism
    initial_perm = np.argsort(degrees, kind='stable')
    
    # Evaluate initial
    current_perm = list(initial_perm)
    current_size, current_count = greedy_decode(current_perm)
    best_perm = current_perm.copy()
    best_count = current_count
    
    # Time budget management
    # We have 120s limit. Let's reserve 10s for safety and final checks.
    deadline = start_time + 110.0
    
    # Simulated Annealing on Permutation
    # State: permutation list
    # Move: Swap two random elements OR Insert a random element at a random position
    # Evaluation: Incremental if possible, else full re-sim (but re-sim is O(n * avg_deg) which is fast for n=200)
    # For n=200, a full re-sim is ~200 * 200 = 40k ops. Very fast in Python/NumPy?
    # Actually, pure Python loop over 200 items is fine.
    
    T_start = 1.0
    T_end = 0.01
    cooling_rate = 0.999
    
    T = T_start
    no_improve = 0
    max_no_improve = 500
    
    # To speed up evaluation, we can cache the "blocked" status logic?
    # No, permutation changes order, so blocked status depends on prefix.
    # Full re-simulation is O(n^2) worst case.
    # n=200 -> 40,000 ops.
    # 110 seconds / (40,000 ops / 1e7 ops/s) ~ 110 / 0.004 = 27,500 iterations.
    # This is feasible.
    
    while time.time() < deadline and no_improve < max_no_improve:
        # Generate a new permutation
        new_perm = current_perm.copy()
        
        # Random Move: Swap or Insert
        move_type = random.random()
        if move_type < 0.5:
            # Swap
            i, j = random.sample(range(n), 2)
            new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
        else:
            # Insert: Pick a random element, remove it, insert at random position
            i = random.randint(0, n - 1)
            v = new_perm.pop(i)
            j = random.randint(0, n - 1)
            new_perm.insert(j, v)
        
        # Evaluate new permutation
        _, new_count = greedy_decode(new_perm)
        
        delta = new_count - current_count
        
        if delta >= 0:
            current_perm = new_perm
            current_count = new_count
            no_improve = 0
            if new_count > best_count:
                best_count = new_count
                best_perm = new_perm.copy()
        else:
            # Accept with probability exp(delta / T)
            # delta is negative
            prob = np.exp(delta / T)
            if random.random() < prob:
                current_perm = new_perm
                current_count = new_count
            no_improve += 1
        
        # Cool down
        T *= cooling_rate
        if T < T_end:
            T = T_end
            
    return best_perm

# Note: The task asks for `priority(adjacency)` to return "one score per vertex".
# The evaluator says: "a fixed greedy decoder picks vertices in descending score order".
# This implies the return value is a vector of scores, not a permutation.
# However, the "inspiration" notes and the strategy suggest optimizing the order.
# If I return a permutation, how do I map it to scores?
# The greedy decoder sorts by score descending.
# So if I have a desired permutation P (where P[0] is picked first),
# I need scores S such that argsort(-S) == P.
# I can assign S[P[0]] = n, S[P[1]] = n-1, ..., S[P[n-1]] = 1.
# This ensures the greedy decoder picks them in the order P.

def priority(adjacency):
    start_time = time.time()
    adj = np.asarray(adjacency, dtype=bool)
    n = adj.shape[0]
    
    # Precompute neighbor lists
    neighbor_lists = []
    for i in range(n):
        neighbors = np.nonzero(adj[i])[0]
        neighbor_lists.append(neighbors)
    
    def greedy_decode(permutation):
        blocked = np.zeros(n, dtype=bool)
        count = 0
        for v in permutation:
            if not blocked[v]:
                count += 1
                blocked[v] = True
                for u in neighbor_lists[v]:
                    blocked[u] = True
        return count

    # Initial heuristic: sort by degree ascending
    degrees = adj.sum(axis=1)
    initial_perm = list(np.argsort(degrees, kind='stable'))
    
    current_perm = initial_perm
    current_count = greedy_decode(current_perm)
    best_perm = current_perm.copy()
    best_count = current_count
    
    deadline = start_time + 110.0
    
    T_start = 1.0
    T_end = 0.01
    cooling_rate = 0.999
    
    T = T_start
    no_improve = 0
    max_no_improve = 1000 # Allow some exploration even if stuck
    
    while time.time() < deadline:
        if no_improve > max_no_improve:
            # Restart with higher temperature if stuck? Or just break?
            # Let's try a random restart of the permutation if stuck for too long
            if random.random() < 0.1:
                # Perturb current best
                current_perm = best_perm.copy()
                for _ in range(10):
                    i, j = random.sample(range(n), 2)
                    current_perm[i], current_perm[j] = current_perm[j], current_perm[i]
                current_count = greedy_decode(current_perm)
                no_improve = 0
                T = T_start * 0.5
            else:
                break
        
        new_perm = current_perm.copy()
        move_type = random.random()
        if move_type < 0.5:
            i, j = random.sample(range(n), 2)
            new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
        else:
            i = random.randint(0, n - 1)
            v = new_perm.pop(i)
            j = random.randint(0, n - 1)
            new_perm.insert(j, v)
        
        new_count = greedy_decode(new_perm)
        delta = new_count - current_count
        
        if delta >= 0:
            current_perm = new_perm
            current_count = new_count
            no_improve = 0
            if new_count > best_count:
                best_count = new_count
                best_perm = new_perm.copy()
        else:
            prob = np.exp(delta / T)
            if random.random() < prob:
                current_perm = new_perm
                current_count = new_count
            no_improve += 1
        
        T *= cooling_rate
        if T < T_end:
            T = T_end

    # Convert best_perm (list of vertices in order of selection) to scores
    # The greedy decoder sorts by score DESCENDING.
    # So vertex at index 0 in best_perm should have the highest score.
    scores = np.zeros(n)
    for rank, v in enumerate(best_perm):
        scores[v] = n - rank
        
    return scores
