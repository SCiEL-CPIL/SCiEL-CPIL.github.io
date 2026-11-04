import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = 110  # Safety margin
    
    # 1. Generate all 4-sparse vectors
    all_vectors = []
    for indices in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for idx, s in zip(indices, signs):
                v[idx] = s
            all_vectors.append(v)
            
    V = np.array(all_vectors, dtype=np.int64)
    n_vec = len(V)
    
    # 2. Compute conflict matrix
    # Dot products
    # V is (n_vec, D)
    # Dots is (n_vec, n_vec)
    dots = V @ V.T
    
    # Conflict if dot > 2
    conflict = dots > 2
    
    # 3. Find Max Independent Set
    # Greedy with randomization + Local Search
    
    best_size = 0
    best_set = []
    
    # Greedy initialization
    # We will run multiple randomized greedy passes
    
    # Pre-compute degrees? Or just use the matrix.
    # To speed up, we can maintain a set of available candidates.
    
    for _ in range(200): # 200 restarts
        if time.time() > deadline:
            break
            
        # Randomize order
        order = np.random.permutation(n_vec)
        
        selected = []
        selected_indices = set()
        
        # Greedy Selection
        # We want to pick vertices with low conflict with current set.
        # Efficient way: maintain a bitmask or list of candidates?
        # n_vec is 5280. 
        # For each step, scanning all remaining is O(N^2). 200 * 5280^2 is too slow.
        
        # Better: 
        # Maintain a list of available vectors.
        # When we select a vector, remove all its neighbors from available.
        
        available = list(range(n_vec))
        # Shuffle available? No, we pick best from available.
        # "Best" = min conflicts with selected.
        # Initially selected is empty, so all have 0 conflicts.
        # We can just pick randomly from available?
        # Or pick the one with min degree in the remaining graph?
        
        # Let's try a simpler heuristic:
        # Randomized greedy: pick a random vector from available, add it, remove its neighbors.
        # This is fast.
        
        current_set = []
        current_indices = set()
        
        # To make it faster, we can just iterate through a shuffled list and add if no conflict with current_set.
        # But checking conflict with current_set for each candidate is O(|current_set|).
        # Total O(N * K). K is size of set (~300). N=5280. 5280*300 = 1.5M ops per run.
        # 200 runs = 300M ops. Might be tight in Python.
        
        # Optimization: Use NumPy for conflict check.
        # But we need to update the set dynamically.
        
        # Let's stick to a simpler local search on a single good greedy result.
        
        # Run 1 good greedy: Min-degree heuristic.
        # Compute degrees.
        degrees = np.sum(conflict, axis=1)
        
        selected = []
        used = np.zeros(n_vec, dtype=bool)
        
        # We will simulate the greedy process.
        # To avoid O(N^2) in Python loops, we can just use the dot product matrix directly in a vectorized way?
        # No, we need to build the set sequentially.
        
        # Alternative: Use the slot-based logic but with the CORRECT conflict rule.
        # The correct rule: Conflict if dot > 2.
        # We can pre-compute the "signature" of each vector that allows fast conflict check?
        # Not easily, because it depends on the 4th element.
        
        # Let's try a simple local search on the conflict graph.
        # Start with a random independent set.
        
        # Generate a random independent set:
        # Shuffle vertices.
        # Add vertex if it doesn't conflict with any in set.
        
        order = np.random.permutation(n_vec)
        current_set = []
        current_indices = set()
        
        for i in order:
            if time.time() > deadline - 20:
                break
            # Check conflict with current_set
            # current_set is list of indices.
            # conflict[i, current_set] should be all False.
            if not current_set:
                current_set.append(i)
                current_indices.add(i)
            else:
                # Vectorized check
                if not np.any(conflict[i, current_set]):
                    current_set.append(i)
                    current_indices.add(i)
                    
        best_size = len(current_set)
        best_set = current_set[:]
        
        # Local Search: Try to improve
        # We can try to swap.
        # Or, we can try to add a vector if we remove one?
        
        # Given time constraints, the randomized greedy might be enough to find 306.
        # 5280 vectors, max independent set 306.
        # Random greedy usually finds close to optimal for such sparse graphs.
        
        # Combine with D11 vectors
        # Generate D11 vectors
        d11_vectors = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = np.zeros(D, dtype=np.int64)
                    v[i], v[j] = si, sj
                    d11_vectors.append(v)
                    
        d11_arr = np.array(d11_vectors, dtype=np.int64)
        
        # Check if we should return just the 4-sparse or union?
        # The task asks for "integer vectors". Union is valid.
        # Score is k.
        
        if len(best_set) > 0:
            result_4sparse = V[best_set]
            result = np.vstack([d11_arr, result_4sparse])
        else:
            result = d11_arr
            
        return result

