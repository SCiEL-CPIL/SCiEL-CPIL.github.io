import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110 # 10s margin
    
    best_vectors = None
    best_size = 0
    
    # Try different k
    for k in [3, 4, 5, 6]:
        if time.time() > deadline:
            break
            
        # Generate all vectors with k non-zeros
        # This might be memory heavy for large k?
        # k=6: C(11,6)*64 = 462*64 = 29568. Small.
        
        vectors = []
        # Efficient generation
        # Use itertools.combinations for indices
        import itertools
        for indices in itertools.combinations(range(11), k):
            # Generate all 2^k sign combinations
            for signs in itertools.product([1, -1], repeat=k):
                v = np.zeros(11, dtype=np.int64)
                for idx, val in zip(indices, signs):
                    v[idx] = val
                vectors.append(v)
                
        vectors = np.array(vectors)
        N = len(vectors)
        
        # Greedy selection
        selected = []
        # Shuffle to avoid bias? Or keep order?
        # Order might matter for greedy. Random order often better.
        np.random.seed(42)
        indices = np.random.permutation(N)
        
        for idx in indices:
            if time.time() > deadline:
                break
            v = vectors[idx]
            if not selected:
                selected.append(v)
            else:
                # Check distance to all selected
                # selected is list of arrays. Convert to array for speed?
                # If selected is large, converting every time is slow.
                # Keep selected as list, but maybe use a matrix?
                # If we use a matrix, appending is O(K).
                # Let's use a list for now, convert to array for check?
                # Or just loop? No, NumPy is faster.
                
                sel_arr = np.array(selected)
                dists = np.linalg.norm(sel_arr - v, axis=1)
                if np.min(dists) >= np.sqrt(k) - 1e-9:
                    selected.append(v)
                    
        if len(selected) > best_size:
            best_size = len(selected)
            best_vectors = np.array(selected)
            
    if best_vectors is None:
        # Fallback to seed
        # ... seed logic ...
        pass
        
    return best_vectors
