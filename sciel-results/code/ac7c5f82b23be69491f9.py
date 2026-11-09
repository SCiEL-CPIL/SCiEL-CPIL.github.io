import numpy as np
import itertools
import time
import random

def construct():
    D = 11
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Generate candidates with exactly 4 non-zero entries of +/- 1
    # Norm R = 2. Constraint: u . v <= 2.
    # Conflict if u . v > 2.
    
    supports = list(itertools.combinations(range(D), 4))
    signs_iter = itertools.product([1, -1], repeat=4)
    
    candidates = []
    for supp in supports:
        for signs in signs_iter:
            v = [0] * D
            for idx, val in zip(supp, signs):
                v[idx] = val
            candidates.append(v)
            
    V = np.array(candidates, dtype=np.int64)
    N = len(V)
    
    # Compute dot products
    # V is (N, D). V @ V.T is (N, N).
    Dots = V @ V.T
    
    # Conflict if dot > 2
    # Note: Diagonal is 4, which is > 2, so self-conflict.
    # We just need to ensure we don't pick the same node twice, which the greedy loop handles.
    
    # Create adjacency bitsets
    # adj[i] is an integer with bit j set if i conflicts with j
    adj = [0] * N
    # To speed up bitset creation, we can use numpy to get indices
    # But N=5280, so N^2 = 27M.
    # Looping in Python to build bitsets might be slow.
    # Let's use a list of lists of neighbors first, then convert?
    # Or just use numpy boolean matrix and convert row by row?
    
    # Efficient bitset construction:
    # For each i, get mask of conflicts.
    # mask = Dots[i] > 2
    # Convert mask to int.
    
    # This loop takes a bit of time but is necessary.
    for i in range(N):
        mask = Dots[i] > 2
        # Convert boolean array to integer bitset
        # This is the bottleneck.
        # We can pack bits using numpy.
        # Or just use a simple loop if N is small enough?
        # 5280 iterations.
        # Inside: 5280 boolean ops.
        # Total 27M ops. NumPy is fast.
        
        # Method:
        # bits = mask.astype(np.uint64)
        # Pack 64 bits at a time?
        
        # Simpler:
        # Use Python int from bytes?
        # mask.tobytes() -> int.from_bytes
        
        b = mask.tobytes()
        # int.from_bytes expects little-endian by default in py3?
        # Yes.
        # But mask is boolean (1 byte per element).
        # We need to pack bits.
        
        # Let's try a different approach for speed:
        # Store neighbors as a list of lists, then use that for greedy?
        # Greedy:
        # selected = set()
        # for i in order:
        #   if not any(j in selected for j in neighbors[i]):
        #       selected.add(i)
        # This is O(N * deg).
        # deg is low (~30-50).
        # N=5280.
        # 5280 * 50 = 264,000 checks per pass.
        # 1000 passes = 264M checks.
        # In Python, this might be too slow (10-20s per 10M checks?).
        # 264M checks ~ 200s. Too slow.
        
        # Bitset is better.
        # Let's optimize bitset creation.
        
        # We can create the bitset by summing powers of 2?
        # No, too slow.
        
        # Let's use the bytes method.
        # mask is shape (N,).
        # We want an integer where bit j is mask[j].
        
        # Trick:
        # view as uint8, then pack.
        # Actually, int.from_bytes(mask.tobytes(), 'little') works if we treat each byte as a bit?
        # No, tobytes gives 1 byte per element.
        # We need to compress.
        
        # Alternative:
        # Use numpy to create the bitset directly?
        # There isn't a direct function.
        
        # Let's just build the neighbor lists and use a hybrid approach?
        # Or accept the cost of bitset creation once.
        
        # Cost of bitset creation:
        # 5280 rows.
        # Each row: 5280 bools.
        # If we use a loop to set bits:
        #   for j in np.where(mask)[0]:
        #       adj[i] |= (1 << j)
        # Number of conflicts is ~75,000.
        # Total iterations over all i: 75,000.
        # This is very fast!
        
        # So:
        # conflicts = np.where(Dots > 2)
        # c_i, c_j = conflicts
        # for i, j in zip(c_i, c_j):
        #     adj[i] |= (1 << j)
        
        # This is O(E). E=75,000. Very fast.
        
    # Re-implementing adj construction efficiently
    adj = [0] * N
    c_i, c_j = np.where(Dots > 2)
    for i, j in zip(c_i, c_j):
        adj[i] |= (1 << j)
        
    # Greedy MIS with random restarts
    best_size = 0
    best_set = []
    
    # Estimate max restarts based on time
    # One pass:
    # N iterations.
    # Bitwise AND: 1 op.
    # Bitwise OR: 1 op.
    # Very fast.
    # 5280 iterations * 1000 restarts = 5M iterations.
    # Should take < 1s.
    
    num_restarts = 2000
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        indices = list(range(N))
        random.shuffle(indices)
        
        # Optional: Sort by degree (popcount of adj[i]) ascending
        # deg = [bin(adj[i]).count('1') for i in indices]
        # indices = [x for _, x in sorted(zip(deg, indices))]
        # This adds O(N log N) per restart.
        # Let's try random first. If it's too slow, add sorting.
        
        selected_bits = 0
        selected_nodes = []
        
        for i in indices:
            if adj[i] & selected_bits:
                continue
            selected_bits |= (1 << i)
            selected_nodes.append(i)
            
        if len(selected_nodes) > best_size:
            best_size = len(selected_nodes)
            best_set = selected_nodes[:]
            
    if best_size == 0:
        # Fallback
        vectors = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors.append(v)
        return np.array(vectors, dtype=np.int64)
        
    result = V[best_set]
    return result
