import numpy as np
import time
import itertools

D = 11

def construct():
    start_time = time.time()
    deadline = 150  # seconds
    
    # Generate candidate vectors
    # We will consider vectors with weight 1 to 5
    # To manage memory and time, we generate them in batches or use a generator
    # But for D=11, weight 5 is 14784 vectors. Weight 4 is 5280. Total < 25000.
    # This is small enough to store in memory.
    
    candidates = []
    
    # Helper to add vector
    def add_vec(v):
        # v is a tuple of ints
        # Check if it's a duplicate (signs matter)
        # We just append, duplicates will be handled or avoided
        candidates.append(v)

    # Generate weight 1
    for i in range(D):
        for s in (1, -1):
            v = [0]*D
            v[i] = s
            add_vec(tuple(v))
            
    # Generate weight 2
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0]*D
                v[i] = si
                v[j] = sj
                add_vec(tuple(v))
                
    # Generate weight 3
    for i, j, k in itertools.combinations(range(D), 3):
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    v = [0]*D
                    v[i] = si
                    v[j] = sj
                    v[k] = sk
                    add_vec(tuple(v))
                    
    # Generate weight 4
    for i, j, k, l in itertools.combinations(range(D), 4):
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    for sl in (1, -1):
                        v = [0]*D
                        v[i] = si
                        v[j] = sj
                        v[k] = sk
                        v[l] = sl
                        add_vec(tuple(v))
                        
    # Generate weight 5
    for i, j, k, l, m in itertools.combinations(range(D), 5):
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    for sl in (1, -1):
                        for sm in (1, -1):
                            v = [0]*D
                            v[i] = si
                            v[j] = sj
                            v[k] = sk
                            v[l] = sl
                            v[m] = sm
                            add_vec(tuple(v))
                            
    # Convert to numpy array for fast ops? 
    # Or keep as list of tuples and use list of numpy arrays for selected set.
    
    # Calculate norms squared
    # Sort by norm descending
    # norm_sq = sum(x*x for x in v)
    
    # To speed up, we can pre-calculate norms
    # And sort
    # Then greedy selection
    
    # For efficiency, we will use a list of numpy arrays for the selected set
    selected = []
    max_norm_sq = 0
    
    # Sort candidates by norm_sq descending
    # This might be slow if we do it for 25k vectors, but it's fine.
    # candidates is a list of tuples.
    
    # Optimization: Group by norm?
    # No, just sort.
    
    # We need to be careful with the deadline.
    # Greedy check is O(N_selected * D).
    # If we select ~500 vectors, and have 25k candidates, 
    # 25k * 500 * 11 ops ~ 137M ops. 
    # In Python, this might take a few seconds. 
    # NumPy vectorization can help.
    
    # Let's use NumPy for the selected set.
    # selected_np = np.array(selected)
    
    # To handle the "max norm" constraint properly with descending sort:
    # The first vector added sets the max norm.
    # Subsequent vectors must have norm <= current max norm (guaranteed by sort).
    # And must be at distance >= sqrt(max_norm_sq) from all selected vectors.
    
    # We will iterate through sorted candidates.
    
    # Pre-compute norms for sorting
    # Using list comprehension
    norm_sq_list = [sum(x*x for x in v) for v in candidates]
    # Sort indices by norm_sq descending
    indices = np.argsort(norm_sq_list)[::-1]
    
    # We will use a numpy array for selected vectors to speed up distance checks
    # But we need to update it as we go.
    # List of numpy arrays is easier to append.
    
    selected_arrays = []
    current_max_norm_sq = 0
    
    # To speed up, we can break early if time is low
    # Or just run and hope it's fast enough.
    # 25k candidates.
    
    # Let's try to vectorize the distance check against the selected set.
    # If selected set is small, loop is fine.
    
    for idx in indices:
        if time.time() > deadline - 10:
            break
            
        v = candidates[idx]
        v_np = np.array(v, dtype=np.float64)
        v_norm_sq = norm_sq_list[idx]
        
        # If this vector has a smaller norm than current max, it's fine.
        # If it has a larger norm, it would have been processed earlier.
        # So v_norm_sq <= current_max_norm_sq is always true for the first few?
        # No, the first vector sets current_max_norm_sq.
        
        if not selected_arrays:
            selected_arrays.append(v_np)
            current_max_norm_sq = v_norm_sq
            continue
            
        # Check compatibility with all selected vectors
        # Distance squared >= current_max_norm_sq
        # Note: current_max_norm_sq is the max norm of the set.
        # Since we sorted descending, v_norm_sq <= current_max_norm_sq.
        # So the constraint is ||u - v||^2 >= current_max_norm_sq.
        
        # Compute distances to all selected
        # selected_arrays is a list of arrays.
        # Convert to array if possible? 
        # For now, loop.
        
        compatible = True
        for u_np in selected_arrays:
            diff = u_np - v_np
            dist_sq = np.dot(diff, diff)
            if dist_sq < current_max_norm_sq:
                compatible = False
                break
                
        if compatible:
            selected_arrays.append(v_np)
            
    # Convert to integer array
    if selected_arrays:
        result = np.array(selected_arrays, dtype=np.int64)
    else:
        result = np.array([], dtype=np.int64).reshape(0, D)
        
    return result

# EVOLVE-BLOCK-START
# (Code above is inside construct, but we need to put it in the block)
# The prompt asks for the COMPLETE program.
# I will structure it properly.
# EVOLVE-BLOCK-END
