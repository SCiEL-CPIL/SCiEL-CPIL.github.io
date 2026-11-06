import numpy as np
import random
import time

N = 64
NUM_POINTS = N * N

def construct():
    # Create point indices and coordinates
    coords = np.array([(x, y) for y in range(N) for x in range(N)], dtype=np.int32)
    points = list(range(NUM_POINTS))
    
    # Precompute squared distances
    # diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    # d2 = np.sum(diff**2, axis=-1)
    # This creates a 4096x4096 array.
    # Memory: 4096*4096*8 bytes = 128MB.
    # Time: Vectorized, should be fast (<1s).
    
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    d2_matrix = np.sum(diff**2, axis=-1)
    
    best_set = []
    start_time = time.time()
    deadline = start_time + 290
    
    # Map point index to coordinate for return
    # The task asks for list of (x, y)
    
    def is_valid_addition(p_idx, S_indices):
        if len(S_indices) < 2:
            return True
        
        # Get distances from p to all in S
        # S_indices is a list of indices
        # We need d2(p, s) for s in S
        # And d2(s1, s2) for s1, s2 in S
        
        n = len(S_indices)
        # Optimization: If we maintain S as a list, accessing d2_matrix[p, s] is fast.
        
        # We can precompute distances from p to all points in S in a small array
        # But d2_matrix is already there.
        
        for i in range(n):
            si = S_indices[i]
            dpi = d2_matrix[p_idx, si]
            for j in range(i + 1, n):
                sj = S_indices[j]
                dpj = d2_matrix[p_idx, sj]
                dsij = d2_matrix[si, sj]
                
                if dpi == dpj or dpi == dsij or dpj == dsij:
                    return False
        return True

    while time.time() < deadline:
        random.shuffle(points)
        S_indices = []
        for p_idx in points:
            if is_valid_addition(p_idx, S_indices):
                S_indices.append(p_idx)
        
        if len(S_indices) > len(best_set):
            best_set = S_indices
            
    # Convert best indices to coordinates
    return [tuple(coords[idx]) for idx in best_set]
