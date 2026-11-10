import numpy as np

def kissing_number11() -> np.ndarray:
    d = 11
    
    # Generate 240 scaled E8 roots in Z^8
    e8_roots = []
    
    # Type 1: Two coordinates +/- 2, others 0
    for i in range(8):
        for j in range(i+1, 8):
            for s1 in [1, -1]:
                for s2 in [1, -1]:
                    v = np.zeros(8, dtype=np.int64)
                    v[i] = 2 * s1
                    v[j] = 2 * s2
                    e8_roots.append(v)
                    
    # Type 2: Eight coordinates +/- 1, even number of -1s
    # Iterate over all 2^8 sign combinations
    for mask in range(256):
        # Count number of 1s in mask (or -1s)
        # Let's map 0->1, 1->-1. 
        # We need even number of -1s.
        num_neg = bin(mask).count('1')
        if num_neg % 2 == 0:
            v = np.zeros(8, dtype=np.int64)
            for k in range(8):
                if (mask >> k) & 1:
                    v[k] = -1
                else:
                    v[k] = 1
            e8_roots.append(v)
            
    # Embed in Z^11 (first 8 dims)
    e8_points = np.zeros((len(e8_roots), 11), dtype=np.int64)
    for i, v in enumerate(e8_roots):
        e8_points[i, :8] = v
        
    # Generate 12 scaled D3 roots in Z^3
    d3_roots = []
    for i in range(3):
        for j in range(i+1, 3):
            for s1 in [1, -1]:
                for s2 in [1, -1]:
                    v = np.zeros(3, dtype=np.int64)
                    v[i] = 2 * s1
                    v[j] = 2 * s2
                    d3_roots.append(v)
                    
    # Embed in Z^11 (last 3 dims)
    d3_points = np.zeros((len(d3_roots), 11), dtype=np.int64)
    for i, v in enumerate(d3_roots):
        d3_points[i, 8:] = v
        
    # Combine
    all_points = np.vstack([e8_points, d3_points])
    
    return all_points
