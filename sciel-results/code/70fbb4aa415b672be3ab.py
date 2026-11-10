import numpy as np

def kissing_number11() -> np.ndarray:
    d = 11
    points = []
    
    # Generate D_11 lattice minimal vectors
    # Vectors with exactly two non-zero entries, which are +/- 1
    for i in range(d):
        for j in range(i + 1, d):
            # 4 sign combinations
            for s1 in [1, -1]:
                for s2 in [1, -1]:
                    v = np.zeros(d, dtype=np.int64)
                    v[i] = s1
                    v[j] = s2
                    points.append(v)
    
    # Convert to array
    arr = np.array(points, dtype=np.int64)
    
    # The task requires rounding and casting, though we generated integers directly.
    # arr = np.around(arr).astype(np.int64)
    
    return arr
