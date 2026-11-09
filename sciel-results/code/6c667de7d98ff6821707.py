import numpy as np

def kissing_number11() -> np.ndarray:
    d = 11
    # Generate all pairs of indices for the non-zero coordinates
    # Each point has exactly two non-zero coordinates, each being +/-1
    # Squared norm = 2 for all points
    # Minimum squared distance between distinct points:
    #   - If they share no indices: dist^2 = 4
    #   - If they share one index: dist^2 = 2 (if same sign) or 4 (if different sign)
    #   Wait, if they share one index with same sign: (1,1,0..) - (1,0,1..) = (0,1,-1,0..) -> dist^2 = 2
    #   If they share one index with different sign: (1,1,0..) - (-1,0,1..) = (2,1,-1,0..) -> dist^2 = 6
    #   If they are the same pair with different signs: (1,1,0..) - (-1,-1,0..) = (2,2,0..) -> dist^2 = 8
    #   Or (1,1,0..) - (-1,1,0..) = (2,0,0..) -> dist^2 = 4
    #   Or (1,1,0..) - (1,-1,0..) = (0,2,0..) -> dist^2 = 4
    # So minimum squared distance is 2, which equals max_norm^2 = 2. Valid!
    
    points = []
    for i in range(d):
        for j in range(i + 1, d):
            for si in (1, -1):
                for sj in (1, -1):
                    v = np.zeros(d, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    points.append(v)
    
    points_array = np.array(points, dtype=np.int64)
    return np.around(points_array).astype(np.int64)
