import numpy as np

def kissing_number11() -> np.ndarray:
    d = 11
    points = []
    
    # Generate all vectors with exactly two non-zero entries, which are +/- 1
    # This corresponds to norm squared 2.
    # Indices i < j
    # Signs si, sj in {-1, 1}
    
    for i in range(d):
        for j in range(i + 1, d):
            for si in [-1, 1]:
                for sj in [-1, 1]:
                    v = np.zeros(d, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    points.append(v)
    
    # Convert to array
    if not points:
        return np.array([[0]*d], dtype=np.int64)
        
    arr = np.array(points, dtype=np.int64)
    
    # Verification logic (optional but good for sanity check in dev, not needed for submission if confident)
    # k = arr.shape[0]
    # norms = np.sum(arr**2, axis=1)
    # max_norm = np.max(norms)
    # # Check pairwise distances
    # # This is O(k^2 * d) which is 220^2 * 11 ~ 500k ops, fast enough.
    # for i in range(k):
    #     for j in range(i+1, k):
    #         diff = arr[i] - arr[j]
    #         dist_sq = np.sum(diff**2)
    #         if dist_sq < max_norm:
    #             print("Invalid config found!")
    #             return arr # Or handle error
    
    return arr
