import numpy as np
import time

def kissing_number11() -> np.ndarray:
    """
    Constructs a collection of 11-dimensional points with integral coordinates.
    Uses a shortened Hamming code of length 15 to get a code of length 11
    with minimum Hamming distance 3, mapped to {+1, -1}^11.
    This yields 128 points with min squared distance 12 and max squared norm 11.
    """
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 600  # 600 seconds limit, use full time if needed but return early if fast
    
    d = 11
    
    # Generate Hamming code of length 15
    # Parity check matrix H is 4x15, columns are 1..15 in binary
    H = np.zeros((4, 15), dtype=np.int64)
    for i in range(15):
        val = i + 1
        H[0, i] = val & 1
        H[1, i] = (val >> 1) & 1
        H[2, i] = (val >> 2) & 1
        H[3, i] = (val >> 3) & 1
        
    # The Hamming code is the null space of H.
    # We can generate all 2^11 = 2048 codewords by iterating over all 11-bit messages
    # and computing c = M @ m, where M is a generator matrix.
    # Alternatively, just iterate all 2^15 vectors and keep those with H @ v = 0?
    # 2^15 = 32768. This is fast.
    
    code = []
    for i in range(32768):
        v = np.zeros(15, dtype=np.int64)
        # Extract bits of i
        for j in range(15):
            v[j] = (i >> j) & 1
        # Check parity
        # H @ v mod 2
        # Since H is 4x15, we can compute dot products
        p0 = np.dot(H[0], v) % 2
        p1 = np.dot(H[1], v) % 2
        p2 = np.dot(H[2], v) % 2
        p3 = np.dot(H[3], v) % 2
        if p0 == 0 and p1 == 0 and p2 == 0 and p3 == 0:
            code.append(v)
            
    # code now has 2048 vectors of length 15
    # Shorten 4 times to get length 11
    # Shortening: keep only those with first bit 0, then remove first bit
    for _ in range(4):
        if not code:
            break
        new_code = []
        for v in code:
            if v[0] == 0:
                new_code.append(v[1:])
        code = new_code
        
    # Now code has 128 vectors of length 11
    # Map to {+1, -1}
    # 0 -> 1, 1 -> -1
    points = np.array(code, dtype=np.int64)
    # Convert 0/1 to 1/-1
    # points = 1 - 2 * points
    points = 1 - 2 * points
    
    # Verify
    if len(points) < 2:
        return np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int64)
        
    # Check time
    if time.time() > deadline - 10:
        return points.astype(np.int64)

    # Optional: Verify validity
    # Norms
    norms = np.sum(points * points, axis=1)
    max_norm = np.max(norms)
    
    # Min distance
    min_dist_sq = np.inf
    n_pts = len(points)
    for i in range(n_pts):
        for j in range(i + 1, n_pts):
            diff = points[i] - points[j]
            dist_sq = np.dot(diff, diff)
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
    if min_dist_sq < max_norm:
        # Fallback to seed
        return np.array([np.ones(11), -1 * np.ones(11)]).astype(np.int64)

    return points.astype(np.int64)
