import numpy as np
import time

def nikodym_set() -> np.ndarray:
    p = 7
    q = p * p
    k = 3  # alpha^2 = 3

    # Generate Unital U
    U = set()
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    if (2 * xa + ya * ya - k * yb * yb) % p == 0:
                        U.add((xa, xb, ya, yb))
                        
    C = set(U)
    
    # Directions
    dirs = []
    for ma in range(p):
        for mb in range(p):
            dirs.append((1, 0, ma, mb))
    dirs.append((0, 0, 1, 0))
    
    start_time = time.time()
    deadline = 600.0
    safety_margin = 5.0
    
    def is_valid_complement(C_set):
        for x in C_set:
            xa, xb, ya, yb = x
            has_empty_line = False
            for v in dirs:
                va_a, va_b, vb_a, vb_b = v
                blocked = False
                for t in range(1, q):
                    nx_a = (xa + t * va_a) % p
                    nx_b = (xb + t * va_b) % p
                    ny_a = (ya + t * vb_a) % p
                    ny_b = (yb + t * vb_b) % p
                    if (nx_a, nx_b, ny_a, ny_b) in C_set:
                        blocked = True
                        break
                if not blocked:
                    has_empty_line = True
                    break
            if not has_empty_line:
                return False, x
        return True, None

    # Greedy reduction
    while time.time() < deadline - safety_margin:
        valid, bad_point = is_valid_complement(C)
        if valid:
            break
        C.remove(bad_point)
        
    # If timeout, return current C (which might be invalid if we cut off mid-check? 
    # No, we only remove if is_valid_complement returns False.
    # If we exit due to time, C might still be invalid.
    # To be safe, if we exit due to time, we should return a KNOWN valid set, e.g., the seed's set.
    # But the seed's set is constructed differently.
    # Let's ensure we don't return an invalid set.
    # If time runs out, we can fall back to a small valid set, e.g., a single line.
    
    # Check if C is valid at the end
    final_valid, _ = is_valid_complement(C)
    if not final_valid:
        # Fallback to a known valid set: A single line
        # C = { (0, y) for y in F_49 }
        C = set()
        for ya in range(p):
            for yb in range(p):
                C.add((0, 0, ya, yb))
        
    points = []
    for pt in C:
        points.append([[pt[0], pt[1]], [pt[2], pt[3]]])
        
    if not points:
        return np.array([[[0, 0], [0, 0]]], dtype=np.int64)

    return np.array(points, dtype=np.int64)
