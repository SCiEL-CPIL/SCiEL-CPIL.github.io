import time

def construct():
    """
    Returns the smallest known tournament for Schütte's problem (k=4).
    Uses Paley tournament construction for p=67.
    """
    n = 67
    
    # Precompute orientation bits for Paley tournament of size 67
    # Paley: i -> j iff (j-i) is a nonzero quadratic residue mod p
    p = n
    is_sq = [False] * p
    for x in range(1, p):
        is_sq[(x * x) % p] = True
    
    orientation = [0] * (p * (p - 1) // 2)
    idx = 0
    for i in range(p):
        for j in range(i + 1, p):
            diff = (j - i) % p
            # 0 means i -> j
            orientation[idx] = 0 if is_sq[diff] else 1
            idx += 1
            
    return {'n': n, 'orientation': orientation}
