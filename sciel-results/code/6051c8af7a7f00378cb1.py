import time

def construct():
    deadline = time.time() + 140
    
    n = 16
    edges = []
    
    # Vertices are 0..15, representing 5-bit strings 0xxxx
    # We check Hamming distance between the 5-bit representations.
    # Note: The "folded" connection means we consider the complement if the first bit is 1,
    # but since we only use 0xxxx, we just check distance 1 or 4 between the 5-bit ints.
    
    def hamming_dist(a, b):
        x = a ^ b
        return bin(x).count('1')
    
    for u in range(n):
        for v in range(u + 1, n):
            d = hamming_dist(u, v)
            if d == 1 or d == 4:
                edges.append([u, v])
                
    return edges
