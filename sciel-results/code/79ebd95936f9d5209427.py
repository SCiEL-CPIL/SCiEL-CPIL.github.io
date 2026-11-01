import numpy as np

def construct():
    # E8 minimal vectors
    e8_vectors = []
    # Integer vectors: two +/-1, rest 0
    for i in range(8):
        for j in range(i+1, 8):
            for s1 in [1, -1]:
                for s2 in [1, -1]:
                    v = [0]*8
                    v[i] = s1
                    v[j] = s2
                    e8_vectors.append(v)
    # Half-integer vectors: all +/-1/2, even number of +1s
    for signs in range(256):
        # Check even number of +1s
        # Represent signs as bits
        bits = bin(signs).count('1')
        if bits % 2 == 0:
            v = [0.5 * (1 if (signs >> i) & 1 else -1) for i in range(8)]
            e8_vectors.append(v)
            
    e8_vectors = np.array(e8_vectors)
    # Scale to integers?
    # The vectors are currently half-integers.
    # We can multiply by 2 to make them integers.
    # If we scale E8 by 2, norm squared becomes 8.
    # If we scale A3 by 2, norm squared becomes 8.
    # Then product vectors have norm squared 8.
    # Max norm is sqrt(8).
    # Distance condition: dist >= sqrt(8).
    # dist^2 >= 8.
    # For E8 vectors, min dist squared is 2 (in original scale) -> 8 in scaled scale.
    # So it works.
    
    e8_int = e8_vectors * 2
    
    # A3 minimal vectors
    a3_vectors = []
    # Permutations of (+/-1, +/-1, 0)
    for i in range(3):
        for j in range(3):
            if i == j: continue
            k = 3 - i - j # This logic is flawed for 3 indices
            # Better: iterate permutations of (0, 1, 2)
    for perm in [(0,1,2), (0,2,1), (1,0,2), (1,2,0), (2,0,1), (2,1,0)]:
        for s1 in [1, -1]:
            for s2 in [1, -1]:
                v = [0]*3
                v[perm[0]] = s1
                v[perm[1]] = s2
                # v[perm[2]] is 0
                a3_vectors.append(v)
                
    a3_vectors = np.array(a3_vectors)
    a3_int = a3_vectors * 2
    
    # Construct product vectors
    # V1: (e8, 0)
    # V2: (0, a3)
    # V3: (e8, a3)
    
    n8 = len(e8_int)
    n3 = len(a3_int)
    
    v1 = np.hstack([e8_int, np.zeros((n8, 3))])
    v2 = np.hstack([np.zeros((n3, 8)), a3_int])
    
    # V3: Cartesian product
    # e8_int shape (240, 8)
    # a3_int shape (12, 3)
    # Result shape (240*12, 11)
    
    # Use broadcasting
    # e8_expanded = e8_int[:, np.newaxis, :] # (240, 1, 8)
    # a3_expanded = a3_int[np.newaxis, :, :] # (1, 12, 3)
    # v3 = np.concatenate([np.broadcast_to(e8_expanded, (240, 12, 8)), 
    #                      np.broadcast_to(a3_expanded, (240, 12, 3))], axis=2)
    # v3 = v3.reshape(-1, 11)
    
    v3_parts = []
    for v8 in e8_int:
        for v3 in a3_int:
            v3_parts.append(np.concatenate([v8, v3]))
    v3 = np.array(v3_parts)
    
    all_vectors = np.vstack([v1, v2, v3])
    
    return all_vectors
