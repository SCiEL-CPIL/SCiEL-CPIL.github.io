import time
import random
import numpy as np

N = 64

def construct():
    """
    Finds a set of points in [0, N)^2 with no isosceles triangles.
    Uses randomized greedy with optimized conflict checking.
    """
    # Precompute squared distances
    # Grid points are indexed 0 to N*N - 1
    # index i corresponds to (i // N, i % N)
    
    # Generate coordinates
    coords = np.array([(x // N, x % N) for x in range(N * N)])
    
    # Compute squared distances matrix
    # dist2[i, j] = (xi - xj)^2 + (yi - yj)^2
    # Using broadcasting
    # coords shape (N*N, 2)
    # diff = coords[:, None, :] - coords[None, :, :]
    # dist2 = np.sum(diff**2, axis=2)
    
    # This creates a (4096, 4096) matrix. 
    # Memory: 4096*4096*4 bytes (int32) = 64 MB. Acceptable.
    # Time: Fast with NumPy.
    
    diff = coords[:, None, :] - coords[None, :, :]
    dist2 = np.sum(diff**2, axis=2).astype(np.int32)
    
    best_set = []
    deadline = time.time() + 250  # Safety margin
    
    while time.time() < deadline:
        points = list(range(N * N))
        random.shuffle(points)
        
        S = []  # List of indices in the current set
        # D[u] is a set of squared distances from u to other points in S
        # We only need D for u in S.
        D = {}
        
        for p in points:
            valid = True
            dists_from_p = set()
            
            # Check against all u in S
            for u in S:
                d = dist2[p, u]
                
                # 1. Isosceles at p: d(p, u) == d(p, v)
                # If d is already in dists_from_p, we have another point v with same distance.
                if d in dists_from_p:
                    valid = False
                    break
                
                dists_from_p.add(d)
                
                # 2. Isosceles at u: d(p, u) == d(u, v) for some v in S
                # Check if this distance is already in D[u]
                if d in D.get(u, ()):
                    valid = False
                    break
            
            if valid:
                S.append(p)
                # Update D
                D[p] = set()
                # Add distances to existing points
                for u in S[:-1]:
                    d = dist2[p, u]
                    D[u].add(d)
                    D[p].add(d)
        
        if len(S) > len(best_set):
            best_set = S.copy()
            # Optional: if we reach a very high score, we might break early?
            # But randomization might find better.
            # Let's just keep going until deadline.
            
    # Convert indices back to (x, y)
    result = [(i // N, i % N) for i in best_set]
    return result
