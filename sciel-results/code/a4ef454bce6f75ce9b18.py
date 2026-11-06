import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    # Precompute subsets of 8
    subsets_8 = list(combinations(range(16), 8))
    
    # Precompute bipartitions for 8 items (indices 0..7)
    # We need all ways to split 0..7 into two sets of 4.
    # To avoid duplicates (A,B) and (B,A), we can fix 0 in A.
    # Then choose 3 from 1..7 for A.
    # C(7,3) = 35.
    bipartitions = []
    for a_rest in combinations(range(1, 8), 3):
        a = (0,) + a_rest
        b = tuple(i for i in range(8) if i not in a)
        bipartitions.append((a, b))
        
    def has_q3(adj):
        # adj: list of 16 ints
        for subset in subsets_8:
            # Quick edge count
            edge_count = 0
            for i in range(8):
                u = subset[i]
                mask = 0
                for j in range(i+1, 8):
                    v = subset[j]
                    mask |= 1 << v
                edge_count += bin(adj[u] & mask).count('1')
                
            if edge_count < 12:
                continue
                
            # Check bipartitions
            for a, b in bipartitions:
                # a, b are tuples of indices in 0..7
                # Map to actual vertices
                va = [subset[i] for i in a]
                vb = [subset[i] for i in b]
                
                # Check degrees
                valid = True
                missing_edges = []
                
                # For each v in va, count neighbors in vb
                for u in va:
                    deg = 0
                    for v in vb:
                        if adj[u] & (1 << v):
                            deg += 1
                        else:
                            missing_edges.append((u, v))
                    if deg != 3:
                        valid = False
                        break
                
                if not valid:
                    continue
                    
                # Check degrees for vb
                for v in vb:
                    deg = 0
                    for u in va:
                        if adj[v] & (1 << u):
                            deg += 1
                    if deg != 3:
                        valid = False
                        break
                        
                if not valid:
                    continue
                
                # Check if missing_edges form a perfect matching
                # missing_edges should have 4 edges
                # Check if no vertex appears twice
                vertices_in_missing = []
                for u, v in missing_edges:
                    if u in vertices_in_missing or v in vertices_in_missing:
                        valid = False
                        break
                    vertices_in_missing.extend([u, v])
                    
                if valid:
                    return True
                    
        return False

    # Initialize
    adj = [0]*16
    edges = []
    for u in range(16):
        for v in range(u+1, 16):
            if random.random() < 0.5:
                adj[u] |= 1 << v
                adj[v] |= 1 << u
                edges.append((u,v))
                
    best_edges = edges[:]
    best_adj = adj[:]
    best_count = len(edges)
    
    while time.time() < deadline:
        u = random.randint(0, 15)
        v = random.randint(0, 15)
        if u == v: continue
        if adj[u] & (1 << v): continue
        
        # Try add
        adj[u] |= 1 << v
        adj[v] |= 1 << u
        
        if not has_q3(adj):
            edges.append((u,v))
            if len(edges) > best_count:
                best_count = len(edges)
                best_edges = edges[:]
                best_adj = adj[:]
        else:
            adj[u] ^= 1 << v
            adj[v] ^= 1 << u

    return [[u, v] for u, v in best_edges]
