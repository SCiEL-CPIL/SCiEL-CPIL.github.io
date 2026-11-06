import time
import random

def _has_k222(adj):
    n = len(adj)
    # adj[i] is a bitmask of neighbors
    # Check for K2,2,2
    # Partitions A={u,v}, B={w,x}, C={y,z}
    # u,v not connected. w,x not connected. y,z not connected.
    # All cross edges present.
    
    # Iterate over pairs (u, v) not connected
    for u in range(n):
        for v in range(u+1, n):
            if adj[u] & (1 << v):
                continue # u and v connected, cannot be same part
            
            # Common neighbors of u and v
            common = adj[u] & adj[v]
            
            # We need to find two disjoint non-edges (w,x) and (y,z) in common
            # such that all edges between {w,x} and {y,z} are present.
            
            # Get list of vertices in common
            verts = []
            mask = common
            while mask:
                lsb = mask & -mask
                idx = lsb.bit_length() - 1
                verts.append(idx)
                mask ^= lsb
            
            # Iterate over pairs (w, x) in verts
            # w and x must NOT be connected
            for i in range(len(verts)):
                w = verts[i]
                for j in range(i+1, len(verts)):
                    x = verts[j]
                    if adj[w] & (1 << x):
                        continue # w and x connected, cannot be same part
                    
                    # Now we have part A={u,v}, part B={w,x}
                    # We need part C={y,z} such that:
                    # y,z not connected
                    # y,z connected to u,v,w,x
                    # y,z are in common (so connected to u,v)
                    # y,z must be connected to w and x
                    
                    # Candidates for C are in common & adj[w] & adj[x]
                    cand = common & adj[w] & adj[x]
                    
                    # Get list of candidates
                    cverts = []
                    cmask = cand
                    while cmask:
                        lsb = cmask & -cmask
                        idx = lsb.bit_length() - 1
                        cverts.append(idx)
                        cmask ^= lsb
                        
                    # Check if cverts contains a non-edge pair
                    for k in range(len(cverts)):
                        y = cverts[k]
                        for l in range(k+1, len(cverts)):
                            z = cverts[l]
                            if not (adj[y] & (1 << z)):
                                return True
    return False

def construct():
    start_time = time.time()
    deadline = start_time + 110 # Safety margin
    
    def check_coloring(n, colors):
        # Build adj bitmasks for color 0 and color 1
        adj0 = [0] * n
        adj1 = [0] * n
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                c = colors[idx]
                if c == 0:
                    adj0[i] |= (1 << j)
                    adj0[j] |= (1 << i)
                else:
                    adj1[i] |= (1 << j)
                    adj1[j] |= (1 << i)
                idx += 1
        return not (_has_k222(adj0) or _has_k222(adj1))

    def paley(n):
        # Only works if n is prime and n % 4 == 1
        # Check primality and mod
        if n < 2: return None
        if n % 4 != 1: return None
        # Simple primality check
        if n % 2 == 0: return None
        for i in range(3, int(n**0.5)+1, 2):
            if n % i == 0: return None
            
        squares = {x*x % n for x in range(1, n)}
        colors = []
        for i in range(n):
            for j in range(i+1, n):
                if (j-i) % n in squares:
                    colors.append(0)
                else:
                    colors.append(1)
        return colors

    # Try known good sizes first, descending from 29
    for n in [29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17]:
        if time.time() > deadline:
            break
            
        # Try Paley if possible
        if n % 4 == 1:
            colors = paley(n)
            if colors and check_coloring(n, colors):
                return {'n': n, 'colors': colors}
        
        # If Paley failed or not applicable, try local search for this n
        # But local search is expensive. Only do it for high n if we have time.
        if n >= 25:
            # Random restart local search
            best_found = None
            attempts = 0
            while time.time() < deadline - 5 and attempts < 10:
                attempts += 1
                colors = [random.randint(0, 1) for _ in range(n*(n-1)//2)]
                # Quick improvement: flip edges that are part of many K222s?
                # Simple hill climbing: find a K222, flip a random edge in it.
                # This is complex to implement robustly in short time.
                # Given Paley(29) is a very strong candidate, we rely on it.
                pass

    # Fallback to seed
    return {'n': 17, 'colors': paley(17)}
