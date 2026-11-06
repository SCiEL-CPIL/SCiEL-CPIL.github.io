import time
import random

def _find_k222(adj):
    """Find a K2,2,2 subgraph in adj. Returns (u,v,w,x,y,z) or None."""
    n = len(adj)
    for u in range(n):
        for v in range(u+1, n):
            if adj[u] & (1 << v):
                continue
            common = adj[u] & adj[v]
            verts = []
            mask = common
            while mask:
                lsb = mask & -mask
                idx = lsb.bit_length() - 1
                verts.append(idx)
                mask ^= lsb
            for i in range(len(verts)):
                w = verts[i]
                for j in range(i+1, len(verts)):
                    x = verts[j]
                    if adj[w] & (1 << x):
                        continue
                    cand = common & adj[w] & adj[x]
                    cverts = []
                    cmask = cand
                    while cmask:
                        lsb = cmask & -cmask
                        idx = lsb.bit_length() - 1
                        cverts.append(idx)
                        cmask ^= lsb
                    for k in range(len(cverts)):
                        y = cverts[k]
                        for l in range(k+1, len(cverts)):
                            z = cverts[l]
                            if not (adj[y] & (1 << z)):
                                return (u, v, w, x, y, z)
    return None

def _has_k222(adj):
    return _find_k222(adj) is not None

def construct():
    start_time = time.time()
    deadline = start_time + 110 # Safety margin
    
    def build_adj(n, colors):
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
        return adj0, adj1

    def check_coloring(n, colors):
        adj0, adj1 = build_adj(n, colors)
        return not (_has_k222(adj0) or _has_k222(adj1))

    def local_search(n, colors, max_time=10.0):
        """Local search: find K2,2,2 violations and flip edges to break them."""
        start = time.time()
        colors = list(colors)
        for iteration in range(10000):
            if time.time() - start > max_time:
                break
            adj0, adj1 = build_adj(n, colors)
            # Check color 0
            k222 = _find_k222(adj0)
            if k222:
                u, v, w, x, y, z = k222
                # Pick a random edge among the 6 cross edges of the K2,2,2
                # Cross edges: (u,w),(u,x),(u,y),(u,z),(v,w),(v,x),(v,y),(v,z),(w,y),(w,z),(x,y),(x,z)
                cross_edges = [(u,w),(u,x),(u,y),(u,z),(v,w),(v,x),(v,y),(v,z),(w,y),(w,z),(x,y),(x,z)]
                edge = random.choice(cross_edges)
                a, b = edge
                if a > b:
                    a, b = b, a
                idx = 0
                for i in range(n):
                    for j in range(i+1, n):
                        if i == a and j == b:
                            colors[idx] = 1 - colors[idx]
                            break
                        idx += 1
                continue
            # Check color 1
            k222 = _find_k222(adj1)
            if k222:
                u, v, w, x, y, z = k222
                cross_edges = [(u,w),(u,x),(u,y),(u,z),(v,w),(v,x),(v,y),(v,z),(w,y),(w,z),(x,y),(x,z)]
                edge = random.choice(cross_edges)
                a, b = edge
                if a > b:
                    a, b = b, a
                idx = 0
                for i in range(n):
                    for j in range(i+1, n):
                        if i == a and j == b:
                            colors[idx] = 1 - colors[idx]
                            break
                        idx += 1
                continue
            # No violations found
            return colors
        return colors

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
        
        # Try local search for this n
        if n >= 20:
            attempts = 0
            while time.time() < deadline - 5 and attempts < 5:
                attempts += 1
                # Start from random or Paley-like initialization
                if n % 4 == 1:
                    base = paley(n)
                    if base:
                        colors = list(base)
                    else:
                        colors = [random.randint(0, 1) for _ in range(n*(n-1)//2)]
                else:
                    colors = [random.randint(0, 1) for _ in range(n*(n-1)//2)]
                
                colors = local_search(n, colors, max_time=8.0)
                if check_coloring(n, colors):
                    return {'n': n, 'colors': colors}

    # Fallback to seed
    return {'n': 17, 'colors': paley(17)}
