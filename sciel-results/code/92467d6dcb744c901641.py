import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    def get_idx(i, j, n):
        return i * (2 * n - i - 3) // 2 + (j - i - 1)

    def find_k222(adj):
        """
        Finds a K2,2,2 in the graph defined by adj bitmasks.
        Returns a list of edge indices (i, j) with i < j forming the K2,2,2, or None.
        """
        n = len(adj)
        # Iterate over pairs (u, v) not connected
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u] & (1 << v):
                    continue  # u and v connected, cannot be same part A
                
                # Common neighbors of u and v
                common = adj[u] & adj[v]
                
                # Get list of vertices in common
                verts = []
                mask = common
                while mask:
                    lsb = mask & -mask
                    idx = lsb.bit_length() - 1
                    verts.append(idx)
                    mask ^= lsb
                
                # Iterate over pairs (w, x) in verts
                # w and x must NOT be connected (part B)
                for i in range(len(verts)):
                    w = verts[i]
                    for j in range(i + 1, len(verts)):
                        x = verts[j]
                        if adj[w] & (1 << x):
                            continue  # w and x connected, cannot be same part B
                        
                        # Now we have part A={u,v}, part B={w,x}
                        # We need part C={y,z} such that:
                        # y,z not connected
                        # y,z connected to u,v (in common)
                        # y,z connected to w and x
                        
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
                        
                        # Check if cverts contains a non-edge pair (y, z)
                        for k in range(len(cverts)):
                            y = cverts[k]
                            for l in range(k + 1, len(cverts)):
                                z = cverts[l]
                                if not (adj[y] & (1 << z)):
                                    # Found K2,2,2 with parts:
                                    # A: u, v
                                    # B: w, x
                                    # C: y, z
                                    
                                    # Collect all 12 edges
                                    edges = []
                                    # A-B
                                    edges.append(get_idx(min(u, w), max(u, w), n))
                                    edges.append(get_idx(min(u, x), max(u, x), n))
                                    edges.append(get_idx(min(v, w), max(v, w), n))
                                    edges.append(get_idx(min(v, x), max(v, x), n))
                                    # A-C
                                    edges.append(get_idx(min(u, y), max(u, y), n))
                                    edges.append(get_idx(min(u, z), max(u, z), n))
                                    edges.append(get_idx(min(v, y), max(v, y), n))
                                    edges.append(get_idx(min(v, z), max(v, z), n))
                                    # B-C
                                    edges.append(get_idx(min(w, y), max(w, y), n))
                                    edges.append(get_idx(min(w, z), max(w, z), n))
                                    edges.append(get_idx(min(x, y), max(x, y), n))
                                    edges.append(get_idx(min(x, z), max(x, z), n))
                                    
                                    return edges
        return None

    def is_valid(n, colors):
        adj0 = [0] * n
        adj1 = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                idx = get_idx(i, j, n)
                c = colors[idx]
                if c == 0:
                    adj0[i] |= (1 << j)
                    adj0[j] |= (1 << i)
                else:
                    adj1[i] |= (1 << j)
                    adj1[j] |= (1 << i)
        
        return find_k222(adj0) is None and find_k222(adj1) is None

    def paley(n):
        if n < 2: return None
        if n % 4 != 1: return None
        # Simple primality check
        if n % 2 == 0: return None
        for i in range(3, int(n**0.5) + 1, 2):
            if n % i == 0: return None
            
        squares = {x * x % n for x in range(1, n)}
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                if (j - i) % n in squares:
                    colors.append(0)
                else:
                    colors.append(1)
        return colors

    def local_search(n, colors, time_limit):
        """
        Attempts to fix the coloring by flipping edges in K2,2,2s.
        Returns valid colors or None.
        """
        adj0 = [0] * n
        adj1 = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                idx = get_idx(i, j, n)
                c = colors[idx]
                if c == 0:
                    adj0[i] |= (1 << j)
                    adj0[j] |= (1 << i)
                else:
                    adj1[i] |= (1 << j)
                    adj1[j] |= (1 << i)
        
        def update_adj(idx, new_color):
            i, j = idx_to_pair(idx, n)
            old = colors[idx]
            if old == 0:
                adj0[i] &= ~(1 << j)
                adj0[j] &= ~(1 << i)
            else:
                adj1[i] &= ~(1 << j)
                adj1[j] &= ~(1 << i)
            
            if new_color == 0:
                adj0[i] |= (1 << j)
                adj0[j] |= (1 << i)
            else:
                adj1[i] |= (1 << j)
                adj1[j] |= (1 << i)
            colors[idx] = new_color

        def idx_to_pair(idx, n):
            # Reverse of get_idx
            # idx = i * (2n - i - 3) // 2 + (j - i - 1)
            # Let k = 2n - 3
            # i * (k - i) // 2 <= idx
            # Approx i ~ sqrt(2*idx)
            # We can just iterate i
            # But for speed, we can precompute or use binary search.
            # Given n is small, linear scan is okay?
            # Actually, we can derive i.
            # Let's just use a precomputed list for speed in the inner loop?
            # No, let's just calculate.
            # For n=30, max idx ~ 435.
            # i is small.
            for i in range(n):
                base = i * (2 * n - i - 3) // 2
                if base <= idx < base + (n - i - 1):
                    j = i + 1 + (idx - base)
                    return i, j
            return 0, 1 # Should not happen

        start = time.time()
        # Check if already valid
        if find_k222(adj0) is None and find_k222(adj1) is None:
            return colors

        while time.time() - start < time_limit:
            k222_0 = find_k222(adj0)
            if k222_0:
                edge_idx = random.choice(k222_0)
                update_adj(edge_idx, 1 - colors[edge_idx])
                continue
            
            k222_1 = find_k222(adj1)
            if k222_1:
                edge_idx = random.choice(k222_1)
                update_adj(edge_idx, 1 - colors[edge_idx])
                continue
            
            # No K222 found
            return colors
        
        return None

    # Try n=29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17
    for n in [29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17]:
        if time.time() > deadline:
            break
        
        best_colors = None
        
        # Try Paley if applicable
        if n % 4 == 1:
            colors = paley(n)
            if colors:
                if is_valid(n, colors):
                    return {'n': n, 'colors': colors}
                # If not valid, try to fix it
                remaining_time = deadline - time.time() - 5
                if remaining_time > 10:
                    fixed = local_search(n, colors, remaining_time)
                    if fixed and is_valid(n, fixed):
                        return {'n': n, 'colors': fixed}
        
        # Try Random initialization
        remaining_time = deadline - time.time() - 5
        if remaining_time > 10:
            colors = [random.randint(0, 1) for _ in range(n * (n - 1) // 2)]
            fixed = local_search(n, colors, remaining_time)
            if fixed and is_valid(n, fixed):
                return {'n': n, 'colors': fixed}
        
        # If we found a valid coloring in local search but didn't return? 
        # The logic above returns immediately.
        # If local search failed, we just try the next n.

    # Fallback to seed
    return {'n': 17, 'colors': paley(17)}
