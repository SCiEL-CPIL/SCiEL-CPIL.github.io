"""Triangle-free graph without independent 10-set: maximize n."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 118  # safety margin
    
    # The Andrasfai graph An(9) on 27 vertices (3-circulant C(27, {1,2,5}))
    # is triangle-free and has independence number 9.
    # We try to extend it by adding vertices one at a time.
    
    # Build Andrasfai graph on 27 vertices
    n_base = 27
    adj = [0] * n_base
    for i in range(n_base):
        for j in [1, 2, 5]:
            v = (i + j) % n_base
            if v != i:
                adj[i] |= (1 << v)
            v2 = (i - j) % n_base
            if v2 != i:
                adj[i] |= (1 << v2)
    
    # Verify base graph is triangle-free and alpha < 10
    # (We trust the construction, but verify alpha)
    
    def max_independent_set_size(adj, n):
        """Compute alpha using branch and bound with bitsets."""
        # Use a greedy coloring upper bound approach
        # For small n, we can use exact search
        
        # First try greedy to get a lower bound
        remaining = (1 << n) - 1
        indep = 0
        while remaining:
            # Pick vertex with minimum degree in remaining
            best_v = -1
            best_deg = float('inf')
            r = remaining
            while r:
                lsb = r & -r
                v = lsb.bit_length() - 1
                deg = (adj[v] & remaining).bit_count()
                if deg < best_deg:
                    best_deg = deg
                    best_v = v
                r ^= lsb
            if best_v < 0:
                break
            indep += 1
            remaining &= ~(1 << best_v) & ~adj[best_v]
        
        # Now try to find if there's an independent set of size indep+1
        # Use branch and bound
        best = indep
        
        def search(remaining, size):
            nonlocal best
            if size > best:
                best = size
            if remaining == 0:
                return
            # Upper bound: size + popcount(remaining)
            if size + remaining.bit_count() <= best:
                return
            
            # Pick a vertex to branch on
            # Use the vertex with highest degree in remaining for pruning
            r = remaining
            pivot = -1
            max_deg = -1
            while r:
                lsb = r & -r
                v = lsb.bit_length() - 1
                deg = (adj[v] & remaining).bit_count()
                if deg > max_deg:
                    max_deg = deg
                    pivot = v
                r ^= lsb
            
            if pivot < 0:
                return
            
            # Branch: include pivot or exclude pivot
            # Include pivot
            new_remaining = remaining & ~(1 << pivot) & ~adj[pivot]
            search(new_remaining, size + 1)
            
            # Exclude pivot
            search(remaining & ~(1 << pivot), size)
        
        search((1 << n) - 1, 0)
        return best
    
    # Check base
    alpha_base = max_independent_set_size(adj, n_base)
    current_n = n_base
    current_adj = adj[:]
    best_n = n_base
    best_adj = adj[:]
    
    # Try to add vertices one by one
    for target_n in range(n_base + 1, 41):
        if time.time() > deadline - 15:
            break
        
        added = False
        # Try multiple attempts to add a vertex
        for attempt in range(100):
            if time.time() > deadline - 15:
                break
            
            new_n = current_n + 1
            new_adj = current_adj[:] + [0]
            new_vertex = current_n
            
            # The new vertex should connect to an independent set S in the current graph
            # such that S is an independent set and the new vertex doesn't form triangles
            # For triangle-free: for any edge (u,v) in current graph, new vertex can't connect to both
            # So S (the neighborhood of new vertex) must be an independent set
            
            # Find a large independent set in current graph
            # Use greedy
            remaining = (1 << current_n) - 1
            S = 0
            while remaining:
                best_v = -1
                best_deg = float('inf')
                r = remaining
                while r:
                    lsb = r & -r
                    v = lsb.bit_length() - 1
                    deg = (current_adj[v] & remaining).bit_count()
                    if deg < best_deg:
                        best_deg = deg
                        best_v = v
                    r ^= lsb
                if best_v < 0:
                    break
                S |= (1 << best_v)
                remaining &= ~(1 << best_v) & ~current_adj[best_v]
            
            # Randomly modify S: remove some vertices, add others
            if random.random() < 0.5 and S:
                # Remove a random vertex from S
                r = S
                while r:
                    lsb = r & -r
                    if random.random() < 0.3:
                        S &= ~lsb
                    r ^= lsb
            
            # Add new vertex connections
            for i in range(current_n):
                if S & (1 << i):
                    new_adj[i] |= (1 << new_vertex)
                    new_adj[new_vertex] |= (1 << i)
            
            # Check triangle-free
            # For each edge (i, new_vertex), i is in S
            # Triangle (i, new_vertex, j) exists if i-j is an edge
            # So for all i in S, i must not be adjacent to any other vertex in S
            # This is guaranteed since S is an independent set
            # But also need to check: for any i in S and any j, if i-j is edge, then j not in S
            # Already satisfied.
            
            # Check alpha of new graph
            alpha_new = max_independent_set_size(new_adj, new_n)
            if alpha_new < 10:
                current_n = new_n
                current_adj = new_adj
                best_n = current_n
                best_adj = current_adj
                added = True
                break
        
        if not added:
            # Try a few more random strategies before giving up
            for attempt in range(50):
                if time.time() > deadline - 15:
                    break
                
                new_n = current_n + 1
                new_adj = current_adj[:] + [0]
                new_vertex = current_n
                
                # Random independent set
                remaining = (1 << current_n) - 1
                S = 0
                while remaining:
                    # Random pick
                    r = remaining
                    count = r.bit_count()
                    target = random.randrange(count)
                    idx = 0
                    while r:
                        lsb = r & -r
                        if idx == target:
                            v = lsb.bit_length() - 1
                            break
                        idx += 1
                        r ^= lsb
                    else:
                        v = -1
                    if v < 0:
                        break
                    S |= (1 << v)
                    remaining &= ~(1 << v) & ~current_adj[v]
                
                for i in range(current_n):
                    if S & (1 << i):
                        new_adj[i] |= (1 << new_vertex)
                        new_adj[new_vertex] |= (1 << i)
                
                alpha_new = max_independent_set_size(new_adj, new_n)
                if alpha_new < 10:
                    current_n = new_n
                    current_adj = new_adj
                    best_n = current_n
                    best_adj = current_adj
                    break
        
        if time.time() > deadline - 15:
            break
    
    # Convert to edge list
    edges = []
    for u in range(best_n):
        for v in range(u + 1, best_n):
            if best_adj[u] & (1 << v):
                edges.append([u, v])
    
    return {'n': best_n, 'edges': edges}

# EVOLVE-BLOCK-END
