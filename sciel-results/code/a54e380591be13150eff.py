import random
import numpy as np
import time

def construct():
    start_time = time.time()
    time_limit = 115  # Leave buffer for overhead
    
    # Try to find a graph for n=35, then 34, etc.
    for n in range(35, 16, -1):
        # Multiple restarts for each n
        for restart in range(200):
            if time.time() - start_time > time_limit:
                break
                
            # Spectral Initialization
            # Generate random symmetric matrix
            M = np.random.rand(n, n)
            M = (M + M.T) / 2
            # Get top eigenvector
            try:
                w, v = np.linalg.eigh(M)
                top_v = v[:, -1]
            except:
                top_v = np.ones(n)
            
            # Binarize with bias
            # If top_v[i] * top_v[j] > 0, higher prob of edge?
            # This creates a "clique-like" structure which is bad for K4.
            # If top_v[i] * top_v[j] < 0, lower prob?
            # Let's just use random but with a slight bias based on the sign product.
            adj_bits = [0] * n
            for i in range(n):
                for j in range(i+1, n):
                    bias = 0.5
                    if top_v[i] * top_v[j] > 0:
                        bias = 0.6
                    else:
                        bias = 0.4
                    if random.random() < bias:
                        adj_bits[i] |= (1 << j)
                        adj_bits[j] |= (1 << i)
            
            # Local Search
            success = local_search(adj_bits, n, start_time, time_limit)
            if success:
                # Convert to edge list
                edges = []
                for i in range(n):
                    for j in range(i+1, n):
                        if adj_bits[i] & (1 << j):
                            edges.append([i, j])
                return {'n': n, 'edges': edges}
            
    # Fallback: Return the best known small graph (Paley 17) if all else fails
    squares = {x * x % 17 for x in range(1, 17)}
    return {'n': 17, 'edges': [[i, j] for i in range(17) for j in range(i + 1, 17) if (j - i) % 17 in squares]}

def local_search(adj_bits, n, start_time, time_limit):
    for step in range(5000):
        if time.time() - start_time > time_limit:
            return False
        
        # Check for K4
        k4 = find_k4_bitset(adj_bits, n)
        if k4:
            # Pick random edge in K4 to remove
            edges_in_k4 = [(k4[0], k4[1]), (k4[0], k4[2]), (k4[0], k4[3]),
                           (k4[1], k4[2]), (k4[1], k4[3]), (k4[2], k4[3])]
            u, v = random.choice(edges_in_k4)
            adj_bits[u] &= ~(1 << v)
            adj_bits[v] &= ~(1 << u)
            continue
        
        # Check for Ind6
        ind6 = find_ind6_bitset(adj_bits, n)
        if ind6:
            # Pick random pair in Ind6 to add edge
            pairs = [(ind6[i], ind6[j]) for i in range(6) for j in range(i+1, 6)]
            u, v = random.choice(pairs)
            adj_bits[u] |= (1 << v)
            adj_bits[v] |= (1 << u)
            continue
        
        # No violations found
        # Verify exactly
        if not find_k4_bitset(adj_bits, n) and not find_ind6_bitset(adj_bits, n):
            return True
            
    return False

def find_k4_bitset(adj_bits, n):
    for i in range(n):
        for j in range(i+1, n):
            if adj_bits[i] & (1 << j):
                common = adj_bits[i] & adj_bits[j]
                # Check if common has an edge
                # Iterate through set bits in common
                c = common
                while c:
                    # Get lowest set bit
                    lsb = c & -c
                    k = lsb.bit_length() - 1
                    c ^= lsb
                    # Check if k is connected to any other in common
                    if adj_bits[k] & common:
                        # Find the other vertex
                        neighbor = (adj_bits[k] & common) & ~lsb
                        if neighbor:
                            lsb2 = neighbor & -neighbor
                            l = lsb2.bit_length() - 1
                            return [i, j, k, l]
    return None

def find_ind6_bitset(adj_bits, n):
    # Backtracking with bitsets
    # candidates: bitmask of vertices that can be added
    # current: bitmask of vertices already in the independent set
    def dfs(candidates, current, depth):
        if depth == 6:
            # Extract vertices
            verts = []
            c = current
            while c:
                lsb = c & -c
                v = lsb.bit_length() - 1
                verts.append(v)
                c ^= lsb
            return verts
        
        if candidates.bit_count() + depth < 6:
            return None
            
        # Heuristic: pick a vertex from candidates
        # To speed up, we can try to pick a vertex with low degree in the induced subgraph?
        # For simplicity, just pick the lowest set bit, or randomize?
        # Randomization helps escape local minima in the search tree.
        # But deterministic is faster. Let's try lowest bit first, then next.
        # Actually, for independent set, we want to branch on vertices that "remove" the most candidates?
        # No, we want to find ANY set.
        
        c = candidates
        # Try all vertices in candidates? That's expensive.
        # We can just iterate through them.
        while c:
            lsb = c & -c
            v = lsb.bit_length() - 1
            c ^= lsb
            
            # New candidates: those in candidates that are NOT neighbors of v
            # And we remove v itself
            new_candidates = candidates & ~adj_bits[v] & ~lsb
            
            res = dfs(new_candidates, current | lsb, depth + 1)
            if res:
                return res
        return None

    return dfs((1 << n) - 1, 0, 0)
