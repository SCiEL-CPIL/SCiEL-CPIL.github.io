import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    def has_independent_set_k(n, adj_bits, k):
        """
        Fast exact check: does there exist an independent set of size k?
        adj_bits[v] is a bitmask of neighbors of v.
        Uses branch and bound with good vertex ordering.
        """
        if k == 0:
            return True
        
        # Order vertices by degree (descending) for better pruning
        degrees = [bin(adj_bits[v]).count('1') for v in range(n)]
        order = sorted(range(n), key=lambda v: -degrees[v])
        
        # Precompute neighbor masks in the ordered sense
        # We'll work with the original vertex indices but pick vertices in 'order'
        
        def search(avail, depth):
            if depth == 0:
                return True
            if avail == 0:
                return False
            if bin(avail).count('1') < depth:
                return False
            
            # Pick vertex with highest degree in the induced subgraph
            best_v = -1
            best_deg = -1
            temp = avail
            while temp:
                lb = temp & -temp
                v = lb.bit_length() - 1
                deg_in_avail = bin(adj_bits[v] & avail).count('1')
                if deg_in_avail > best_deg:
                    best_deg = deg_in_avail
                    best_v = v
                temp ^= lb
            
            v = best_v
            v_bit = 1 << v
            
            # Branch 1: Include v
            new_avail = avail & ~(adj_bits[v] | v_bit)
            if search(new_avail, depth - 1):
                return True
            
            # Branch 2: Exclude v
            new_avail2 = avail & ~v_bit
            if search(new_avail2, depth):
                return True
            
            return False
        
        avail = (1 << n) - 1
        return search(avail, k)

    def is_c4_free(n, adj_bits):
        """Check if graph is C4-free: no two vertices share 2+ common neighbors."""
        for u in range(n):
            for v in range(u+1, n):
                common = adj_bits[u] & adj_bits[v]
                if bin(common).count('1') >= 2:
                    return False
        return True

    def build_adj_bits(n, edges):
        adj_bits = [0] * n
        for u, v in edges:
            adj_bits[u] |= (1 << v)
            adj_bits[v] |= (1 << u)
        return adj_bits

    def edges_from_adj_bits(n, adj_bits):
        edges = []
        for u in range(n):
            for v in range(u+1, n):
                if adj_bits[u] & (1 << v):
                    edges.append([u, v])
        return edges

    # Fallback: 30-vertex graph (10 disjoint triangles)
    edges_fallback = []
    for t in range(10):
        a, b, c = 3*t, 3*t+1, 3*t+2
        edges_fallback += [[a,b], [a,c], [b,c]]
    best_graph = {'n': 30, 'edges': edges_fallback}
    best_n = 30

    # Try to construct C4-free graphs with small independence number
    # Strategy: Use a greedy randomized construction with verification
    
    for n in range(43, best_n, -1):
        if time.time() > deadline - 15:
            break
        
        found = False
        for trial in range(20):
            if time.time() > deadline - 10:
                break
            
            # Random initialization with controlled density
            # C4-free graphs have O(n^{3/2}) edges, so density ~ n^{-1/2}
            density = 1.5 / (n ** 0.5)
            
            adj_bits = [0] * n
            edge_set = set()
            
            for u in range(n):
                for v in range(u+1, n):
                    if random.random() < density:
                        adj_bits[u] |= (1 << v)
                        adj_bits[v] |= (1 << u)
                        edge_set.add((u, v))
            
            # Remove C4s by removing edges
            max_iters = 500
            for _ in range(max_iters):
                if time.time() > deadline - 8:
                    break
                
                # Find a pair with >= 2 common neighbors
                c4_pair = None
                for u in range(n):
                    for v in range(u+1, n):
                        common = adj_bits[u] & adj_bits[v]
                        if bin(common).count('1') >= 2:
                            c4_pair = (u, v, common)
                            break
                    if c4_pair:
                        break
                
                if not c4_pair:
                    break
                
                u, v, common = c4_pair
                # Pick a common neighbor w and remove one edge
                temp = common
                lb = temp & -temp
                w = lb.bit_length() - 1
                
                # Remove edge (u, w)
                adj_bits[u] &= ~(1 << w)
                adj_bits[w] &= ~(1 << u)
                edge_set.discard((min(u,w), max(u,w)))
            
            # Check if C4-free
            if not is_c4_free(n, adj_bits):
                continue
            
            # Check independence number < 11
            if not has_independent_set_k(n, adj_bits, 11):
                edges = edges_from_adj_bits(n, adj_bits)
                best_graph = {'n': n, 'edges': edges}
                best_n = n
                found = True
                break
        
        if found:
            # Try to go higher
            continue
    
    return best_graph

# EVOLVE-BLOCK-END
