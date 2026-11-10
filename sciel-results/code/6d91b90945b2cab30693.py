import time
import random
import numpy as np

def construct():
    """
    Constructs a C6-free graph on 40 vertices maximizing edges.
    Uses greedy insertion with kicks and optimized C6 detection.
    """
    N = 40
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    def can_add(adj_bits, u, v):
        """Check if adding edge (u,v) creates a C6.
        A C6 forms if there's a path u-a-b-c-d-v of length 5 (4 intermediate nodes).
        """
        # Find all nodes at distance 4 from u (avoiding v), check if any adjacent to v
        # Use BFS layer by layer with bitmask tracking
        
        # Layer 0: {u}
        # Layer 1: neighbors of u (excluding v)
        # Layer 2: neighbors of layer 1 (excluding u, v, and layer 0/1 nodes for path simplicity)
        # Actually, we need simple paths of length 4 from u, then check adjacency to v.
        # For efficiency, use the fact that we only need to detect existence.
        
        # BFS from u, tracking visited set per path is expensive.
        # Alternative: enumerate all 4-walks from u avoiding v, check if endpoint adjacent to v.
        # But walks may revisit nodes. We need simple paths.
        
        # Efficient approach: for each pair (a, d) where a is neighbor of u, d is neighbor of v,
        # check if there's a path a-b-c-d of length 3 (2 intermediate nodes b, c)
        # with all distinct and not u, v.
        
        # Even better: precompute for the current graph.
        # But since graph changes, do it on the fly.
        
        # Fast check: enumerate neighbors of u (call them A) and neighbors of v (call them D)
        # For each a in A, d in D, check if there exist b, c such that:
        #   a-b, b-c, c-d are edges, and {a,b,c,d} disjoint from {u,v} and all distinct.
        
        neighbors_u = adj_bits[u] & ~(1 << v)
        neighbors_v = adj_bits[v] & ~(1 << u)
        
        if neighbors_u == 0 or neighbors_v == 0:
            return True  # No path of length 5 possible
        
        # For each a in neighbors_u, for each d in neighbors_v,
        # check if there's a path of length 3 from a to d avoiding u, v, a, d.
        # Path: a - b - c - d, where b, c not in {u, v, a, d} and b != c.
        
        # To speed up: for each a, compute set of nodes reachable from a in 2 steps (avoiding u, v, a)
        # Then check if any such node is adjacent to d.
        
        # Precompute 2-step reachability from each node? Too expensive per call.
        
        # Direct approach: iterate over possible b, c
        # Number of neighbors is at most ~10-15 for dense graphs, so this is manageable.
        
        for a_bit in range(N):
            if not (neighbors_u & (1 << a_bit)):
                continue
            a = a_bit
            # Find common neighbors of a and d for various d
            # For each d in neighbors_v, check if there exists b, c with a-b-c-d
            # Equivalently: for each d, check if (neighbors[a] & neighbors[d]) has a node c
            # such that c is adjacent to some b which is adjacent to a... 
            # Actually: a-b-c-d means b is neighbor of a, c is neighbor of b and d.
            # So for fixed a, d: exists b in N(a), c in N(b) ∩ N(d), with b,c distinct, 
            # b not in {u,v,a,d}, c not in {u,v,a,d,b}.
            
            # Simpler: for fixed a, compute all nodes at distance 2 from a (via one intermediate)
            # avoiding u, v, a. Then for each d in N(v), check if d is adjacent to any node
            # at distance 2 from a, where the intermediate node is valid.
            
            # Let's just do a bounded search:
            # For each b in N(a) \ {u, v, a, d for any d in N(v)}... this is getting complex.
            
            # Direct triple loop with early exit:
            # For each d in N(v):
            #   For each b in N(a) \ {u,v,a,d}:
            #     For each c in N(b) \ {u,v,a,b,d}:
            #       If c is in N(d): return False
            # This is O(deg(u) * deg(v) * deg^2) which could be slow.
            
            # Better: precompute for each a, the set of nodes reachable in exactly 2 steps
            # (with one intermediate), avoiding u and v.
            
            two_step_from_a = 0
            na = adj_bits[a] & ~(1 << u) & ~(1 << v) & ~(1 << a)
            while na:
                b_bit = (na & -na).bit_length() - 1
                na &= na - 1
                b = b_bit
                # Nodes reachable from b in one step, avoiding u, v, a
                nb = adj_bits[b] & ~(1 << u) & ~(1 << v) & ~(1 << a) & ~(1 << b)
                two_step_from_a |= nb
            
            # Now check if any node in two_step_from_a is adjacent to some d in N(v)
            # and the path is valid (b and c distinct, etc.)
            # Actually, two_step_from_a includes all c such that there exists b with a-b-c.
            # We need c adjacent to d, with b != c (guaranteed since c is at distance 2 from a via b,
            # and we excluded b from nb... wait, we excluded b from nb, so c != b).
            # Also need c != a (excluded), c != u, c != v (excluded).
            # And b != d? b is intermediate, d is endpoint. b could equal d? No, d is in N(v) and 
            # b is in N(a), they could be the same node. If b == d, then path is a-d-c-d which is 
            # not a simple path. So we need b != d.
            
            # Let me redo: for the path a-b-c-d to be a valid part of a C6 (u-a-b-c-d-v),
            # we need all of u,a,b,c,d,v distinct. So b != d is required.
            
            # My two_step_from_a computation doesn't track which b led to which c.
            # Let me do it differently:
            
            # For each d in N(v):
            #   For each b in N(a) \ {u,v,a,d}:
            #     c_set = N(b) \ {u,v,a,b,d}
            #     if c_set & N(d) != 0: return False
            
            # This is O(deg(a) * deg(b)) per d, and we do this for each d.
            # Total: O(deg(v) * deg(a) * deg(b)) per a. With deg ~ 10, that's ~1000 ops per a,
            # and deg(u) ~ 10 a's, so ~10000 ops per can_add call. Acceptable.
            
            nv = neighbors_v
            while nv:
                d_bit = (nv & -nv).bit_length() - 1
                nv &= nv - 1
                d = d_bit
                if d == a:
                    continue
                # Find b in N(a) \ {u,v,a,d}
                ba = adj_bits[a] & ~(1 << u) & ~(1 << v) & ~(1 << a) & ~(1 << d)
                while ba:
                    b_bit = (ba & -ba).bit_length() - 1
                    ba &= ba - 1
                    b = b_bit
                    # c in N(b) \ {u,v,a,b,d}
                    cb = adj_bits[b] & ~(1 << u) & ~(1 << v) & ~(1 << a) & ~(1 << b) & ~(1 << d)
                    if cb & adj_bits[d]:
                        return False
        
        return True
    
    # Initialize with a denser starting structure
    # Start with 8 K5s (80 edges) and try to add inter-clique edges
    adj_bits = [0] * N
    
    for k in range(8):
        base = 5 * k
        for a in range(5):
            for b in range(a + 1, 5):
                u, v = base + a, base + b
                adj_bits[u] |= (1 << v)
                adj_bits[v] |= (1 << u)
    
    def edge_count(bits):
        return sum(bin(b).count('1') for b in bits) // 2
    
    current_edges = edge_count(adj_bits)
    best_edges = current_edges
    best_adj_bits = adj_bits[:]
    
    # Build missing edges list
    all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    missing_edges = [p for p in all_pairs if not (adj_bits[p[0]] & (1 << p[1]))]
    
    # Sort missing edges by a heuristic: prefer edges between nodes in different cliques
    # and with lower degree (more likely to be addable)
    def edge_priority(u, v):
        # Prefer inter-clique edges, and edges involving lower-degree nodes
        clique_u = u // 5
        clique_v = v // 5
        inter_clique = 0 if clique_u != clique_v else 1
        deg_u = bin(adj_bits[u]).count('1')
        deg_v = bin(adj_bits[v]).count('1')
        return (inter_clique, deg_u + deg_v)
    
    while time.time() < deadline:
        # Sort missing edges by priority
        missing_edges.sort(key=lambda e: edge_priority(e[0], e[1]))
        
        added_in_pass = 0
        new_missing = []
        
        for u, v in missing_edges:
            if time.time() > deadline:
                break
                
            if can_add(adj_bits, u, v):
                adj_bits[u] |= (1 << v)
                adj_bits[v] |= (1 << u)
                added_in_pass += 1
                current_edges += 1
            else:
                new_missing.append((u, v))
                
        if current_edges > best_edges:
            best_edges = current_edges
            best_adj_bits = adj_bits[:]
        
        missing_edges = new_missing
        
        if added_in_pass == 0:
            # Stuck. Perform a kick.
            if current_edges > 10:
                # Remove edges that are most constraining
                # Heuristic: remove edges whose removal would allow many new edges
                # Approximate: remove random edges, or edges in high-degree regions
                kick_size = random.randint(3, 8)
                
                existing_edges = []
                for u in range(N):
                    for v in range(u + 1, N):
                        if adj_bits[u] & (1 << v):
                            existing_edges.append((u, v))
                
                if not existing_edges:
                    break
                
                # Prefer removing edges between high-degree nodes
                existing_edges.sort(key=lambda e: -(bin(adj_bits[e[0]]).count('1') + bin(adj_bits[e[1]]).count('1')))
                
                # Take a mix: some from top, some random
                for i in range(min(kick_size, len(existing_edges))):
                    if i < kick_size // 2 and i < len(existing_edges):
                        u, v = existing_edges[i]
                    else:
                        idx = random.randint(0, len(existing_edges) - 1)
                        u, v = existing_edges[idx]
                    # Avoid duplicates
                    if (u, v) in missing_edges:
                        continue
                    adj_bits[u] &= ~(1 << v)
                    adj_bits[v] &= ~(1 << u)
                    current_edges -= 1
                    missing_edges.append((u, v))
                    existing_edges.remove((u, v))
                    if not existing_edges:
                        break
    
    # Final polish: try one more pass
    if time.time() < deadline - 5:
        missing_edges.sort(key=lambda e: edge_priority(e[0], e[1]))
        for u, v in missing_edges:
            if time.time() > deadline - 2:
                break
            if can_add(adj_bits, u, v):
                adj_bits[u] |= (1 << v)
                adj_bits[v] |= (1 << u)
                current_edges += 1
        
        if current_edges > best_edges:
            best_edges = current_edges
            best_adj_bits = adj_bits[:]
    
    # Return best graph
    final_edges = []
    for u in range(N):
        for v in range(u + 1, N):
            if best_adj_bits[u] & (1 << v):
                final_edges.append([u, v])
    return final_edges

if __name__ == "__main__":
    e = construct()
    print(f"Edges: {len(e)}")
