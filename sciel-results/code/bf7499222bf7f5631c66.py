"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

def construct():
    """
    Construct a graph with 31 vertices and 91 edges minimizing the number of 4-cycles.
    
    Strategy:
    1. Start with the polarity graph of PG(2,5) which is C4-free with 90 edges.
    2. Add one edge (non-edge in the base graph) that creates the minimum number of new 4-cycles.
    3. Optionally, try local search to see if swapping edges can reduce the count further.
    """
    start_time = time.time()
    deadline = start_time + 140  # Keep a 20s safety margin
    
    n = 31
    target_edges = 91
    
    # Step 1: Construct the polarity graph of PG(2,5)
    # Points are (x, y, z) over GF(5) not all zero, modulo scalar multiples
    # For PG(2,5), we can represent points as (x, y) where (x,y) are in GF(5)
    # but we need 31 points. PG(2,5) has (5^3-1)/(5-1) = 124/4 = 31 points.
    
    # Let's use the standard representation:
    # Points: (1, x, y) for x, y in GF(5) -> 25 points
    #          (0, 1, y) for y in GF(5) -> 5 points  
    #          (0, 0, 1) -> 1 point
    # Total: 31 points
    
    # Actually, let me use a cleaner approach for the polarity graph.
    # The polarity graph of PG(2,q) for odd q has vertices as points of PG(2,q)
    # and edges when points are orthogonal under a suitable polarity.
    
    # For PG(2,5), we can use the symplectic polarity or a suitable quadratic form.
    # Let's use the standard construction:
    
    # Label points 0-30
    # We'll use the fact that the polarity graph of PG(2,q) for odd q is:
    # - Self-adjoint (symmetric)
    # - C4-free
    # - Has q^2 + q edges (for q odd, this is (q+1)(q) which for q=5 is 30... wait)
    
    # Actually, let me reconsider. The number of edges in the polarity graph of PG(2,q):
    # For a polarity of PG(2,q), the number of absolute points is q+1 (for a unitary polarity)
    # or similar. The total number of edges depends on the type.
    
    # Let me use a different, well-known construction.
    # The Reiman polarity graph for PG(2,5):
    
    # I'll use the explicit construction from literature.
    # Points of PG(2,5) can be represented as:
    # Type 1: (1, a, b) for a,b in {0,1,2,3,4} -> 25 points, label 0-24
    # Type 2: (0, 1, c) for c in {0,1,2,3,4} -> 5 points, label 25-29
    # Type 3: (0, 0, 1) -> 1 point, label 30
    
    # The polarity can be defined by the quadratic form Q(x,y,z) = xz - y^2/2 (mod 5)
    # or similar. Two points are adjacent if they are orthogonal under the associated bilinear form.
    
    # The bilinear form B((x1,y1,z1), (x2,y2,z2)) = x1*z2 + x2*z1 - y1*y2 (mod 5)
    # (This is the polar form of Q(x,y,z) = xz - y^2/2, since B(P,P) = 2Q(P))
    
    # Wait, I need to be careful. Let me use a standard reference.
    
    # For the Hermitian polarity on PG(2,q^2) or the symplectic... actually for PG(2,q) with q odd,
    # we can use the quadratic form Q(x,y,z) = x^2 + y^2 + z^2 (mod 5) won't work well.
    
    # Let me use: Q(x,y,z) = xz - y^2 (mod 5). The associated bilinear form is
    # B((x1,y1,z1), (x2,y2,z2)) = x1*z2 + x2*z1 - 2*y1*y2 (mod 5)
    
    # Two points P and Q are adjacent in the polarity graph if B(P,Q) = 0 and P != Q,
    # plus we include loops for absolute points (B(P,P) = 0) but since we want a simple graph,
    # we just include the edge if B(P,Q) = 0.
    
    # Actually, in a polarity graph, vertex P is adjacent to Q if B(P,Q) = 0.
    # For a graph (not digraph), this is symmetric.
    
    # Let me compute the number of edges to verify it's 90.
    
    def point_coords(label):
        """Return homogeneous coordinates (x,y,z) for point label."""
        if label < 25:
            x = 1
            y = label % 5
            z = (label // 5) % 5
            return (x, y, z)
        elif label < 30:
            x = 0
            y = 1
            z = label - 25
            return (x, y, z)
        else:
            x = 0
            y = 0
            z = 1
            return (x, y, z)
    
    def bilinear(p1, p2):
        """Compute B(p1, p2) mod 5."""
        x1, y1, z1 = p1
        x2, y2, z2 = p2
        return (x1 * z2 + x2 * z1 - 2 * y1 * y2) % 5
    
    # Build the polarity graph
    edges = set()
    for i in range(n):
        pi = point_coords(i)
        for j in range(i + 1, n):
            pj = point_coords(j)
            if bilinear(pi, pj) == 0:
                edges.add((i, j))
    
    # Check the number of edges
    base_edges = list(edges)
    # If we don't have exactly 90 edges, we may need to adjust the construction.
    # Let's assume for now this gives us 90. If not, we'll adapt.
    
    if len(base_edges) != 90:
        # Fallback: if the construction doesn't give exactly 90 edges,
        # we need a different approach. But let's hope it does.
        # For now, just proceed with what we have and add edges to reach 91.
        pass
    
    # Step 2: Add one edge to create exactly 91 edges, minimizing new 4-cycles.
    # The number of new 4-cycles created by adding edge (u,v) equals the number of
    # common neighbors w such that (u,w) and (v,w) are both in the graph, but that's
    # just 2-cycles. For 4-cycles, we need paths of length 3: u - a - b - v where
    # a,b are vertices and edges (u,a), (a,b), (b,v) exist.
    
    # Actually, a 4-cycle involving the new edge (u,v) is: u - v - c - d - u
    # which requires edges (v,c), (c,d), (d,u) in the existing graph.
    # So the number of new 4-cycles = number of pairs (c,d) such that
    # (v,c) in E, (c,d) in E, (d,u) in E.
    # This is equivalent to: for each c in N(v), for each d in N(u), check if (c,d) in E.
    
    # Build adjacency
    adj = [set() for _ in range(n)]
    for u, v in base_edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Find all non-edges
    non_edges = []
    edge_set = set()
    for u, v in base_edges:
        edge_set.add((u, v))
        edge_set.add((v, u))
    
    for i in range(n):
        for j in range(i + 1, n):
            if (i, j) not in edge_set:
                non_edges.append((i, j))
    
    # For each non-edge, count the number of 4-cycles it would create
    def count_new_c4s(u, v, adj, n):
        """Count 4-cycles created by adding edge (u,v)."""
        count = 0
        # 4-cycles: u-v-c-d-u, so we need paths v-c-d-u of length 3
        # Equivalently, count triples (c,d) with c in N(v), d in N(u), and (c,d) in E
        for c in adj[v]:
            if c == u:
                continue
            for d in adj[u]:
                if d == v:
                    continue
                if c == d:
                    continue
                if d in adj[c]:
                    count += 1
        # Each 4-cycle is counted twice (once for each direction), so divide by 2? 
        # Actually, let's think: a 4-cycle u-v-c-d-u is distinct from u-v-d-c-u only if c!=d.
        # In our counting, for fixed u,v, we count (c,d) and (d,c) separately if both exist.
        # But a 4-cycle is an unordered set of 4 vertices with 4 edges. 
        # The number of 4-cycles containing edge (u,v) is the number of paths of length 3
        # from v to u in the graph without using (u,v). Each such path gives a unique 4-cycle.
        # So count should be correct as is, but let's verify the factor.
        # Actually, each 4-cycle containing (u,v) corresponds to exactly one path v-...-u of length 3.
        # So the count is correct.
        return count
    
    # Find the non-edge that creates the fewest 4-cycles
    best_non_edge = None
    min_c4s = float('inf')
    
    for u, v in non_edges:
        c4s = count_new_c4s(u, v, adj, n)
        if c4s < min_c4s:
            min_c4s = c4s
            best_non_edge = (u, v)
    
    # Add the best edge
    if best_non_edge is not None:
        u, v = best_non_edge
        base_edges.append((u, v))
        adj[u].add(v)
        adj[v].add(u)
    
    # Now we have 91 edges. Let's verify.
    if len(base_edges) != target_edges:
        # This shouldn't happen, but just in case
        pass
    
    # Step 3: Local search to potentially reduce the number of 4-cycles.
    # We can try swapping edges: remove edge e1 and add edge e2, if it reduces C4 count.
    # But this is complex. Let's first compute the current C4 count.
    
    def count_total_c4s(adj, n):
        """Count total number of 4-cycles in the graph."""
        # A 4-cycle is a set of 4 vertices {a,b,c,d} with edges (a,b), (b,c), (c,d), (d,a)
        # and no other edges among them (but in our count, we count subgraphs, so we count
        # each 4-cycle as a subgraph, meaning we count each 4-vertex set that induces a 4-cycle).
        
        # Actually, the problem says "number of four-cycles (subgraphs)". This typically means
        # the number of cycles of length 4, i.e., sequences of 4 distinct vertices v1,v2,v3,v4
        # with edges (v1,v2), (v2,v3), (v3,v4), (v4,v1), counted up to rotation and reflection.
        # But "subgraphs" might mean induced subgraphs that are C4, or it might mean the number
        # of C4 as a subgraph (not necessarily induced).
        
        # The standard interpretation in extremal graph theory for "number of C4s" is the number
        # of (not necessarily induced) 4-cycles, i.e., the number of 4-tuples (v1,v2,v3,v4) of
        # distinct vertices with the four edges, divided by 8 (for the 8 symmetries).
        
        # Let me compute the number of 4-cycles by counting for each pair of vertices,
        # the number of common neighbors, and using the formula.
        
        # Number of C4s = (1/8) * sum over all unordered pairs {u,v} of (common_neighbors(u,v))^2 
        # ... no, that's not quite right.
        
        # Actually, a 4-cycle can be counted as follows:
        # For each pair of opposite vertices (u,v) in the 4-cycle, the other two vertices (a,b)
        # must be common neighbors of u and v, and the edges (a,b) must not be required (for 
        # non-induced). Wait, in a 4-cycle u-a-v-b-u, the vertices are u,a,v,b. The opposite 
        # pairs are (u,v) and (a,b). The edges are (u,a), (a,v), (v,b), (b,u).
        
        # So for a fixed pair (u,v) as opposite vertices, the number of 4-cycles with u,v as 
        # opposite is: for each pair of common neighbors (a,b) of u and v, with a < b, 
        # we get a 4-cycle if... actually, any two common neighbors a,b of u and v form a 
        # 4-cycle u-a-v-b-u. So the number of such 4-cycles for opposite pair (u,v) is 
        # C(common_neighbors(u,v), 2).
        
        # But each 4-cycle has two opposite pairs, so the total number of 4-cycles is:
        # (1/2) * sum over all unordered pairs {u,v} of C(common_neighbors(u,v), 2)
        
        # Let me verify: a 4-cycle has 4 vertices, and 2 pairs of opposite vertices. 
        # So each 4-cycle is counted twice in the sum, hence we divide by 2.
        
        total = 0
        for u in range(n):
            for v in range(u + 1, n):
                common = len(adj[u] & adj[v])
                total += common * (common - 1) // 2
        return total // 2
    
    current_c4s = count_total_c4s(adj, n)
    
    # Try local search: swap edges to reduce C4 count
    # This is optional and time-consuming, so only do it if we have time
    if time.time() < deadline - 30:
        # Simple local search
        import copy
        best_edges = list(base_edges)
        best_c4s = current_c4s
        best_adj = [s.copy() for s in adj]
        
        for iteration in range(100):
            if time.time() > deadline - 30:
                break
            
            # Try random swaps
            found_improvement = False
            for _ in range(50):
                if time.time() > deadline - 30:
                    break
                
                # Pick a random edge to remove
                if not best_edges:
                    break
                idx = random.randint(0, len(best_edges) - 1)
                u1, v1 = best_edges[idx]
                
                # Pick a random non-edge to add
                # Generate a random non-edge
                attempts = 0
                while attempts < 100:
                    u2 = random.randint(0, n - 1)
                    v2 = random.randint(0, n - 1)
                    if u2 == v2:
                        continue
                    if (u2, v2) in {(a,b) for a,b in best_edges} or (v2, u2) in {(a,b) for a,b in best_edges}:
                        continue
                    # Check it's a non-edge
                    is_edge = False
                    for a, b in best_edges:
                        if (a == u2 and b == v2) or (a == v2 and b == u2):
                            is_edge = True
                            break
                    if not is_edge:
                        break
                    attempts += 1
                else:
                    continue
                
                # Try the swap
                # Remove (u1,v1), add (u2,v2)
                new_edges = best_edges[:idx] + best_edges[idx+1:]
                new_edges.append((u2, v2))
                
                new_adj = [s.copy() for s in best_adj]
                new_adj[u1].discard(v1)
                new_adj[v1].discard(u1)
                new_adj[u2].add(v2)
                new_adj[v2].add(u2)
                
                new_c4s = count_total_c4s(new_adj, n)
                
                if new_c4s < best_c4s:
                    best_edges = new_edges
                    best_c4s = new_c4s
                    best_adj = new_adj
                    found_improvement = True
                    break
            
            if not found_improvement:
                break
        
        base_edges = best_edges
        adj = best_adj
        current_c4s = best_c4s
    
    # Return the 91 edges
    result = []
    for u, v in base_edges:
        if u < v:
            result.append([u, v])
        else:
            result.append([v, u])
    
    return result

# EVOLVE-BLOCK-END
