import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    n = 20
    
    # --- Phase 1: Deterministic Algebraic Construction ---
    # Use a Paley-type graph on p=23 vertices, then delete 3 vertices to get n=20.
    # Paley graph P(p) has vertices 0..p-1, edge (u,v) iff (v-u) is a quadratic residue mod p.
    # For p=23, the quadratic residues are:
    # 1^2=1, 2^2=4, 3^2=9, 4^2=16, 5^2=2, 6^2=13, 7^2=3, 8^2=18, 9^2=12, 10^2=8, 11^2=6
    # QR = {1, 2, 3, 4, 6, 8, 9, 12, 13, 16, 18}
    
    p = 23
    qr = set([1, 2, 3, 4, 6, 8, 9, 12, 13, 16, 18])
    
    # Create full Paley graph on 23 vertices
    adj_full = np.zeros((p, p), dtype=np.int8)
    for u in range(p):
        for v in range(u + 1, p):
            diff = (v - u) % p
            if diff in qr:
                adj_full[u, v] = 1
                adj_full[v, u] = 1
                
    # Select 20 vertices to keep. We can try a few deterministic subsets or just the first 20.
    # The first 20 vertices (0..19) form a subgraph.
    # Let's verify if this subgraph is K3,3 free. It should be, as subgraphs of K3,3-free are K3,3-free.
    # Is Paley(23) K3,3 free?
    # Paley graphs are strongly regular. Girth of Paley(23) is 4? No, Paley graphs have triangles.
    # However, the property "no 3 vertices have 3 common neighbors" is equivalent to no K3,3.
    # Let's just use the subgraph on vertices 0..19.
    
    keep_vertices = list(range(20))
    adj = np.zeros((n, n), dtype=np.int8)
    for i in range(n):
        for j in range(i + 1, n):
            if adj_full[keep_vertices[i], keep_vertices[j]]:
                adj[i, j] = 1
                adj[j, i] = 1

    # --- Phase 2: Local Search Optimization ---
    # We will use a local search to add edges and perform swaps.
    # Constraint: For any triple (a,b,c), |N(a) ∩ N(b) ∩ N(c)| <= 2.
    
    # Precompute triples for fast checking
    triples = list(combinations(range(n), 3))
    # Map vertex to list of triples containing it
    vertex_to_triples = [[] for _ in range(n)]
    for idx, t in enumerate(triples):
        for v in t:
            vertex_to_triples[v].append(idx)

    def is_k33_free(adj):
        # Check all triples
        for a, b, c in triples:
            common = adj[a] & adj[b] & adj[c]
            if np.sum(common) > 2:
                return False
        return True

    def edge_count(adj):
        return int(np.sum(adj)) // 2

    def can_add_edge(adj, u, v):
        # Check if adding edge (u,v) violates K3,3
        # Only triples containing u or v are affected.
        # Union of triples containing u and v
        affected = set(vertex_to_triples[u])
        affected.update(vertex_to_triples[v])
        
        for idx in affected:
            a, b, c = triples[idx]
            # Simulate adding edge
            # We need to check if common neighbors > 2
            # Common neighbors of a,b,c in current graph + potential new edges involving u,v
            
            # Efficient check:
            # Current common: adj[a] & adj[b] & adj[c]
            # If we add (u,v), adj[u] gets v, adj[v] gets u.
            # So for any triple, the common neighbors might increase if u or v is in the triple.
            
            ma = adj[a].copy()
            mb = adj[b].copy()
            mc = adj[c].copy()
            
            # Apply hypothetical edge
            if a == u:
                ma[v] = 1
            elif a == v:
                ma[u] = 1
            if b == u:
                mb[v] = 1
            elif b == v:
                mb[u] = 1
            if c == u:
                mc[v] = 1
            elif c == v:
                mc[u] = 1
                
            common = ma & mb & mc
            if np.sum(common) > 2:
                return False
        return True

    def try_add_edges(adj):
        # Greedily add all possible edges
        changed = True
        while changed:
            changed = False
            for u in range(n):
                for v in range(u + 1, n):
                    if not adj[u, v]:
                        if can_add_edge(adj, u, v):
                            adj[u, v] = 1
                            adj[v, u] = 1
                            changed = True
        return adj

    def local_search_swap(adj):
        # Try 1-for-1 swaps: remove edge (u,v), add edge (x,y)
        # This is expensive, so we limit it.
        # Better: Try to add an edge, if it fails, try removing one edge from a violating triple.
        
        # For now, just do greedy add. The algebraic start is strong enough that
        # simple greedy add might reach the optimum or near optimum.
        # We can add a simple hill climbing step if time permits.
        pass

    # Initial check and count
    if not is_k33_free(adj):
        # This should not happen for Paley subgraph, but safety check
        # If it happens, we need to repair.
        # Simple repair: remove edges until valid
        while not is_k33_free(adj):
            # Find a violation
            viol = None
            for a, b, c in triples:
                common = adj[a] & adj[b] & adj[c]
                if np.sum(common) > 2:
                    # Remove an edge incident to one of the common neighbors
                    # Let's remove edge (a, x) where x is a common neighbor
                    common_indices = np.where(common)[0]
                    if len(common_indices) > 0:
                        x = common_indices[0]
                        # Remove edge (a, x)
                        adj[a, x] = 0
                        adj[x, a] = 0
                        viol = True
                        break
            if not viol:
                break
            
    # Phase 2: Greedy Add
    adj = try_add_edges(adj)
    
    # Phase 3: Local Search (Time permitting)
    # We can try to improve by swapping edges.
    # But given the time budget and the strength of the algebraic start,
    # let's do a few random restarts of the greedy add from slightly perturbed starts?
    # No, keep it deterministic for now.
    
    # Let's try a simple local search: 
    # Iterate over all non-edges. If it can be added, add it.
    # This is already done in try_add_edges.
    
    # Can we do better? 
    # Maybe the subgraph 0..19 is not optimal. 
    # Let's try a different set of 20 vertices from the 23.
    # But 0..19 is a natural choice.
    
    # Let's convert to list of edges
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j]:
                edges.append([i, j])
                
    return edges
