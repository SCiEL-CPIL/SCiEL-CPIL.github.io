import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s kill
    
    n = 16
    all_edges = list(combinations(range(n), 2))
    edge_set = set(all_edges)
    
    # Precompute all Q3 embeddings (8-vertex subsets with Q3 structure)
    # Q3 has vertices 0-7, edges between vertices differing in exactly 1 bit
    q3_vertices = list(range(8))
    q3_edges = set()
    for a in range(8):
        for b in range(8):
            if a != b and bin(a ^ b).count('1') == 1:
                q3_edges.add((min(a,b), max(a,b)))
    
    # For each 8-subset of 16 vertices, check if it contains Q3 as a subgraph
    # We need to find all injective mappings from Q3 vertices to the 8-subset
    # such that all 12 Q3 edges are present
    
    # Better: precompute all Q3 subgraphs in K16
    # A Q3 subgraph is determined by choosing 8 vertices and a labeling
    # But we just need to know which 12-edge sets form Q3
    
    # Actually, let's use a different approach for efficiency:
    # For the local search, we need to quickly check if adding an edge creates a Q3.
    # An edge (u,v) creates a Q3 if there exist 6 other vertices such that
    # the 8 vertices form a Q3 containing edge (u,v).
    
    # Let me precompute: for each edge, which Q3 subgraphs contain it.
    # But that's expensive. Instead, let's use a direct check.
    
    def is_q3_subset(vertices, adj):
        """Check if the induced subgraph on 'vertices' (8 vertices) contains Q3 as a subgraph."""
        # vertices is a tuple of 8 vertex IDs
        # We need to check if there's an isomorphism from Q3 to this subgraph
        # Try all 8! / automorphisms... too slow.
        # Instead: Q3 is bipartite with parts of size 4.
        # In Q3, each vertex has degree 3.
        # Check if the subgraph contains a 3-regular bipartite graph with 4+4 vertices.
        
        vlist = list(vertices)
        # Build adjacency within these 8 vertices
        local_adj = {v: set() for v in vlist}
        for i in range(8):
            for j in range(i+1, 8):
                u, w = vlist[i], vlist[j]
                if (min(u,w), max(u,w)) in adj:
                    local_adj[u].add(w)
                    local_adj[w].add(u)
        
        # Q3 is 3-regular bipartite on 8 vertices
        # Check if there's a bipartition into two sets of 4, each vertex degree 3 in the subgraph
        # and all edges go between the parts
        
        # Try all ways to split into two sets of 4: C(8,4)/2 = 35 possibilities
        for part_a in combinations(range(8), 4):
            part_b = tuple(range(8))
            part_b = tuple(x for x in range(8) if x not in set(part_a))
            if part_b == part_a:
                continue
            # Check bipartiteness: no edges within part_a or part_b
            valid = True
            for i in range(4):
                for j in range(i+1, 4):
                    if vlist[part_a[i]] in local_adj[vlist[part_a[j]]]:
                        valid = False
                        break
                    if vlist[part_b[i]] in local_adj[vlist[part_b[j]]]:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                continue
            
            # Check each vertex has degree exactly 3 to the other part
            valid = True
            for idx in part_a:
                v = vlist[idx]
                deg = 0
                for bidx in part_b:
                    if vlist[bidx] in local_adj[v]:
                        deg += 1
                if deg != 3:
                    valid = False
                    break
            if not valid:
                continue
            for idx in part_b:
                v = vlist[idx]
                deg = 0
                for aidx in part_a:
                    if vlist[aidx] in local_adj[v]:
                        deg += 1
                if deg != 3:
                    valid = False
                    break
            if valid:
                return True
        return False
    
    def count_q3_in_subgraph(vertices, adj):
        """Count Q3 subgraphs in the 8-vertex induced subgraph."""
        return 1 if is_q3_subset(vertices, adj) else 0
    
    def graph_has_q3(adj):
        """Check if graph with adjacency set 'adj' contains Q3."""
        for verts in combinations(range(n), 8):
            if is_q3_subset(verts, adj):
                return True
        return False
    
    # Start with a random dense graph and iteratively remove Q3 edges
    def remove_q3_edges(adj, max_iter=5000):
        """Iteratively remove edges that are in Q3 subgraphs."""
        adj = set(adj)
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            # Find a Q3 and remove an edge from it
            found = False
            for verts in combinations(range(n), 8):
                if is_q3_subset(verts, adj):
                    # Find edges in this Q3 and remove one
                    vlist = list(verts)
                    # Find which edges form the Q3
                    q3_edges_here = []
                    for i in range(8):
                        for j in range(i+1, 8):
                            u, w = vlist[i], vlist[j]
                            e = (min(u,w), max(u,w))
                            if e in adj:
                                q3_edges_here.append(e)
                    # Remove a random edge from the Q3
                    if q3_edges_here:
                        e_to_remove = random.choice(q3_edges_here)
                        adj.discard(e_to_remove)
                        found = True
                        break
            if not found:
                break
        return adj
    
    def try_add_edges(adj, max_iter=2000):
        """Try to add edges that don't create Q3."""
        adj = set(adj)
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            # Try to add a random missing edge
            missing = [e for e in all_edges if e not in adj]
            if not missing:
                break
            random.shuffle(missing)
            added = False
            for e in missing:
                new_adj = adj | {e}
                # Check if adding e creates a Q3
                # Only need to check 8-sets containing both endpoints of e
                u, v = e
                other = [x for x in range(n) if x != u and x != v]
                for six in combinations(other, 6):
                    verts = tuple(sorted((u, v) + six))
                    if is_q3_subset(verts, new_adj):
                        break
                else:
                    adj.add(e)
                    added = True
                    break
            if not added:
                break
        return adj
    
    best_edges = 0
    best_adj = set()
    
    num_restarts = 50
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Start with a random graph with varying density
        density = 0.7 + 0.25 * random.random()
        adj = set()
        for e in all_edges:
            if random.random() < density:
                adj.add(e)
        
        # Remove Q3 edges
        adj = remove_q3_edges(adj)
        
        # Try to add edges
        adj = try_add_edges(adj)
        
        if len(adj) > best_edges:
            best_edges = len(adj)
            best_adj = set(adj)
    
    # Also try starting from the seed (K3,13) and improving
    seed_adj = set()
    for a in range(3):
        for b in range(3, 16):
            seed_adj.add((min(a,b), max(a,b)))
    
    if len(seed_adj) > best_edges:
        best_edges = len(seed_adj)
        best_adj = set(seed_adj)
    
    # Try to improve the best found
    if time.time() < deadline - 10:
        improved = try_add_edges(best_adj, max_iter=5000)
        if len(improved) > best_edges:
            best_edges = len(improved)
            best_adj = improved
    
    # Convert to list of [u, v] pairs
    result = [[u, v] for (u, v) in best_adj]
    return result
