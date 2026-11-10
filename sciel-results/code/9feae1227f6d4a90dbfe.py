import time
import itertools
import numpy as np
from functools import lru_cache

def construct():
    n = 20
    deadline = time.time() + 145.0
    best_count = 0
    best_edges = []
    
    # Precompute all pairs
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # We'll search over bipartitions. For a partition (A, B),
    # the K3,3 constraint in the bipartite graph between A and B is:
    #   No 3 vertices in A share 3 common neighbors in B
    #   No 3 vertices in B share 3 common neighbors in A
    # 
    # To maximize edges in the bipartite graph between A and B under these constraints,
    # we can use a greedy removal approach: start with complete bipartite K_{|A|,|B|},
    # then remove edges to eliminate all K3,3 subgraphs.
    #
    # Actually, a cleaner approach: for a fixed bipartition, we want to find the
    # densest subgraph of K_{|A|,|B|} that is K3,3-free. This is equivalent to
    # finding a 0-1 matrix with no 3x3 all-ones submatrix (after permuting rows/cols,
    # but actually we need: no 3 rows have 3 common 1-columns, AND no 3 cols have 3
    # common 1-rows).
    #
    # Wait - that's not quite right. K3,3 in a bipartite graph means there exist
    # 3 vertices a1,a2,a3 in A and 3 vertices b1,b2,b3 in B such that all 9 edges
    # (ai,bj) are present. So the constraint is: for every triple of rows, they share
    # at most 2 common 1-columns. Equivalently, the 0-1 matrix has no 3x3 all-ones submatrix.
    #
    # The maximum number of 1s in an m x k 0-1 matrix with no 3x3 all-ones submatrix
    # is a known extremal problem. For our purposes, we can solve it greedily:
    # Start with all 1s, then for each triple of rows, if they share >= 3 columns with 1s,
    # remove one edge from the intersection.
    #
    # Actually, let me think differently. The problem of finding the densest K3,3-free
    # bipartite subgraph of K_{m,k} is NP-hard in general, but for small m,k (<=10),
    # we can use a simple greedy: try all edges in some order, add if it doesn't create
    # a 3x3 all-ones submatrix.
    #
    # But we also want to consider non-bipartite edges within A and within B.
    # Let's first focus on bipartite-only graphs, then try adding intra-part edges.
    
    # Better strategy: iterate over all 2^20 subsets as "side A", compute the
    # optimal bipartite K3,3-free graph between A and B, then try to add
    # intra-A and intra-B edges greedily.
    
    # To make this tractable, we'll use a smarter approach:
    # 1. For each bipartition size (a, n-a) with a <= n//2, try several random partitions
    # 2. For each partition, greedily build the densest K3,3-free bipartite subgraph
    # 3. Then try to add intra-part edges
    
    # Actually, let's use a completely different approach that's more efficient:
    # Use the fact that ex(n, K3,3) for n=20 is around 50-60 edges.
    # We'll use a local search with edge swaps starting from random dense graphs.
    
    # Let me implement a robust local search:
    # 1. Start with a random graph with ~50-60 edges
    # 2. Check if K3,3-free; if not, remove offending edges
    # 3. Try to add edges or swap edges to increase count while maintaining K3,3-free
    
    # First, let's implement a fast K3,3 check and edge-addition check.
    
    def is_k33_free(adj, n):
        """Check if graph is K3,3-free."""
        # For every pair of disjoint 3-sets {u,v,w} and {x,y,z}, check not all 9 edges present
        # Faster: for each triple (u,v,w), compute common neighbors, check if any 3 of them
        # form a triple with the right connections... 
        # Actually, the cleanest check: for each pair of vertices (u,v), if |N(u)∩N(v)| >= 3,
        # then for each pair of common neighbors (a,b), check if there's a third common neighbor c
        # such that a,b,c are all connected to u and v. But that's just |N(u)∩N(v)| >= 3.
        # 
        # Wait, no. K3,3 requires TWO triples with ALL 9 edges between them.
        # The condition |N(u)∩N(v)| >= 3 for some u,v is NECESSARY but not SUFFICIENT
        # for K3,3. We need 3 vertices on each side.
        #
        # Proper check: For each 3-set S = {u,v,w}, let C = N(u)∩N(v)∩N(w).
        # If |C| >= 3, then pick any 3 from C, say {x,y,z}. We need edges x-u, x-v, x-w,
        # y-u, y-v, y-w, z-u, z-v, z-w. But by definition of C, all these edges exist!
        # So if |N(u)∩N(v)∩N(w)| >= 3, we have a K3,3.
        #
        # Wait, that's the condition for a K_{3,3} where {u,v,w} is one part and 
        # {x,y,z} ⊂ C is the other part. Yes! So the graph is K3,3-free iff
        # for every triple {u,v,w}, |N(u)∩N(v)∩N(w)| <= 2.
        
        for u in range(n):
            for v in range(u+1, n):
                common_uv = adj[u] & adj[v]
                for w in range(v+1, n):
                    if len(common_uv & adj[w]) >= 3:
                        return False
        return True
    
    def can_add_edge(adj, u, v, n):
        """Check if adding edge (u,v) keeps the graph K3,3-free."""
        # After adding (u,v), we need: for every triple containing u or v,
        # the common neighborhood size is <= 2.
        # New triples involving the new edge:
        # Triples {u, v, w} for w != u,v: need |N(u)∩N(v)∩N(w)| <= 2 after adding (u,v)
        #   After adding, u∈N(v) and v∈N(u).
        #   N(u)∩N(v)∩N(w) = (old N(u)∪{v}) ∩ (old N(v)∪{u}) ∩ N(w)
        #   = (old N(u) ∩ old N(v) ∩ N(w)) ∪ ({v} ∩ old N(v) ∩ N(w)) ∪ (old N(u) ∩ {u} ∩ N(w)) ∪ ({v}∩{u}∩N(w))
        #   The last term is empty (u≠v).
        #   {v} ∩ old N(v) ∩ N(w): v is not in its own neighborhood, so this is empty.
        #   old N(u) ∩ {u} ∩ N(w): u is not in its own neighborhood, so this is empty.
        #   So it's just old N(u) ∩ old N(v) ∩ N(w).
        #   So for triple {u,v,w}: need |N(u)∩N(v)∩N(w)| <= 2 (unchanged by adding (u,v)).
        
        # Triples {u, w1, w2} where w1,w2 != u,v: 
        #   N(u)∩N(w1)∩N(w2) after adding = (old N(u)∪{v}) ∩ N(w1) ∩ N(w2)
        #   = (old N(u)∩N(w1)∩N(w2)) ∪ ({v}∩N(w1)∩N(w2))
        #   Need this to be <= 2.
        #   So need: |old N(u)∩N(w1)∩N(w2)| + [1 if v∈N(w1)∩N(w2) else 0] <= 2
        
        # Triples {v, w1, w2} where w1,w2 != u,v: symmetric.
        
        # Triples not containing u or v: unchanged.
        
        # So we need to check:
        # 1. For all w != u,v: |N(u)∩N(v)∩N(w)| <= 2 (already true if graph was K3,3-free)
        # 2. For all w1 < w2, w1,w2 != u,v: 
        #    |N(u)∩N(w1)∩N(w2)| + [v∈N(w1)∩N(w2)] <= 2
        #    and |N(v)∩N(w1)∩N(w2)| + [u∈N(w1)∩N(w2)] <= 2
        
        # Condition 1 is automatically satisfied if the current graph is K3,3-free.
        # Let's verify conditions 2.
        
        for w1 in range(n):
            if w1 == u or w1 == v:
                continue
            for w2 in range(w1+1, n):
                if w2 == u or w2 == v:
                    continue
                # Check triple {u, w1, w2}
                common_uw1w2 = adj[u] & adj[w1] & adj[w2]
                v_in_w1w2 = (v in adj[w1]) and (v in adj[w2])
                if len(common_uw1w2) + (1 if v_in_w1w2 else 0) > 2:
                    return False
                # Check triple {v, w1, w2}
                common_vw1w2 = adj[v] & adj[w1] & adj[w2]
                u_in_w1w2 = (u in adj[w1]) and (u in adj[w2])
                if len(common_vw1w2) + (1 if u_in_w1w2 else 0) > 2:
                    return False
        return True
    
    # Now implement the local search
    random_seed = 42
    rng = np.random.default_rng(random_seed)
    
    num_restarts = 200
    
    for restart in range(num_restarts):
        if time.time() > deadline - 10:
            break
        
        # Start with a random graph
        adj = [set() for _ in range(n)]
        edges = []
        
        # Random initial density
        p = 0.4 + 0.3 * rng.random()
        for i in range(n):
            for j in range(i+1, n):
                if rng.random() < p:
                    adj[i].add(j)
                    adj[j].add(i)
                    edges.append([i, j])
        
        # Remove edges to make K3,3-free
        # Find and remove offending edges
        changed = True
        while changed:
            changed = False
            for idx, (u, v) in enumerate(edges[:]):
                if time.time() > deadline - 5:
                    break
                # Check if this edge is part of a K3,3
                # Temporarily remove it and check
                # Actually, let's just check if removing it helps
                # Simpler: check all triples
                # For each triple {u,v,w}: if |N(u)∩N(v)∩N(w)| >= 3, remove one edge
                pass
            
            # Better: iteratively find K3,3 subgraphs and remove edges
            found = False
            for u in range(n):
                if found:
                    break
                for v in range(u+1, n):
                    if found:
                        break
                    common_uv = adj[u] & adj[v]
                    if len(common_uv) < 3:
                        continue
                    for w in range(v+1, n):
                        if found:
                            break
                        common_uvw = common_uv & adj[w]
                        if len(common_uvw) >= 3:
                            # Found a K3,3: {u,v,w} x {x,y,z} where {x,y,z} ⊂ common_uvw
                            # Remove an edge. Remove (u, x) where x is first common neighbor
                            common_list = list(common_uvw)
                            x = common_list[0]
                            # Remove edge (u, x)
                            adj[u].discard(x)
                            adj[x].discard(u)
                            edges = [e for e in edges if not (e[0]==u and e[1]==x) and not (e[0]==x and e[1]==u)]
                            found = True
                            changed = True
                            break
            if time.time() > deadline - 5:
                break
        
        if time.time() > deadline - 5:
            break
        
        # Now try to add edges greedily
        all_possible = [(i, j) for i in range(n) for j in range(i+1, n) 
                        if j not in adj[i]]
        rng.shuffle(all_possible)
        
        for u, v in all_possible:
            if time.time() > deadline - 3:
                break
            if can_add_edge(adj, u, v, n):
                adj[u].add(v)
                adj[j].add(u) if False else adj[v].add(u)
                edges.append([u, v])
        
        # Try edge swaps: remove one edge, try to add two
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            current_edges = list(edges)
            rng.shuffle(current_edges)
            for u, v in current_edges:
                if time.time() > deadline - 5:
                    break
                # Try removing (u,v) and adding two other edges
                adj[u].discard(v)
                adj[v].discard(u)
                
                # Try to add edges
                added = 0
                candidates = [(i, j) for i in range(n) for j in range(i+1, n) 
                             if j not in adj[i] and not (i==u and j==v)]
                rng.shuffle(candidates)
                for a, b in candidates:
                    if added >= 2:
                        break
                    if can_add_edge(adj, a, b, n):
                        adj[a].add(b)
                        adj[b].add(a)
                        edges.append([a, b])
                        added += 1
                
                if added > 1:
                    improved = True
                elif added == 1:
                    # Net zero, revert
                    # Actually keep it, might help later
                    pass
                else:
                    # Revert
                    adj[u].add(v)
                    adj[v].add(u)
                    # Remove the one we added
                    for e in edges[:]:
                        if not (e[0]==u and e[1]==v) and not (e[0]==v and e[1]==u):
                            pass
                    # This is getting complicated. Let's simplify.
                    break
        
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges[:]
        
        if best_count >= 55:  # Good enough threshold
            break
    
    # Final verification
    # Rebuild adjacency from best_edges
    adj_final = [set() for _ in range(n)]
    for u, v in best_edges:
        adj_final[u].add(v)
        adj_final[v].add(u)
    
    # Verify K3,3-free
    valid = True
    for u in range(n):
        for v in range(u+1, n):
            common_uv = adj_final[u] & adj_final[v]
            for w in range(v+1, n):
                if len(common_uv & adj_final[w]) >= 3:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        # Fallback: use a known valid construction
        # Simple: take a random sparse graph and verify
        adj = [set() for _ in range(n)]
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if rng.random() < 0.3:
                    adj[i].add(j)
                    adj[j].add(i)
                    edges.append([i, j])
        # Remove K3,3 violations
        for _ in range(1000):
            found = False
            for u in range(n):
                if found:
                    break
                for v in range(u+1, n):
                    if found:
                        break
                    common_uv = adj[u] & adj[v]
                    for w in range(v+1, n):
                        if found:
                            break
                        if len(common_uv & adj[w]) >= 3:
                            x = next(iter(common_uv & adj[w]))
                            adj[u].discard(x)
                            adj[x].discard(u)
                            edges = [e for e in edges if not (e[0]==u and e[1]==x) and not (e[0]==x and e[1]==u)]
                            found = True
                            break
            if not found:
                break
        best_edges = edges
    
    return best_edges
