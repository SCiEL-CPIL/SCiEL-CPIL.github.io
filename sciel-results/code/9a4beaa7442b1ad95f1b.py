import time
import sys
import numpy as np
from scipy.sparse.csgraph import laplacian
from scipy.sparse.linalg import eigs
from random import Random

def construct():
    start_time = time.time()
    DEADLINE = 115.0  # Safety margin for 120s limit

    def random_graph(n, p, seed):
        rng = Random(seed)
        edges = set()
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
        return edges

    def solve_instance(edges_set, n):
        if not edges_set:
            return []
        
        # Build adjacency matrix
        adj_matrix = np.zeros((n, n), dtype=int)
        for u, v in edges_set:
            adj_matrix[u, v] = 1
            adj_matrix[v, u] = 1
        
        # Compute eigenvector centrality
        # Use power iteration or scipy eigs
        # For small n, we can use dense eigen decomposition
        # But for n=64, sparse is fine. Let's use dense for simplicity and speed on small n.
        try:
            # Symmetric matrix
            # Laplacian L = D - A
            # We want the principal eigenvector of A
            # Using numpy for n<=100 is fast
            eigvals, eigvecs = np.linalg.eig(adj_matrix)
            # Find index of largest real eigenvalue
            # For symmetric matrices, eigenvalues are real
            idx = np.argmax(eigvals)
            centrality = eigvecs[:, idx]
            # Ensure positive
            if np.sum(centrality) < 0:
                centrality = -centrality
        except:
            # Fallback: degree centrality
            centrality = adj_matrix.sum(axis=1)
        
        # Sort vertices by centrality
        vertices_sorted = np.argsort(-centrality)
        
        # Greedy Clique Extraction
        # We maintain a set of remaining edges
        remaining_edges = set(edges_set)
        selected_cliques = []
        
        # Adjacency sets for fast intersection
        adj_sets = [set() for _ in range(n)]
        for u, v in edges_set:
            adj_sets[u].add(v)
            adj_sets[v].add(u)
        
        # We will iteratively pick a clique.
        # Heuristic: Start with the vertex with highest centrality that still has edges.
        # Expand to form the largest possible clique containing it?
        # Or, use a greedy set cover approach on the "cliques" we find.
        
        # Let's try a different structural approach:
        # 1. Identify a "core" vertex (highest centrality).
        # 2. Find the maximum clique containing this vertex? No, that's hard.
        # 3. Instead, let's just greedily form cliques.
        
        # Better Heuristic for Partitioning:
        # 1. Sort vertices by centrality.
        # 2. For each vertex v in sorted order:
        #    - If v has no remaining incident edges, skip.
        #    - Find the largest clique C containing v, using only remaining edges.
        #      - To do this, we can greedily add neighbors of v that are connected to each other and to v.
        #      - Start with C = {v}.
        #      - Candidates = neighbors of v in remaining graph.
        #      - Sort candidates by centrality.
        #      - For each candidate u in Candidates:
        #          - If u is connected to all vertices in C (in remaining graph), add u to C.
        #    - Add C to selected_cliques.
        #    - Remove all edges within C from remaining_edges.
        #    - Remove vertices in C from consideration? No, just remove edges.
        
        # This is a "Greedy Clique Cover" variant.
        # Is it guaranteed to find a valid partition?
        # Yes, as long as we remove edges.
        # Does it minimize count? No, but it's a different strategy.
        
        # Let's refine the "Largest Clique Containing v" step.
        # A simple greedy expansion might not yield the largest clique.
        # But for dense graphs, greedy expansion is often good.
        
        # Optimization: Precompute candidate order.
        # We need to be careful with the "remaining edges" update.
        
        # Data structures:
        # remaining_adj: list of sets
        for i in range(n):
            remaining_adj[i] = set(adj_sets[i])
        
        # Sort vertices by centrality descending
        v_order = list(vertices_sorted)
        
        # We iterate through vertices.
        # If a vertex has no remaining edges, we skip it.
        # When we form a clique C, we remove edges between all pairs in C.
        
        # To speed up "is u connected to all in C", we can maintain intersection of neighborhoods.
        
        # Let's implement this.
        
        # Note: The order of processing vertices matters.
        # High centrality vertices are processed first.
        
        # However, if we pick a clique C, some vertices in C might have lower centrality.
        # That's fine.
        
        # One issue: If we process v, and form C={v, u, w}, we remove edges (v,u), (v,w), (u,w).
        # Later, when we process u (if it hasn't been "consumed"), it will have fewer edges.
        # But we don't "consume" vertices, only edges.
        # So u might still have other edges.
        # But u is already in a clique. Can u be in another clique?
        # Yes, if it has other edges.
        # But the condition is "every edge lies in exactly one clique".
        # Vertices can be in multiple cliques, as long as edges are disjoint.
        
        # So the algorithm is:
        # For v in v_order:
        #   if not remaining_adj[v]: continue
        #   C = [v]
        #   # Greedily add neighbors
        #   # Candidates are neighbors of v in remaining graph
        #   # Sorted by centrality
        #   candidates = [u for u in remaining_adj[v] if centrality[u] > centrality[v]] # Optional: only higher? No, all.
        #   # Actually, just all remaining neighbors.
        #   candidates = list(remaining_adj[v])
        #   candidates.sort(key=lambda x: -centrality[x])
        #   
        #   for u in candidates:
        #       # Check if u is connected to all in C
        #       # C contains v and possibly others.
        #       # We can check if remaining_adj[u] is a superset of C (excluding u itself, but u not in C yet)
        #       # Specifically, for all w in C, w must be in remaining_adj[u]
        #       if all(w in remaining_adj[u] for w in C):
        #           C.append(u)
        #           # Note: We don't remove edges yet. We remove them after the clique is formed.
        #           # But wait, if we add u to C, does it affect the check for the next candidate?
        #           # Yes, the next candidate must be connected to u as well.
        #           # So the check `all(w in remaining_adj[u] for w in C)` handles this because C grows.
        
        #   # Now we have a clique C.
        #   # Is it maximal? Not necessarily, but it's a valid clique.
        #   # We want to cover as many edges as possible.
        #   # Greedy expansion tends to produce reasonably large cliques in dense graphs.
        
        #   # Remove edges in C
        #   for i in range(len(C)):
        #       for j in range(i+1, len(C)):
        #           u, w = C[i], C[j]
        #           if (u, w) in remaining_edges:
        #               remaining_edges.remove((u, w))
        #           elif (w, u) in remaining_edges:
        #               remaining_edges.remove((w, u))
        #           # Update adjacency
        #           remaining_adj[u].discard(w)
        #           remaining_adj[w].discard(u)
        
        #   if len(C) >= 2:
        #       selected_cliques.append(C)
        
        # After loop, if remaining_edges is not empty, we have a problem.
        # Why? Because every edge (u,v) has u and v.
        # When u was processed, if v was a neighbor, it might have been added to u's clique.
        # If not, it means v was not connected to all of u's clique members.
        # But the edge (u,v) itself is still there.
        # Wait, if (u,v) is in remaining_edges, it means it wasn't removed.
        # If u was processed, and v was a candidate, why wasn't it added?
        # Because v wasn't connected to some w in C.
        # So (u,v) remains.
        # Later, when v is processed, u might be a candidate for v.
        # If u is added to v's clique, the edge (u,v) is removed.
        # So eventually, all edges should be covered?
        # Not necessarily. Consider a triangle u-v-w.
        # Process u. C={u}. Candidates v, w.
        # Add v? Check if v connected to u. Yes. C={u,v}.
        # Add w? Check if w connected to u and v. Yes. C={u,v,w}.
        # Remove edges (u,v), (u,w), (v,w). Done.
        
        # Consider a path u-v-w.
        # Process u. C={u}. Candidate v.
        # Add v? Yes. C={u,v}.
        # Candidate w? Is w connected to u? No. So w not added.
        # C={u,v}. Remove (u,v).
        # Process v. Remaining neighbors: w. (u is gone).
        # C={v}. Candidate w.
        # Add w? Yes. C={v,w}.
        # Remove (v,w).
        # Process w. No remaining neighbors.
        # All edges covered.
        
        # It seems this greedy approach covers all edges.
        # Why?
        # For any edge (u,v), assume u is processed before v in the order.
        # When u is processed, v is a neighbor (unless the edge was already removed, which means it's covered).
        # If the edge is not covered, v is in remaining_adj[u].
        # v is considered as a candidate.
        # If v is added to C, edge is covered.
        # If v is NOT added to C, it's because v is not connected to some w in C.
        # But this doesn't prevent v from being processed later.
        # When v is processed, u is a neighbor (edge still there).
        # u is considered as a candidate for v.
        # If u is added to v's clique, edge is covered.
        # If u is NOT added, it's because u is not connected to some x in v's clique.
        # This logic suggests that an edge might survive if both endpoints fail to add each other to their respective cliques due to other constraints.
        # Is that possible?
        # Edge (u,v). u processed first.
        # C_u = {u, ...}. v not in C_u because v not connected to w in C_u.
        # v processed later.
        # C_v = {v, ...}. u not in C_v because u not connected to x in C_v.
        # Edge (u,v) remains.
        # This would be a failure.
        
        # To ensure coverage, we can add a fallback:
        # If after the main loop, remaining_edges is not empty,
        # we can cover them individually or with small cliques.
        # But we want to minimize count.
        # If we have a failure, we can just add the edge as a 2-clique.
        # This is valid.
        
        # So the algorithm is safe.
        
        # Let's implement it.
        
        # One optimization: 
        # When checking `all(w in remaining_adj[u] for w in C)`, 
        # we can maintain the intersection of neighborhoods.
        
        # Let's stick to the simple version first, n is small.
        
        # Re-initialize remaining_adj
        remaining_adj = [set(adj_sets[i]) for i in range(n)]
        remaining_edges = set(edges_set)
        selected_cliques = []
        
        v_order = list(vertices_sorted)
        
        for v in v_order:
            if time.time() > DEADLINE:
                # Fallback for remaining edges
                for e in remaining_edges:
                    selected_cliques.append([e[0], e[1]])
                break
            
            if not remaining_adj[v]:
                continue
            
            C = [v]
            candidates = list(remaining_adj[v])
            candidates.sort(key=lambda x: -centrality[x])
            
            for u in candidates:
                # Check if u is connected to all in C
                # Note: u is not in C yet
                is_connected = True
                for w in C:
                    if w not in remaining_adj[u]:
                        is_connected = False
                        break
                if is_connected:
                    C.append(u)
            
            if len(C) >= 2:
                selected_cliques.append(C)
                
                # Remove edges in C
                for i in range(len(C)):
                    for j in range(i+1, len(C)):
                        u, w = C[i], C[j]
                        edge = (u, w) if u < w else (w, u)
                        if edge in remaining_edges:
                            remaining_edges.remove(edge)
                        remaining_adj[u].discard(w)
                        remaining_adj[w].discard(u)
        
        # Fallback for any remaining edges
        if remaining_edges:
            for e in remaining_edges:
                selected_cliques.append([e[0], e[1]])
        
        return selected_cliques

    # Instance 1
    n1, p1, s1 = 32, 0.6, 601
    e1 = random_graph(n1, p1, s1)
    c1 = solve_instance(e1, n1)
    
    # Instance 2
    n2, p2, s2 = 64, 0.55, 602
    e2 = random_graph(n2, p2, s2)
    c2 = solve_instance(e2, n2)
    
    return [c1, c2]

if __name__ == "__main__":
    res = construct()
    print(len(res[0]), len(res[1]))
