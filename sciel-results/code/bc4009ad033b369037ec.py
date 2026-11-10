import time
import random
import itertools
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n = 20
    deadline = time.time() + 145
    
    # We work on the complement graph G_bar.
    # G is K3,3-free iff for all disjoint triples A, B, there is at least one edge in G_bar between A and B.
    # Equivalent: For all triples A, the set S_A of vertices in V\A with no edges to A in G_bar has size <= 2.
    
    def is_valid(adj_bar):
        """
        Check if adj_bar (list of sets) satisfies the condition.
        For every triple A, |S_A| <= 2, where S_A = {v in V\A | v not in N(A)}
        """
        vertices = range(n)
        # Precompute all triples
        triples = list(itertools.combinations(vertices, 3))
        
        for A in triples:
            A_set = set(A)
            # Find N(A) in G_bar
            # N(A) = union of neighbors of u in A
            N_A = set()
            for u in A:
                N_A.update(adj_bar[u])
            
            # S_A = {v in V \ A | v not in N_A}
            # Note: v in A is not in N_A (no self loops), but we exclude A anyway.
            # We need v such that v not in A and v not in N_A.
            S_A = [v for v in vertices if v not in A_set and v not in N_A]
            
            if len(S_A) > 2:
                return False, A, S_A
        
        return True, None, None

    def count_edges(adj_bar):
        count = 0
        for u in range(n):
            for v in adj_bar[u]:
                if u < v:
                    count += 1
        return count

    def greedy_construct():
        adj_bar = [set() for _ in range(n)]
        
        # Optional: Start with a small random graph to break symmetry?
        # Or start empty.
        
        while time.time() < deadline - 5:
            valid, A, S_A = is_valid(adj_bar)
            if valid:
                break
            
            # Find the edge to add that maximally reduces violations
            # A violation is a triple A with |S_A| > 2.
            # We can re-compute all violations to find the best edge.
            
            # List of all triples
            triples = list(itertools.combinations(range(n), 3))
            
            # For each edge (u,v), count how many violations it fixes.
            # An edge (u,v) fixes triple A if:
            # (u in A and v not in A and v in S_A) OR (v in A and u not in A and u in S_A)
            # Note: S_A depends on current adj_bar.
            
            # To be efficient, we can just pick a random violating triple A,
            # and pick a random edge from A to S_A.
            # This is a simple greedy heuristic.
            
            # Let's try to be a bit smarter: pick the violating A with largest S_A,
            # and pick the edge (u,v) in A x S_A that appears in the most "potential" fixes?
            # For now, random edge from A to S_A.
            
            u = random.choice(A)
            v = random.choice(S_A)
            
            adj_bar[u].add(v)
            adj_bar[v].add(u)
        
        # Minimization phase
        # Try to remove edges
        changed = True
        while changed and time.time() < deadline - 2:
            changed = False
            # Get list of edges
            edges = []
            for u in range(n):
                for v in adj_bar[u]:
                    if u < v:
                        edges.append((u, v))
            
            random.shuffle(edges)
            
            for u, v in edges:
                if time.time() > deadline - 2:
                    break
                # Remove edge
                adj_bar[u].remove(v)
                adj_bar[v].remove(u)
                
                # Check validity
                valid, _, _ = is_valid(adj_bar)
                
                if valid:
                    changed = True
                else:
                    # Add back
                    adj_bar[u].add(v)
                    adj_bar[v].add(u)
        
        return adj_bar

    best_adj_bar = None
    best_edge_count = n * (n - 1) // 2 # Max possible edges in G_bar, so G has 0
    
    num_restarts = 20
    
    for i in range(num_restarts):
        if time.time() > deadline - 10:
            break
            
        adj_bar = greedy_construct()
        
        # Verify one last time
        valid, _, _ = is_valid(adj_bar)
        if not valid:
            continue
            
        e_bar = count_edges(adj_bar)
        e_g = n * (n - 1) // 2 - e_bar
        
        if e_g > best_edge_count:
            best_edge_count = e_g
            best_adj_bar = [set(s) for s in adj_bar]
            
    # Convert best_adj_bar to edge list of G
    if best_adj_bar is None:
        # Fallback: Empty G? No, must be valid.
        # If we failed, return a known valid sparse graph.
        # K2,18 is K3,3 free?
        # Parts {0,1} and rest.
        # K3,3 requires 3 in each part. We only have 2 in first part. So yes.
        edges = []
        for i in [0, 1]:
            for j in range(2, n):
                edges.append([i, j])
        return edges
        
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            # Edge (u,v) is in G if it is NOT in G_bar
            if v not in best_adj_bar[u]:
                edges.append([u, v])
                
    return edges
# EVOLVE-BLOCK-END
