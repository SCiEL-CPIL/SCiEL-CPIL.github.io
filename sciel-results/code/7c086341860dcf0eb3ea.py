"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import itertools

def construct():
    def generate_edges(n, p, seed):
        rng = random.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def solve_instance(n, p, seed):
        edges = generate_edges(n, p, seed)
        uncovered = set(edges)
        # Build adjacency sets for fast lookup
        adj = {i: set() for i in range(n)}
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        partition = []
        
        # To make the greedy more effective, we can try to find cliques of size > 2
        # We iterate through edges, but we need to be careful about the order.
        # A simple randomized greedy:
        # While uncovered edges exist:
        #   Pick a random edge from uncovered
        #   Grow it into a maximal clique using the uncovered subgraph
        #   Add clique to partition
        #   Remove all edges in clique from uncovered
        
        # Optimization: Instead of picking random, we can pick edges that are part of many triangles
        # But for simplicity and speed, random selection with greedy growth is often sufficient.
        
        # We will use a fixed seed for the solver's internal randomness to be deterministic?
        # The task says "Minimize", implying a specific algorithm. Randomness might vary score.
        # However, the task contract implies the input graphs are fixed.
        # If we use random in the solver, the result might vary. 
        # It's better to use a deterministic heuristic or fix the random seed for the solver.
        solver_rng = random.Random(42)
        
        while uncovered:
            # Pick an edge. To improve, we can pick an edge that allows for a larger clique.
            # Heuristic: Pick the edge (u,v) that maximizes the number of common neighbors in the uncovered graph.
            # This is O(E^2) or worse. 
            # Let's just pick a random edge to keep it fast, or sort by degree.
            
            # Let's try picking the edge with the highest "clique potential"
            # potential(u,v) = number of w such that (u,w) and (v,w) are in uncovered
            # We can precompute this? It changes as we remove edges.
            
            # Simple approach: Pick a random edge.
            u, v = solver_rng.choice(list(uncovered))
            
            # Grow clique starting from {u, v}
            clique = [u, v]
            # Candidates must be connected to all vertices in clique via uncovered edges
            candidates = set()
            # Initial candidates: neighbors of u and v in uncovered
            # We need to check edges in 'uncovered'
            
            # Helper to get neighbors of x in uncovered graph
            def get_uncovered_neighbors(x):
                return {y for (a, b) in uncovered if a == x for y in [b] if y != x} | \
                       {y for (a, b) in uncovered if b == x for y in [a] if y != x}
            
            # This is slow. Let's maintain an adjacency set for the uncovered graph.
            # Re-initializing adj_uncovered every time is O(E). 
            # Let's maintain a dynamic adjacency list.
            
            # Actually, let's just use the static adj and check if edge is in uncovered.
            # common_neighbors = adj[u].intersection(adj[v])
            # candidates = [w for w in common_neighbors if (min(u,w), max(u,w)) in uncovered and (min(v,w), max(v,w)) in uncovered]
            
            common = adj[u].intersection(adj[v])
            candidates = []
            for w in common:
                e1 = (min(u, w), max(u, w))
                e2 = (min(v, w), max(v, w))
                if e1 in uncovered and e2 in uncovered:
                    candidates.append(w)
            
            while candidates:
                # Pick candidate w that maximizes connections to current clique in uncovered graph
                best_w = None
                max_score = -1
                for w in candidates:
                    score = 0
                    # Check how many other candidates are connected to w in uncovered graph
                    # This is a greedy approximation
                    for v2 in clique:
                        e = (min(w, v2), max(w, v2))
                        if e in uncovered:
                            score += 1
                    # Also consider connections to other candidates?
                    # For now, just pick the one with most connections to existing clique (should be all, by definition of candidates)
                    # Actually, all candidates are connected to all clique vertices.
                    # So we should pick the one that allows the most future extensions.
                    # Score: number of common neighbors with all clique vertices that are also in candidates
                    # This is expensive. 
                    
                    # Let's just pick the first one or a random one? 
                    # To be deterministic, pick the one with smallest ID?
                    # Or the one with highest degree in the candidate subgraph?
                    
                    # Let's calculate degree in candidate subgraph
                    deg = 0
                    for w2 in candidates:
                        if w == w2: continue
                        e = (min(w, w2), max(w, w2))
                        if e in uncovered:
                            deg += 1
                    if deg > max_score:
                        max_score = deg
                        best_w = w
                
                if best_w is None:
                    break
                    
                clique.append(best_w)
                
                # Update candidates: must be connected to best_w in uncovered graph
                new_candidates = []
                for w in candidates:
                    if w == best_w: continue
                    e = (min(best_w, w), max(best_w, w))
                    if e in uncovered:
                        new_candidates.append(w)
                candidates = new_candidates
                
            # Add clique to partition
            partition.append(clique)
            
            # Remove all edges in this clique from uncovered
            for u2, v2 in itertools.combinations(clique, 2):
                e = (min(u2, v2), max(u2, v2))
                if e in uncovered:
                    uncovered.remove(e)
                    
        return partition

    p1 = solve_instance(32, 0.6, 601)
    p2 = solve_instance(64, 0.55, 602)
    
    return [p1, p2]

# EVOLVE-BLOCK-END
