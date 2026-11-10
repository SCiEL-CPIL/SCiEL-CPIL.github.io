import random
import networkx as nx

def construct():
    def generate_graph(n, p, seed):
        rng = random.Random(seed)
        G = nx.Graph()
        G.add_nodes_from(range(n))
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    G.add_edge(u, v)
        return G

    def solve_instance(G):
        adj = {u: set(G.neighbors(u)) for u in G.nodes()}
        degrees = {u: len(adj[u]) for u in G.nodes()}
        
        edges = list(G.edges())
        # Sort by degree sum descending
        edges.sort(key=lambda e: degrees[e[0]] + degrees[e[1]], reverse=True)
        
        covered_edges = set()
        cliques = [] # List of lists of vertices
        
        for u, v in edges:
            e_key = (min(u, v), max(u, v))
            if e_key in covered_edges:
                continue
            
            placed = False
            for i, clique in enumerate(cliques):
                # Check if we can add u, v to this clique
                # New vertex set
                new_v = set(clique)
                new_v.add(u)
                new_v.add(v)
                
                # Check 1: Is it a clique?
                is_clique = True
                # We need u and v to connect to all in clique, and u to v
                # u-v is an edge.
                for x in clique:
                    if x not in adj[u] or x not in adj[v]:
                        is_clique = False
                        break
                
                if not is_clique:
                    continue
                
                # Check 2: Are any edges in the new clique already covered?
                # We only need to check edges that involve u or v, 
                # because edges within 'clique' are already covered by this clique (by definition).
                # Wait, if 'clique' was formed by previous additions, all its internal edges are covered.
                # So we just need to check edges (u, x) and (v, x) for x in clique.
                # And (u, v) is the current edge, not covered.
                
                can_add = True
                for x in clique:
                    eu = (min(u, x), max(u, x))
                    ev = (min(v, x), max(v, x))
                    if eu in covered_edges or ev in covered_edges:
                        can_add = False
                        break
                
                if can_add:
                    # Add to clique
                    cliques[i] = list(new_v) # Update the clique vertex list
                    # Mark all new edges as covered
                    covered_edges.add(e_key)
                    for x in clique:
                        covered_edges.add((min(u, x), max(u, x)))
                        covered_edges.add((min(v, x), max(v, x)))
                    placed = True
                    break
            
            if not placed:
                # Create new clique
                cliques.append([u, v])
                covered_edges.add(e_key)
                
        return cliques

    G1 = generate_graph(32, 0.6, 601)
    G2 = generate_graph(64, 0.55, 602)
    
    C1 = solve_instance(G1)
    C2 = solve_instance(G2)
    
    return [C1, C2]
