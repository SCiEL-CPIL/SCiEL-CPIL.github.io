import networkx as nx
import random
import time

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 145
    
    best_n = float('inf')
    best_edges = []
    
    # Fallback to seed logic if no better found
    def get_seed_graph(k, g):
        # ... seed code ...
        pass

    # Try to find a better graph
    # Start with n=936
    for n in range(936, 1000, 2):
        if time.time() > deadline:
            break
            
        # Try a few random starts
        for attempt in range(10):
            if time.time() > deadline:
                break
                
            G = nx.random_regular_graph(k, n)
            if not nx.is_connected(G):
                continue
                
            # Local search
            for step in range(500):
                if time.time() > deadline:
                    break
                    
                girth = nx.girth(G)
                if girth >= g:
                    best_n = n
                    best_edges = list(G.edges())
                    break
                
                # Find shortest cycle
                # BFS from each node to find shortest cycle
                # This is O(N^2) which is 10^6, ok.
                # But we can optimize:
                # Just pick a random node, do BFS, find shortest cycle from it.
                # If girth is small, likely find it.
                
                # Let's just do random swaps for simplicity and speed.
                # Random swaps might not increase girth, but sometimes do.
                
                # Better: Directed swaps.
                # Find a cycle.
                # We'll use a simple heuristic:
                # Pick a random edge (u, v).
                # Try to swap it with a random non-edge (x, y).
                
                # To maintain 3-regularity:
                # Remove (u, v) and (x, y). Add (u, x) and (v, y).
                # Conditions:
                # 1. (x, y) is not an edge.
                # 2. (u, x) is not an edge.
                # 3. (v, y) is not an edge.
                # 4. u != x, u != y, v != x, v != y.
                
                # Let's try this.
                edges = list(G.edges())
                if len(edges) < 2:
                    break
                
                u, v = random.choice(edges)
                # Find a candidate (x, y)
                # x is a random node not u or v
                # y is a random node not u or v
                # But we need (x, y) to be a non-edge and (u, x), (v, y) to be non-edges.
                
                # This is hard to pick randomly.
                # Instead, pick a random node x.
                # Pick a random node y.
                # Check conditions.
                
                # Simpler:
                # Pick two random edges (u, v) and (x, y) that are disjoint.
                # Swap them.
                
                # Let's try to find disjoint edges.
                swapped = False
                for _ in range(10):
                    e1 = random.choice(edges)
                    e2 = random.choice(edges)
                    u1, v1 = e1
                    u2, v2 = e2
                    if len({u1, v1, u2, v2}) == 4:
                        # Disjoint
                        # Remove e1, e2
                        G.remove_edge(u1, v1)
                        G.remove_edge(u2, v2)
                        # Add (u1, u2) and (v1, v2)
                        G.add_edge(u1, u2)
                        G.add_edge(v1, v2)
                        swapped = True
                        break
                        
                if not swapped:
                    continue

            if best_n != float('inf'):
                break

    if best_n == float('inf'):
        # Use seed
        best_n, best_edges = get_seed_graph(k, g)
        
    return best_n, best_edges
