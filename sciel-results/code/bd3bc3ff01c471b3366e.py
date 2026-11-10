import itertools

def construct():
    # Parts of size 4
    P0 = [0, 1, 2, 3]
    P1 = [4, 5, 6, 7]
    P2 = [8, 9, 10, 11]
    P3 = [12, 13, 14, 15]
    parts = [P0, P1, P2, P3]
    
    edges = []
    
    # Define edges to remove for each pair of parts
    # We remove 2 edges incident to the first vertex of the first part
    # This ensures the max degree in that bipartite subgraph is 3 for that vertex,
    # but actually we want to ensure no 3-regular subgraph exists.
    # Removing 2 edges from u0 makes deg(u0)=2.
    # Thus u0 cannot be in any 3-regular subgraph.
    # Since Q3 requires 4 vertices from U, and u0 is the only one with deg<3?
    # Wait, if we remove 2 edges from u0, u0 has deg 2.
    # u1, u2, u3 have deg 4.
    # Can we find a Q3 using u1, u2, u3 and some other u? No, we only have 4 u's.
    # So any Q3 must use u0.
    # But u0 has deg 2, so it can't be in a 3-regular subgraph.
    # So no Q3 exists.
    
    removals = {
        (0, 1): [(0, 4), (0, 5)],  # P0-P1: remove (0,4), (0,5)
        (0, 2): [(0, 8), (0, 9)],  # P0-P2
        (0, 3): [(0, 12), (0, 13)],# P0-P3
        (1, 2): [(4, 8), (4, 9)],  # P1-P2
        (1, 3): [(4, 12), (4, 13)],# P1-P3
        (2, 3): [(8, 12), (8, 13)] # P2-P3
    }
    
    removed_edges = set()
    for key, rem_list in removals.items():
        for e in rem_list:
            removed_edges.add((min(e), max(e)))
            
    # Generate all edges between different parts
    for i in range(4):
        for j in range(i + 1, 4):
            for u in parts[i]:
                for v in parts[j]:
                    edge = (min(u, v), max(u, v))
                    if edge not in removed_edges:
                        edges.append([u, v])
                        
    return edges
