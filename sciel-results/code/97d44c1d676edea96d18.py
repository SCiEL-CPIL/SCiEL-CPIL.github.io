import random
import time

def construct():
    n = 20
    start_time = time.time()
    deadline = start_time + 110  # 10 second safety margin
    
    # Use a set of edges for fast lookup
    # And a list of neighbors for each vertex
    edges = set()
    neighbors = [set() for _ in range(n)]
    
    def add_edge(u, v):
        if u == v: return
        if (u, v) in edges or (v, u) in edges: return
        edges.add((u, v))
        neighbors[u].add(v)
        neighbors[v].add(u)
        
    def remove_edge(u, v):
        if (u, v) not in edges and (v, u) not in edges: return
        if (u, v) in edges:
            edges.remove((u, v))
        else:
            edges.remove((v, u))
        neighbors[u].discard(v)
        neighbors[v].discard(u)

    def is_add_safe(u, v):
        if (u, v) in edges or (v, u) in edges:
            return True
        # Check if adding (u, v) creates a K3,3
        # u and v must be in opposite parts of the K3,3
        # Part A: {u, a1, a2}, Part B: {v, b1, b2}
        # Need a1, a2 in N(u) and b1, b2 in N(v) such that K2,2 exists between {a1,a2} and {b1,b2}
        # Also need to consider the case where u is in B and v is in A, which is symmetric.
        
        nu = neighbors[u]
        nv = neighbors[v]
        
        # We need to find a1, a2 in nu and b1, b2 in nv such that
        # all edges (a1,b1), (a1,b2), (a2,b1), (a2,b2) exist.
        
        # Iterate over pairs in nu
        nu_list = list(nu)
        for i in range(len(nu_list)):
            a1 = nu_list[i]
            for j in range(i+1, len(nu_list)):
                a2 = nu_list[j]
                # Common neighbors of a1 and a2 in nv
                common = neighbors[a1].intersection(neighbors[a2])
                common_in_nv = common.intersection(nv)
                if len(common_in_nv) >= 2:
                    # We found b1, b2
                    return False
        return True

    def find_k33():
        # Find a K3,3 in the current graph
        # Iterate over all triples A
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    common = neighbors[i].intersection(neighbors[j]).intersection(neighbors[k])
                    if len(common) >= 3:
                        # Pick any 3 from common
                        common_list = list(common)
                        b1, b2, b3 = common_list[0], common_list[1], common_list[2]
                        return [(i, b1), (i, b2), (i, b3), (j, b1), (j, b2), (j, b3), (k, b1), (k, b2), (k, b3)]
        return None

    # Initialize with a random graph
    random.seed(42)
    p = 0.5
    for u in range(n):
        for v in range(u+1, n):
            if random.random() < p:
                add_edge(u, v)
                
    # Remove K3,3s
    while time.time() < deadline:
        k33 = find_k33()
        if k33 is None:
            break
        # Remove a random edge from the K3,3
        e = random.choice(k33)
        remove_edge(e[0], e[1])
        
    # Now try to add edges
    best_edges = len(edges)
    best_graph = set(edges)
    
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        # Try to add a random edge
        u = random.randint(0, n-1)
        v = random.randint(0, n-1)
        if u == v: continue
        if (u, v) in edges or (v, u) in edges: continue
        
        if is_add_safe(u, v):
            add_edge(u, v)
            if len(edges) > best_edges:
                best_edges = len(edges)
                best_graph = set(edges)
        else:
            # If not safe, maybe try to remove an edge and add another?
            # For now, just skip.
            pass
            
        # Occasionally, try to restart or perturb if stuck
        if attempts % 1000 == 0:
            # Check if we are stuck (no improvements in last 1000 attempts)
            # We can try to remove a random edge and see if we can add more
            if random.random() < 0.1:
                if edges:
                    e = random.choice(list(edges))
                    remove_edge(e[0], e[1])
                    
    # Return the best graph found
    # Convert set of tuples to list of lists
    result = []
    for u, v in best_graph:
        result.append([u, v])
    return result
