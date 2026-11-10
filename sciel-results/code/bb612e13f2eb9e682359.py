import time
import random
import sys

def construct():
    def instance(n, p, seed):
        rng = random.Random(seed)
        adj = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        return adj

    def count_bits(x):
        return bin(x).count("1")

    def greedy_line_graph_coloring(adj, n, seed=None):
        """
        Treats edges of the original graph as vertices in a line graph.
        We want to partition these 'vertices' (original edges) into cliques 
        (original cliques). This is equivalent to edge coloring where 
        adjacent edges (sharing a vertex) can go in the same color ONLY IF 
        they form a clique with the rest of the color's edges.
        
        Actually, the constraint is: A set of edges E' forms a clique in G 
        if and only if the vertices involved in E' form a clique in G.
        
        Strategy: 
        1. List all edges.
        2. Iterate through edges. For each edge (u,v), try to insert it into 
           an existing clique C.
           Insertion is valid if:
           - u and v are already in C (impossible, simple graph)
           - u is in C and v is connected to all nodes in C
           - v is in C and u is connected to all nodes in C
           - neither is in C, but u and v are connected to all nodes in C
           
        To optimize, we maintain for each clique a bitmask of its vertices.
        An edge (u,v) fits in clique C (with vertex set V_C) if:
        (adj[u] & V_C) == V_C AND (adj[v] & V_C) == V_C
        Note: If u is in V_C, then (adj[u] & V_C) == V_C is true because 
        u is connected to all others in V_C (since C is a clique).
        So the condition simply checks if u and v are universal to the current clique.
        
        This is a valid clique extension.
        
        Heuristic: 
        - Order edges by some heuristic (e.g., sum of degrees of endpoints).
        - Try to fit into existing cliques. 
        - To minimize count, we should prefer filling existing cliques.
        - Which existing clique? 
          Option A: First fit (fast, but might be suboptimal).
          Option B: Best fit (e.g., clique with max size, or max overlap).
          Option C: Randomized restarts with different edge orderings.
        """
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random.Random()
            
        # Get all edges
        edges = []
        for u in range(n):
            v = adj[u]
            while v:
                lsb = v & -v
                w = lsb.bit_length() - 1
                if u < w:
                    edges.append((u, w))
                v ^= lsb
        
        # Shuffle edges based on seed to explore different orderings
        # Or sort by degree sum? Let's try shuffling for diversity in restarts.
        # A fixed good ordering might be better, but random restarts allow exploration.
        rng.shuffle(edges)
        
        cliques = [] # List of lists of vertices
        clique_masks = [] # List of bitmasks of vertices
        
        for u, v in edges:
            placed = False
            # Try to place in existing cliques
            # Iterate in reverse? Or forward? 
            # Forward is standard. 
            for i in range(len(cliques)):
                mask = clique_masks[i]
                # Check if u and v connect to all vertices in mask
                # adj[u] & mask == mask means u is connected to all in mask
                # adj[v] & mask == mask means v is connected to all in mask
                if (adj[u] & mask) == mask and (adj[v] & mask) == mask:
                    # Valid insertion
                    cliques[i].append(u)
                    cliques[i].append(v)
                    # Update mask
                    clique_masks[i] = mask | (1 << u) | (1 << v)
                    placed = True
                    break
            
            if not placed:
                # Open new clique
                cliques.append([u, v])
                clique_masks.append((1 << u) | (1 << v))
        
        return cliques

    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    adj_1 = instance(32, 0.6, 601)
    adj_2 = instance(64, 0.55, 602)
    
    best_cliques_1 = None
    best_cliques_2 = None
    best_len_1 = float('inf')
    best_len_2 = float('inf')
    
    seed_counter = 0
    while time.time() < deadline:
        # Run greedy for instance 1
        cliques_1 = greedy_line_graph_coloring(adj_1, 32, seed=seed_counter)
        if len(cliques_1) < best_len_1:
            best_len_1 = len(cliques_1)
            best_cliques_1 = cliques_1
        
        # Run greedy for instance 2
        cliques_2 = greedy_line_graph_coloring(adj_2, 64, seed=seed_counter)
        if len(cliques_2) < best_len_2:
            best_len_2 = len(cliques_2)
            best_cliques_2 = cliques_2
        
        seed_counter += 1
        
        # Check time more frequently if many seeds
        if seed_counter % 100 == 0 and time.time() > deadline:
            break

    if best_cliques_1 is None:
        # Fallback: individual edges
        cliques = []
        for u in range(32):
            v = adj_1[u]
            while v:
                lsb = v & -v
                w = lsb.bit_length() - 1
                if u < w:
                    cliques.append([u, w])
                v ^= lsb
        best_cliques_1 = cliques
        
    if best_cliques_2 is None:
        cliques = []
        for u in range(64):
            v = adj_2[u]
            while v:
                lsb = v & -v
                w = lsb.bit_length() - 1
                if u < w:
                    cliques.append([u, w])
                v ^= lsb
        best_cliques_2 = cliques

    return [best_cliques_1, best_cliques_2]

if __name__ == "__main__":
    res = construct()
    print(len(res[0]), len(res[1]))
