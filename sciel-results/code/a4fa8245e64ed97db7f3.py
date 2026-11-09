import random
import time
import itertools
import sys

def construct():
    """
    Constructs a graph with no K5 and no independent 5-set.
    Returns (n, edges) where n is the number of vertices and edges is a list of (i, j) pairs.
    """
    start_time = time.time()
    deadline = start_time + 280  # Safety margin before 300s limit
    
    n_target = 42
    
    def get_paley_graph():
        q = 37
        residues = {(x * x) % q for x in range(1, q)}
        edges = [(i, j) for i in range(q) for j in range(i + 1, q) if (j - i) % q in residues]
        return q, edges

    def initialize_graph_from_paley():
        """
        Creates a 42-vertex graph by taking Paley(37) and adding 5 vertices.
        """
        q, paley_edges = get_paley_graph()
        n = q + 5
        adj = [0] * n
        
        # Add Paley edges
        for i, j in paley_edges:
            adj[i] |= (1 << j)
            adj[j] |= (1 << i)
            
        # Add 5 new vertices (37..41)
        # Connect them randomly to existing vertices and each other
        for i in range(q, n):
            for j in range(q):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
                    
        return n, adj

    def get_edges_from_adj(adj, n):
        edges = []
        for i in range(n):
            mask = adj[i]
            j = i + 1
            while j < n:
                if (mask >> j) & 1:
                    edges.append((i, j))
                j += 1
        return edges

    def find_violation_deterministic(adj, n):
        """
        Finds a 5-clique or 5-independent set deterministically.
        Returns ('clique', [v1..v5]) or ('ind', [v1..v5]) or None.
        """
        # Check for 5-cliques
        # A 5-clique contains a vertex v and a 4-clique in N(v)
        for v in range(n):
            neighbors = [u for u in range(n) if u != v and (adj[v] >> u) & 1]
            if len(neighbors) < 4:
                continue
            
            for combo in itertools.combinations(neighbors, 4):
                a, b, c, d = combo
                if ((adj[a] >> b) & 1) and ((adj[a] >> c) & 1) and ((adj[a] >> d) & 1) and \
                   ((adj[b] >> c) & 1) and ((adj[b] >> d) & 1) and ((adj[c] >> d) & 1):
                    return ('clique', [v, a, b, c, d])
        
        # Check for 5-independent sets
        # A 5-independent set contains a vertex v and a 4-independent set in M(v)
        for v in range(n):
            non_neighbors = [u for u in range(n) if u != v and not ((adj[v] >> u) & 1)]
            if len(non_neighbors) < 4:
                continue
            
            for combo in itertools.combinations(non_neighbors, 4):
                a, b, c, d = combo
                if not ((adj[a] >> b) & 1) and not ((adj[a] >> c) & 1) and not ((adj[a] >> d) & 1) and \
                   not ((adj[b] >> c) & 1) and not ((adj[b] >> d) & 1) and not ((adj[c] >> d) & 1):
                    return ('ind', [v, a, b, c, d])
        
        return None

    n = n_target
    adj = initialize_graph_from_paley()[1]
    
    while time.time() < deadline:
        viol = find_violation_deterministic(adj, n)
        
        if viol is None:
            edges = get_edges_from_adj(adj, n)
            return n, edges

        v_type, vertices = viol
        if v_type == 'clique':
            i, j = random.sample(vertices, 2)
            adj[i] &= ~(1 << j)
            adj[j] &= ~(1 << i)
        elif v_type == 'ind':
            i, j = random.sample(vertices, 2)
            adj[i] |= (1 << j)
            adj[j] |= (1 << i)

    return get_paley_graph()
