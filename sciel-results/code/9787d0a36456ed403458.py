import time
import random

def _is_triangle_free(adj, u, v):
    """Check if adding edge (u, v) creates a triangle."""
    # If u and v are already connected, it's not a triangle creation issue, but we only call this for non-edges
    # Triangle exists if they share a common neighbor
    common = adj[u] & adj[v]
    return len(common) == 0

def _greedy_add_edges(adj, n, deadline):
    """
    Greedily add edges to adj (list of sets) to maximize density 
    while maintaining triangle-free property.
    Returns the final adj.
    """
    # List of all possible non-edges
    potential_edges = [(i, j) for i in range(n) for j in range(i+1, n) if j not in adj[i]]
    
    # Shuffle to break symmetry and find different local maxima
    random.shuffle(potential_edges)
    
    changed = True
    while changed and time.time() < deadline:
        changed = False
        # We iterate through a shuffled list each time? Or just try to add any valid edge?
        # For speed, we can just scan the potential list.
        # Since n=16, potential edges <= 120.
        # We can just loop through all pairs and try to add if valid.
        # To be robust, we do a pass.
        for u, v in potential_edges:
            if v in adj[u]:
                continue
            if _is_triangle_free(adj, u, v):
                adj[u].add(v)
                adj[v].add(u)
                changed = True
                # Remove from potential list? Or just let the 'v in adj[u]' check handle it.
                # The check 'if v in adj[u]' is fast.
                
    return adj

def _max_independent_set(adj, n):
    """
    Find a maximum independent set in a graph with n <= 16.
    Uses Bron-Kerbosch with pivot or simple branch and bound.
    """
    # For n=15, a simple backtracking is fast enough.
    # We want the largest set S such that no two vertices in S are adjacent.
    # Equivalent to Max Clique in complement graph.
    
    # Complement adjacency
    comp_adj = [set() for _ in range(n)]
    for u in range(n):
        for v in range(u+1, n):
            if v not in adj[u]:
                comp_adj[u].add(v)
                comp_adj[v].add(u)
    
    best_clique = set()
    
    def bronk(R, P, X):
        nonlocal best_clique
        if not P and not X:
            if len(R) > len(best_clique):
                best_clique = set(R)
            return
        
        # Pivot selection: choose u in P U X with max degree in P
        pivot_set = P | X
        if not pivot_set:
            return
        u = max(pivot_set, key=lambda x: len(comp_adj[x] & P))
        
        # Candidates: P \ N(u)
        candidates = P - comp_adj[u]
        
        for v in candidates:
            bronk(R | {v}, P & comp_adj[v], X & comp_adj[v])
            P.remove(v)
            X.add(v)
            
    bronk(set(), set(range(n)), set())
    
    # best_clique in complement is Independent Set in original
    return best_clique

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    n = 16
    
    # 1. Construct Seed: Mycielski graph of K_{2,2,2,1}
    # K_{2,2,2,1} vertices 0..6
    # Parts: A={0,1}, B={2,3}, C={4,5}, D={6}
    base_edges = []
    # Edges between A and B
    for u in [0, 1]:
        for v in [2, 3]:
            base_edges.append((u, v))
    # Edges between A and C
    for u in [0, 1]:
        for v in [4, 5]:
            base_edges.append((u, v))
    # Edges between A and D
    for u in [0, 1]:
        base_edges.append((u, 6))
    # Edges between B and C
    for u in [2, 3]:
        for v in [4, 5]:
            base_edges.append((u, v))
    # Edges between B and D
    for u in [2, 3]:
        base_edges.append((u, 6))
    # Edges between C and D
    for u in [4, 5]:
        base_edges.append((u, 6))
        
    # Build Adjacency for Base (7 vertices)
    base_adj = [set() for _ in range(7)]
    for u, v in base_edges:
        base_adj[u].add(v)
        base_adj[v].add(u)
        
    # Mycielski Construction
    # Original: 0..6
    # Copies: 7..13 (i -> i+7)
    # Apex: 14
    # New N = 15
    m_adj = [set() for _ in range(15)]
    
    # 1. Copy original edges
    for u in range(7):
        for v in base_adj[u]:
            if u < v: # avoid duplicates if we iterate all, but sets handle it
                m_adj[u].add(v)
                m_adj[v].add(u)
                
    # 2. Connect copies to original neighbors
    # v' is connected to N(v)
    for v in range(7):
        v_copy = v + 7
        for u in base_adj[v]:
            m_adj[v_copy].add(u)
            m_adj[u].add(v_copy)
            
    # 3. Connect Apex to all copies
    apex = 14
    for v in range(7):
        v_copy = v + 7
        m_adj[apex].add(v_copy)
        m_adj[v_copy].add(apex)
        
    # 4. Add 16th vertex (15)
    # Extend adj to size 16
    full_adj = [set() for _ in range(16)]
    for i in range(15):
        full_adj[i] = m_adj[i]
        
    # Connect vertex 15 to Max Independent Set of m_adj
    # MIS of m_adj (15 vertices)
    # Note: m_adj is triangle-free, so MIS is well-defined.
    mis = _max_independent_set(m_adj, 15)
    
    for v in mis:
        full_adj[15].add(v)
        full_adj[v].add(15)
        
    # 5. Greedy Edge Addition
    # We try multiple random shuffles to escape local optima
    best_edge_count = sum(len(s) for s in full_adj) // 2
    best_adj = [set(s) for s in full_adj]
    
    # Run greedy optimization a few times
    for _ in range(50):
        if time.time() > deadline:
            break
        curr_adj = [set(s) for s in best_adj]
        curr_adj = _greedy_add_edges(curr_adj, 16, deadline)
        curr_count = sum(len(s) for s in curr_adj) // 2
        if curr_count > best_edge_count:
            best_edge_count = curr_count
            best_adj = curr_adj
            
    # Convert to list of edges
    edges = []
    for u in range(16):
        for v in best_adj[u]:
            if u < v:
                edges.append([u, v])
                
    return edges
