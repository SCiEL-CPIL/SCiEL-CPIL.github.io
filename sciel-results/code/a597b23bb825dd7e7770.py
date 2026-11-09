"""C6-free graph on 40 vertices: maximize edges via fast local search."""
import time
import random

N = 40
DEADLINE = 115.0  # safety margin under 120s wall-clock


def can_add_edge(adj, u, v):
    """
    Return True if adding edge (u,v) to the graph represented by adj (bitmasks)
    does NOT create a 6-cycle.
    
    A 6-cycle is created if there exists a path of length 5 between u and v.
    Path: u - x1 - y1 - y2 - x4 - v
    x1 in N(u), x4 in N(v)
    y1 in N(x1), y2 in N(x4)
    (y1, y2) is an edge.
    """
    nu = adj[u]
    nv = adj[v]
    
    # Iterate x1 in N(u)
    m_u = nu
    while m_u:
        lsb_u = m_u & (-m_u)
        x1 = lsb_u.bit_length() - 1
        m_u ^= lsb_u
        
        # Iterate x4 in N(v)
        m_v = nv
        while m_v:
            lsb_v = m_v & (-m_v)
            x4 = lsb_v.bit_length() - 1
            m_v ^= lsb_v
            
            # Check if exists y1 in N(x1) such that N(y1) intersects N(x4)
            nx1 = adj[x1]
            m_nx1 = nx1
            while m_nx1:
                lsb_nx1 = m_nx1 & (-m_nx1)
                y1 = lsb_nx1.bit_length() - 1
                m_nx1 ^= lsb_nx1
                
                # Check intersection of N(y1) and N(x4)
                if adj[y1] & adj[x4]:
                    return False # C6 found
        
    return True


def _edge_count(adj, n):
    c = 0
    for i in range(n):
        c += adj[i].bit_count()
    return c // 2


def construct():
    start = time.time()
    deadline = start + DEADLINE
    
    # Initialize empty graph
    adj = [0] * N
    
    def add_edge(u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    def remove_edge(u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        
    # 1. Greedy Random Construction
    # Shuffle vertex order to break symmetry
    vertices = list(range(N))
    random.seed(42)
    random.shuffle(vertices)
    
    # Try to add all possible edges in random order
    pairs = [(u, v) for u in range(N) for v in range(u + 1, N)]
    random.shuffle(pairs)
    
    for u, v in pairs:
        if time.time() > deadline - 30: # Reserve time for local search
            break
        if can_add_edge(adj, u, v):
            add_edge(u, v)
            
    best_edges = _edge_count(adj, N)
    
    # 2. Local Search
    # We will try to add edges, and if stuck, do swaps.
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try single edge additions
        # Shuffle pairs to avoid bias
        missing_pairs = [(u, v) for u in range(N) for v in range(u + 1, N) if not (adj[u] & (1 << v))]
        random.shuffle(missing_pairs)
        
        for u, v in missing_pairs:
            if time.time() > deadline:
                break
            if can_add_edge(adj, u, v):
                add_edge(u, v)
                best_edges = _edge_count(adj, N)
                improved = True
                # Restart loop to check new possibilities
                break
                
        if not improved:
            # Try 1-out-1-in swaps
            existing_edges = [(i, j) for i in range(N) for j in range(i + 1, N) if adj[i] & (1 << j)]
            missing_edges = [(i, j) for i in range(N) for j in range(i + 1, N) if not (adj[i] & (1 << j))]
            
            if not existing_edges or not missing_edges:
                break
                
            swap_found = False
            for ru, rv in existing_edges:
                if time.time() > deadline:
                    break
                remove_edge(ru, rv)
                
                for au, av in missing_edges:
                    # Skip if same edge
                    if (ru, rv) == (au, av) or (ru, rv) == (av, au):
                        continue
                    
                    if can_add_edge(adj, au, av):
                        add_edge(au, av)
                        swap_found = True
                        break
                        
                if swap_found:
                    break
                else:
                    # Restore
                    add_edge(ru, rv)
                    
            if not swap_found:
                break
            # If swap found, loop back to try additions

    # Return list of edges
    edges = []
    for i in range(N):
        for j in range(i + 1, N):
            if adj[i] & (1 << j):
                edges.append([i, j])
                
    return edges
