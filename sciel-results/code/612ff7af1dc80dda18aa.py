import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    # Moore bound calculation for lower bound
    # For even girth g=2m, n >= k + k(k-1) + ... + k(k-1)^(m-1) + (k-1)(k-1)^(m-1)... wait
    # Standard Moore bound for k-regular, girth g:
    # If g is even (g=2m): n >= 2 * sum_{i=0}^{m-1} (k-1)^i
    # If g is odd (g=2m+1): n >= 1 + k * sum_{i=0}^{m-1} (k-1)^i
    if g % 2 == 0:
        m = g // 2
        moore = 2 * sum((k - 1) ** i for i in range(m))
    else:
        m = (g - 1) // 2
        moore = 1 + k * sum((k - 1) ** i for i in range(m))
        
    # Candidate n values to try
    # We know for k=4, g=10, best is 320.
    # Let's try n from moore to moore + 200
    candidates = []
    for n in range(moore, moore + 300):
        # Check if n can be decomposed into Nb * M
        # We need a k-regular base graph of size Nb
        # Small k-regular graphs exist for Nb >= k+1 (if k+1 is even for bipartite? No, just regular)
        # k=4: Nb can be 5, 6, 8, 10, 12...
        # Let's collect valid Nb
        valid_nb = []
        for nb in range(k + 1, n + 1):
            if n % nb == 0:
                # Check if a k-regular graph of size nb exists
                # Necessary condition: nb * k is even
                if (nb * k) % 2 == 0:
                    # Sufficient for small nb? 
                    # For nb=k+1, only K_{k+1} works (degree k).
                    # For nb > k+1, usually yes.
                    # We will generate specific base graphs for known small sizes
                    valid_nb.append(nb)
        
        # Prioritize smaller Nb for larger M (more randomness?) or specific known good combos
        # Actually, smaller Nb means smaller base, harder to avoid short cycles?
        # Larger Nb means more structure.
        # Let's just iterate
        if valid_nb:
            candidates.append((n, valid_nb))
            
    # Sort candidates by n (smallest first)
    candidates.sort(key=lambda x: x[0])
    
    # Base graph generators
    def get_base_graph(nb, k):
        """Returns list of edges for a k-regular graph on nb vertices (0..nb-1)"""
        if nb == k + 1:
            # Complete graph K_{k+1}
            edges = []
            for i in range(nb):
                for j in range(i + 1, nb):
                    edges.append((i, j))
            return edges
        elif nb == 8 and k == 4:
            # Cube Q3 or K_{4,4}
            # Let's use K_{4,4}
            edges = []
            for i in range(4):
                for j in range(4):
                    edges.append((i, 4 + j))
            return edges
        elif nb == 10 and k == 4:
            # K_{10} minus K_{5,5} (bipartite complement)
            # Vertices 0..4 and 5..9
            # Edges within 0..4 and within 5..9
            edges = []
            for i in range(5):
                for j in range(i + 1, 5):
                    edges.append((i, j))
            for i in range(5, 10):
                for j in range(i + 1, 10):
                    edges.append((i, j))
            return edges
        elif nb == 6 and k == 4:
            # K6 minus perfect matching
            edges = []
            for i in range(6):
                for j in range(i + 1, 6):
                    if (i % 2 == 0 and j % 2 == 1 and abs(i-j) < 2) or (i % 2 == 1 and j % 2 == 0 and abs(i-j) < 2):
                        # This logic is messy. 
                        # Perfect matching: (0,1), (2,3), (4,5)
                        pass
            # Explicit:
            # 0-2, 0-3, 0-4, 0-5
            # 1-2, 1-3, 1-4, 1-5
            # 2-4, 2-5
            # 3-4, 3-5
            # 4-0, 4-1, 4-2, 4-3
            # 5-0, 5-1, 5-2, 5-3
            # Let's just use a standard construction or skip if complex
            # K6 is 5-reg. Remove (0,1), (2,3), (4,5).
            all_edges = [(i, j) for i in range(6) for j in range(i + 1, 6)]
            match = {(0,1), (2,3), (4,5)}
            edges = [e for e in all_edges if e not in match]
            return edges
        else:
            return None

    def check_girth(edges, n, g):
        """Returns True if girth >= g"""
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        for start in range(n):
            # BFS
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            q = deque([start])
            
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        q.append(v)
                    elif v != parent[u]:
                        # Found a cycle
                        # Length is dist[u] + dist[v] + 1
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < g:
                            return False
            # If BFS completes without finding cycle < g, continue to next start
            # Note: If graph is connected, one BFS might not cover all? 
            # No, BFS from start covers all reachable. 
            # We need to check ALL starts to ensure no short cycle anywhere.
            # Optimization: If we find a cycle of length < g, return False.
            # If we finish all BFS and no cycle < g found, return True.
            
        return True

    def generate_cover(base_edges, nb, M, seed):
        """
        Generates a cover graph using Z_M as voltage group.
        Base vertices 0..nb-1.
        Cover vertices (v, g) -> v * M + g.
        """
        random.seed(seed)
        # Assign voltages to base edges
        # We need to handle orientation. 
        # Let's orient base edges (u, v) with u < v.
        # Voltage w_e in Z_M.
        # Cover edge: (u, g) -- (v, g + w) if u < v.
        # And (v, g) -- (u, g - w) ? 
        # In voltage graphs, usually we define a directed graph.
        # For undirected base, we can just assign a voltage to each undirected edge.
        # The cover is bipartite if base is? No.
        # Let's use the standard construction:
        # For each edge (u, v) in base, assign voltage w.
        # In cover: (u, g) is connected to (v, g + w).
        # This implies (v, g+w) is connected to (u, g).
        # So the edge is undirected.
        
        cover_edges = []
        # Map base edge to voltage
        edge_voltages = {}
        for u, v in base_edges:
            w = random.randint(0, M - 1)
            edge_voltages[(u, v)] = w
            # Note: (v, u) is same edge
            
        for u, v in base_edges:
            w = edge_voltages[(u, v)]
            for g in range(M):
                # Vertex u, layer g
                # Vertex v, layer (g + w) % M
                u_idx = u * M + g
                v_idx = v * M + ((g + w) % M)
                cover_edges.append((u_idx, v_idx))
                
        return cover_edges

    best_n = None
    best_edges = None
    
    # Search loop
    for n, valid_nbs in candidates:
        if time.time() > deadline - 20:
            break
            
        for nb in valid_nbs:
            if time.time() > deadline - 20:
                break
                
            base_edges = get_base_graph(nb, k)
            if base_edges is None:
                continue
                
            M = n // nb
            
            # Try a few random seeds
            # For small M, fewer seeds needed. For large M, more.
            attempts = 10 if M > 20 else 50
            
            for seed in range(attempts):
                if time.time() > deadline - 20:
                    break
                    
                cover_edges = generate_cover(base_edges, nb, M, seed)
                # Check girth
                if check_girth(cover_edges, n, g):
                    best_n = n
                    best_edges = cover_edges
                    return n, cover_edges

    # Fallback: LUW Graph from seed
    # Re-implement seed logic as fallback
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges
