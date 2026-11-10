import time
import random
from itertools import combinations

START_TIME = time.time()
TIME_LIMIT = 145  # safety margin before 150s kill

def build_adj(edges, n=16):
    adj = [0] * n
    for u, v in edges:
        adj[u] |= 1 << v
        adj[v] |= 1 << u
    return adj

def is_triangle_free_bitmask(adj, n=16):
    for u in range(n):
        mask = adj[u]
        while mask:
            lsb = mask & -mask
            v = lsb.bit_length() - 1
            mask ^= lsb
            common = adj[v] & adj[u]
            if common & ~(1 << u) & ~(1 << v):
                return False
    return True

def is_not_3_colorable_exact(adj, n=16):
    """Exact backtracking 3-colorability check."""
    colors = [-1] * n
    
    def dfs(uncolored_mask):
        if uncolored_mask == 0:
            return False
        
        # Select vertex with most colored neighbors (MRV heuristic)
        best_v = -1
        best_score = -1
        m = uncolored_mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            m ^= lsb
            score = 0
            nb_mask = adj[v]
            mm = nb_mask
            while mm:
                lsb_nb = mm & -mm
                nb = lsb_nb.bit_length() - 1
                mm ^= lsb_nb
                if colors[nb] != -1:
                    score += 1
            if score > best_score:
                best_score = score
                best_v = v
        
        v = best_v
        
        forbidden = 0
        nb_mask = adj[v]
        mm = nb_mask
        while mm:
            lsb_nb = mm & -mm
            nb = lsb_nb.bit_length() - 1
            mm ^= lsb_nb
            if colors[nb] != -1:
                forbidden |= 1 << colors[nb]
        
        for c in range(3):
            if not (forbidden & (1 << c)):
                colors[v] = c
                if not dfs(uncolored_mask & ~(1 << v)):
                    return False
                colors[v] = -1
        
        return True
    
    initial_mask = (1 << n) - 1
    return dfs(initial_mask)

def greedy_3_colorable(adj, n=16, max_trials=30):
    """Quick greedy check - if it succeeds, graph is 3-colorable."""
    for _ in range(max_trials):
        order = list(range(n))
        random.shuffle(order)
        colors = [-1] * n
        for node in order:
            used = 0
            nb_mask = adj[node]
            mm = nb_mask
            while mm:
                lsb = mm & -mm
                nb = lsb.bit_length() - 1
                mm ^= lsb
                if colors[nb] != -1:
                    used |= 1 << colors[nb]
            for c in range(3):
                if not (used & (1 << c)):
                    colors[node] = c
                    break
            else:
                break
        else:
            return True
    return False

def is_not_3_colorable_fast(adj, n=16):
    """Fast check: greedy first, then exact if needed."""
    if greedy_3_colorable(adj, n, max_trials=15):
        return False
    return is_not_3_colorable_exact(adj, n)

def build_clebsch_edges(n=16):
    """Build edges of the Clebsch graph (16 vertices, 40 edges)."""
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            xor = u ^ v
            dist = bin(xor).count('1')
            if dist in (1, 4):
                edges.append([u, v])
    return edges

def get_clebsch_adj(n=16):
    edges = build_clebsch_edges(n)
    adj = [0] * n
    for u, v in edges:
        adj[u] |= 1 << v
        adj[v] |= 1 << u
    return adj

def generate_random_triangle_free(n=16, edge_prob=0.3, seed=None):
    """Generate a random triangle-free graph."""
    if seed is not None:
        random.seed(seed)
    adj = [0] * n
    edges = []
    for u in range(n):
        for v in range(u+1, n):
            if random.random() < edge_prob:
                # Check if adding this edge creates a triangle
                if (adj[u] & adj[v]) == 0:
                    adj[u] |= 1 << v
                    adj[v] |= 1 << u
                    edges.append([u, v])
    return adj, edges

def local_search(adj, current_edges, n=16, deadline=None, max_iterations=1000):
    """Local search: try to add edges while maintaining constraints."""
    improved = True
    iterations = 0
    while improved and iterations < max_iterations and (deadline is None or time.time() < deadline):
        improved = False
        iterations += 1
        
        # Find all candidate edges that would maintain triangle-free
        candidates = []
        for u in range(n):
            for v in range(u+1, n):
                if not (adj[u] & (1 << v)):
                    if (adj[u] & adj[v]) == 0:
                        candidates.append((u, v))
        
        random.shuffle(candidates)
        
        for u, v in candidates:
            if deadline and time.time() > deadline:
                return current_edges
            
            # Try adding edge
            adj[u] |= 1 << v
            adj[v] |= 1 << u
            current_edges.append([u, v])
            
            if is_not_3_colorable_fast(adj, n):
                improved = True
            else:
                # Revert
                adj[u] ^= 1 << v
                adj[v] ^= 1 << u
                current_edges.pop()
                
        if not improved:
            break
    
    return current_edges

def construct():
    global START_TIME, TIME_LIMIT
    START_TIME = time.time()
    TIME_LIMIT = 145
    
    n = 16
    deadline = START_TIME + TIME_LIMIT
    
    best_edges = []
    best_count = 0
    
    # Strategy 1: Start from Clebsch graph
    clebsch_adj = get_clebsch_adj(n)
    clebsch_edges = build_clebsch_edges(n)
    
    if is_triangle_free_bitmask(clebsch_adj, n) and is_not_3_colorable_fast(clebsch_adj, n):
        current_edges = list(clebsch_edges)
        # Intensive local search from this strong seed
        for trial in range(20):
            if time.time() > deadline:
                break
            random.seed(100 + trial * 13)
            current_edges = local_search(clebsch_adj, current_edges, n, deadline, max_iterations=300)
            if len(current_edges) > best_count:
                best_count = len(current_edges)
                best_edges = list(current_edges)
                clebsch_adj = build_adj(best_edges, n)
                
    # Strategy 2: Random restarts with dense triangle-free graphs
    for trial in range(10):
        if time.time() > deadline:
            break
            
        random.seed(42 + trial * 100)
        adj, edges = generate_random_triangle_free(n, edge_prob=0.35, seed=42 + trial * 100)
        
        if not is_triangle_free_bitmask(adj, n):
            continue
            
        if is_not_3_colorable_fast(adj, n):
            current_edges = list(edges)
            current_edges = local_search(adj, current_edges, n, deadline, max_iterations=300)
            if len(current_edges) > best_count:
                best_count = len(current_edges)
                best_edges = list(current_edges)
        else:
            # Graph is 3-colorable, try to make it non-3-colorable by adding edges
            # This is harder; skip for now
            pass
    
    # Strategy 3: If we have a good candidate, do more intensive local search
    if best_edges:
        adj = build_adj(best_edges, n)
        current_edges = list(best_edges)
        
        # Multiple rounds of local search with different random seeds
        for trial in range(5):
            if time.time() > deadline:
                break
            random.seed(1000 + trial * 777)
            current_edges = local_search(adj, current_edges, n, deadline, max_iterations=500)
            if len(current_edges) > best_count:
                best_count = len(current_edges)
                best_edges = list(current_edges)
                adj = build_adj(best_edges, n)
    
    # Strategy 4: If still not good enough, try more aggressive random generation
    if best_count < 40:
        for trial in range(20):
            if time.time() > deadline:
                break
            
            random.seed(5000 + trial * 13)
            adj, edges = generate_random_triangle_free(n, edge_prob=0.4, seed=5000 + trial * 13)
            
            if not is_triangle_free_bitmask(adj, n):
                continue
                
            # Check if non-3-colorable
            if is_not_3_colorable_fast(adj, n):
                current_edges = list(edges)
                current_edges = local_search(adj, current_edges, n, deadline, max_iterations=200)
                if len(current_edges) > best_count:
                    best_count = len(current_edges)
                    best_edges = list(current_edges)
    
    # Final safety check
    if best_edges:
        adj = build_adj(best_edges, n)
        if not is_triangle_free_bitmask(adj, n):
            # Fallback to a known valid graph
            base_edges = build_groetzsch_edges()
            adj = build_adj(base_edges, n)
            current_edges = list(base_edges)
            for v in range(11, 16):
                candidates = list(range(11))
                random.shuffle(candidates)
                for c in candidates:
                    if time.time() > deadline:
                        return current_edges
                    if adj[v] & adj[c] == 0:
                        current_edges.append([v, c])
                        adj[v] |= 1 << c
                        adj[c] |= 1 << v
            best_edges = current_edges
    
    return best_edges
