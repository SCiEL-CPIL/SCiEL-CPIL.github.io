import random
import time
import itertools

def is_k4_free(adj):
    n = len(adj)
    for a in range(n):
        for b in range(a+1, n):
            if not (adj[a] & (1 << b)): continue
            for c in range(b+1, n):
                if not (adj[a] & (1 << c)) or not (adj[b] & (1 << c)): continue
                if adj[a] & adj[b] & adj[c]:
                    return False
    return True

def _has_triangle_in_subset(adj, subset_mask, subset_list):
    """Check if the induced subgraph on subset_list (with mask subset_mask) has a triangle."""
    for i, u in enumerate(subset_list):
        # Only consider edges (u, v) where v comes after u in subset_list
        for j in range(i+1, len(subset_list)):
            v = subset_list[j]
            if adj[u] & (1 << v):
                # Check if u and v share a common neighbor in the subset
                if adj[u] & adj[v] & subset_mask:
                    return True
    return False

def get_largest_triangle_free(adj, upper_bound=None):
    n = len(adj)
    start_k = upper_bound if upper_bound is not None else n
    for k in range(start_k, 0, -1):
        for subset in itertools.combinations(range(n), k):
            mask = 0
            for v in subset:
                mask |= (1 << v)
            if not _has_triangle_in_subset(adj, mask, subset):
                return k
    return 0

def has_tf_subset_of_size(adj, target_size, n):
    """Check if there exists a triangle-free induced subgraph of size >= target_size."""
    for k in range(target_size, n+1):
        for subset in itertools.combinations(range(n), k):
            mask = 0
            for v in subset:
                mask |= (1 << v)
            if not _has_triangle_in_subset(adj, mask, subset):
                return True
    return False

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    n = 18
    best_l = 18
    best_graph = None
    
    # Helper to check K4 free and return violation if any
    def find_k4(adj):
        for a in range(n):
            for b in range(a+1, n):
                if not (adj[a] & (1 << b)): continue
                for c in range(b+1, n):
                    if not (adj[a] & (1 << c)) or not (adj[b] & (1 << c)): continue
                    common = adj[a] & adj[b] & adj[c]
                    if common:
                        return (a, b, c, (common & -common).bit_length() - 1)
        return None

    def gen_random_k4_free():
        adj = [0] * n
        # Start with random p=0.6
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < 0.6:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        # Repair
        for _ in range(50):
            viol = find_k4(adj)
            if not viol:
                return adj
            a, b, c, d = viol
            # Remove edge (a, d)
            adj[a] &= ~(1 << d)
            adj[d] &= ~(1 << a)
        return adj

    # Try multiple random restarts for initial graph
    for _ in range(20):
        if time.time() > deadline:
            break
        adj = gen_random_k4_free()
        l = get_largest_triangle_free(adj)
        if l < best_l:
            best_l = l
            best_graph = adj[:]
    
    # If no valid graph found yet, use first one
    if best_graph is None:
        adj = gen_random_k4_free()
        best_l = get_largest_triangle_free(adj)
        best_graph = adj[:]
    
    adj = best_graph[:]
    
    # Local search with decision-based evaluation
    consecutive_rejections = 0
    while time.time() < deadline:
        # Local search: try flipping edges
        u = random.randint(0, n-1)
        v = random.randint(0, n-1)
        if u == v: continue
        
        # Check if edge exists
        has_edge = bool(adj[u] & (1 << v))
        
        # Try flipping
        new_adj = adj[:]
        if has_edge:
            new_adj[u] &= ~(1 << v)
            new_adj[v] &= ~(1 << u)
        else:
            new_adj[u] |= (1 << v)
            new_adj[v] |= (1 << u)
            
        # Check K4 free
        if not is_k4_free(new_adj):
            continue
        
        # Use decision check: can we beat best_l?
        # If there's no TF subset of size best_l, then new_l < best_l
        if not has_tf_subset_of_size(new_adj, best_l, n):
            # new_l < best_l, compute exact value
            new_l = get_largest_triangle_free(new_adj, upper_bound=best_l - 1)
            if new_l < best_l:
                best_l = new_l
                best_graph = new_adj
                adj = new_adj
                consecutive_rejections = 0
            else:
                # Shouldn't happen, but be safe
                adj = new_adj
        else:
            # L >= best_l, compute to see if equal or worse
            # Only compute full L occasionally to allow exploration
            if random.random() < 0.3:
                new_l = get_largest_triangle_free(new_adj)
                if new_l == best_l:
                    # Accept to explore
                    if random.random() < 0.4:
                        adj = new_adj
                        consecutive_rejections = 0
                    else:
                        consecutive_rejections += 1
                else:
                    consecutive_rejections += 1
            else:
                consecutive_rejections += 1
        
        # Random restart if stuck
        if consecutive_rejections > 200:
            adj = gen_random_k4_free()
            l = get_largest_triangle_free(adj)
            if l < best_l:
                best_l = l
                best_graph = adj[:]
            consecutive_rejections = 0

    # Convert to edges
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if best_graph[i] & (1 << j):
                edges.append([i, j])
    return edges
