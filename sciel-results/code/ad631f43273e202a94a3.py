"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def _compute_max_book(adj_masks, edge_list, n=40):
    """Compute the largest book size using bitmask adjacency."""
    max_book = 0
    for u, v in edge_list:
        common = (adj_masks[u] & adj_masks[v]).bit_count()
        if common > max_book:
            max_book = common
    return max_book

def _build_structural_init(a_size, seed=42):
    """
    Build graph with complete bipartite A-B + triangle-free subgraph on B.
    A = vertices 0..a_size-1, B = vertices a_size..39.
    """
    random.seed(seed)
    n = 40
    target_edges = 401
    
    A = list(range(a_size))
    B = list(range(a_size, n))
    
    edges = []
    edge_set = set()
    
    # 1. Add all A-B edges
    for a in A:
        for b in B:
            edges.append((a, b))
            edge_set.add((a, b))
    
    needed_in_B = target_edges - len(edges)
    
    # 2. Split B into B1 and B2 to make B-subgraph bipartite (triangle-free)
    mid = len(B) // 2
    B1 = B[:mid]
    B2 = B[mid:]
    
    # Generate all potential B1-B2 edges
    potential = []
    for u in B1:
        for v in B2:
            potential.append((u, v))
    
    random.shuffle(potential)
    
    # Greedy add edges to keep degrees balanced
    deg_B = {v: 0 for v in B}
    added = 0
    
    # Sort potential edges by current degree sum to balance
    def degree_sum(edge):
        return deg_B[edge[0]] + deg_B[edge[1]]
    
    for u, v in potential:
        if added == needed_in_B:
            break
        edges.append((u, v))
        edge_set.add((u, v))
        deg_B[u] += 1
        deg_B[v] += 1
        added += 1
    
    # If we didn't add enough, add remaining (shouldn't happen)
    if added < needed_in_B:
        for u, v in potential:
            if added == needed_in_B:
                break
            if (u, v) not in edge_set:
                edges.append((u, v))
                edge_set.add((u, v))
                deg_B[u] += 1
                deg_B[v] += 1
                added += 1
    
    return edges, edge_set

def _sa_optimize(initial_edges, n=40, time_limit=105.0, seed=42):
    """Simulated annealing with efficient bitmask updates."""
    start_time = time.time()
    random.seed(seed)
    
    current_edges = [list(e) for e in initial_edges]
    edge_set = set((min(u,v), max(u,v)) for u,v in current_edges)
    
    # Build adjacency masks
    adj = [0] * n
    for u, v in current_edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    # Compute initial book sizes for all edges
    book_sizes = {}
    current_max = 0
    for u, v in current_edges:
        key = (min(u,v), max(u,v))
        bs = (adj[u] & adj[v]).bit_count()
        book_sizes[key] = bs
        if bs > current_max:
            current_max = bs
            
    best_edges = [list(e) for e in current_edges]
    best_max = current_max
    
    # Precompute all possible edges for random selection
    all_possible = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    # SA parameters
    T0 = 3.0
    Tend = 0.01
    alpha = 0.9998
    
    T = T0
    no_improve = 0
    
    while T > Tend and time.time() - start_time < time_limit:
        # Pick a random edge to remove
        remove_idx = random.randint(0, len(current_edges) - 1)
        ru, rv = current_edges[remove_idx]
        rkey = (min(ru, rv), max(ru, rv))
        
        # Pick a random non-edge to add
        ckey = None
        for _ in range(20):
            cu, cv = random.choice(all_possible)
            ckey = (min(cu, cv), max(cu, cv))
            if ckey not in edge_set:
                break
        if ckey is None or ckey in edge_set:
            continue
            
        cu, cv = ckey
        
        # Identify affected edges: all edges incident to ru, rv, cu, cv
        affected = set()
        affected.add(rkey)
        affected.add(ckey)
        
        for v in range(n):
            if v in (ru, rv, cu, cv):
                continue
            for w in (ru, rv, cu, cv):
                k = (min(v, w), max(v, w))
                if k in edge_set:
                    affected.add(k)
        
        # Save old book sizes for affected edges
        old_books = {}
        for k in affected:
            if k in book_sizes:
                old_books[k] = book_sizes[k]
        
        # Apply swap
        # Remove rkey
        del book_sizes[rkey]
        edge_set.remove(rkey)
        adj[ru] &= ~(1 << rv)
        adj[rv] &= ~(1 << ru)
        
        # Add ckey
        edge_set.add(ckey)
        adj[cu] |= (1 << cv)
        adj[cv] |= (1 << cu)
        current_edges[remove_idx] = [cu, cv]
        
        # Recompute book sizes for affected edges
        new_max_affected = 0
        for k in affected:
            if k in edge_set:
                u, v = k
                bs = (adj[u] & adj[v]).bit_count()
                book_sizes[k] = bs
                if bs > new_max_affected:
                    new_max_affected = bs
            else:
                if k in book_sizes:
                    del book_sizes[k]
        
        # Determine new global max
        if rkey in old_books and old_books[rkey] == current_max:
            # The removed edge was a max edge, need to recompute global max
            new_max = max(book_sizes.values()) if book_sizes else 0
        else:
            # Check if any affected edge now exceeds current_max
            new_max = current_max
            if new_max_affected > new_max:
                new_max = new_max_affected
        
        # Accept or reject
        if new_max <= current_max:
            # Accept (improvement or neutral)
            current_max = new_max
            if new_max < best_max:
                best_max = new_max
                best_edges = [list(e) for e in current_edges]
                no_improve = 0
            else:
                no_improve += 1
        else:
            # Reject with probability exp(-(new_max - current_max)/T)
            delta = new_max - current_max
            if random.random() < (2.718281828 ** (-delta / T)):
                # Accept worse solution
                current_max = new_max
                no_improve += 1
            else:
                # Revert
                # Remove ckey
                if ckey in book_sizes:
                    del book_sizes[ckey]
                edge_set.remove(ckey)
                adj[cu] &= ~(1 << cv)
                adj[cv] &= ~(1 << cu)
                current_edges[remove_idx] = [ru, rv]
                
                # Add rkey back
                edge_set.add(rkey)
                adj[ru] |= (1 << rv)
                adj[rv] |= (1 << ru)
                
                # Restore book sizes
                for k, bs in old_books.items():
                    if k in edge_set:
                        book_sizes[k] = bs
        
        # Cool down
        T *= alpha
        
        # Periodic restart if stuck
        if no_improve > 500:
            # Perturb: make 5 random swaps
            for _ in range(5):
                ri = random.randint(0, len(current_edges) - 1)
                ru2, rv2 = current_edges[ri]
                rk2 = (min(ru2, rv2), max(ru2, rv2))
                
                ck2 = None
                for _ in range(20):
                    cu2, cv2 = random.choice(all_possible)
                    ck2 = (min(cu2, cv2), max(cu2, cv2))
                    if ck2 not in edge_set:
                        break
                if ck2 is None or ck2 in edge_set:
                    continue
                
                # Simple swap without full book size tracking
                edge_set.remove(rk2)
                edge_set.add(ck2)
                adj[ru2] &= ~(1 << rv2)
                adj[rv2] &= ~(1 << ru2)
                adj[cu2] |= (1 << cv2)
                adj[cv2] |= (1 << cu2)
                current_edges[ri] = [cu2, cv2]
            
            # Recompute all book sizes
            book_sizes = {}
            current_max = 0
            for u, v in current_edges:
                key = (min(u,v), max(u,v))
                bs = (adj[u] & adj[v]).bit_count()
                book_sizes[key] = bs
                if bs > current_max:
                    current_max = bs
            
            if current_max < best_max:
                best_max = current_max
                best_edges = [list(e) for e in current_edges]
            
            no_improve = 0
            T = max(T, 1.0)
    
    return best_edges, best_max

def construct():
    start_time = time.time()
    n = 40
    target_edges = 401
    time_limit = 115.0  # Safety margin before 120s kill
    
    # Try multiple structural initializations with different A sizes
    best_overall = None
    best_overall_max = float('inf')
    
    # Primary: a=9 gives max book ~9
    a_sizes_to_try = [9, 10, 8]
    
    for a_size in a_sizes_to_try:
        if time.time() - start_time > time_limit - 5.0:
            break
        
        # Build structural initialization
        initial_edges, edge_set = _build_structural_init(a_size, seed=a_size)
        
        if len(initial_edges) != target_edges:
            continue
        
        # Run simulated annealing
        remaining_time = time_limit - (time.time() - start_time) - 3.0
        if remaining_time < 5.0:
            break
        
        best_edges, best_max = _sa_optimize(
            initial_edges, n=n, time_limit=remaining_time, seed=a_size * 1000
        )
        
        if best_max < best_overall_max:
            best_overall_max = best_max
            best_overall = best_edges
    
    # Fallback if something went wrong
    if best_overall is None or len(best_overall) != target_edges:
        # Use the a=9 structural init directly
        initial_edges, _ = _build_structural_init(9, seed=42)
        best_overall = [list(e) for e in initial_edges]
    
    # Ensure exactly target_edges edges
    if len(best_overall) != target_edges:
        # Truncate or pad (shouldn't happen)
        best_overall = best_overall[:target_edges]
    
    return best_overall

# EVOLVE-BLOCK-END
