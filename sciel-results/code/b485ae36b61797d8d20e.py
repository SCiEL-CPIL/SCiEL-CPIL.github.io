"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def _compute_book_sizes(edges, n=40):
    """Compute the book size for each edge in the graph."""
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    edge_set = set()
    for u, v in edges:
        edge_set.add((min(u,v), max(u,v)))
        
    book_sizes = {}
    max_book = 0
    for u, v in edges:
        key = (min(u,v), max(u,v))
        common = adj[u] & adj[v]
        book_sizes[key] = len(common)
        if book_sizes[key] > max_book:
            max_book = book_sizes[key]
            
    return max_book, book_sizes, adj, edge_set

def _structural_init(n=40, target_edges=401, seed=42):
    """
    Create a structurally good initial graph using a 20/20 partition
    with degree-constrained intra-partition edges.
    
    Strategy:
    - Partition: A = 0..19 (20 vertices), B = 20..39 (20 vertices)
    - Add all A-B edges: 20*20 = 400 edges
    - Add 1 edge within B (or A) to reach 401
    - This is close to the extremal K20,20 + 1 edge construction
    """
    random.seed(seed)
    
    A = list(range(20))
    B = list(range(20, 40))
    
    edges = set()
    
    # Add all edges between A and B (400 edges)
    for a in A:
        for b in B:
            edges.add((a, b))
    
    needed = target_edges - len(edges)  # 401 - 400 = 1
    added = 0
    
    # Add 1 intra-B edge with minimum book size
    # For an edge (u,v) within B, book size = number of common neighbors in A
    # Since all A vertices connect to all B vertices, any intra-B edge has book size 20
    # So we just add one random intra-B edge
    B_pairs = [(u, v) for u in range(20, 40) for v in range(u+1, 40)]
    random.shuffle(B_pairs)
    
    for u, v in B_pairs:
        if added >= needed:
            break
        key = (min(u, v), max(u, v))
        if key not in edges:
            edges.add(key)
            added += 1
    
    # Convert to list of [u, v] pairs
    final_edges = []
    for u, v in edges:
        if u < v:
            final_edges.append([u, v])
        else:
            final_edges.append([v, u])
    
    # Ensure exactly target_edges
    if len(final_edges) > target_edges:
        final_edges = final_edges[:target_edges]
    elif len(final_edges) < target_edges:
        # Add more random edges
        all_possible = [(i, j) for i in range(n) for j in range(i+1, n)]
        existing = set((min(u,v), max(u,v)) for u, v in final_edges)
        random.shuffle(all_possible)
        for u, v in all_possible:
            if len(final_edges) >= target_edges:
                break
            key = (u, v)
            if key not in existing:
                final_edges.append([u, v])
                existing.add(key)
    
    return final_edges

def construct():
    start_time = time.time()
    n = 40
    target_edges = 401
    time_limit = 115.0  # Safety margin
    
    # Use structural initialization with 9/31 partition
    initial_edges = _structural_init(n, target_edges)
    
    max_book, book_sizes, adj, edge_set = _compute_book_sizes(initial_edges, n)
    
    best_edges = list(initial_edges)
    best_score = max_book
    
    # Simulated Annealing parameters
    T = 3.0
    T_min = 0.1
    cooling_rate = 0.9995
    
    current_edges = list(initial_edges)
    current_max = max_book
    current_book_sizes = dict(book_sizes)
    current_adj = {k: set(v) for k, v in adj.items()}
    current_edge_set = set(edge_set)
    
    # Build index from edge key to position in current_edges
    edge_index = {}
    for i, e in enumerate(current_edges):
        key = (min(e[0], e[1]), max(e[0], e[1]))
        edge_index[key] = i
    
    all_possible_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    # Track stagnation
    no_improve_count = 0
    stagnation_limit = 3000
    
    iterations = 0
    while time.time() - start_time < time_limit:
        iterations += 1
        
        # Check time more frequently
        if time.time() - start_time > time_limit - 5:
            break
            
        # Stagnation handling
        if no_improve_count > stagnation_limit:
            # Perturbation: make 5 random swaps
            for _ in range(5):
                idx = random.randint(0, len(current_edges) - 1)
                u1, v1 = current_edges[idx]
                rem_key = (min(u1, v1), max(u1, v1))
                
                while True:
                    add_u, add_v = random.choice(all_possible_edges)
                    add_key = (min(add_u, add_v), max(add_u, add_v))
                    if add_key not in current_edge_set:
                        break
                
                # Apply swap without checking acceptance
                current_edge_set.remove(rem_key)
                current_edge_set.add(add_key)
                current_adj[u1].discard(v1)
                current_adj[v1].discard(u1)
                current_adj[add_u].add(add_v)
                current_adj[add_v].add(add_u)
                current_edges[idx] = [add_u, add_v]
                
                # Update edge index
                del edge_index[rem_key]
                edge_index[add_key] = idx
                
                # Update book sizes
                affected = {rem_key, add_key}
                for w in list(current_adj[u1]):
                    k = (min(u1, w), max(u1, w))
                    if k in current_edge_set:
                        affected.add(k)
                for w in list(current_adj[v1]):
                    k = (min(v1, w), max(v1, w))
                    if k in current_edge_set:
                        affected.add(k)
                for w in list(current_adj[add_u]):
                    k = (min(add_u, w), max(add_u, w))
                    if k in current_edge_set:
                        affected.add(k)
                for w in list(current_adj[add_v]):
                    k = (min(add_v, w), max(add_v, w))
                    if k in current_edge_set:
                        affected.add(k)
                
                for k in affected:
                    if k in current_edge_set:
                        u, v = k
                        current_book_sizes[k] = len(current_adj[u] & current_adj[v])
                    elif k in current_book_sizes:
                        del current_book_sizes[k]
                
                current_max = max(current_book_sizes.values()) if current_book_sizes else 0
            
            no_improve_count = 0
            T = max(T * 2, 3.0)  # Reheat
        
        # Pick an edge to remove - bias toward high-book edges
        if random.random() < 0.7 and current_book_sizes:
            # Pick from edges with above-average book size
            avg_book = sum(current_book_sizes.values()) / len(current_book_sizes)
            high_book_edges = [k for k, v in current_book_sizes.items() if v > avg_book]
            if high_book_edges:
                rem_key = random.choice(high_book_edges)
            else:
                rem_key = random.choice(list(current_book_sizes.keys()))
        else:
            rem_key = random.choice(list(current_book_sizes.keys()))
        
        u1, v1 = rem_key
        # Find index using the index
        idx = edge_index.get(rem_key, -1)
        if idx == -1:
            continue
        
        # Pick a non-edge to add - bias toward edges that won't create large books
        # Sample more candidates and also consider edges that reduce book sizes
        candidates = []
        for _ in range(30):  # Sample 30 candidates for better selection
            add_u, add_v = random.choice(all_possible_edges)
            add_key = (min(add_u, add_v), max(add_u, add_v))
            if add_key not in current_edge_set:
                # Estimate book size if added
                common = current_adj[add_u] & current_adj[add_v]
                est_book = len(common)
                # Penalize if this edge would create a large book
                # Also consider the impact on existing high-book edges
                candidates.append((est_book, add_key))
        
        if not candidates:
            continue
        
        # Pick the candidate with minimum estimated book size
        candidates.sort()
        _, add_key = candidates[0]
        add_u, add_v = add_key
        
        # Calculate affected edges
        affected_keys = set()
        affected_keys.add(rem_key)
        affected_keys.add(add_key)
        
        for w in list(current_adj[u1]):
            if w != v1:
                k = (min(u1, w), max(u1, w))
                if k in current_edge_set:
                    affected_keys.add(k)
        for w in list(current_adj[v1]):
            if w != u1:
                k = (min(v1, w), max(v1, w))
                if k in current_edge_set:
                    affected_keys.add(k)
        for w in list(current_adj[add_u]):
            if w != add_v:
                k = (min(add_u, w), max(add_u, w))
                if k in current_edge_set:
                    affected_keys.add(k)
        for w in list(current_adj[add_v]):
            if w != add_u:
                k = (min(add_v, w), max(add_v, w))
                if k in current_edge_set:
                    affected_keys.add(k)
        
        old_values = {k: current_book_sizes.get(k, 0) for k in affected_keys}
        old_max = current_max
        
        # Apply swap
        current_edge_set.remove(rem_key)
        current_adj[u1].remove(v1)
        current_adj[v1].remove(u1)
        current_edges[idx] = [add_u, add_v]
        
        current_edge_set.add(add_key)
        current_adj[add_u].add(add_v)
        current_adj[add_v].add(add_u)
        
        # Update edge index
        del edge_index[rem_key]
        edge_index[add_key] = idx
        
        # Update book sizes
        for k in affected_keys:
            if k in current_edge_set:
                u, v = k
                current_book_sizes[k] = len(current_adj[u] & current_adj[v])
            else:
                if k in current_book_sizes:
                    del current_book_sizes[k]
        
        new_max = max(current_book_sizes.values()) if current_book_sizes else 0
        
        # Acceptance
        delta = new_max - current_max
        accept = False
        if delta < 0:
            accept = True
        elif delta == 0:
            accept = True
        else:
            accept = random.random() < pow(2.718, -delta / T)
        
        if accept:
            current_max = new_max
            if new_max < best_score:
                best_score = new_max
                best_edges = list(current_edges)
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            # Revert
            current_edge_set.remove(add_key)
            current_edge_set.add(rem_key)
            
            current_adj[add_u].remove(add_v)
            current_adj[add_v].remove(add_u)
            current_adj[u1].add(v1)
            current_adj[v1].add(u1)
            
            current_edges[idx] = [u1, v1]
            
            # Update edge index
            del edge_index[add_key]
            edge_index[rem_key] = idx
            
            for k, val in old_values.items():
                if k in current_edge_set:
                    current_book_sizes[k] = val
                else:
                    if k in current_book_sizes:
                        del current_book_sizes[k]
            
            current_max = old_max
        
        T = max(T * cooling_rate, T_min)
        
    # Final polish pass: try to reduce max book by swapping edges
    if time.time() - start_time < time_limit - 10:
        polish_start = time.time()
        polish_edges = list(best_edges)
        polish_max, polish_books, polish_adj, polish_edge_set = _compute_book_sizes(polish_edges, n)
        
        for _ in range(200):
            if time.time() - polish_start > time_limit - 10 - (time.time() - start_time):
                break
            
            # Find the edge with maximum book size
            if not polish_books:
                break
            max_edge = max(polish_books, key=polish_books.get)
            max_book_val = polish_books[max_edge]
            u1, v1 = max_edge
            
            # Try to replace this edge with a better one
            best_swap = None
            best_swap_score = max_book_val
            
            for _ in range(50):
                add_u, add_v = random.choice(all_possible_edges)
                add_key = (min(add_u, add_v), max(add_u, add_v))
                if add_key in polish_edge_set:
                    continue
                
                # Calculate what the max book would be after this swap
                # Remove max_edge, add add_key
                affected = {max_edge, add_key}
                for w in list(polish_adj[u1]):
                    k = (min(u1, w), max(u1, w))
                    if k in polish_edge_set:
                        affected.add(k)
                for w in list(polish_adj[v1]):
                    k = (min(v1, w), max(v1, w))
                    if k in polish_edge_set:
                        affected.add(k)
                for w in list(polish_adj[add_u]):
                    k = (min(add_u, w), max(add_u, w))
                    if k in polish_edge_set:
                        affected.add(k)
                for w in list(polish_adj[add_v]):
                    k = (min(add_v, w), max(add_v, w))
                    if k in polish_edge_set:
                        affected.add(k)
                
                # Temporarily apply swap
                old_vals = {k: polish_books.get(k, 0) for k in affected}
                
                polish_edge_set.remove(max_edge)
                polish_adj[u1].remove(v1)
                polish_adj[v1].remove(u1)
                
                polish_edge_set.add(add_key)
                polish_adj[add_u].add(add_v)
                polish_adj[add_v].add(add_u)
                
                new_max = 0
                for k in affected:
                    if k in polish_edge_set:
                        uu, vv = k
                        polish_books[k] = len(polish_adj[uu] & polish_adj[vv])
                        if polish_books[k] > new_max:
                            new_max = polish_books[k]
                    else:
                        if k in polish_books:
                            del polish_books[k]
                
                # Check overall max
                if polish_books:
                    overall_max = max(polish_books.values())
                else:
                    overall_max = 0
                
                if overall_max < best_swap_score:
                    best_swap_score = overall_max
                    best_swap = (max_edge, add_key, {k: v for k, v in polish_books.items()}, 
                                 {k: set(v) for k, v in polish_adj.items()},
                                 set(polish_edge_set))
                
                # Revert
                polish_edge_set.remove(add_key)
                polish_edge_set.add(max_edge)
                
                polish_adj[add_u].remove(add_v)
                polish_adj[add_v].remove(add_u)
                polish_adj[u1].add(v1)
                polish_adj[v1].add(u1)
                
                for k, val in old_vals.items():
                    if k in polish_edge_set:
                        polish_books[k] = val
                    else:
                        if k in polish_books:
                            del polish_books[k]
            
            if best_swap is None:
                break
            
            # Apply the best swap
            rem_k, add_k, new_books, new_adj, new_edge_set = best_swap
            
            # Update polish_edges
            for i, e in enumerate(polish_edges):
                key = (min(e[0], e[1]), max(e[0], e[1]))
                if key == rem_k:
                    polish_edges[i] = [add_k[0], add_k[1]]
                    break
            
            polish_books = new_books
            polish_adj = new_adj
            polish_edge_set = new_edge_set
            
            if best_swap_score < polish_max:
                polish_max = best_swap_score
                best_edges = list(polish_edges)
    
    return best_edges

# EVOLVE-BLOCK-END
