import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 150s limit
    
    n = 40
    target_edges = 401
    total_possible_edges = n * (n - 1) // 2
    
    # Initialize masks for bitset representation
    # mask[i] is an integer where bit j is 1 if edge (i, j) exists
    masks = [0] * n
    edges = []
    non_edges = []
    
    # Helper to add edge
    def add_edge(u, v):
        if u == v: return False
        if (masks[u] >> v) & 1: return False
        masks[u] |= (1 << v)
        masks[v] |= (1 << u)
        edges.append((u, v))
        return True
        
    # Helper to remove edge
    def remove_edge(u, v):
        if u == v: return False
        if not ((masks[u] >> v) & 1): return False
        masks[u] &= ~(1 << v)
        masks[v] &= ~(1 << u)
        # Remove from edges list - O(n) but n is small
        try:
            edges.remove((u, v))
        except ValueError:
            # Try reversed
            try:
                edges.remove((v, u))
            except ValueError:
                pass
        return True

    # Helper to check edge
    def has_edge(u, v):
        return bool((masks[u] >> v) & 1)
        
    # Helper to get book size of edge (u, v)
    def book_size(u, v):
        if not has_edge(u, v): return 0
        return (masks[u] & masks[v]).bit_count()

    # 1. Generate a random 20-regular graph
    # We will use a simple heuristic: random matching of stubs or just random swaps to regularize
    # Start with random graph with 400 edges, then regularize
    
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    random.shuffle(all_pairs)
    
    # Add 400 edges
    for u, v in all_pairs[:400]:
        add_edge(u, v)
        
    # Regularize to 20-regular
    # Target degree: 20
    for _ in range(5000):
        degs = [masks[i].bit_count() for i in range(n)]
        # Find max and min degree
        u = degs.index(max(degs))
        v = degs.index(min(degs))
        if degs[u] - degs[v] <= 1:
            break
            
        # Find w1 in N(u) \ N(v) and w2 in N(v) \ N(u)
        # Swap (u, w1) with (v, w2) -> (u, w2) and (v, w1)
        # This maintains degrees
        
        nu = [i for i in range(n) if i != u and has_edge(u, i)]
        nv = [i for i in range(n) if i != v and has_edge(v, i)]
        
        candidates_w1 = [w for w in nu if not has_edge(v, w)]
        candidates_w2 = [w for w in nv if not has_edge(u, w)]
        
        if candidates_w1 and candidates_w2:
            w1 = random.choice(candidates_w1)
            w2 = random.choice(candidates_w2)
            # Check if (u, w2) and (v, w1) are non-edges
            if not has_edge(u, w2) and not has_edge(v, w1):
                remove_edge(u, w1)
                remove_edge(v, w2)
                add_edge(u, w2)
                add_edge(v, w1)
                
    # Now we have a 20-regular graph (400 edges).
    # Add one more edge to reach 401.
    # Pick two non-adjacent vertices.
    remaining_pairs = [p for p in all_pairs if not has_edge(p[0], p[1])]
    if remaining_pairs:
        u, v = random.choice(remaining_pairs)
        add_edge(u, v)
        
    # Build non_edges list
    non_edges = [p for p in all_pairs if not has_edge(p[0], p[1])]
    
    # Compute initial book sizes for all edges
    # We will track current_max and the edge achieving it
    # To be efficient, we don't store all book sizes, we compute on the fly for candidates
    # But for SA, we need to evaluate the new max.
    # Since n is small, we can recompute max book size in O(m * popcount) which is fast.
    
    def get_max_book():
        max_b = 0
        max_e = None
        for u, v in edges:
            b = (masks[u] & masks[v]).bit_count()
            if b > max_b:
                max_b = b
                max_e = (u, v)
        return max_b, max_e
        
    current_max, current_edge = get_max_book()
    best_max = current_max
    best_masks = masks.copy()
    
    T = 1.0
    cooling_rate = 0.999
    
    while time.time() < deadline:
        # Pick a move
        # 70% chance: Targeted move on current max book edge
        # 30% chance: Random swap
        
        if random.random() < 0.7 and current_edge:
            u, v = current_edge
            # Common neighbors
            common = masks[u] & masks[v]
            # Get list of common neighbors
            common_list = [i for i in range(n) if (common >> i) & 1]
            
            if common_list:
                # Pick a common neighbor k to break
                k = random.choice(common_list)
                
                # We want to remove an edge incident to u or v to reduce book(u,v)
                # Removing (u, k) reduces book(u, k) and book(v, k) ?
                # Actually, removing (u, k) reduces book(u, v) by 1 if k was common.
                # Let's remove (u, k).
                # We must add a new edge (a, b) from non_edges.
                
                # Pick a random non-edge
                if not non_edges: continue
                a, b = random.choice(non_edges)
                
                # Check validity of swap:
                # Remove (u, k), Add (a, b)
                # Ensure (a, b) is not (u, k)
                if (a == u and b == k) or (a == k and b == u):
                    continue
                
                # Perform swap
                remove_edge(u, k)
                add_edge(a, b)
                
                # Update non_edges
                try:
                    non_edges.remove((a, b))
                except ValueError:
                    pass
                non_edges.append((u, k))
                
                new_max, new_edge = get_max_book()
                
                delta = new_max - current_max
                
                if delta < 0 or random.random() < math.exp(-delta / T):
                    current_max = new_max
                    current_edge = new_edge
                    if current_max < best_max:
                        best_max = current_max
                        best_masks = masks.copy()
                else:
                    # Revert
                    remove_edge(a, b)
                    add_edge(u, k)
                    
                    try:
                        non_edges.remove((u, k))
                    except ValueError:
                        pass
                    non_edges.append((a, b))
            else:
                # No common neighbors, this edge is not a bottleneck.
                # Do a random swap
                pass
        else:
            # Random swap
            if not edges or not non_edges: continue
            u, v = random.choice(edges)
            a, b = random.choice(non_edges)
            
            if (u == a and v == b) or (u == b and v == a):
                continue
            
            remove_edge(u, v)
            add_edge(a, b)
            
            try:
                non_edges.remove((a, b))
            except ValueError:
                pass
            non_edges.append((u, v))
            
            new_max, new_edge = get_max_book()
            
            delta = new_max - current_max
            
            if delta < 0 or random.random() < math.exp(-delta / T):
                current_max = new_max
                current_edge = new_edge
                if current_max < best_max:
                    best_max = current_max
                    best_masks = masks.copy()
            else:
                # Revert
                remove_edge(a, b)
                add_edge(u, v)
                
                try:
                    non_edges.remove((u, v))
                except ValueError:
                    pass
                non_edges.append((a, b))
        
        T *= cooling_rate
        if T < 0.01:
            T = 0.01

    # Reconstruct result from best_masks
    result = []
    for i in range(n):
        for j in range(i+1, n):
            if (best_masks[i] >> j) & 1:
                result.append([i, j])
                
    # Verify edge count
    if len(result) != target_edges:
        # Fallback to a valid random graph
        # This should rarely happen if logic is correct
        random.seed(42)
        all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_pairs)
        result = [list(e) for e in all_pairs[:target_edges]]
        
    return result
