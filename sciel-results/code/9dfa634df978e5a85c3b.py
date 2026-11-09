import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin from 150s limit
    
    n = 40
    target_edges = 401
    
    # 1. Structural Initialization: Use K_{20,20} base with deterministic greedy edge addition
    A = list(range(20))
    B = list(range(20, 40))
    
    edges = set()
    
    # Add all edges between A and B (20 * 20 = 400 edges)
    for a in A:
        for b in B:
            edges.add((a, b))
            
    # Need 1 more edge to reach 401
    # Add one intra-partition edge. Choose it to minimize book size impact.
    # For K_{20,20}, adding edge (u,v) within A creates triangles with all common neighbors in B.
    # Since all A vertices connect to all B vertices, any intra-A edge (u,v) has 20 common neighbors in B.
    # That's a book of size 20! Way too high.
    
    # Better approach: Use the A/B split from the current program but optimize the intra-B edges.
    # Reset and use the proven structure.
    edges = set()
    A = list(range(9))
    B = list(range(9, 40))
    
    # Add all edges between A and B (9 * 31 = 279 edges)
    for a in A:
        for b in B:
            edges.add((a, b))
            
    # Add 122 edges within B
    B1 = list(range(9, 24))   # 15 vertices
    B2 = list(range(24, 40))  # 16 vertices
    
    needed = 122
    added = 0
    
    # Deterministic greedy: add edges to maximize minimum remaining slack
    # Track degrees to keep distribution balanced
    deg = {v: 0 for v in B}
    
    # Generate all possible intra-B pairs, sorted by a heuristic
    all_pairs = []
    for u in B:
        for v in B:
            if u < v:
                all_pairs.append((u, v))
    
    # Heuristic: prefer pairs where both vertices have low degree (to balance)
    # Use a priority queue approach: always pick the pair with minimum sum of degrees
    import heapq
    
    # Initialize heap with all pairs, keyed by (deg[u] + deg[v], u, v)
    heap = []
    for u, v in all_pairs:
        heapq.heappush(heap, (deg[u] + deg[v], u, v))
    
    while added < needed and heap:
        key, u, v = heapq.heappop(heap)
        if (u, v) in edges:
            continue
        edges.add((u, v))
        deg[u] += 1
        deg[v] += 1
        added += 1
        
        # Push updated neighbors of u and v back into heap
        for x in B:
            if x != u and (u, x) not in edges and (x, u) not in edges:
                heapq.heappush(heap, (deg[u] + deg[x], min(u, x), max(u, x)))
        for x in B:
            if x != v and (v, x) not in edges and (x, v) not in edges:
                heapq.heappush(heap, (deg[v] + deg[x], min(v, x), max(v, x)))
    
    if added < needed:
        # Fallback: fill remaining
        for u, v in all_pairs:
            if added == needed:
                break
            if (u, v) not in edges:
                edges.add((u, v))
                added += 1

    # 2. Efficient Data Structures for Local Search
    adj_masks = [0] * n
    
    def update_adj(u, v, add):
        if add:
            adj_masks[u] |= (1 << v)
            adj_masks[v] |= (1 << u)
        else:
            adj_masks[u] &= ~(1 << v)
            adj_masks[v] &= ~(1 << u)
            
    for u, v in edges:
        update_adj(u, v, True)
        
    def get_book_size(u, v):
        if u == v:
            return 0
        return (adj_masks[u] & adj_masks[v]).bit_count()
        
    def get_max_book_info():
        max_b = -1
        max_edges = []
        for u, v in edges:
            b = get_book_size(u, v)
            if b > max_b:
                max_b = b
                max_edges = [(u, v)]
            elif b == max_b:
                max_edges.append((u, v))
        return max_b, max_edges

    current_max_book, current_max_edges = get_max_book_info()
    
    # 3. Local Search with improved swap heuristic
    no_improve_count = 0
    max_no_improve = 500
    
    # Precompute non-edges for faster lookup
    all_pairs_set = set()
    for u in range(n):
        for v in range(u + 1, n):
            all_pairs_set.add((u, v))
    
    while time.time() < deadline:
        if current_max_book <= 6:
            break
        if current_max_book == 7 and no_improve_count > 200:
            break

        if not current_max_edges:
            break

        best_swap = None

        num_candidates = min(5, len(current_max_edges))
        candidates = random.sample(current_max_edges, num_candidates)

        for (u, v) in candidates:
            # Find best non-edge to swap with
            # Criterion: minimize the new max book after swap
            # Quick estimate: for each non-edge (x,y), compute its book size
            # and check if it's lower than current max, or if removing (u,v) reduces max
            
            best_score = float('inf')
            best_x, best_y = -1, -1
            best_estimated_max = float('inf')
            best_estimated_max_count = float('inf')
            
            # Sample non-edges for efficiency
            non_edges = []
            for x in range(n):
                for y in range(x + 1, n):
                    if (x, y) not in edges and (y, x) not in edges:
                        non_edges.append((x, y))
            
            random.shuffle(non_edges)
            sample_size = min(200, len(non_edges))
            sampled_non_edges = non_edges[:sample_size]
            
            for (x, y) in sampled_non_edges:
                # Skip if x or y is u or v (would create issues)
                if x == u or x == v or y == u or y == v:
                    continue
                    
                # Compute book size of new edge (x,y)
                b_new = get_book_size(x, y)
                
                # Estimate new max: 
                # - If (u,v) was the unique max and b_new < current_max, new max might drop
                # - Otherwise new max is max of other edges' books and b_new
                # Quick heuristic: if b_new < current_max_book, this swap is promising
                # Also prefer swaps where b_new is much smaller than current max
                
                # Score: prefer low b_new, and bonus if b_new < current_max_book
                score = b_new
                if b_new < current_max_book:
                    score -= 10  # Strong preference for reducing max
                
                if score < best_score:
                    best_score = score
                    best_x, best_y = x, y
                    best_estimated_max = max(b_new, current_max_book - 1 if len(current_max_edges) == 1 else current_max_book)
                    best_estimated_max_count = 1 if b_new < current_max_book else len(current_max_edges)
            
            if best_x == -1:
                continue

            # Perform trial swap
            edges.remove((u, v))
            update_adj(u, v, False)
            edges.add((best_x, best_y))
            update_adj(best_x, best_y, True)

            new_max, new_max_edges = get_max_book_info()

            # Rollback
            edges.add((u, v))
            update_adj(u, v, True)
            edges.remove((best_x, best_y))
            update_adj(best_x, best_y, False)

            # Track best swap
            if best_swap is None:
                best_swap = (new_max, new_max_edges, u, v, best_x, best_y)
            else:
                if new_max < best_swap[0]:
                    best_swap = (new_max, new_max_edges, u, v, best_x, best_y)
                elif new_max == best_swap[0] and len(new_max_edges) < len(best_swap[1]):
                    best_swap = (new_max, new_max_edges, u, v, best_x, best_y)

        if best_swap is None:
            break

        new_max, new_max_edges, u, v, best_x, best_y = best_swap
        old_max = current_max_book
        old_max_count = len(current_max_edges)

        # Acceptance
        accepted = False
        if new_max < old_max:
            accepted = True
            no_improve_count = 0
        elif new_max == old_max:
            if len(new_max_edges) < old_max_count:
                accepted = True
                no_improve_count = 0
            elif random.random() < 0.05:
                accepted = True
        else:
            accepted = False

        if accepted:
            edges.remove((u, v))
            update_adj(u, v, False)
            edges.add((best_x, best_y))
            update_adj(best_x, best_y, True)
            current_max_book = new_max
            current_max_edges = new_max_edges
        else:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                # Perturbation: randomly swap 3 edges
                if time.time() < deadline - 10:
                    for _ in range(3):
                        if not current_max_edges:
                            break
                        (u, v) = random.choice(current_max_edges)
                        non_edges = [(x, y) for x in range(n) for y in range(x + 1, n) 
                                    if (x, y) not in edges and (y, x) not in edges]
                        if not non_edges:
                            break
                        (x, y) = random.choice(non_edges)
                        edges.remove((u, v))
                        update_adj(u, v, False)
                        edges.add((x, y))
                        update_adj(x, y, True)
                    current_max_book, current_max_edges = get_max_book_info()
                    no_improve_count = 0
                else:
                    break

    result = [[u, v] for u, v in edges]
    return result
