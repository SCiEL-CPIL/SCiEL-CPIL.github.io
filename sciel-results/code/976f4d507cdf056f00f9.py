import time
import random
import numpy as np

def construct():
    N = 40
    M = 401
    # Safe deadline to avoid timeout (evaluator kills at 150s, use 140s margin)
    deadline = time.time() + 140
    
    def get_edges(masks):
        edges = []
        for i in range(N):
            m = masks[i]
            for j in range(i + 1, N):
                if (m >> j) & 1:
                    edges.append([i, j])
        return edges

    def compute_books(masks):
        books = {}
        max_b = 0
        for i in range(N):
            m_i = masks[i]
            for j in range(i + 1, N):
                if (m_i >> j) & 1:
                    b = (m_i & masks[j]).bit_count()
                    books[(i, j)] = b
                    if b > max_b:
                        max_b = b
        return books, max_b

    # --- Phase 1: Deterministic Greedy Construction ---
    # Based on the insight that greedy construction with binary search on target book size
    # can directly target the min-max objective. We construct a graph by adding edges
    # only if they don't violate a book bound B. We try to find the smallest B feasible.
    
    def greedy_construct(B, seed):
        """
        Construct a graph with M edges such that no edge is in more than B triangles.
        Returns masks if successful, else None.
        """
        random.seed(seed)
        masks = [0] * N
        edge_count = 0
        
        # Pre-compute all possible edges
        all_edges = [(i, j) for i in range(N) for j in range(i + 1, N)]
        random.shuffle(all_edges)
        
        # We need exactly M edges. 
        # Strategy: Try to add edges in random order. 
        # If adding an edge (u,v) would create a triangle count > B for (u,v) or any 
        # edge involved in the new triangles, we might still add it if we have no choice, 
        # but ideally we skip it. 
        # However, skipping might lead to < M edges. 
        # A better greedy: 
        # 1. Start with empty graph.
        # 2. Iterate through shuffled edges.
        # 3. If edge (u,v) can be added such that:
        #    - book(u,v) <= B
        #    - For all common neighbors w of u and v, book(u,w) + 1 <= B and book(v,w) + 1 <= B
        #    Actually, adding (u,v) increases book(u,w) by 1 for each w in N(u)∩N(v).
        #    And sets book(u,v) = |N(u)∩N(v)|.
        #    So we need |N(u)∩N(v)| <= B AND for all w in N(u)∩N(v), book(u,w) < B and book(v,w) < B.
        
        # This is a strict constraint. If we can't fill M edges, we fail.
        
        # To make it more robust, we can use a "soft" greedy or allow some violations 
        # but try to minimize max book. 
        # Let's stick to the hard constraint first and see if it finds a graph.
        # If not, we can relax.
        
        # Actually, the theorem says min max book is at least 7. 
        # Let's try B=7, then 8, etc.
        
        added = []
        for u, v in all_edges:
            if edge_count >= M:
                break
            
            # Check if we can add (u,v)
            common = (masks[u] & masks[v]).bit_count()
            if common > B:
                continue
            
            # Check neighbors
            ok = True
            common_mask = masks[u] & masks[v]
            # Iterate through common neighbors
            w = 0
            while common_mask:
                if common_mask & 1:
                    # w is a common neighbor
                    # Adding (u,v) creates triangles (u,v,w)
                    # This increases book(u,w) by 1 and book(v,w) by 1
                    # We need current book(u,w) + 1 <= B
                    # book(u,w) is (masks[u] & masks[w]).bit_count()
                    # But wait, masks[u] already includes v? No, we haven't added it yet.
                    # So book(u,w) is correct as is.
                    if (masks[u] & masks[w]).bit_count() >= B:
                        ok = False
                        break
                    if (masks[v] & masks[w]).bit_count() >= B:
                        ok = False
                        break
                    common_mask &= common_mask - 1
                    w += 1
                else:
                    common_mask >>= 1
                    w += 1
            
            if ok:
                masks[u] |= (1 << v)
                masks[v] |= (1 << u)
                edge_count += 1
                added.append((u, v))
        
        if edge_count == M:
            return masks
        return None

    best_masks = None
    best_max_book = float('inf')
    
    # Try to find a graph with max book <= 7 using greedy
    # We might need multiple seeds to succeed
    for B in [7, 8, 9]:
        for seed in range(50):
            if time.time() > deadline - 30: # Leave time for local search
                break
            masks = greedy_construct(B, seed)
            if masks is not None:
                _, max_b = compute_books(masks)
                if max_b < best_max_book:
                    best_max_book = max_b
                    best_masks = masks[:]
        
        if best_masks is not None and best_max_book <= B:
            # We found a graph with max book <= B. 
            # Since we are iterating B upwards, the first success at B=7 is ideal.
            # But we should check if we can do better? No, 7 is the lower bound.
            # If we found one with B=7, max_b will be 7.
            # If we found one with B=8, max_b will be 8.
            # We can break if we found a 7.
            if best_max_book == 7:
                break

    # --- Phase 2: Local Search Refinement ---
    # If greedy found a graph, refine it. If not, use random init.
    
    if best_masks is None:
        # Fallback to random init
        masks = [0] * N
        all_possible = [(i, j) for i in range(N) for j in range(i + 1, N)]
        random.shuffle(all_possible)
        for u, v in all_possible[:M]:
            masks[u] |= (1 << v)
            masks[v] |= (1 << u)
        best_masks = masks
        _, best_max_book = compute_books(masks)
    else:
        masks = best_masks[:]

    current_books, current_max_book = compute_books(masks)
    
    # Local search loop
    # We use the same logic as the original but with tighter constraints and more restarts if needed
    # The original used random swaps. We can improve by targeting specific edges.
    
    max_steps_per_restart = 200
    num_restarts = 0
    max_restarts = 10
    
    while time.time() < deadline - 5 and num_restarts < max_restarts:
        num_restarts += 1
        
        # If we started from greedy, we might already be good. 
        # Let's do a few local search passes.
        
        stuck_count = 0
        steps = 0
        
        while steps < max_steps_per_restart and time.time() < deadline - 5:
            steps += 1
            
            # Find max edge
            max_edge = None
            max_val = -1
            for e, b in current_books.items():
                if b > max_val:
                    max_val = b
                    max_edge = e
            
            if max_val < best_max_book:
                best_max_book = max_val
                best_masks = masks[:]
                
            if max_val <= 7:
                break # Optimal found
            
            u, v = max_edge
            improved = False
            
            # Try to remove (u,v) and add a non-edge that minimizes the increase in other books
            # Or just random swaps as before, but more targeted.
            
            # Let's try 10 random swaps
            for _ in range(10):
                x = random.randint(0, N - 1)
                y = random.randint(0, N - 1)
                if x == y:
                    continue
                if x > y:
                    x, y = y, x
                if x == u and y == v:
                    continue
                if (masks[x] >> y) & 1:
                    continue
                
                # Perform Swap
                masks[u] ^= (1 << v)
                masks[v] ^= (1 << u)
                masks[x] |= (1 << y)
                masks[y] |= (1 << x)
                
                new_books, new_max_book = compute_books(masks)
                
                if new_max_book < max_val:
                    improved = True
                    current_books = new_books
                    max_val = new_max_book
                    stuck_count = 0
                    break
                elif new_max_book == max_val:
                    if random.random() < 0.3: # Lower acceptance for neutral
                        improved = True
                        current_books = new_books
                        stuck_count += 1
                        break
                    else:
                        masks[x] ^= (1 << y)
                        masks[y] ^= (1 << x)
                        masks[u] |= (1 << v)
                        masks[v] |= (1 << u)
                else:
                    masks[x] ^= (1 << y)
                    masks[y] ^= (1 << x)
                    masks[u] |= (1 << v)
                    masks[v] |= (1 << u)
            
            if not improved:
                stuck_count += 1
                if stuck_count > 10:
                    break # Restart local search with new randomization or new init
            
            # If stuck, maybe re-initialize from a slightly perturbed version of best_masks
            if stuck_count > 10:
                # Perturb best_masks
                masks = best_masks[:]
                # Remove a random edge from a high-book edge and add a random non-edge
                # Just restart local search on the same masks but with new random seed for swaps
                # Or break to outer loop to try a new restart strategy
                break

        # If we broke due to stuck, try a different initialization for the next restart
        # If we broke due to max_val <= 7, we are done
        if best_max_book == 7:
            break

        # For the next restart, let's try a fresh random graph if greedy didn't give a good start,
        # or a perturbation of the best found so far.
        if num_restarts < max_restarts:
            # Mix of random and best
            if random.random() < 0.5:
                # Fresh random
                masks = [0] * N
                all_possible = [(i, j) for i in range(N) for j in range(i + 1, N)]
                random.shuffle(all_possible)
                for u, v in all_possible[:M]:
                    masks[u] |= (1 << v)
                    masks[v] |= (1 << u)
            else:
                # Perturb best
                masks = best_masks[:]
                # Remove 5 random edges
                edges_to_remove = []
                all_edges_in_mask = []
                for i in range(N):
                    m = masks[i]
                    for j in range(i + 1, N):
                        if (m >> j) & 1:
                            all_edges_in_mask.append((i, j))
                random.shuffle(all_edges_in_mask)
                for u, v in all_edges_in_mask[:5]:
                    masks[u] ^= (1 << v)
                    masks[v] ^= (1 << u)
                    edges_to_remove.append((u, v))
                # Add 5 random non-edges
                added = 0
                attempts = 0
                while added < 5 and attempts < 100:
                    attempts += 1
                    x = random.randint(0, N - 1)
                    y = random.randint(0, N - 1)
                    if x == y:
                        continue
                    if x > y:
                        x, y = y, x
                    if (masks[x] >> y) & 1:
                        continue
                    masks[x] |= (1 << y)
                    masks[y] |= (1 << x)
                    added += 1
            
            current_books, current_max_book = compute_books(masks)

    return get_edges(best_masks)
