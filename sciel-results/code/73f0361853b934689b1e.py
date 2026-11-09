import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    n = 40
    target_edges = 401
    
    # We use a bitmask representation for speed.
    # mask[i] is an integer where bit j is 1 if edge (i, j) exists.
    masks = [0] * n
    
    # We need to track the current book size of every existing edge to update them efficiently.
    # book[u][v] is the book size of edge (u, v).
    # Since n=40, we can use a 40x40 matrix.
    book = np.zeros((n, n), dtype=np.int32)
    
    # List of existing edges for quick access if needed, but we can just iterate over all pairs.
    # To speed up the greedy step, we need to evaluate all possible (u, v) pairs.
    # Total possible edges = 40*39/2 = 780.
    # We need to pick 401 of them.
    
    # We will maintain a set of existing edges to avoid duplicates.
    existing_edges = set()
    
    # Helper to get common neighbors count (book size of new edge)
    def get_common_count(u, v):
        return bin(masks[u] & masks[v]).count('1')
    
    # Greedy Construction
    # We iterate 401 times.
    # In each iteration, we find the edge (u, v) that minimizes the resulting maximum book size.
    
    current_max_book = 0
    
    for _ in range(target_edges):
        if time.time() > deadline:
            # If we run out of time, we must return what we have, but we need exactly 401 edges.
            # If we haven't reached 401, we just add random edges to fill up.
            # This is a fallback for safety, but the greedy loop is fast.
            break
            
        best_u = -1
        best_v = -1
        best_new_max = float('inf')
        best_common = 0
        
        # Evaluate all non-existing edges
        # Optimization: We can skip edges where the potential book size is already higher than best_new_max
        for u in range(n):
            for v in range(u + 1, n):
                if (u, v) in existing_edges:
                    continue
                
                # Calculate common neighbors
                common_mask = masks[u] & masks[v]
                k = bin(common_mask).count('1')
                
                # If k is already greater than the current best potential max, 
                # we might still need to check if it's better than other options, 
                # but if k > best_new_max, it can't be the minimum *unless* best_new_max is inf.
                # Actually, the new max is max(current_max_book, k, max_incident_books + 1).
                # So if k > best_new_max, it's not a candidate for minimum.
                if k > best_new_max:
                    continue
                
                # Calculate the increase in book sizes for incident edges
                # The incident edges are (u, w) and (v, w) for all w in common_mask.
                # We need to find the max book size among these incident edges.
                max_incident = 0
                # Iterate through set bits in common_mask
                cm = common_mask
                while cm:
                    # Get lowest set bit index
                    # w = (cm & -cm).bit_length() - 1
                    lsb = cm & -cm
                    w = lsb.bit_length() - 1
                    cm ^= lsb
                    
                    # Check book size of (u, w) and (v, w)
                    # Note: (u, w) and (v, w) must exist because w is a common neighbor.
                    buw = book[u, w]
                    bvw = book[v, w]
                    if buw > max_incident:
                        max_incident = buw
                    if bvw > max_incident:
                        max_incident = bvw
                        
                # The new book sizes for these incident edges will be book+1.
                # So the new max contributed by incident edges is max_incident + 1.
                # The new book size of (u, v) is k.
                # The global current max is current_max_book.
                
                potential_max = max(current_max_book, k, max_incident + 1)
                
                if potential_max < best_new_max:
                    best_new_max = potential_max
                    best_u = u
                    best_v = v
                    best_common = k
                    # If we found a potential max equal to current_max_book, 
                    # it means we didn't increase the max. This is ideal.
                    # We can break early if best_new_max == current_max_book?
                    # No, because we want to minimize the *future* growth as well.
                    # But for the immediate step, any edge with potential_max == current_max_book is equally good.
                    # We might prefer one that has lower 'k' or lower 'max_incident' to keep options open.
                    # For simplicity, we keep the first one found or the one with min k.
                    # Let's refine: if potential_max == best_new_max, prefer smaller k.
                    # Or just keep the first.
        
        if best_u == -1:
            # No edge found? Should not happen unless graph is complete.
            break
            
        # Apply the chosen edge
        u, v = best_u, best_v
        
        # Update masks
        masks[u] |= (1 << v)
        masks[v] |= (1 << u)
        
        # Update book sizes
        # 1. The new edge (u, v) has book size best_common
        book[u, v] = best_common
        book[v, u] = best_common
        
        # 2. Increase book sizes of incident edges (u, w) and (v, w) for w in common neighbors
        common_mask = masks[u] & masks[v]
        cm = common_mask
        while cm:
            lsb = cm & -cm
            w = lsb.bit_length() - 1
            cm ^= lsb
            
            book[u, w] += 1
            book[w, u] += 1
            book[v, w] += 1
            book[w, v] += 1
            
        existing_edges.add((u, v))
        
        # Update global max
        if best_new_max > current_max_book:
            current_max_book = best_new_max
        
        # Early termination if we hit the theoretical lower bound
        if current_max_book == 7:
            # We can stop adding edges if we are confident we can't go lower?
            # No, we must add all 401 edges.
            # But if we are at 7, adding more edges might push it to 8.
            # The greedy algorithm will try to keep it at 7.
            pass

    # If we didn't finish the loop due to time, fill with random edges
    if len(existing_edges) < target_edges:
        edges_needed = target_edges - len(existing_edges)
        # Generate random edges
        while edges_needed > 0:
            u = np.random.randint(0, n)
            v = np.random.randint(0, n)
            if u == v:
                continue
            if u > v:
                u, v = v, u
            if (u, v) in existing_edges:
                continue
            
            # Add edge
            masks[u] |= (1 << v)
            masks[v] |= (1 << u)
            
            # Update books (simplified, just count common neighbors for the new edge)
            # This is less accurate but ensures we return a valid graph
            k = bin(masks[u] & masks[v]).count('1')
            book[u, v] = k
            book[v, u] = k
            
            # Note: This fallback doesn't update incident books correctly, 
            # but it's only for timeout safety.
            
            existing_edges.add((u, v))
            edges_needed -= 1

    # Construct the final list of edges
    result = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u, v) in existing_edges:
                result.append([u, v])
                
    # Verification: Ensure exactly 401 edges
    if len(result) != 401:
        # If something went wrong, return a valid 401 edge graph (e.g., K_20,20 + 1)
        # This is a fallback for correctness
        fallback = []
        for i in range(20):
            for j in range(20, 40):
                fallback.append([i, j])
        # Add 1 more edge
        fallback.append([0, 1])
        return fallback

    return result
