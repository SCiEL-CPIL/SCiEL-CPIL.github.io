import time
import random
import heapq

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    n = 40
    m = 401

    # Initialize with a random graph
    # To get a good starting point, let's try to make it regular-ish
    # Just random is fine for SA
    
    edges = set()
    adj = [set() for _ in range(n)]
    
    # Generate random m edges
    all_possible = [(i, j) for i in range(n) for j in range(i+1, n)]
    random.shuffle(all_possible)
    for u, v in all_possible[:m]:
        edges.add((u, v))
        adj[u].add(v)
        adj[v].add(u)
        
    # Calculate initial book sizes
    # book[(u,v)] = number of common neighbors
    book = {}
    for u, v in edges:
        common = len(adj[u] & adj[v])
        book[(u, v)] = common
        
    # Initialize heap for max book size
    # Heap stores (-book_size, u, v)
    heap = [(-book[e], e[0], e[1]) for e in edges]
    heapq.heapify(heap)
    
    current_max_book = -heap[0][0]
    
    best_max_book = current_max_book
    best_edges = set(edges)
    
    # Simulated Annealing
    T = 5.0
    T_min = 0.1
    cooling_rate = 0.999
    
    while time.time() < deadline:
        if T < T_min:
            T = T_min
            
        # Choose a random edge to remove
        if not edges:
            break
        u, v = random.choice(list(edges))
        
        # Choose a random non-edge to add
        # To speed this up, we can pick random vertices and check if they are connected
        # Or maintain a list of non-edges. With 401 edges out of 780, non-edges are ~379.
        # Picking random vertices is efficient enough.
        x, y = random.sample(range(n), 2)
        if x > y: x, y = y, x
        if (x, y) in edges:
            continue
            
        # Check if (u,v) and (x,y) are the same (shouldn't be, but just in case)
        if (u, v) == (x, y):
            continue

        # Perform the move tentatively
        # 1. Remove (u, v)
        edges.remove((u, v))
        adj[u].remove(v)
        adj[v].remove(u)
        
        # Update book sizes for edges incident to u or v
        # Edges (u, w) where w != v
        # Common neighbors of (u, w) lose v if v was connected to w
        # Since (u,v) is removed, v is no longer a neighbor of u.
        # But for the book size of (u, w), we count common neighbors.
        # If v was a common neighbor of u and w, it means (v,w) existed.
        # Now u is not connected to v, so v is not a neighbor of u.
        # So v is no longer a common neighbor.
        # So book[(u,w)] decreases by 1 if (v,w) is in edges.
        
        changed_edges = set()
        
        # Update edges incident to u
        for w in list(adj[u]):
            if w == v: continue
            e = (u, w) if u < w else (w, u)
            # If (v, w) is an edge, then v was a common neighbor of u and w
            # Now u-v is gone, so v is not a neighbor of u.
            # So v is not a common neighbor.
            # But wait. The definition of book size is common neighbors.
            # Common neighbors of u and w are vertices z such that (u,z) and (w,z) are edges.
            # Previously, if (v,w) was an edge, and (u,v) was an edge, v was a common neighbor.
            # Now (u,v) is not an edge. So v is not a common neighbor.
            # So book[e] decreases by 1.
            if (v, w) in edges or (w, v) in edges:
                book[e] -= 1
                changed_edges.add(e)
                
        # Update edges incident to v
        for w in list(adj[v]):
            if w == u: continue
            e = (v, w) if v < w else (w, v)
            if (u, w) in edges or (w, u) in edges:
                book[e] -= 1
                changed_edges.add(e)
                
        # 2. Add (x, y)
        edges.add((x, y))
        adj[x].add(y)
        adj[y].add(x)
        
        # Calculate book size for new edge (x, y)
        common_xy = len(adj[x] & adj[y])
        book[(x, y)] = common_xy
        changed_edges.add((x, y))
        
        # Update edges incident to x
        for w in adj[x]:
            if w == y: continue
            e = (x, w) if x < w else (w, x)
            # If (y, w) is an edge, then y is a common neighbor of x and w.
            # Previously, x-y was not an edge, so y was not a neighbor of x.
            # Now x-y is an edge.
            # So y becomes a common neighbor.
            # So book[e] increases by 1.
            if (y, w) in edges or (w, y) in edges:
                book[e] += 1
                changed_edges.add(e)
                
        # Update edges incident to y
        for w in adj[y]:
            if w == x: continue
            e = (y, w) if y < w else (w, y)
            if (x, w) in edges or (w, x) in edges:
                book[e] += 1
                changed_edges.add(e)
        
        # Push updated values to heap
        for e in changed_edges:
            heapq.heappush(heap, (-book[e], e[0], e[1]))
            
        # Get new max book size
        while heap:
            neg_b, u_h, v_h = heap[0]
            e_h = (u_h, v_h) if u_h < v_h else (v_h, u_h)
            if e_h in edges and book.get(e_h) == -neg_b:
                new_max_book = -neg_b
                break
            else:
                heapq.heappop(heap)
        else:
            # Should not happen
            new_max_book = 0

        # Acceptance criterion
        delta = new_max_book - current_max_book
        if delta < 0 or random.random() < pow(2.71828, -delta / T):
            # Accept
            current_max_book = new_max_book
            if current_max_book < best_max_book:
                best_max_book = current_max_book
                best_edges = set(edges)
                # If we hit the lower bound, we can stop
                if best_max_book <= 7:
                    break
        else:
            # Reject: Rollback
            # Remove (x, y)
            edges.remove((x, y))
            adj[x].remove(y)
            adj[y].remove(x)
            del book[(x, y)]
            
            # Re-add (u, v)
            edges.add((u, v))
            adj[u].add(v)
            adj[v].add(u)
            
            # Restore book sizes
            # We don't strictly need to restore the dict values if we recompute them,
            # but the heap has stale entries. 
            # Actually, the easiest way to rollback is to just recompute the book sizes 
            # for the changed edges and push them to the heap.
            # But we already pushed the new values. 
            # So we need to push the OLD values back? 
            # Or just recompute and push.
            
            # Let's just recompute book sizes for changed_edges and push them.
            # This ensures the heap has valid entries for the current state.
            for e in changed_edges:
                 if e in edges:
                    # Recompute
                    u_e, v_e = e
                    # Note: adj has been rolled back
                    book[e] = len(adj[u_e] & adj[v_e])
                    heapq.heappush(heap, (-book[e], e[0], e[1]))
                 else:
                    # e is (x,y) which was removed
                    # It's not in edges, so it won't be considered for max
                    pass
            
            # Note: (u,v) was in changed_edges? 
            # (u,v) was removed. Its book size is no longer relevant.
            # But it might be in the heap from before.
            # The "while heap" loop will skip it because (u,v) is not in edges (wait, we re-added it).
            # So (u,v) IS in edges.
            # So we need to recompute its book size too.
            # Was (u,v) in changed_edges? 
            # The loop `for w in adj[u]` did not include v (skipped).
            # So (u,v) was NOT in changed_edges.
            # But its book size didn't change? 
            # Wait. When we removed (u,v), we didn't update book[(u,v)] because it was removed.
            # When we re-add it, we need to calculate it.
            # Let's just add (u,v) to changed_edges for rollback safety.
            # Actually, simpler: just recompute book sizes for ALL edges in the neighborhood of u,v,x,y.
            
            # To be safe and simple, let's just recompute the max book size from scratch 
            # if we reject? No, that's slow.
            # Let's just ensure the heap is consistent.
            # The heap contains entries for the "new" state.
            # We rolled back to the "old" state.
            # So the entries in the heap for the changed edges are now stale (they reflect the new state).
            # The "while heap" loop will pop them if they don't match the current book dict.
            # But we updated the book dict? No, we didn't update the book dict in the reject branch 
            # except for the recompute loop I just wrote.
            # In the recompute loop, I updated `book[e]` for `e in changed_edges`.
            # And I pushed them to heap.
            # So the heap now has both the "new" entries (stale) and "old" entries (valid).
            # The "while heap" loop will find the valid ones.
            # This works.
            
            # One issue: (u,v) book size.
            # It was not in changed_edges. So its book value in the dict is still the OLD value.
            # And it was not pushed to heap in the "new" state phase (since it was removed).
            # So the heap might have an old entry for (u,v) from previous steps.
            # That entry is still valid! Because (u,v) is back and its book size is the same as before.
            # So we don't need to do anything for (u,v).

        T *= cooling_rate

    # Return best edges found
    return [[u, v] for u, v in best_edges]

# EVOLVE-BLOCK-START
# The construct function is defined above.
# EVOLVE-BLOCK-END
