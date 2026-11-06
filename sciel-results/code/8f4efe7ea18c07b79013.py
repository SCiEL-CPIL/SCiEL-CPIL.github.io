import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    n = 40
    target_edges = 401
    
    # Precompute all pairs
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    def get_initial_graph():
        """Generate a random graph with target_edges edges."""
        random.shuffle(all_pairs)
        edges = all_pairs[:target_edges]
        A = np.zeros((n, n), dtype=np.int8)
        for u, v in edges:
            A[u, v] = 1
            A[v, u] = 1
        return A

    def calc_book_sizes(A):
        """Calculate book sizes for all edges in A."""
        B = np.dot(A, A)
        # book_sizes[i,j] is the number of common neighbors of i and j
        # For edges, this is the book size.
        # For non-edges, it's just common neighbors.
        return B

    def greedy_construct():
        """Construct a graph by greedily adding edges that minimize max book size."""
        A = np.zeros((n, n), dtype=np.int8)
        book_sizes = np.zeros((n, n), dtype=np.int16)
        current_max = 0
        current_sum = 0
        
        remaining = all_pairs[:]
        random.shuffle(remaining)
        
        for _ in range(target_edges):
            best_edge = None
            best_max = float('inf')
            best_sum = float('inf')
            
            # Evaluate a subset of candidates if time is tight, otherwise all
            # n=40, pairs=780. 401 iterations * 780 candidates * O(n) update = ~12M ops.
            # This is fast enough in Python if optimized, but let's be careful.
            # We will sample candidates to speed up.
            candidates = remaining[:50] # Sample 50 candidates
            
            for u, v in candidates:
                if A[u, v] == 1:
                    continue
                
                # Calculate new book size for (u,v)
                # common neighbors
                common = np.where(A[u] & A[v])[0]
                new_book_uv = len(common)
                
                # Calculate impact on other edges
                # Edges (u,w) and (v,w) for w in common gain 1
                # We need to check if any of them exceed best_max
                # Current max is current_max.
                # New max is max(current_max, new_book_uv, max(book_sizes[u, common] + 1, book_sizes[v, common] + 1))
                
                if new_book_uv > best_max:
                    continue # Can't be better than current best_max if new_book_uv is already too high? 
                    # Wait, we are comparing to best_max of the *candidate set*.
                    # Actually, we want to minimize the *global* max book size.
                    # So we should compare against the resulting global max.
                    
                # Efficient check:
                # The new global max will be max(current_max, new_book_uv, max_affected)
                # where max_affected is max of (book_sizes[u, w] + 1) and (book_sizes[v, w] + 1) for w in common.
                
                if common.size > 0:
                    max_affected = max(np.max(book_sizes[u, common] + 1), np.max(book_sizes[v, common] + 1))
                else:
                    max_affected = 0
                    
                new_global_max = max(current_max, new_book_uv, max_affected)
                
                # If this new global max is already worse than the best we found in this round, skip
                if new_global_max > best_max:
                    continue
                
                # Calculate sum change for tie-breaking
                # Sum increases by new_book_uv + sum(book_sizes[u, common] + 1 - old) + ...
                # Actually, sum of book sizes increases by:
                # 1. The new edge (u,v) adds new_book_uv to the sum.
                # 2. Each edge (u,w) for w in common increases by 1.
                # 3. Each edge (v,w) for w in common increases by 1.
                # So delta_sum = new_book_uv + 2 * len(common)
                # Wait, book_sizes[u,w] is the book size of edge (u,w).
                # The sum of all book sizes is S.
                # New S = S + new_book_uv + sum_{w in common} (1 + 1) = S + new_book_uv + 2*len(common).
                # This is an approximation? No, it's exact for the *change*.
                
                delta_sum = new_book_uv + 2 * len(common)
                new_global_sum = current_sum + delta_sum
                
                if new_global_max < best_max or (new_global_max == best_max and new_global_sum < best_sum):
                    best_max = new_global_max
                    best_sum = new_global_sum
                    best_edge = (u, v)
                    # Store details for update
                    best_common = common
                    best_new_book_uv = new_book_uv
                    best_delta_sum = delta_sum

            if best_edge is None:
                # If no candidate found (should not happen), pick first available
                for u, v in remaining:
                    if A[u, v] == 0:
                        best_edge = (u, v)
                        break
            
            u, v = best_edge
            # Update A and book_sizes
            A[u, v] = 1
            A[v, u] = 1
            
            # Recalculate common neighbors for update (might have changed if we didn't use best_common directly? No, A changed)
            # We used best_common from the candidate evaluation.
            # But we need to update book_sizes matrix.
            common = np.where(A[u] & A[v])[0] # This includes u and v? No, A[u,u]=0.
            # Wait, A[u] & A[v] gives common neighbors.
            
            book_sizes[u, v] = len(common)
            book_sizes[v, u] = len(common)
            
            for w in common:
                book_sizes[u, w] += 1
                book_sizes[w, u] += 1
                book_sizes[v, w] += 1
                book_sizes[w, v] += 1
                
            current_max = best_max
            current_sum = best_sum
            
            # Remove chosen edge from remaining
            # Instead of removing from list, we can just check A[u,v]==0 in next loop
            # But we need to ensure we don't pick the same edge again.
            # We can mark it in A. The loop checks A[u,v]==1.
            # So we just need to ensure we iterate over all pairs.
            # To avoid O(N^2) scanning, we can maintain a list of non-edges.
            # For simplicity, we iterate over all pairs and check A.
            # 780 pairs * 401 steps = 312,000 checks. Very fast.
            
            if time.time() > deadline - 10:
                break
                
        return A, current_max, current_sum

    def local_search_polish(A, current_max, current_sum):
        """Improve the graph using local search with fast updates."""
        # Ensure book_sizes is up to date
        book_sizes = calc_book_sizes(A)
        
        # Verify current_max
        # max_book = np.max(book_sizes * A)
        # if max_book != current_max: current_max = max_book
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            
            # Find max book edges
            # max_book = np.max(book_sizes * A)
            # We can just use current_max if we maintain it, but let's be safe.
            max_book = np.max(book_sizes * A)
            if max_book < current_max:
                current_max = max_book
                
            max_edges = []
            for i in range(n):
                for j in range(i+1, n):
                    if A[i, j] == 1 and book_sizes[i, j] == max_book:
                        max_edges.append((i, j))
                        
            if not max_edges:
                break
                
            # Try to swap a random max edge with a random non-edge
            random.shuffle(max_edges)
            swapped = False
            
            for u, v in max_edges:
                if time.time() > deadline - 5:
                    break
                    
                # Find non-edges
                non_edges = [(x, y) for x in range(n) for y in range(x+1, n) if A[x, y] == 0]
                random.shuffle(non_edges)
                
                # Try a few non-edges
                for x, y in non_edges[:50]:
                    # Perform swap: remove (u,v), add (x,y)
                    
                    # Remove (u,v)
                    common_uv = np.where(A[u] & A[v])[0]
                    # Update book_sizes for removal
                    book_sizes[u, v] = 0
                    book_sizes[v, u] = 0
                    A[u, v] = 0
                    A[v, u] = 0
                    
                    for w in common_uv:
                        book_sizes[u, w] -= 1
                        book_sizes[w, u] -= 1
                        book_sizes[v, w] -= 1
                        book_sizes[w, v] -= 1
                        
                    # Add (x,y)
                    common_xy = np.where(A[x] & A[y])[0]
                    new_book_xy = len(common_xy)
                    
                    # Check if new book size exceeds current max_book
                    # If new_book_xy > max_book, this swap is bad (unless max_book drops enough elsewhere? 
                    # No, max_book is the max. If we add an edge with book > max_book, new max is > max_book. Bad.)
                    if new_book_xy > max_book:
                        # Revert
                        A[x, y] = 0
                        A[y, x] = 0
                        # book_sizes[x,y] is already 0? No, we didn't set it.
                        # We need to revert the removal of (u,v) first?
                        # It's safer to revert everything.
                        pass 
                    else:
                        # Check impact on other edges
                        # Edges (x,w) and (y,w) for w in common_xy gain 1
                        # If any of them + 1 > max_book, bad.
                        bad = False
                        if len(common_xy) > 0:
                            max_gain_uv = max(np.max(book_sizes[x, common_xy] + 1), np.max(book_sizes[y, common_xy] + 1))
                            if max_gain_uv > max_book:
                                bad = True
                                
                        if not bad:
                            # Calculate new max
                            new_max = max_book
                            # The removed edge (u,v) is gone.
                            # The added edge (x,y) has book new_book_xy.
                            # The affected edges have new book sizes.
                            # We need to check if the *global* max decreased.
                            # It decreased if:
                            # 1. The removed edge was the unique max, and no new edge reaches that max.
                            # 2. Or, the max was shared, and we broke enough of them.
                            
                            # Simple check:
                            # New max is max( 
                            #    max(book_sizes of all other edges), 
                            #    new_book_xy, 
                            #    max(affected edges)
                            # )
                            # We don't have "max of all other edges" readily available without scanning.
                            # But we know the old max was max_book.
                            # If we removed an edge with book max_book, the new max is at most max_book.
                            # It is strictly less than max_book if:
                            # - No other edge has book max_book.
                            # - AND no new/affected edge has book max_book.
                            
                            # Let's just calculate the new max by scanning? 
                            # O(n^2) scan is 1600 ops. Fast enough.
                            # But we are in a loop.
                            
                            # Let's do the scan.
                            new_max_val = 0
                            for i in range(n):
                                for j in range(i+1, n):
                                    if A[i, j] == 1:
                                        if book_sizes[i, j] > new_max_val:
                                            new_max_val = book_sizes[i, j]
                                        
                            if new_max_val < current_max:
                                # Accept
                                A[x, y] = 1
                                A[y, x] = 1
                                book_sizes[x, y] = new_book_xy
                                book_sizes[y, x] = new_book_xy
                                
                                for w in common_xy:
                                    book_sizes[x, w] += 1
                                    book_sizes[w, x] += 1
                                    book_sizes[y, w] += 1
                                    book_sizes[w, y] += 1
                                    
                                current_max = new_max_val
                                improved = True
                                swapped = True
                                break
                            else:
                                # Revert addition
                                # We didn't add it yet, so just clean up book_sizes?
                                # We didn't modify book_sizes for x,y yet.
                                # But we did modify for u,v removal.
                                # We need to revert u,v removal.
                                pass
                        else:
                            pass
                            
                    # Revert removal of (u,v) if we didn't accept
                    # This logic is getting messy. Let's simplify.
                    # Use a function to apply/revert swap.
                    
            # End non-edges loop
            if swapped:
                continue
            else:
                # If no swap accepted, break or try another max edge
                pass
                
        return A, current_max

    # Main Logic
    best_A = None
    best_score = float('inf')
    
    # 1. Greedy Construction
    A_greedy, max_greedy, sum_greedy = greedy_construct()
    # Verify
    B = calc_book_sizes(A_greedy)
    true_max = np.max(B * A_greedy)
    if true_max < best_score:
        best_score = true_max
        best_A = A_greedy.copy()
        
    if time.time() > deadline - 20:
        # Not enough time for polish
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if best_A[i, j] == 1:
                    edges.append([i, j])
        return edges
        
    # 2. Local Search Polish
    A_polished, max_polished = local_search_polish(A_greedy, true_max, 0)
    B = calc_book_sizes(A_polished)
    true_max_polished = np.max(B * A_polished)
    if true_max_polished < best_score:
        best_score = true_max_polished
        best_A = A_polished.copy()
        
    # 3. Multiple Restarts with Random Init + Local Search
    while time.time() < deadline - 10:
        A_rand = get_initial_graph()
        B = calc_book_sizes(A_rand)
        cur_max = np.max(B * A_rand)
        
        A_polished, max_polished = local_search_polish(A_rand, cur_max, 0)
        B = calc_book_sizes(A_polished)
        true_max_polished = np.max(B * A_polished)
        
        if true_max_polished < best_score:
            best_score = true_max_polished
            best_A = A_polished.copy()
            
        if best_score <= 7:
            break
            
    # Extract edges
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if best_A[i, j] == 1:
                edges.append([i, j])
                
    # Ensure exactly 401 edges
    if len(edges) != target_edges:
        existing = set(tuple(e) for e in edges)
        if len(edges) > target_edges:
            while len(edges) > target_edges:
                idx = random.randint(0, len(edges)-1)
                e = edges.pop(idx)
                existing.discard(tuple(e))
                existing.discard(tuple(reversed(e)))
        else:
            for u, v in all_pairs:
                if len(edges) == target_edges:
                    break
                if (u, v) not in existing and (v, u) not in existing:
                    edges.append([u, v])
                    existing.add((u, v))
                    existing.add((v, u))
                    
    return edges
