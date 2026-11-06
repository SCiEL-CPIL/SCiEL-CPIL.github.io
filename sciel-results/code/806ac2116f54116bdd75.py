import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    n = 40
    target_edges = 401

    # Fast book computation: books[u,v] = (A@A)[u,v] for edge (u,v)
    # We maintain books incrementally when adding/removing edges.

    def compute_books(A):
        return A @ A

    def max_book_from_books(A, books):
        masked = np.where(A > 0, books, 0)
        return int(masked.max())

    # --- Phase 1: Fast greedy construction using incremental updates ---
    # When adding edge (u,v) to graph with adjacency A and book matrix B:
    # New B' = B + outer(A[:,u], A[:,v]) + outer(A[:,v], A[:,u])
    # But only for the symmetric case. Actually:
    # B[u,v] = sum_k A[u,k]*A[k,v]
    # After adding (u,v): A[u,v]=A[v,u]=1
    # For any pair (i,j), B'[i,j] = B[i,j] + A[i,u]*A[v,j] + A[i,v]*A[u,j]
    # Wait, that's not quite right. Let me think more carefully.
    #
    # B = A @ A, so B[i,j] = sum_k A[i,k]*A[k,j]
    # After adding edge (u,v): A'[u,v] = A'[v,u] = 1, rest same.
    # B'[i,j] = sum_k A'[i,k]*A'[k,j]
    # The only changes are when k=u or k=v.
    # B'[i,j] = B[i,j] + A[i,u]*A'[u,j] + A'[i,v]*A'[v,j] - (corrections if i or j is u or v)
    #
    # Actually simpler: B' = A' @ A' = (A + E) @ (A + E) where E has 1s at (u,v),(v,u)
    # B' = A@A + A@E + E@A + E@E
    # A@E: (A@E)[i,j] = sum_k A[i,k]*E[k,j] = A[i,u]*E[u,j] + A[i,v]*E[v,j] = A[i,u]*(1 if j==v else 0) + A[i,v]*(1 if j==u else 0)
    # So A@E has 1s at (i,v) for all i where A[i,u]=1, and 1s at (i,u) for all i where A[i,v]=1.
    # Similarly E@A.
    # E@E has 1s at (u,u) and (v,v) (since E[u,v]*E[v,u] = 1 for (u,u), etc.)
    #
    # This is getting complex. For n=40, let's just use the fact that A@A is fast.
    # But we can avoid recomputing the full matrix by noting that only O(n) entries change.
    #
    # Better approach: Use the formula that adding edge (u,v) increases book of edge (i,j)
    # by the number of new triangles formed. A new triangle is formed for each common neighbor.
    # Actually, the book of edge (i,j) is the number of common neighbors of i and j.
    # When we add edge (u,v), new common neighbors are created for pairs (i,j) where
    # one of i,j is u or v, or where u/v become common neighbors.
    #
    # For simplicity and correctness with n=40, let me use a different strategy:
    # Start from a balanced bipartite-like graph and do local search.

    # --- Strategy: Start from a near-complete bipartite graph ---
    # Split 40 vertices into two sets of 20. Complete bipartite has 400 edges.
    # Add 1 more edge within a part. This gives a good starting point.
    # The max book in K_{20,20} is 19 (each edge has 19 common neighbors on the other side... wait)
    # In K_{20,20}, edge (u in L, v in R): common neighbors are all other 19 in L and all other 19 in R? No.
    # Common neighbors of u (in L) and v (in R): vertices adjacent to both. 
    # u is adjacent to all in R. v is adjacent to all in L. 
    # A vertex w is common neighbor if w~u and w~v.
    # If w in L: w~v (yes, all L connected to v in R), w~u? u in L, w in L, no edge in L. So no.
    # If w in R: w~u (yes, all R connected to u in L), w~v? v in R, w in R, no edge in R. So no.
    # So common neighbors = 0! The book size of every edge in K_{20,20} is 0.
    # 
    # So K_{20,20} has max book 0. Adding one intra-part edge (u1,u2) in L:
    # New edges: (u1,u2). 
    # Book of (u1,u2): common neighbors of u1 and u2. Both connected to all 20 in R. So 20 common neighbors.
    # Book of edge (u1, r) for r in R: common neighbors of u1 and r. 
    #   u1 adjacent to: all R, and u2 (new). r adjacent to: all L.
    #   Common: vertices adjacent to both. u2 is adjacent to u1 and to r (since u2 in L, r in R). So u2 is common neighbor.
    #   Also any other? w in L: w~r (yes), w~u1? only if w=u2. So just u2. Book = 1.
    #   w in R: w~u1 (yes), w~r? no (same part). So no.
    # So book of (u1,r) = 1 for each r. Similarly for (u2,r).
    # Book of edge (l, r) where l in L\{u1,u2}: common neighbors of l and r.
    #   l adjacent to all R. r adjacent to all L.
    #   w in L: w~r (yes), w~l? no unless w is... no intra-L edges except (u1,u2). So no.
    #   w in R: w~l (yes), w~r? no. So no.
    #   Wait, what about u1 and u2? u1~l? no. u2~l? no. So book = 0.
    #
    # So after adding (u1,u2): max book = 20 (for edge (u1,u2)).
    # That's terrible. We need to add 1 more edge, not 1.
    #
    # Better: Start from K_{20,20} (400 edges, max book 0) and add 1 edge carefully.
    # But we showed adding any intra-part edge creates a book of size 20.
    # 
    # Alternative: Start from a different base. 
    # What about a cycle or other sparse structure?
    #
    # Let me think about what graph minimizes max book with 401 edges on 40 vertices.
    # The theorem says min max book >= ceil(40/6) = 7.
    # 
    # A good heuristic: try to make the graph as "bipartite-like" as possible while having 401 edges.
    # K_{20,20} has 400 edges. We need 1 more.
    # If we add an intra-part edge, we get max book 20. Bad.
    # 
    # What if we use K_{19,21} = 19*21 = 399 edges? Then add 2 intra-part edges.
    # Or K_{20,20} minus some edges plus more intra-part edges to balance?
    #
    # Actually, let's try a different approach: use simulated annealing / local search
    # starting from K_{20,20} + 1 random intra edge, then optimize.
    #
    # Even better: start from a random 401-edge graph and do local search with fast book updates.

    # Let me implement fast incremental book updates.
    # 
    # Key insight: When we add edge (u,v), the book sizes change as follows:
    # For any edge (i,j) in the graph (including the new one), its book size changes by:
    #   number of new common neighbors created by adding (u,v)
    # 
    # A vertex w becomes a new common neighbor of i and j if:
    #   w was not previously a common neighbor, but now is.
    #   This happens if: (w~i and w~v and v==j) or (w~j and w~v and v==i) or similar.
    # 
    # More precisely, adding (u,v) creates new triangles: (u,v,w) for each w adjacent to both u and v.
    # Each such triangle contributes to the book of edges (u,v), (u,w), (v,w).
    # 
    # So: 
    #   book(u,v) += num_common_neighbors(u,v)  [the new edge's book]
    #   For each w in N(u) ∩ N(v): book(u,w) += 1, book(v,w) += 1
    #   Also, for other edges (i,j) not involving u or v: does their book change?
    #   Edge (i,j)'s book = |N(i) ∩ N(j)|. Adding (u,v) doesn't change N(i) or N(j) unless i or j is u or v.
    #   So only edges involving u or v have their books changed!
    #
    # This is the key: adding edge (u,v) only affects books of edges incident to u or v.
    # Number of such edges: deg(u) + deg(v) - 1 (the edge (u,v) itself counted once).
    # For n=40, max degree ~40, so at most ~80 edges affected. Much faster than full matrix mult!

    def add_edge_update_books(A, books, u, v):
        """Add edge (u,v) and update books incrementally. Returns None (modifies in place)."""
        # Find common neighbors of u and v (these form new triangles)
        common = np.logical_and(A[u] > 0, A[v] > 0)
        common[u] = 0  # exclude u itself (though A[u,u]=0 already)
        common[v] = 0
        common_list = np.where(common)[0].tolist()
        num_common = len(common_list)
        
        # Add the edge
        A[u, v] = 1
        A[v, u] = 1
        
        # The new edge (u,v) has book size = num_common
        books[u, v] = num_common
        books[v, u] = num_common
        
        # For each common neighbor w, edges (u,w) and (v,w) gain 1 to their book
        for w in common_list:
            books[u, w] += 1
            books[w, u] += 1
            books[v, w] += 1
            books[w, v] += 1
        
        # Also, edges (u, x) for x not in common: do they change?
        # book(u,x) = |N(u) ∩ N(x)|. After adding (u,v), N(u) gains v.
        # So book(u,x) increases by 1 if v is in N(x), i.e., if A[v,x]=1.
        # Similarly for book(v,x).
        # Wait, I need to be more careful.
        # Before adding: book(u,x) = |{w : A[u,w]=1 and A[x,w]=1}|
        # After adding (u,v): N(u) now includes v. So book'(u,x) = |{w : A'[u,w]=1 and A[x,w]=1}|
        #   = |{w : A[u,w]=1 and A[x,w]=1}| + (1 if A[v,x]=1 and A[v,x] was just added... wait)
        #   The new neighbor of u is v. So book'(u,x) = book(u,x) + (1 if v ∈ N(x) else 0)
        #   But v ∈ N(x) means A[x,v]=1, which is true if there's an edge (x,v).
        #   However, we just added (u,v), not (x,v). So A[x,v] is unchanged unless x==u.
        #   For x != u,v: book'(u,x) = book(u,x) + (1 if A[x,v]==1 else 0)
        #   But A[x,v]==1 means x and v were already connected.
        #   Hmm, but this double-counts with the common neighbor update above!
        #
        # Let me reconsider. The common neighbor update already handles the case where w is a common neighbor.
        # For edge (u,w) where w is a common neighbor: book(u,w) increases by 1 because v becomes a new common neighbor of u and w? 
        # No wait. book(u,w) = |N(u) ∩ N(w)|. Before: some set S. After adding (u,v): N(u) gains v.
        # So book'(u,w) = |S ∪ {v if v ∈ N(w)}| = |S| + (1 if v ∈ N(w) else 0).
        # v ∈ N(w) means A[v,w]=1. But w is a common neighbor of u and v, so A[v,w]=1 by definition.
        # So for w in common_list, book(u,w) increases by 1. ✓ (already handled)
        #
        # For x NOT in common_list (and x != u,v): 
        #   book'(u,x) = book(u,x) + (1 if A[v,x]==1 else 0)
        #   But A[v,x]==1 means x is a neighbor of v. If x is also a neighbor of u, then x would be a common neighbor!
        #   Since x is NOT in common_list, either A[u,x]=0 or A[v,x]=0.
        #   If A[u,x]=0, then (u,x) is not an edge, so we don't care about its book.
        #   If A[v,x]=0, then the increment is 0.
        #   So for non-common-neighbor x with A[u,x]=1, we must have A[v,x]=0, so no increment.
        #
        # Therefore, the common neighbor update is sufficient! No other edges' books change.
        #
        # Wait, what about edge (v,w) for w in common_list? Same logic: book(v,w) increases by 1 because u becomes a new common neighbor of v and w. ✓
        #
        # What about edges not incident to u or v? Their books don't change because N(i) and N(j) don't change. ✓

    def remove_edge_update_books(A, books, u, v):
        """Remove edge (u,v) and update books incrementally."""
        # Find common neighbors of u and v (excluding the edge itself)
        common = np.logical_and(A[u] > 0, A[v] > 0)
        common[u] = 0
        common[v] = 0
        common_list = np.where(common)[0].tolist()
        num_common = len(common_list)
        
        # The edge (u,v) currently has book size = num_common
        books[u, v] = 0
        books[v, u] = 0
        
        # For each common neighbor w, edges (u,w) and (v,w) lose 1 from their book
        for w in common_list:
            books[u, w] -= 1
            books[w, u] -= 1
            books[v, w] -= 1
            books[w, v] -= 1
        
        # Remove the edge
        A[u, v] = 0
        A[v, u] = 0

    def max_book(A, books):
        masked = np.where(A > 0, books, 0)
        return int(masked.max())

    # --- Phase 1: Construct initial graph ---
    # Start from K_{20,20} (400 edges) and add 1 edge, then optimize.
    # But K_{20,20} + 1 intra edge has max book 20. 
    # Alternative: start from a random 401-edge graph.
    # Or: start from K_{20,20} and do local search to reduce the max book.
    
    # Let's start from K_{20,20} + 1 random intra-part edge.
    A = np.zeros((n, n), dtype=np.int8)
    # Create K_{20,20}: left = 0..19, right = 20..39
    A[:20, 20:] = 1
    A[20:, :20] = 1
    
    # Add one random intra-part edge
    left_nodes = list(range(20))
    right_nodes = list(range(20, 40))
    
    # Add edge within left part
    u1, v1 = random.sample(left_nodes, 2)
    A[u1, v1] = 1
    A[v1, u1] = 1
    
    # Now we have 401 edges. Compute books.
    books = compute_books(A)
    current_max = max_book(A, books)
    
    # --- Phase 2: Local search to minimize max book ---
    best_max = current_max
    best_A = A.copy()
    best_books = books.copy()
    
    # Get list of all edges and non-edges for faster sampling
    def get_edge_list(A):
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if A[i,j] == 1:
                    edges.append((i,j))
        return edges
    
    def get_nonedge_list(A):
        nonedges = []
        for i in range(n):
            for j in range(i+1, n):
                if A[i,j] == 0:
                    nonedges.append((i,j))
        return nonedges
    
    # Precompute edge and non-edge lists (will update during search)
    edge_list = get_edge_list(A)
    nonedge_list = get_nonedge_list(A)
    
    # Local search: try to swap an edge with a non-edge to reduce max book
    # Focus on edges with high book sizes.
    
    # Track temperature for simulated annealing-like behavior
    temp = 1.0
    no_improve_count = 0
    
    while time.time() < deadline:
        if best_max <= 7:
            break  # Theoretical minimum
        
        # Find edges with max book
        masked_books = np.where(A > 0, books, 0)
        max_val = int(masked_books.max())
        
        if max_val < best_max:
            best_max = max_val
            best_A = A.copy()
            best_books = books.copy()
            no_improve_count = 0
        elif max_val == best_max:
            no_improve_count += 1
        else:
            # Shouldn't happen if we only accept non-increasing moves
            no_improve_count += 1
        
        # If stuck for too long, try a random perturbation
        if no_improve_count > 500:
            # Do a random swap
            if edge_list and nonedge_list:
                eu, ev = random.choice(edge_list)
                xu, xv = random.choice(nonedge_list)
                if (eu, ev) == (xu, xv) or (eu, ev) == (xv, xu):
                    pass
                else:
                    # Tentative swap
                    remove_edge_update_books(A, books, eu, ev)
                    add_edge_update_books(A, books, xu, xv)
                    # Update lists
                    edge_list.remove((eu, ev))
                    nonedge_list.remove((xu, xv))
                    edge_list.append((xu, xv))
                    nonedge_list.append((eu, ev))
                    no_improve_count = 0
            continue
        
        # Pick a random edge with high book (prefer max-book edges)
        # Get all edges and their book sizes
        # For efficiency, sample from max-book edges first
        max_edges = []
        for i in range(n):
            for j in range(i+1, n):
                if A[i,j] == 1 and books[i,j] >= max_val - 2:  # near-max edges
                    max_edges.append((i,j))
        
        if not max_edges:
            max_edges = edge_list[:]
        
        # Try several random swaps
        found_improvement = False
        num_tries = min(30, len(max_edges) * 5)
        
        for _ in range(num_tries):
            # Pick an edge to remove (biased toward high-book edges)
            if random.random() < 0.7 and max_edges:
                ru, rv = random.choice(max_edges)
            elif edge_list:
                ru, rv = random.choice(edge_list)
            else:
                break
            
            # Pick a non-edge to add
            if not nonedge_list:
                break
            xu, xv = random.choice(nonedge_list)
            
            if (ru, rv) == (xu, xv) or (ru, rv) == (xv, xu):
                continue
            
            # Tentative swap
            remove_edge_update_books(A, books, ru, rv)
            add_edge_update_books(A, books, xu, xv)
            
            new_max = max_book(A, books)
            
            if new_max <= max_val:
                # Accept (equal or better)
                # Update lists
                edge_list.remove((ru, rv))
                nonedge_list.remove((xu, xv))
                edge_list.append((xu, xv))
                nonedge_list.append((ru, rv))
                found_improvement = True
                if new_max < max_val:
                    break  # Found improvement, restart
            else:
                # Revert
                remove_edge_update_books(A, books, xu, xv)
                add_edge_update_books(A, books, ru, rv)
        
        if found_improvement:
            no_improve_count = 0
    
    # Extract final graph
    final_A = best_A
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if final_A[i,j] == 1:
                edges.append([i, j])
    
    # Ensure exactly 401 edges
    if len(edges) != target_edges:
        # Fallback: rebuild from scratch
        A2 = np.zeros((n, n), dtype=np.int8)
        A2[:20, 20:] = 1
        A2[20:, :20] = 1
        u1, v1 = random.sample(list(range(20)), 2)
        A2[u1, v1] = 1
        A2[v1, u1] = 1
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if A2[i,j] == 1:
                    edges.append([i, j])
    
    return edges
