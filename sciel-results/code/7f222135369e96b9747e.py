import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    n = 40
    target_edges = 401
    
    # Build initial graph: K_{20,20} plus one extra edge
    # K_{20,20} has 400 edges, each vertex degree 20
    # Adding one edge makes 401 edges, two vertices degree 21
    
    def make_initial_graph():
        A = np.zeros((n, n), dtype=np.int8)
        for i in range(20):
            for j in range(20, 40):
                A[i, j] = 1
                A[j, i] = 1
        # Add one random edge within a partition
        u = random.randint(0, 19)
        v = random.randint(0, 19)
        if u != v:
            A[u, v] = 1
            A[v, u] = 1
        else:
            A[0, 1] = 1
            A[1, 0] = 1
        return A
    
    def compute_books(A):
        A2 = A @ A
        books = A2 * A
        return books
    
    def get_max_book(books):
        return int(np.max(books))
    
    def get_edges_list(A):
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if A[i, j] == 1:
                    edges.append((i, j))
        return edges
    
    def get_nonedges_list(A):
        nonedges = []
        for i in range(n):
            for j in range(i+1, n):
                if A[i, j] == 0:
                    nonedges.append((i, j))
        return nonedges
    
    def randomize_graph(A, num_swaps=5000):
        """Apply random edge swaps to explore the space of 401-edge graphs."""
        A = A.copy()
        for _ in range(num_swaps):
            # Pick random edge to remove
            edges = get_edges_list(A)
            if not edges:
                break
            u1, v1 = random.choice(edges)
            
            # Pick random non-edge to add
            nonedges = get_nonedges_list(A)
            if not nonedges:
                break
            u2, v2 = random.choice(nonedges)
            
            # Check: after removing (u1,v1) and adding (u2,v2), 
            # we need no multi-edges and the swap is valid
            # (u2,v2) must not be (u1,v1) or (v1,u1) - already guaranteed by non-edge
            # After removing (u1,v1), (u2,v2) is still a non-edge unless 
            # (u2,v2) == (u1,v1) which is impossible since (u2,v2) is non-edge
            A[u1, v1] = 0
            A[v1, u1] = 0
            A[u2, v2] = 1
            A[v2, u2] = 1
        return A
    
    def local_search(A, max_iters=5000):
        """Greedy + simulated annealing local search using fast book updates."""
        A = A.copy()
        books = compute_books(A)
        current_score = get_max_book(books)
        best_score = current_score
        best_A = A.copy()
        
        T = 2.0
        T_min = 0.01
        cool = 0.998
        
        edges = get_edges_list(A)
        nonedges = get_nonedges_list(A)
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            if not edges or not nonedges:
                break
            
            u1, v1 = random.choice(edges)
            u2, v2 = random.choice(nonedges)
            
            # Compute delta in max book size
            # We need to track which edges' book sizes change
            # Removing (u1,v1): affects books of all edges incident to u1 or v1
            # Adding (u2,v2): affects books of all edges incident to u2 or v2
            
            # For efficiency, compute new books only for affected edges
            # Affected edges: those sharing a vertex with u1, v1, u2, or v2
            
            # Get neighbors
            neigh_u1 = set(np.where(A[u1] == 1)[0].tolist())
            neigh_v1 = set(np.where(A[v1] == 1)[0].tolist())
            neigh_u2 = set(np.where(A[u2] == 1)[0].tolist())
            neigh_v2 = set(np.where(A[v2] == 1)[0].tolist())
            
            # Edges that will have their book size change
            # When removing (u1,v1): edges (x,u1) for x in neigh_u1\{v1} and (x,v1) for x in neigh_v1\{u1}
            # When adding (u2,v2): edges (x,u2) for x in neigh_u2\{v2} and (x,v2) for x in neigh_v2\{u2}
            # Plus the new edge (u2,v2) itself
            
            # Build set of affected edges
            affected = set()
            for x in neigh_u1:
                if x != v1:
                    affected.add((min(x, u1), max(x, u1)))
            for x in neigh_v1:
                if x != u1:
                    affected.add((min(x, v1), max(x, v1)))
            for x in neigh_u2:
                if x != v2:
                    affected.add((min(x, u2), max(x, u2)))
            for x in neigh_v2:
                if x != u2:
                    affected.add((min(x, v2), max(x, v2)))
            affected.add((min(u2, v2), max(u2, v2)))
            affected.add((min(u1, v1), max(u1, v1)))
            
            # Compute new book sizes for affected edges
            # After swap: remove (u1,v1), add (u2,v2)
            # New adjacency: A' where A'[u1,v1]=A'[v1,u1]=0, A'[u2,v2]=A'[v2,u2]=1, rest same
            
            # For each affected edge (e1, e2), compute new common neighbors
            # new_book(e1,e2) = sum_k A'[e1,k] * A'[e2,k]
            # This equals old_book(e1,e2) + delta
            # delta from removing (u1,v1): 
            #   If e1==u1 or e2==u1: the term k=v1 was A[e1,v1]*A[u1,v1]=A[e1,v1]*1, now 0 => -A[e1,v1] (if e1==u1) or -A[e2,v1] (if e2==u1)
            #   Similarly for v1
            # delta from adding (u2,v2):
            #   If e1==u2 or e2==u2: +A[e1,v2] (if e1==u2) or +A[e2,v2] (if e2==u2)
            #   Similarly for v2
            
            # Let's just compute directly for affected edges
            new_books_affected = {}
            for (a, b) in affected:
                if a == b:
                    continue
                # Compute common neighbors in new graph
                # A'[a,k] * A'[b,k] for all k
                # Start with old books[a,b]
                old_val = books[a, b]
                new_val = old_val
                
                # Effect of removing (u1, v1):
                # For edge (a,b), the term k=v1 contributed A[a,v1]*A[b,v1] if (u1,v1) was... 
                # Wait, removing (u1,v1) changes A[u1,v1] and A[v1,u1] to 0.
                # This affects common(a,b) only if a==u1 or b==u1 or a==v1 or b==v1.
                # If a==u1: term k=v1 was A[u1,v1]*A[b,v1] = 1*A[b,v1]. Now A[u1,v1]=0, so lose A[b,v1].
                #   But also, for k=u1: A[u1,u1]*A[b,u1] = 0*A[b,u1] = 0, no change.
                #   Actually, the term for k in the sum is A[a,k]*A[b,k].
                #   When we change A[u1,v1] from 1 to 0:
                #     If a==u1 and k==v1: A[u1,v1]*A[b,v1] changes from A[b,v1] to 0. Delta = -A[b,v1].
                #     If b==u1 and k==v1: A[a,v1]*A[u1,v1] changes from A[a,v1] to 0. Delta = -A[a,v1].
                #     If a==v1 and k==u1: A[v1,u1]*A[b,u1] changes from A[b,u1] to 0. Delta = -A[b,u1].
                #     If b==v1 and k==u1: A[a,u1]*A[v1,u1] changes from A[a,u1] to 0. Delta = -A[a,u1].
                #   Note: if a==u1 and b==v1, both apply, but that's the edge being removed.
                
                # Effect of adding (u2, v2):
                #   If a==u2 and k==v2: A[u2,v2]*A[b,v2] changes from 0 to A[b,v2]. Delta = +A[b,v2].
                #   If b==u2 and k==v2: A[a,v2]*A[u2,v2] changes from 0 to A[a,v2]. Delta = +A[a,v2].
                #   If a==v2 and k==u2: A[v2,u2]*A[b,u2] changes from 0 to A[b,u2]. Delta = +A[b,u2].
                #   If b==v2 and k==u2: A[a,u2]*A[v2,u2] changes from 0 to A[a,u2]. Delta = +A[a,u2].
                
                # Remove (u1,v1) effects:
                if a == u1 and b != v1:
                    new_val -= A[b, v1]
                elif b == u1 and a != v1:
                    new_val -= A[a, v1]
                if a == v1 and b != u1:
                    new_val -= A[b, u1]
                elif b == v1 and a != u1:
                    new_val -= A[a, u1]
                
                # Add (u2,v2) effects:
                if a == u2 and b != v2:
                    new_val += A[b, v2]
                elif b == u2 and a != v2:
                    new_val += A[a, v2]
                if a == v2 and b != u2:
                    new_val += A[b, u2]
                elif b == v2 and a != u2:
                    new_val += A[a, u2]
                
                # Special case: if (a,b) == (u1,v1), this edge is removed, so its book size is irrelevant
                # If (a,b) == (u2,v2), this is the new edge
                if (a, b) == (u1, v1) or (a, b) == (v1, u1):
                    # Edge removed, don't care
                    continue
                if (a, b) == (u2, v2) or (a, b) == (v2, u2):
                    # New edge, compute directly
                    # common(u2,v2) = sum_k A[u2,k]*A[v2,k] in new graph
                    # = old common + 0 (since u2,v2 were not connected before, no shared neighbor change from the swap itself except...)
                    # Actually, the new edge's book size is just the number of common neighbors of u2 and v2 in the new graph.
                    # In the new graph, u2 and v2 are connected, but their other neighbors are the same as before
                    # (since we only removed (u1,v1) which doesn't affect u2 or v2's neighbor sets unless u2/v2 are u1/v1)
                    # So new_book = old common(u2,v2) in the graph before adding (u2,v2)
                    # = sum_k A[u2,k]*A[v2,k] where A is the state before adding but after removing (u1,v1)
                    # Since (u1,v1) removal doesn't change A[u2,:] or A[v2,:] (unless u2/v2 are u1/v1),
                    # it's just the old books[u2,v2] if (u2,v2) was a non-edge... 
                    # But books[u2,v2] was 0 since it was a non-edge.
                    # The actual common neighbors: sum_k A[u2,k]*A[v2,k]
                    # This is what we computed above as new_val, but let's verify.
                    # For the new edge (u2,v2), a=u2, b=v2:
                    # Remove (u1,v1) effects: u2 != u1, u2 != v1, v2 != u1, v2 != v1 (assuming distinct)
                    #   So no change from removal.
                    # Add (u2,v2) effects: a==u2, b==v2, but b==v2 so skip first, a==v2? No.
                    #   Actually for the new edge itself, the "add" effects are about other edges, not itself.
                    #   The book size of the new edge is just the number of common neighbors.
                    #   In the new graph, common(u2,v2) = sum_k A'[u2,k]*A'[v2,k]
                    #   A' differs from A only at (u1,v1) and (u2,v2).
                    #   For k not in {u1,v1,u2,v2}: A'[u2,k]=A[u2,k], A'[v2,k]=A[v2,k].
                    #   For k=u1: A'[u2,u1]=A[u2,u1], A'[v2,u1]=A[v2,u1]. No change unless u1==u2 or u1==v2.
                    #   For k=v1: A'[u2,v1]=A[u2,v1], A'[v2,v1]=A[v2,v1]. No change unless v1==u2 or v1==v2.
                    #   For k=u2: A'[u2,u2]=0, A'[v2,u2]=1 (new edge). So term is 0*1=0.
                    #   For k=v2: A'[u2,v2]=1 (new edge), A'[v2,v2]=0. Term is 1*0=0.
                    #   So common(u2,v2) = sum_{k not in {u2,v2}} A[u2,k]*A[v2,k]
                    #   = sum_k A[u2,k]*A[v2,k] - A[u2,u2]*A[v2,u2] - A[u2,v2]*A[v2,v2]
                    #   = sum_k A[u2,k]*A[v2,k] - 0 - 0 = sum_k A[u2,k]*A[v2,k]
                    #   But in the old graph, A[u2,v2]=0, so sum_k A[u2,k]*A[v2,k] already excludes the (u2,v2) term.
                    #   So new_book = sum_k A[u2,k]*A[v2,k] in the graph after removing (u1,v1) but before adding (u2,v2).
                    #   If (u1,v1) doesn't involve u2 or v2, this is just the old common neighbors.
                    #   We can compute it as: np.sum(A[u2] * A[v2]) where A is after removing (u1,v1).
                    #   But we haven't updated A yet. So use old A, and adjust if needed.
                    #   For simplicity, just compute: 
                    pass  # We'll handle below
                
                new_books_affected[(a, b)] = new_val
            
            # Compute the new edge's book size directly
            # After removing (u1,v1) and adding (u2,v2):
            # common(u2,v2) = sum_k A_new[u2,k]*A_new[v2,k]
            # A_new[u2,k] = A[u2,k] except if k==v1 and u2==u1 (but u2!=u1 typically)
            # Let's just compute it:
            # Temporarily consider the state
            common_u2v2 = 0
            for k in range(n):
                a_u2_k = A[u2, k]
                a_v2_k = A[v2, k]
                # Adjust for removal of (u1,v1)
                if u2 == u1 and k == v1:
                    a_u2_k = 0
                if v2 == u1 and k == v1:
                    a_v2_k = 0
                if u2 == v1 and k == u1:
                    a_u2_k = 0
                if v2 == v1 and k == u1:
                    a_v2_k = 0
                # Adjust for addition of (u2,v2)
                if u2 == u2 and k == v2:
                    a_u2_k = 1  # A[u2,v2] becomes 1
                if v2 == u2 and k == v2:
                    a_v2_k = 1  # A[v2,u2] becomes 1
                if u2 == v2 and k == u2:
                    a_u2_k = 1
                if v2 == v2 and k == u2:
                    a_v2_k = 1
                common_u2v2 += a_u2_k * a_v2_k
            
            new_books_affected[(min(u2,v2), max(u2,v2))] = common_u2v2
            
            # Now compute new max book size
            # The new max is max of:
            # 1. New books for affected edges
            # 2. Old max for unaffected edges (but we need to be careful)
            
            # For simplicity, recompute max from scratch if affected set is large,
            # otherwise use the new values
            if len(affected) < n:
                # Compute max over all edges
                new_max = 0
                for i in range(n):
                    for j in range(i+1, n):
                        if A[i, j] == 1 and not (i == u1 and j == v1) and not (i == v1 and j == u1):
                            # This edge exists after swap
                            if (i, j) in new_books_affected:
                                val = new_books_affected[(i, j)]
                            else:
                                val = books[i, j]
                            if val > new_max:
                                new_max = val
                        elif (i, j) == (u2, v2) or (i, j) == (v2, u2):
                            # New edge
                            val = new_books_affected.get((min(u2,v2), max(u2,v2)), 0)
                            if val > new_max:
                                new_max = val
                new_score = new_max
            else:
                # Just recompute fully
                # Apply swap to a copy
                A_temp = A.copy()
                A_temp[u1, v1] = 0
                A_temp[v1, u1] = 0
                A_temp[u2, v2] = 1
                A_temp[v2, u2] = 1
                books_temp = compute_books(A_temp)
                new_score = get_max_book(books_temp)
            
            delta = new_score - current_score
            if delta < 0 or random.random() < np.exp(-delta / T):
                # Accept
                A[u1, v1] = 0
                A[v1, u1] = 0
                A[u2, v2] = 1
                A[v2, u2] = 1
                
                # Update books
                for (a, b), val in new_books_affected.items():
                    if a != b and ((A[a, b] == 1) or (a == u2 and b == v2) or (a == v2 and b == u2)):
                        books[a, b] = val
                        books[b, a] = val
                # Clear books for removed edge
                books[u1, v1] = 0
                books[v1, u1] = 0
                
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_A = A.copy()
                
                # Update edge lists
                edges = get_edges_list(A)
                nonedges = get_nonedges_list(A)
            
            T = max(T * cool, T_min)
        
        return best_A, best_score
    
    best_A = None
    best_score = float('inf')
    
    # Multiple restarts with randomization
    num_restarts = 5
    for r in range(num_restarts):
        if time.time() > deadline:
            break
        
        A = make_initial_graph()
        A = randomize_graph(A, num_swaps=3000)
        
        A, score = local_search(A, max_iters=3000)
        
        if score < best_score:
            best_score = score
            best_A = A.copy()
    
    # Convert best_A to list of edges
    if best_A is None:
        return [[a, b] for a in range(20) for b in range(20, 40)] + [[0, 1]]
    
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if best_A[i, j] == 1:
                edges.append([i, j])
    
    return edges
