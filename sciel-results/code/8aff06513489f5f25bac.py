import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 147.0  # safety margin before 150s kill

    n = 31
    q = 5
    mod = q

    # Build polarity graph on PG(2,q)
    points = []
    for y in range(q):
        for z in range(q):
            points.append((1, y, z))
    for z in range(q):
        points.append((0, 1, z))
    points.append((0, 0, 1))
    
    # Adjacency matrix
    A = np.zeros((n, n), dtype=np.int8)
    
    # Base edges
    base_edges = []
    for i in range(n):
        for j in range(i+1, n):
            x1, y1, z1 = points[i]
            x2, y2, z2 = points[j]
            if (x1*x2 + y1*y2 + z1*z2) % mod == 0:
                A[i, j] = 1
                A[j, i] = 1
                base_edges.append((i, j))

    m_base = len(base_edges)
    target_m = 91
    to_add = target_m - m_base

    # Lambda matrix: Lambda[i,j] = number of common neighbors of i and j
    # Lambda = A @ A (since A is symmetric and 0-1)
    # Diagonal is degree, off-diagonal is common neighbors
    Lambda = (A @ A).astype(np.int32)
    
    # Helper to get neighbors
    def get_neighbors(u):
        return np.where(A[u] == 1)[0]

    # Function to calculate score from Lambda
    # Score = 0.5 * sum_{i<j} C(Lambda[i,j], 2)
    def calc_score(L):
        # We only care about off-diagonal
        # Triangular part
        iu, ju = np.triu_indices(n, k=1)
        vals = L[iu, ju]
        # C(x, 2) = x*(x-1)/2
        # Sum C(x,2) / 2
        return np.sum(vals * (vals - 1)) // 4

    current_score = calc_score(Lambda)
    
    # Phase 1: Greedily add edges
    # We need to add 'to_add' edges
    # Cost of adding (u,v) is the number of length-3 paths between u and v
    # which is sum_{w in N(v)} Lambda[w, u]
    
    # Precompute non-edges
    non_edges = [(i, j) for i in range(n) for j in range(i+1, n) if A[i, j] == 0]
    
    for _ in range(to_add):
        if time.time() > deadline - 5.0:
            break
            
        best_cost = float('inf')
        best_edge = None
        
        # Evaluate all non-edges
        for u, v in non_edges:
            # Cost = sum_{w in N(v)} Lambda[w, u]
            # N(v) is where A[v] == 1
            # We can use dot product: A[v] @ Lambda[:, u]
            # But A[v] is 0/1, so it's just sum of Lambda[w, u] for w in N(v)
            cost = np.dot(A[v], Lambda[:, u])
            
            if cost < best_cost:
                best_cost = cost
                best_edge = (u, v)
        
        if best_edge is None:
            break
            
        u, v = best_edge
        # Add edge
        A[u, v] = 1
        A[v, u] = 1
        
        # Update Lambda
        # Adding (u,v) means u and v are now neighbors.
        # This increases common neighbor count for pairs (a,b) where a~u, b~u (u is common neighbor)
        # and pairs (a,b) where a~v, b~v.
        # Specifically:
        # For all a in N(u), b in N(u): Lambda[a,b] += 1
        # For all a in N(v), b in N(v): Lambda[a,b] += 1
        
        nu = get_neighbors(u)
        nv = get_neighbors(v)
        
        # Update for u
        for a in nu:
            for b in nu:
                Lambda[a, b] += 1
                Lambda[b, a] += 1
        # Update for v
        for a in nv:
            for b in nv:
                Lambda[a, b] += 1
                Lambda[b, a] += 1
        
        # Remove from non_edges
        non_edges.remove(best_edge)
        
        current_score = calc_score(Lambda)

    # Phase 2: Local Search (Swaps)
    # Try to find a swap (remove e1, add e2) that reduces score
    
    best_score = current_score
    best_A = A.copy()
    best_Lambda = Lambda.copy()
    
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() < deadline - 2.0:
        if no_improve_count > max_no_improve:
            # Perturbation
            # Pick random edge to remove, random non-edge to add
            if len(non_edges) == 0:
                break
                
            e1 = base_edges[np.random.randint(len(base_edges))] # Just pick any existing edge? 
            # Better: pick random existing edge
            existing_edges = [(i, j) for i in range(n) for j in range(i+1, n) if A[i, j] == 1]
            if not existing_edges:
                break
            e1 = existing_edges[np.random.randint(len(existing_edges))]
            e2 = non_edges[np.random.randint(len(non_edges))]
            
            # Apply swap
            u1, v1 = e1
            u2, v2 = e2
            
            # Remove e1
            A[u1, v1] = 0
            A[v1, u1] = 0
            nu1 = get_neighbors(u1)
            nv1 = get_neighbors(v1)
            for a in nu1:
                for b in nu1:
                    Lambda[a, b] -= 1
                    Lambda[b, a] -= 1
            for a in nv1:
                for b in nv1:
                    Lambda[a, b] -= 1
                    Lambda[b, a] -= 1
            
            # Add e2
            A[u2, v2] = 1
            A[v2, u2] = 1
            nu2 = get_neighbors(u2)
            nv2 = get_neighbors(v2)
            for a in nu2:
                for b in nu2:
                    Lambda[a, b] += 1
                    Lambda[b, a] += 1
            for a in nv2:
                for b in nv2:
                    Lambda[a, b] += 1
                    Lambda[b, a] += 1
            
            non_edges.remove(e2)
            # e1 is now a non-edge
            non_edges.append(e1)
            
            no_improve_count = 0
            continue

        improved = False
        
        # Evaluate swaps
        # To make it fast, we only consider a subset of removals and additions
        existing_edges = [(i, j) for i in range(n) for j in range(i+1, n) if A[i, j] == 1]
        
        # Sort non_edges by "potential" (low cost to add)
        # Cost to add (u,v) is sum_{w in N(v)} Lambda[w, u]
        add_costs = []
        for u, v in non_edges:
            cost = np.dot(A[v], Lambda[:, u])
            add_costs.append((cost, (u, v)))
        add_costs.sort(key=lambda x: x[0])
        
        # Sort existing edges by "potential" (high cost to remove? i.e. low C4 contribution)
        # Contribution of edge (u,v) to C4 count is roughly the number of C4s it is part of.
        # Which is the cost to add it? No, it's the number of length-3 paths between u and v *excluding* the edge itself?
        # Actually, the number of C4s containing (u,v) is the number of paths of length 3 between u and v.
        # So we want to remove edges with LOW C4 count.
        rem_costs = []
        for u, v in existing_edges:
            # Temporarily remove to get accurate count?
            # Approx: cost to add was the number of C4s created.
            # So we can just use the add cost as a proxy for removal benefit.
            # But Lambda has changed.
            # Let's just compute it: sum_{w in N(v)} Lambda[w, u]
            # This counts paths v-w-x-u.
            # If (u,v) is an edge, this includes paths that go through (u,v)? No, length 3 paths don't use (u,v) directly as a middle edge.
            # Wait, v-w-x-u. If w=u, then v-u-x-u? No, simple paths.
            # The formula sum_{w in N(v)} Lambda[w, u] counts all length 3 walks.
            # For a simple graph, this is the number of C4s containing (u,v).
            cost = np.dot(A[v], Lambda[:, u])
            rem_costs.append((cost, (u, v)))
        rem_costs.sort(key=lambda x: x[0])
        
        # Try top 10 removals and top 10 additions
        found_improvement = False
        for rc, re in rem_costs[:10]:
            if found_improvement:
                break
            u1, v1 = re
            for ac, ae in add_costs[:10]:
                u2, v2 = ae
                # Calculate delta score
                # We need to update Lambda for the swap and calc new score
                # This is expensive if done for all 100 pairs.
                # But n=31, so Lambda update is small.
                
                # Save state
                old_Lambda = Lambda.copy()
                
                # Remove (u1, v1)
                A[u1, v1] = 0
                A[v1, u1] = 0
                nu1 = get_neighbors(u1)
                nv1 = get_neighbors(v1)
                for a in nu1:
                    for b in nu1:
                        Lambda[a, b] -= 1
                        Lambda[b, a] -= 1
                for a in nv1:
                    for b in nv1:
                        Lambda[a, b] -= 1
                        Lambda[b, a] -= 1
                        
                # Add (u2, v2)
                A[u2, v2] = 1
                A[v2, u2] = 1
                nu2 = get_neighbors(u2)
                nv2 = get_neighbors(v2)
                for a in nu2:
                    for b in nu2:
                        Lambda[a, b] += 1
                        Lambda[b, a] += 1
                for a in nv2:
                    for b in nv2:
                        Lambda[a, b] += 1
                        Lambda[b, a] += 1
                        
                new_score = calc_score(Lambda)
                
                if new_score < current_score:
                    current_score = new_score
                    if current_score < best_score:
                        best_score = current_score
                        best_A = A.copy()
                        best_Lambda = Lambda.copy()
                    
                    # Commit swap
                    non_edges.remove(ae)
                    non_edges.append(re)
                    found_improvement = True
                    improved = True
                    break
                else:
                    # Revert
                    A[u1, v1] = 1
                    A[v1, u1] = 1
                    A[u2, v2] = 0
                    A[v2, u2] = 0
                    Lambda = old_Lambda

        if improved:
            no_improve_count = 0
        else:
            no_improve_count += 1

    # Final check
    if calc_score(best_Lambda) < best_score:
         best_score = calc_score(best_Lambda)
         
    # Return edges from best_A
    final_edges = []
    for i in range(n):
        for j in range(i+1, n):
            if best_A[i, j] == 1:
                final_edges.append([i, j])
                
    return final_edges
