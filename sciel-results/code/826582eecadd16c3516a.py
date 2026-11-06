import time
import random

def construct():
    n = 20
    start_time = time.time()
    # Use a conservative deadline to ensure we return before the 120s kill
    deadline = start_time + 100.0 

    def is_add_valid(adj, u, v):
        """
        Fast check: does adding edge (u,v) create a K3,3?
        
        Key insight: Before adding (u,v), the graph was K3,3-free.
        After adding (u,v), a K3,3 can only be formed if there exists a triple
        {x,y,z} such that {u,v} becomes part of a 3x3 bipartite subgraph.
        
        Specifically, a new K3,3 must involve u or v as part of one partition
        and two other vertices as part of the other partition, where u and v
        serve as two of the three common neighbors.
        
        Case 1: u and v are in the same partition (both connect to {a,b,c}).
          Need a,b,c all adjacent to both u and v.
          So common neighbors of u and v (excluding each other) must have size >= 3.
          
        Case 2: u is in partition A, v is in partition B.
          Need two more vertices a,b in A (adjacent to v and to all of {v}∪... wait)
          Actually: partition A = {u, a, b}, partition B = {v, c, d}
          Edges needed: u-v, u-c, u-d, a-v, a-c, a-d, b-v, b-c, b-d
          So a,b must be adjacent to v (already) and to c,d
          And c,d must be adjacent to u (already) and to a,b
          
        Let me think more carefully. A K3,3 has partitions {x1,x2,x3} and {y1,y2,y3}.
        The new edge (u,v) must be one of the 9 edges.
        
        If u~v is the new edge, then either:
        (a) u ∈ {x1,x2,x3}, v ∈ {y1,y2,y3} (or vice versa)
        
        Sub-case (a1): u is in partition X, v is in partition Y.
          Then we need two more vertices a,b in X that are adjacent to all of Y={v,c,d}
          And two more vertices c,d in Y that are adjacent to all of X={u,a,b}
          
          For a to be in X: a~v, a~c, a~d (a~v is given since a ∈ N(v))
          For c to be in Y: c~u, c~a, c~b (c~u is given since c ∈ N(u))
          
          So we need: a,b ∈ N(v) \ {u,v}, c,d ∈ N(u) \ {u,v}
          such that: a~c, a~d, b~c, b~d (the cross edges between {a,b} and {c,d})
          Plus: a~v, b~v (given), c~u, d~u (given), u~v (new)
          
          So the condition is: there exist a,b ∈ N(v)\{u,v} and c,d ∈ N(u)\{u,v}
          such that all four cross edges a-c, a-d, b-c, b-d exist.
          
        Sub-case (a2): v is in partition X, u is in partition Y.
          Symmetric: need a,b ∈ N(u)\{u,v} and c,d ∈ N(v)\{u,v}
          such that all four cross edges exist.
          
        Sub-case (a3): Both u and v are in the same partition, say X={u,v,a}.
          Then Y={b,c,d} and we need u~b,c,d and v~b,c,d and a~b,c,d.
          So b,c,d must all be in N(u) ∩ N(v) ∩ N(a) for some a.
          Equivalently: |N(u) ∩ N(v) \ {u,v}| >= 3.
          
        Wait, that's not quite right. If X={u,v,a}, then we need b,c,d ∈ Y such that
        each of u,v,a connects to each of b,c,d.
        So b,c,d must be in N(u) ∩ N(v) ∩ N(a).
        
        So the full check is:
        1. |N(u) ∩ N(v) \ {u,v}| >= 3  (same partition case)
        2. There exist a,b ∈ N(v)\{u,v} and c,d ∈ N(u)\{u,v} with all cross edges
        3. There exist a,b ∈ N(u)\{u,v} and c,d ∈ N(v)\{u,v} with all cross edges
        
        But cases 2 and 3 are the same condition (just swap u and v)!
        
        So we need to check:
        - common = (nu & nv) & ~(1<<u) & ~(1<<v); if common.bit_count() >= 3: return False
        - For a,b in N(v)\{u,v} and c,d in N(u)\{u,v}: check if a~c, a~d, b~c, b~d
        
        For the second check, we can iterate over pairs (a,b) in N(v) and check
        if there are at least 2 vertices c in N(u) that are adjacent to both a and b.
        I.e., for each pair (a,b) in N(v)\{u,v}, count = (nu & adj[a] & adj[b]).bit_count()
        If count >= 2, we have a K3,3.
        
        Wait, that's exactly what the original code does! Let me re-examine.
        
        Actually the original code checks:
        - For pairs (a,b) in N(v): if (nu & adj[a] & adj[b]).bit_count() >= 3
        - For pairs (a,b) in N(u): if (nv & adj[a] & adj[b]).bit_count() >= 3
        
        But this is checking for 3 common neighbors, not 2. Let me re-derive.
        
        If u ∈ X, v ∈ Y, a,b ∈ X, c,d ∈ Y:
        We need a~c, a~d, b~c, b~d.
        c,d ∈ N(u) means c~u, d~u.
        a,b ∈ N(v) means a~v, b~v.
        
        For a fixed pair (a,b) ∈ N(v), the vertices c that work are those in N(u) ∩ N(a) ∩ N(b).
        We need at least 2 such vertices.
        
        So the check should be: for pairs (a,b) in N(v)\{u,v}, if (nu & adj[a] & adj[b]).bit_count() >= 2, return False.
        
        Similarly for the symmetric case.
        
        But wait, the original code checks >= 3. Is that because adj already has the edge (u,v) added?
        If adj has (u,v), then nu includes v and nv includes u.
        So (nu & adj[a] & adj[b]) might include v if v~a and v~b (which it does since a,b ∈ N(v)).
        So the count includes v. We need count >= 2 from N(u), but one of them might be v.
        So we need (nu & adj[a] & adj[b] & ~(1<<v)).bit_count() >= 2.
        Which is equivalent to (nu & adj[a] & adj[b]).bit_count() >= 3 (since v is always in that set).
        
        So the original check of >= 3 is correct when adj already has the edge.
        
        The optimization: instead of checking all pairs in N(v), we can early-exit
        and use bit operations more efficiently.
        """
        nu = adj[u]
        nv = adj[v]
        
        # Case 1: u and v in same partition
        common_uv = nu & nv & ~(1 << u) & ~(1 << v)
        if common_uv.bit_count() >= 3:
            return False
        
        # Case 2: u in X, v in Y (and symmetric)
        # For pairs (a,b) in N(v)\{u,v}: check (nu & adj[a] & adj[b]).bit_count() >= 3
        vn_mask = nv & ~(1 << u) & ~(1 << v)
        vn_list = []
        m = vn_mask
        while m:
            lb = m & (-m)
            a = lb.bit_length() - 1
            vn_list.append(a)
            m ^= lb
        
        for i in range(len(vn_list)):
            a = vn_list[i]
            adj_a = adj[a]
            for j in range(i + 1, len(vn_list)):
                b = vn_list[j]
                common = nu & adj_a & adj[b]
                if common.bit_count() >= 3:
                    return False
        
        # Symmetric case
        un_mask = nu & ~(1 << v) & ~(1 << u)
        un_list = []
        m = un_mask
        while m:
            lb = m & (-m)
            a = lb.bit_length() - 1
            un_list.append(a)
            m ^= lb
        
        for i in range(len(un_list)):
            a = un_list[i]
            adj_a = adj[a]
            for j in range(i + 1, len(un_list)):
                b = un_list[j]
                common = nv & adj_a & adj[b]
                if common.bit_count() >= 3:
                    return False
        
        return True

    def count_edges(adj):
        return sum(a.bit_count() for a in adj) // 2

    def greedy_fill(adj):
        """
        Greedily adds edges in random order.
        Returns the number of edges in the final adj.
        """
        pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
        random.shuffle(pairs)
        for u, v in pairs:
            if time.time() > deadline:
                break
            if adj[u] & (1 << v):
                continue
            
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
            if is_add_valid(adj, u, v):
                pass
            else:
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
        
        return count_edges(adj)

    def get_edges(adj):
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u] & (1 << v):
                    edges.append((u, v))
        return edges

    def try_add_edge(adj, u, v):
        """Try to add edge (u,v), return True if successful."""
        if adj[u] & (1 << v):
            return False
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        if is_add_valid(adj, u, v):
            return True
        else:
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            return False

    def local_search(adj, max_iter=200):
        """
        Local search: try to add any non-edge that improves the count.
        If stuck, try 1-edge swaps (remove one, add one).
        Returns improved count.
        """
        current_count = count_edges(adj)
        
        # Phase 1: Try adding any valid non-edge
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for u in range(n):
                for v in range(u + 1, n):
                    if time.time() > deadline:
                        return current_count
                    if adj[u] & (1 << v):
                        continue
                    if try_add_edge(adj, u, v):
                        current_count += 1
                        improved = True
            if improved:
                continue
        
        # Phase 2: Try 1-edge swaps if no more additions possible
        for it in range(max_iter):
            if time.time() > deadline:
                break
            
            edges = get_edges(adj)
            non_edges = [(u, v) for u in range(n) for v in range(u + 1, n) 
                        if not (adj[u] & (1 << v))]
            
            if not edges or not non_edges:
                break
            
            improved = False
            # Try all non-edges with each edge removal (limited for speed)
            random.shuffle(edges)
            for ru, rv in edges[:10]:
                if time.time() > deadline:
                    return current_count
                adj[ru] &= ~(1 << rv)
                adj[rv] &= ~(1 << ru)
                
                for nu_, nv_ in non_edges:
                    if time.time() > deadline:
                        adj[ru] |= (1 << rv)
                        adj[rv] |= (1 << ru)
                        return current_count
                    if try_add_edge(adj, nu_, nv_):
                        current_count += 1
                        improved = True
                        break
                    else:
                        adj[nu_] &= ~(1 << nv_)
                        adj[nv_] &= ~(1 << nu_)
                
                if not improved:
                    adj[ru] |= (1 << rv)
                    adj[rv] |= (1 << ru)
                # If improved, keep the state (edge removed, new edge added)
            
            if not improved:
                break
        
        return current_count

    best_adj = [0] * n
    best_count = 0
    current_adj = [0] * n
    
    # Initial greedy fill from empty graph
    current_count = greedy_fill(current_adj)
    if current_count > best_count:
        best_count = current_count
        best_adj = current_adj[:]

    # Local search / Restart strategy
    iterations = 0
    no_improve_count = 0
    while time.time() < deadline:
        iterations += 1
        current_adj = best_adj[:]
        
        # Adaptive removal: start with fewer, increase if stuck
        if no_improve_count < 10:
            num_removals = random.randint(1, 3)
        elif no_improve_count < 30:
            num_removals = random.randint(2, 6)
        else:
            num_removals = random.randint(3, 10)
        
        edges_to_remove = get_edges(current_adj)
        
        if not edges_to_remove:
            continue
            
        random.shuffle(edges_to_remove)
        for u, v in edges_to_remove[:num_removals]:
            current_adj[u] &= ~(1 << v)
            current_adj[v] &= ~(1 << u)
        
        current_count = greedy_fill(current_adj)
        
        # Also try local search on the result
        if current_count >= best_count:
            current_count = local_search(current_adj)
        
        if current_count > best_count:
            best_count = current_count
            best_adj = current_adj[:]
            no_improve_count = 0
        else:
            no_improve_count += 1
            
        # If stuck for too long, do a full restart
        if no_improve_count >= 50:
            current_adj = [0] * n
            current_count = greedy_fill(current_adj)
            current_count = local_search(current_adj)
            if current_count > best_count:
                best_count = current_count
                best_adj = current_adj[:]
            no_improve_count = 0

    # Final polish: try adding any missing edge that doesn't create K3,3
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for u in range(n):
            for v in range(u + 1, n):
                if time.time() > deadline:
                    break
                if best_adj[u] & (1 << v):
                    continue
                # Try adding
                best_adj[u] |= (1 << v)
                best_adj[v] |= (1 << u)
                if is_add_valid(best_adj, u, v):
                    improved = True
                else:
                    best_adj[u] &= ~(1 << v)
                    best_adj[v] &= ~(1 << u)
            if time.time() > deadline:
                break

    # Convert best_adj to list of edges
    result = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_adj[u] & (1 << v):
                result.append([u, v])
    return result
