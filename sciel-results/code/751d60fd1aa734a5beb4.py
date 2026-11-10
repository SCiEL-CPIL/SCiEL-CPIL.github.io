import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    n = 20

    def time_ok():
        return time.time() < deadline

    def build_adjacency(edge_list):
        adj = [0] * n
        for u, v in edge_list:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj

    def edge_count(adj):
        total = 0
        for i in range(n):
            total += adj[i].bit_count()
        return total // 2

    def count_violations_for_edge(adj, u, v):
        """Count how many K3,3 violations involve edge (u,v) as a common-neighbor edge.
        A violation is a triple (i,j,k) with >=3 common neighbors.
        Edge (u,v) 'fixes' a violation if removing it reduces common neighbors below 3
        for some triple. We count triples where u and v are both common neighbors.
        Actually, for minimum deletion: we want to remove edges that are in the most
        violation triples' common-neighbor sets.
        
        Simpler: for each pair (a,b) of vertices, count triples (i,j,k) containing 
        at least one of a,b where common neighbors >= 3.
        
        Most direct: for edge (u,v), count the number of triples (i,j,k) such that
        u and v are both in N(i) ∩ N(j) ∩ N(k) AND |N(i) ∩ N(j) ∩ N(k)| >= 3.
        Removing (u,v) doesn't directly help unless u or v is in {i,j,k}.
        
        Correct approach: An edge (u,v) is in a violation if there exist vertices i,j,k
        (possibly including u or v) such that u and v are both common neighbors of i,j,k
        and |common| >= 3. But actually, removing edge (u,v) only affects triples where
        u or v is one of the three vertices, OR where u and v are both common neighbors.
        
        Let me think more carefully. A K3,3 violation is a set of 3 vertices {i,j,k} 
        with at least 3 common neighbors. To fix it by removing one edge, we need to 
        remove an edge such that the common neighbor count drops below 3.
        
        If we remove edge (u,v), it affects:
        1. Triples containing u: {u,j,k} - common neighbors lose v (if v was common)
        2. Triples containing v: {i,v,k} - common neighbors lose u (if u was common)
        3. Triples not containing u or v: common neighbors might lose u and v if both
           were common neighbors of i,j,k.
        
        For case 3: if u and v are both common neighbors of i,j,k, removing (u,v) 
        doesn't remove u or v from the common neighbor set (they're still adjacent to 
        i,j,k individually). Wait, no - removing edge (u,v) doesn't change whether u 
        is adjacent to i,j,k. It only changes the adjacency between u and v.
        
        So removing edge (u,v) only affects triples where u or v is one of the three 
        vertices! Because the common neighbors of {i,j,k} depend on edges from i,j,k 
        to other vertices, not on edges between i,j,k themselves (unless i,j,k include 
        u or v).
        
        So: edge (u,v) can only fix violations involving u or v as one of the triple members.
        """
        count = 0
        # Triples containing u: {u, j, k} for j < k, j,k != u
        for j in range(n):
            if j == u:
                continue
            for k in range(j + 1, n):
                if k == u:
                    continue
                # Common neighbors of u, j, k
                cn = adj[u] & adj[j] & adj[k]
                if cn.bit_count() >= 3:
                    # Check if removing (u,v) would reduce cn below 3
                    # v is a common neighbor of u,j,k iff adj[u] has v, adj[j] has v, adj[k] has v
                    if (adj[u] & (1 << v)) and (adj[j] & (1 << v)) and (adj[k] & (1 << v)):
                        # v is in cn, removing (u,v) removes v from cn
                        if cn.bit_count() == 3:
                            count += 1
        # Triples containing v: {i, v, k} for i < k, i,k != v
        for i in range(n):
            if i == v:
                continue
            for k in range(i + 1, n):
                if k == v:
                    continue
                cn = adj[i] & adj[v] & adj[k]
                if cn.bit_count() >= 3:
                    if (adj[i] & (1 << u)) and (adj[v] & (1 << u)) and (adj[k] & (1 << u)):
                        if cn.bit_count() == 3:
                            count += 1
        return count

    def find_violation_triples(adj):
        """Find all triples (i,j,k) with >= 3 common neighbors."""
        viols = []
        for i in range(n):
            ai = adj[i]
            for j in range(i + 1, n):
                aij = ai & adj[j]
                for k in range(j + 1, n):
                    if (aij & adj[k]).bit_count() >= 3:
                        viols.append((i, j, k))
        return viols

    def is_k33_free(adj):
        """Check if graph is K3,3-free."""
        for i in range(n):
            ai = adj[i]
            for j in range(i + 1, n):
                aij = ai & adj[j]
                for k in range(j + 1, n):
                    if (aij & adj[k]).bit_count() >= 3:
                        return False
        return True

    def min_deletion_search(adj):
        """
        Start from current adjacency, iteratively remove the edge that fixes 
        the most violations until K3,3-free.
        """
        adj = adj[:]
        max_iter = 5000
        for _ in range(max_iter):
            if not time_ok():
                return None
            
            viols = find_violation_triples(adj)
            if not viols:
                return adj
            
            # For each violation, find edges whose removal would fix it
            # An edge (u,v) fixes violation (i,j,k) if:
            # - u or v is in {i,j,k}, and the other is a common neighbor with count exactly 3
            # OR more generally, removing (u,v) reduces some triple's common neighbors below 3
            
            # Score each edge by how many violations it can fix
            edge_score = {}
            for (i, j, k) in viols:
                cn = adj[i] & adj[j] & adj[k]
                cn_count = cn.bit_count()
                if cn_count < 3:
                    continue
                need_to_remove = cn_count - 2  # need to remove this many common neighbors
                
                # For each common neighbor w of i,j,k, removing edge (w, x) where x in {i,j,k}
                # would remove w from the common neighbor set if w is adjacent to x
                # Actually, removing edge (w, i) removes w from N(i), so w is no longer 
                # a common neighbor of i,j,k.
                # Similarly for (w,j) and (w,k).
                
                if need_to_remove <= 0:
                    continue
                
                # Each common neighbor w can be removed by deleting one of (w,i), (w,j), (w,k)
                # We want to pick the edge that helps the most violations
                for w_bit in range(n):
                    if not (cn & (1 << w_bit)):
                        continue
                    w = w_bit
                    # Edge (w,i): if w != i and w != j and w != k
                    if w != i:
                        key = (min(w, i), max(w, i))
                        edge_score[key] = edge_score.get(key, 0) + 1
                    if w != j:
                        key = (min(w, j), max(w, j))
                        edge_score[key] = edge_score.get(key, 0) + 1
                    if w != k:
                        key = (min(w, k), max(w, k))
                        edge_score[key] = edge_score.get(key, 0) + 1
            
            if not edge_score:
                # Fallback: remove a random edge
                all_edges = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if adj[i] & (1 << j):
                            all_edges.append((i, j))
                if not all_edges:
                    return adj
                u, v = random.choice(all_edges)
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                continue
            
            # Pick the edge with highest score
            best_edge = max(edge_score, key=edge_score.get)
            u, v = best_edge
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
        
        # Check if we succeeded
        if is_k33_free(adj):
            return adj
        return None

    def local_search_refine(adj):
        """
        Try to add back edges that were removed, using local search.
        Also try edge swaps to improve.
        """
        if not time_ok():
            return adj
        
        adj = adj[:]
        current_count = edge_count(adj)
        
        # Try to add edges back
        improved = True
        while improved and time_ok():
            improved = False
            # Get all non-edges
            non_edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if not (adj[i] & (1 << j)):
                        non_edges.append((i, j))
            
            random.shuffle(non_edges)
            for u, v in non_edges:
                if not time_ok():
                    break
                # Try adding edge (u,v)
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                if is_k33_free(adj):
                    improved = True
                    current_count += 1
                else:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
        
        # Edge swap local search
        for _ in range(50):
            if not time_ok():
                break
            
            edges = []
            non_edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if adj[i] & (1 << j):
                        edges.append((i, j))
                    else:
                        non_edges.append((i, j))
            
            if not edges or not non_edges:
                break
            
            random.shuffle(edges)
            random.shuffle(non_edges)
            
            for a, b in edges:
                if not time_ok():
                    break
                for c, d in non_edges:
                    if not time_ok():
                        break
                    # Remove (a,b), add (c,d)
                    adj[a] &= ~(1 << b)
                    adj[b] &= ~(1 << a)
                    adj[c] |= (1 << d)
                    adj[d] |= (1 << c)
                    
                    if is_k33_free(adj):
                        # Keep the swap
                        break
                    else:
                        # Revert
                        adj[a] |= (1 << b)
                        adj[b] |= (1 << a)
                        adj[c] &= ~(1 << d)
                        adj[d] &= ~(1 << c)
                        continue
                    break
        
        return adj

    # Start with complete graph
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            adj[i] |= (1 << j)
            adj[j] |= (1 << i)
    
    best_adj = adj[:]
    best_count = edge_count(adj)
    
    # Phase 1: Minimum deletion search
    if time_ok():
        result = min_deletion_search(adj)
        if result is not None:
            best_adj = result
            best_count = edge_count(result)
    
    # Phase 2: Local search refinement
    if time_ok():
        refined = local_search_refine(best_adj)
        refined_count = edge_count(refined)
        if refined_count > best_count:
            best_adj = refined
            best_count = refined_count
    
    # Phase 3: Try multiple restarts with perturbation
    if time_ok():
        for restart in range(20):
            if not time_ok():
                break
            
            # Perturb the best solution: randomly remove some edges
            perturbed = best_adj[:]
            for i in range(n):
                for j in range(i + 1, n):
                    if perturbed[i] & (1 << j) and random.random() < 0.05:
                        perturbed[i] &= ~(1 << j)
                        perturbed[j] &= ~(1 << i)
            
            # Run min deletion search on perturbed
            result = min_deletion_search(perturbed)
            if result is not None:
                result_count = edge_count(result)
                if result_count > best_count:
                    best_adj = result
                    best_count = result_count
            
            # Local search
            if time_ok():
                refined = local_search_refine(best_adj)
                refined_count = edge_count(refined)
                if refined_count > best_count:
                    best_adj = refined
                    best_count = refined_count
    
    # Convert to edge list
    edge_list = []
    for i in range(n):
        for j in range(i + 1, n):
            if best_adj[i] & (1 << j):
                edge_list.append([i, j])
    
    return edge_list
