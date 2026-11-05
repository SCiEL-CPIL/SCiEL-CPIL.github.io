import time
import random
from itertools import combinations

def _er_graph(q):
    """
    Generate the Erdős–Rényi orthogonality graph on PG(2, q).
    Vertices are points of PG(2, q).
    Two points are adjacent if x1*x2 + y1*y2 - z1*z2 = 0 (mod q).
    """
    pts = []
    for x in range(q):
        for y in range(q):
            pts.append((x, y, 1))
    for x in range(q):
        pts.append((x, 1, 0))
    pts.append((1, 0, 0))
    
    n = len(pts)
    edges = []
    
    for i in range(n):
        x1, y1, z1 = pts[i]
        for j in range(i + 1, n):
            x2, y2, z2 = pts[j]
            val = (x1 * x2 + y1 * y2 - z1 * z2) % q
            if val == 0:
                edges.append([i, j])
                
    return {'n': n, 'edges': edges}

def _is_c4_free(adj, n):
    """Check if graph is C4-free: no two vertices share more than one common neighbor."""
    for u in range(n):
        for v in range(u + 1, n):
            if (adj[u] & adj[v]).bit_count() > 1:
                return False
    return True

def _has_indep_11_bitmask(adj, n, deadline):
    """
    Check if an independent set of size 11 exists using bitmask backtracking.
    Returns True if such a set exists (bad), False if not (good).
    """
    def backtrack(remaining, current_size):
        if time.time() > deadline:
            return False  # Assume no independent set (optimistic)
        if current_size >= 11:
            return True
        if not remaining:
            return False
        if current_size + remaining.bit_count() < 11:
            return False
        
        # Pick vertex with minimum degree in remaining subgraph
        min_v = -1
        min_deg = 100
        temp = remaining
        while temp:
            v = (temp & -temp).bit_length() - 1
            deg = (adj[v] & remaining).bit_count()
            if deg < min_deg:
                min_deg = deg
                min_v = v
            temp &= temp - 1
            
        if min_v == -1:
            return False
            
        # Branch 1: include min_v in independent set
        new_remaining = remaining & ~(adj[min_v] | (1 << min_v))
        if backtrack(new_remaining, current_size + 1):
            return True
            
        # Branch 2: exclude min_v
        remaining &= ~(1 << min_v)
        if backtrack(remaining, current_size):
            return True
            
        return False

    return backtrack((1 << n) - 1, 0)

def _try_add_vertex(adj, n, deadline):
    """
    Try to add a new vertex to the graph while maintaining C4-freeness.
    Returns the new adjacency list and new n if successful, else None.
    """
    if time.time() > deadline:
        return None
    
    # For each existing vertex, we need to decide if the new vertex is connected to it.
    # Let S be the set of neighbours of the new vertex.
    # C4-free conditions:
    # 1. For any two existing vertices u, v, if both are in S, then |N(u) ∩ N(v)| must be 0 (since they would share the new vertex and potentially another).
    #    Actually, if u, v in S, then new vertex is a common neighbour. If they have another common neighbour w, we have a C4: u-new-v-w-u.
    #    So N(u) ∩ N(v) must be empty for any u, v in S.
    # 2. For any existing vertex u not in S, and any v in S, u and v must not share any common neighbour? No.
    #    The condition is: no two vertices have >1 common neighbour.
    #    Pairs involving the new vertex w:
    #    - (w, u): Common neighbours are N(u) ∩ S. This must have size <= 1.
    #    Pairs of existing vertices (u, v):
    #    - Common neighbours are (N(u) ∩ N(v)) ∪ ({w} if u,v in S else {}).
    #    - If u, v in S, common neighbours are (N(u) ∩ N(v)) ∪ {w}. Size must be <= 1. Since w is always there, N(u) ∩ N(v) must be empty.
    #    - If u in S, v not in S, common neighbours are N(u) ∩ N(v). Size must be <= 1. (This is already satisfied in the original graph).
    #    - If u, v not in S, common neighbours are N(u) ∩ N(v). Size must be <= 1. (Already satisfied).
    
    # So the constraints on S are:
    # A. For all u, v in S, u != v: N(u) ∩ N(v) == 0.
    # B. For all u not in S: |N(u) ∩ S| <= 1.
    
    # We can try to build S greedily or search for a valid S.
    # Since we want to maximize n, we just need *some* valid S.
    # S can be empty? If S is empty, the new vertex is isolated.
    # If S is empty, alpha increases by 1. We need alpha < 11.
    # If we add an isolated vertex, it's always C4-free.
    # So the only constraint for S=empty is that the current alpha < 11.
    
    # Let's first try to add an isolated vertex.
    # If the current graph has alpha < 11, we can add an isolated vertex and alpha becomes alpha+1 < 12.
    # But we need alpha < 11 in the *final* graph.
    # So if current alpha is 10, we can add an isolated vertex? No, alpha becomes 11. That's invalid (independent set of 11).
    # So we can only add an isolated vertex if current alpha <= 9.
    
    # Let's check current alpha.
    # This is expensive, but we do it once per step.
    # Actually, we can just try to add a vertex with a non-empty S first, which might keep alpha lower.
    # Or we can just try S=empty and check alpha later.
    
    # Strategy:
    # 1. Try to find a non-empty S satisfying A and B.
    # 2. If found, add vertex with neighbourhood S.
    # 3. If not found, try S=empty.
    # 4. Check if new graph has alpha < 11.
    
    # Let's try to find a valid S.
    # Start with S = {}.
    # Try to add vertices to S one by one.
    # When adding v to S:
    # - Check A: for all u in S, N(u) ∩ N(v) == 0.
    # - Check B: for all w not in S (and w != v), |N(w) ∩ (S ∪ {v})| <= 1.
    #   Note: |N(w) ∩ S| is already <= 1. So we just need to ensure that if v is in N(w), then N(w) ∩ S must be empty.
    
    # Let's iterate through vertices and try to add them to S.
    # We can randomize the order to find different S.
    
    for _ in range(100):
        if time.time() > deadline:
            return None
        
        order = list(range(n))
        random.shuffle(order)
        
        S = set()
        valid = True
        for v in order:
            # Check if v can be added to S
            can_add = True
            # Check A
            for u in S:
                if (adj[u] & adj[v]) != 0:
                    can_add = False
                    break
            if not can_add:
                continue
            
            # Check B
            for w in range(n):
                if w in S or w == v:
                    continue
                # If v is a neighbour of w, then N(w) ∩ S must be empty
                if (adj[w] >> v) & 1:
                    if (adj[w] & sum([1 << u for u in S])) != 0:
                        can_add = False
                        break
            
            if can_add:
                S.add(v)
        
        if S:
            # We found a non-empty S.
            # Let's verify it fully to be safe.
            # A: for u, v in S, N(u) & N(v) == 0
            s_list = list(S)
            ok = True
            for i in range(len(s_list)):
                for j in range(i+1, len(s_list)):
                    if adj[s_list[i]] & adj[s_list[j]]:
                        ok = False
                        break
                if not ok: break
            
            if ok:
                # B: for w not in S, |N(w) & S| <= 1
                s_mask = 0
                for v in S: s_mask |= (1 << v)
                for w in range(n):
                    if not (s_mask >> w) & 1:
                        if (adj[w] & s_mask).bit_count() > 1:
                            ok = False
                            break
            
            if ok:
                # Construct new adj
                new_adj = adj[:] + [0]
                new_n = n + 1
                for v in S:
                    new_adj[v] |= (1 << new_n)
                    new_adj[new_n] |= (1 << v)
                return new_adj, new_n
    
    # If no non-empty S found, try S = empty
    # We can always add an isolated vertex if we don't care about alpha yet.
    # But we must check alpha afterwards.
    # Let's just return the isolated vertex candidate.
    new_adj = adj[:] + [0]
    new_n = n + 1
    return new_adj, new_n

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill
    
    # Fallback: 10 disjoint triangles (n=30, alpha=10, C4-free)
    fallback_edges = []
    for t in range(10):
        a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
        fallback_edges += [[a, b], [a, c], [b, c]]
    
    best_result = {'n': 30, 'edges': fallback_edges}
    
    # Primary: Try G(5) - orthogonality graph on PG(2,5), n=31
    g5 = _er_graph(5)
    n = g5['n']
    edges = g5['edges']
    
    # Build adjacency bitmask
    adj = [0] * n
    for u, v in edges:
        adj[u] |= 1 << v
        adj[v] |= 1 << u
        
    # Check C4-free
    if _is_c4_free(adj, n):
        # Check no independent set of size 11
        has_indep_11 = _has_indep_11_bitmask(adj, n, deadline)
        if not has_indep_11:
            # We have a valid graph with n=31.
            # Now try to expand it using _try_add_vertex.
            current_adj = adj[:]
            current_n = n
            current_edges = edges[:]
            
            # We need to track edges to return them.
            # Or we can just return the final graph.
            # Let's try to add vertices until we can't or time runs out.
            # We must ensure alpha < 11 at each step?
            # Ideally, we want the final graph to have alpha < 11.
            # Adding vertices can only increase or keep alpha the same (if isolated) or decrease (if connected).
            # Actually, adding a vertex can increase alpha by at most 1.
            # If current alpha is A, new alpha is in [A, A+1].
            # So if we start with alpha < 11, we can add at most (10 - alpha) isolated vertices before hitting 11.
            # But we can also add connected vertices which might not increase alpha.
            
            # Let's check alpha of current graph.
            # We already know it's < 11.
            # Let's try to add vertices.
            
            for _ in range(20): # Max 20 additions
                if time.time() > deadline - 5:
                    break
                
                res = _try_add_vertex(current_adj, current_n, deadline)
                if res is None:
                    break
                
                new_adj, new_n = res
                
                # Check C4-free
                if not _is_c4_free(new_adj, new_n):
                    break
                
                # Check alpha < 11
                # This is the expensive part.
                # We only need to do this if we want to accept the new graph.
                # If it fails, we discard the new graph and stop? Or try a different addition?
                # _try_add_vertex is stochastic. We could retry.
                # But for simplicity, if it fails, we stop expanding.
                
                # Optimization: Only check alpha if new_n is large or if we are close to the limit.
                # Actually, alpha can only increase by 1 per step.
                # If we start with alpha <= 9, we can add up to 1 isolated vertex.
                # If we start with alpha = 10, we can't add any isolated vertices.
                # But we can add connected vertices.
                
                # Let's just check alpha every time.
                has_indep_11 = _has_indep_11_bitmask(new_adj, new_n, deadline)
                if not has_indep_11:
                    # Accept new graph
                    current_adj = new_adj
                    current_n = new_n
                    # Update edges list
                    # Find the new edges
                    # The new vertex is index new_n - 1
                    new_v = new_n - 1
                    for u in range(new_v):
                        if (current_adj[new_v] >> u) & 1:
                            current_edges.append([u, new_v])
                    
                    # Update best result
                    best_result = {'n': current_n, 'edges': current_edges}
                else:
                    # Alpha is 11, invalid. Stop.
                    break
            
            return best_result
    
    # Secondary: Try a different polarity or q value
    # Try q=5 with a different polarity: x1*x2 + y1*y2 + z1*z2 = 0
    pts = []
    for x in range(5):
        for y in range(5):
            pts.append((x, y, 1))
    for x in range(5):
        pts.append((x, 1, 0))
    pts.append((1, 0, 0))
    
    n2 = len(pts)
    edges2 = []
    for i in range(n2):
        x1, y1, z1 = pts[i]
        for j in range(i + 1, n2):
            x2, y2, z2 = pts[j]
            val = (x1 * x2 + y1 * y2 + z1 * z2) % 5
            if val == 0:
                edges2.append([i, j])
    
    adj2 = [0] * n2
    for u, v in edges2:
        adj2[u] |= 1 << v
        adj2[v] |= 1 << u
        
    if _is_c4_free(adj2, n2):
        has_indep_11 = _has_indep_11_bitmask(adj2, n2, deadline)
        if not has_indep_11:
            # Expand this one too
            current_adj = adj2[:]
            current_n = n2
            current_edges = edges2[:]
            
            for _ in range(20):
                if time.time() > deadline - 5:
                    break
                
                res = _try_add_vertex(current_adj, current_n, deadline)
                if res is None:
                    break
                
                new_adj, new_n = res
                
                if not _is_c4_free(new_adj, new_n):
                    break
                
                has_indep_11 = _has_indep_11_bitmask(new_adj, new_n, deadline)
                if not has_indep_11:
                    current_adj = new_adj
                    current_n = new_n
                    new_v = new_n - 1
                    for u in range(new_v):
                        if (current_adj[new_v] >> u) & 1:
                            current_edges.append([u, new_v])
                    
                    if current_n > best_result['n']:
                        best_result = {'n': current_n, 'edges': current_edges}
                else:
                    break
            
            return best_result
    
    # Return best result found
    return best_result

