import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    n = 31
    q = 5
    
    # Build polarity graph of PG(2,5)
    points = []
    for x in range(q):
        for y in range(q):
            points.append((1, x, y))
    for z in range(q):
        points.append((0, 1, z))
    points.append((0, 0, 1))
    
    def dot(p, r):
        return (p[0]*r[0] + p[1]*r[1] + p[2]*r[2]) % q
    
    edge_set = set()
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            if dot(points[i], points[j]) == 0:
                edge_set.add((i, j))
                adj[i].add(j)
                adj[j].add(i)
    
    # Add the best edge
    best_edge = None
    min_cost = float('inf')
    
    for i in range(n):
        for j in range(i+1, n):
            if (i, j) in edge_set:
                continue
            cost = 0
            for a in adj[i]:
                cost += len(adj[a] & adj[j])
            if cost < min_cost:
                min_cost = cost
                best_edge = (i, j)
    
    if best_edge:
        u, v = best_edge
        edge_set.add((u, v))
        adj[u].add(v)
        adj[v].add(u)
    
    # Precompute common neighbor counts C[i][j]
    C = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            c = len(adj[i] & adj[j])
            C[i][j] = c
            C[j][i] = c
    
    def count_c4_from_C(C, n):
        total = 0
        for i in range(n):
            for j in range(i+1, n):
                if C[i][j] >= 2:
                    total += C[i][j] * (C[i][j] - 1) // 2
        return total // 2
    
    current_c4 = count_c4_from_C(C, n)
    best_c4 = current_c4
    best_edge_set = set(edge_set)
    
    # Local search with efficient marginal delta computation
    edges = list(edge_set)
    non_edges = [(i, j) for i in range(n) for j in range(i+1, n) if (i, j) not in edge_set]
    
    random.seed(42)
    no_improve = 0
    max_no_improve = 100000
    
    while no_improve < max_no_improve:
        if time.time() > deadline - 3:
            break
            
        if not edges or not non_edges:
            break
            
        idx_e = random.randint(0, len(edges)-1)
        idx_n = random.randint(0, len(non_edges)-1)
        
        u, v = edges[idx_e]
        x, y = non_edges[idx_n]
        
        # Compute marginal change in C4 count for swapping (u,v) -> (x,y)
        # delta = C4_after - C4_before
        # We compute this by tracking changes to C[i][j] values
        
        # The C4 count is sum over all pairs (i,j) of C[i][j]*(C[i][j]-1)/2
        # When we remove edge (u,v), C[i][j] decreases by 1 for each common neighbor of u and v
        # When we add edge (x,y), C[i][j] increases by 1 for each common neighbor of x and y
        
        # More precisely:
        # Removing edge (u,v): for each w in N(u) ∩ N(v), C[u][w] and C[v][w] each decrease by 1
        #   But also for each pair (a,b) where a in N(u)\{v} and b in N(v)\{u} and a != b,
        #   C[a][b] is unaffected directly, but the paths through u and v change.
        
        # Let me think more carefully. C[i][j] = |N(i) ∩ N(j)|.
        # Removing edge (u,v):
        #   - u is no longer in N(v), v is no longer in N(u)
        #   - For any w: if w in N(u) and w in N(v), then C[u][w] decreases by 1 (since v was in N(u) and w in N(v), but now v not in N(u))
        #     Wait, C[u][w] = |N(u) ∩ N(w)|. If v in N(u) and v in N(w), then removing edge (u,v) removes v from N(u), so C[u][w] decreases by 1 if v in N(w).
        #     Similarly C[v][w] decreases by 1 if u in N(w).
        #   - For any pair (a,b) not involving u or v: C[a][b] unchanged.
        #   - For pair (u,v): C[u][v] changes from |N(u)∩N(v)| to |N(u)∩N(v)| (but u not in N(v) anymore, v not in N(u))
        #     Actually C[u][v] = |N(u) ∩ N(v)|. After removal, N(u) loses v, N(v) loses u.
        #     So C[u][v] = |(N(u)\{v}) ∩ (N(v)\{u})| = |N(u)∩N(v)| - [u in N(v) ? 1 : 0] - [v in N(u) ? 1 : 0] + [u in N(v) and v in N(u) ? 1 : 0]
        #     Since u in N(v) and v in N(u) (they were adjacent), C[u][v] decreases by 1... wait no.
        #     N(u) ∩ N(v): u is not in N(u), v is not in N(v). So u in N(v) means v in N(u) (symmetric).
        #     The intersection N(u)∩N(v) doesn't contain u or v (since u∉N(u), v∉N(v)).
        #     So removing edge (u,v) doesn't change which vertices are in N(u)∩N(v) except that v leaves N(u) and u leaves N(v).
        #     But v was already not in N(u)∩N(v) because v∉N(v). Similarly u was not in N(u)∩N(v) because u∉N(u).
        #     So C[u][v] is unchanged!
        
        # Let me reconsider. C[i][j] = |N(i) ∩ N(j)|.
        # When we remove edge (u,v):
        #   N(u) becomes N(u) \ {v}, N(v) becomes N(v) \ {u}
        #   For pair (u, w) where w ≠ v: C[u][w] = |(N(u)\{v}) ∩ N(w)| = |N(u)∩N(w)| - [v in N(w) ? 1 : 0]
        #   For pair (v, w) where w ≠ u: C[v][w] = |(N(v)\{u}) ∩ N(w)| = |N(v)∩N(w)| - [u in N(w) ? 1 : 0]
        #   For pair (u, v): C[u][v] = |(N(u)\{v}) ∩ (N(v)\{u})| = |N(u)∩N(v)| (since v∉N(v) and u∉N(u) already)
        #     Wait: (N(u)\{v}) ∩ (N(v)\{u}) = (N(u)∩N(v)) \ {v} \ {u} = N(u)∩N(v) since v∉N(v) and u∉N(u)
        #     So C[u][v] unchanged.
        #   For pair (w1, w2) where w1,w2 ∉ {u,v}: unchanged.
        
        # So the affected C values when removing (u,v) are:
        #   C[u][w] for w in N(u)\{v}: decrease by 1 if v in N(w)
        #   C[v][w] for w in N(v)\{u}: decrease by 1 if u in N(w)
        
        # Similarly for adding (x,y):
        #   C[x][w] for w in N(x): increase by 1 if y in N(w)
        #   C[y][w] for w in N(y): increase by 1 if x in N(w)
        
        # The change in C4 count is:
        # sum over affected pairs of [new_C*(new_C-1)/2 - old_C*(old_C-1)/2]
        
        # Let's compute this incrementally.
        
        # First, handle removal of (u,v)
        delta = 0
        
        # For each w in N(u) \ {v}:
        #   if v in N(w): C[u][w] decreases by 1
        for w in adj[u]:
            if w == v:
                continue
            if v in adj[w]:
                old_c = C[u][w]
                new_c = old_c - 1
                delta += new_c*(new_c-1)//2 - old_c*(old_c-1)//2
                C[u][w] = new_c
                C[w][u] = new_c
        
        # For each w in N(v) \ {u}:
        #   if u in N(w): C[v][w] decreases by 1
        for w in adj[v]:
            if w == u:
                continue
            if u in adj[w]:
                old_c = C[v][w]
                new_c = old_c - 1
                delta += new_c*(new_c-1)//2 - old_c*(old_c-1)//2
                C[v][w] = new_c
                C[w][v] = new_c
        
        # Now handle addition of (x,y)
        # For each w in N(x):
        #   if y in N(w): C[x][w] increases by 1
        for w in adj[x]:
            if w == y:
                continue
            if y in adj[w]:
                old_c = C[x][w]
                new_c = old_c + 1
                delta += new_c*(new_c-1)//2 - old_c*(old_c-1)//2
                C[x][w] = new_c
                C[w][x] = new_c
        
        # For each w in N(y):
        #   if x in N(w): C[y][w] increases by 1
        for w in adj[y]:
            if w == x:
                continue
            if x in adj[w]:
                old_c = C[y][w]
                new_c = old_c + 1
                delta += new_c*(new_c-1)//2 - old_c*(old_c-1)//2
                C[y][w] = new_c
                C[w][y] = new_c
        
        # Also need to handle C[u][v] and C[x][y] if they were affected
        # C[u][v]: was |N(u)∩N(v)|, now |(N(u)\{v})∩(N(v)\{u})| = same (as argued)
        # C[x][y]: was |N(x)∩N(y)|, now |(N(x)∪{y})∩(N(y)∪{x})|
        #   = |N(x)∩N(y) ∪ {y if y in N(x)} ∪ {x if x in N(y)} ∪ {x,y if both}|
        #   Actually: (N(x)∪{y}) ∩ (N(y)∪{x}) = (N(x)∩N(y)) ∪ (N(x)∩{x}) ∪ ({y}∩N(y)) ∪ ({y}∩{x})
        #   = (N(x)∩N(y)) ∪ (∅ since x∉N(x)) ∪ (∅ since y∉N(y)) ∪ ({x} if x==y else ∅)
        #   = N(x)∩N(y) since x≠y
        #   So C[x][y] unchanged!
        
        # But wait, we also need to account for the fact that when we add (x,y), 
        # x is added to N(y) and y is added to N(x). This affects C for pairs involving x or y.
        # I already handled that above for w in N(x) and w in N(y).
        # But what about C[x][y] itself? As shown, it's unchanged.
        
        # However, there's a subtlety: when I iterate over w in adj[x], adj[x] doesn't include y yet.
        # And when I iterate over w in adj[y], adj[y] doesn't include x yet.
        # So I'm correctly using the old adjacency lists.
        
        # Also need to check: does the removal of (u,v) affect C[x][w] or C[y][w] for some w?
        # Only if x or y is u or v, which we should handle.
        # For now, assume u,v,x,y are distinct. If not, the logic breaks.
        # Let's check if they're distinct and skip if not.
        
        if u == x or u == y or v == x or v == y:
            # Revert all C changes
            for w in adj[u]:
                if w == v:
                    continue
                if v in adj[w]:
                    C[u][w] += 1
                    C[w][u] += 1
            for w in adj[v]:
                if w == u:
                    continue
                if u in adj[w]:
                    C[v][w] += 1
                    C[w][v] += 1
            for w in adj[x]:
                if w == y:
                    continue
                if y in adj[w]:
                    C[x][w] -= 1
                    C[w][x] -= 1
            for w in adj[y]:
                if w == x:
                    continue
                if x in adj[w]:
                    C[y][w] -= 1
                    C[w][y] -= 1
            no_improve += 1
            continue
        
        if delta < 0:
            # Accept the swap
            current_c4 += delta
            adj[u].discard(v)
            adj[v].discard(u)
            adj[x].add(y)
            adj[y].add(x)
            
            # Update C[u][v] and C[x][y] (they should be unchanged, but let's verify)
            # Actually they are unchanged as argued, so no update needed.
            
            edges[idx_e] = (x, y)
            non_edges[idx_n] = (u, v)
            edge_set.discard((u, v))
            edge_set.add((x, y))
            
            if current_c4 < best_c4:
                best_c4 = current_c4
                best_edge_set = set(edge_set)
            
            no_improve = 0
        else:
            # Revert C changes
            for w in adj[u]:
                if w == v:
                    continue
                if v in adj[w]:
                    C[u][w] += 1
                    C[w][u] += 1
            for w in adj[v]:
                if w == u:
                    continue
                if u in adj[w]:
                    C[v][w] += 1
                    C[w][v] += 1
            for w in adj[x]:
                if w == y:
                    continue
                if y in adj[w]:
                    C[x][w] -= 1
                    C[w][x] -= 1
            for w in adj[y]:
                if w == x:
                    continue
                if x in adj[w]:
                    C[y][w] -= 1
                    C[w][y] -= 1
            
            no_improve += 1
    
    # Final polish: try all possible single-edge swaps
    if time.time() < deadline - 5:
        # Do a few more passes of best-improvement
        for _ in range(3):
            improved = True
            while improved:
                improved = False
                for i in range(len(edges)):
                    if time.time() > deadline - 3:
                        break
                    u, v = edges[i]
                    for j in range(len(non_edges)):
                        x, y = non_edges[j]
                        if u == x or u == y or v == x or v == y:
                            continue
                        
                        # Compute delta
                        delta = 0
                        for w in adj[u]:
                            if w == v:
                                continue
                            if v in adj[w]:
                                old_c = C[u][w]
                                new_c = old_c - 1
                                delta += new_c*(new_c-1)//2 - old_c*(old_c-1)//2
                        for w in adj[v]:
                            if w == u:
                                continue
                            if u in adj[w]:
                                old_c = C[v][w]
                                new_c = old_c - 1
                                delta += new_c*(new_c-1)//2 - old_c*(old_c-1)//2
                        for w in adj[x]:
                            if w == y:
                                continue
                            if y in adj[w]:
                                old_c = C[x][w]
                                new_c = old_c + 1
                                delta += new_c*(new_c-1)//2 - old_c*(old_c-1)//2
                        for w in adj[y]:
                            if w == x:
                                continue
                            if x in adj[w]:
                                old_c = C[y][w]
                                new_c = old_c + 1
                                delta += new_c*(new_c-1)//2 - old_c*(old_c-1)//2
                        
                        if delta < 0:
                            # Apply
                            current_c4 += delta
                            for w in adj[u]:
                                if w == v:
                                    continue
                                if v in adj[w]:
                                    C[u][w] -= 1
                                    C[w][u] -= 1
                            for w in adj[v]:
                                if w == u:
                                    continue
                                if u in adj[w]:
                                    C[v][w] -= 1
                                    C[w][v] -= 1
                            for w in adj[x]:
                                if w == y:
                                    continue
                                if y in adj[w]:
                                    C[x][w] += 1
                                    C[w][x] += 1
                            for w in adj[y]:
                                if w == x:
                                    continue
                                if x in adj[w]:
                                    C[y][w] += 1
                                    C[w][y] += 1
                            
                            adj[u].discard(v)
                            adj[v].discard(u)
                            adj[x].add(y)
                            adj[y].add(x)
                            
                            edges[i] = (x, y)
                            non_edges[j] = (u, v)
                            edge_set.discard((u, v))
                            edge_set.add((x, y))
                            
                            if current_c4 < best_c4:
                                best_c4 = current_c4
                                best_edge_set = set(edge_set)
                            
                            improved = True
                            break
                    if time.time() > deadline - 3:
                        break
                if time.time() > deadline - 3:
                    break
    
    final_edges = list(best_edge_set)
    edges_out = [[u, v] for u, v in final_edges]
    
    return edges_out
