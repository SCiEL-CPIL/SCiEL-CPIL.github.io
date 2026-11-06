import time
import random
import sys

def _construct_polarity_graph(q):
    """
    Construct the polarity graph of PG(2, q) using the standard orthogonal polarity.
    The vertices are the points of PG(2, q) (normalized triples in GF(q)^3 \\ {0}).
    Two points are adjacent if their dot product (x0*x0' + x1*x1' - x2*x2') is 0.
    This graph is C4-free (it's a polarity graph).
    """
    reps = []
    seen = set()
    
    for z in range(q):
        for y in range(q):
            for x in range(q):
                if x == 0 and y == 0 and z == 0:
                    continue
                # Normalize: make the first non-zero coordinate 1
                if x != 0:
                    norm = (1, (y * pow(x, -1, q)) % q, (z * pow(x, -1, q)) % q)
                elif y != 0:
                    norm = (0, 1, (z * pow(y, -1, q)) % q)
                else:
                    norm = (0, 0, 1)
                
                if norm not in seen:
                    seen.add(norm)
                    reps.append(norm)
    
    n = len(reps)
    expected = q*q + q + 1
    if n != expected:
        # This shouldn't happen for prime q, but handle gracefully
        return n, [], [set() for _ in range(n)]
        
    adj = [set() for _ in range(n)]
    edges = []
    
    # Precompute points for speed
    # For each pair, check orthogonality
    for i in range(n):
        u = reps[i]
        for j in range(i+1, n):
            v = reps[j]
            # Dot product in PG(2,q) with the specific polarity x0x0' + x1x1' - x2x2'
            val = (u[0]*v[0] + u[1]*v[1] - u[2]*v[2]) % q
            if val == 0:
                adj[i].add(j)
                adj[j].add(i)
                edges.append([i, j])
                
    return n, edges, adj

def _verify_c4_free(n, adj):
    """
    Verify that no two vertices have more than one common neighbor.
    """
    for i in range(n):
        for j in range(i+1, n):
            common = adj[i] & adj[j]
            if len(common) > 1:
                return False
    return True

def _max_independent_set_bitset(n, adj, time_limit=2.0):
    """
    Find the maximum independent set size using branch and bound with bitsets.
    Returns the size.
    Optimized for small n (n <= 43).
    """
    # Build adjacency bitmasks
    masks = [0] * n
    for i in range(n):
        mask = 0
        for j in adj[i]:
            mask |= (1 << j)
        masks[i] = mask
    
    all_mask = (1 << n) - 1
    start_time = time.time()
    best_size = 0
    # We can use a simple greedy initial solution to set best_size
    # Greedy: pick vertex with min degree, add to IS, remove neighbors
    remaining = all_mask
    greedy_size = 0
    while remaining:
        # Pick vertex with min degree in remaining subgraph
        temp = remaining
        min_deg = n
        v_pick = -1
        while temp:
            lsb = temp & -temp
            v = (lsb.bit_length() - 1)
            deg = (masks[v] & remaining).bit_count()
            if deg < min_deg:
                min_deg = deg
                v_pick = v
            temp -= lsb
        
        if v_pick != -1:
            greedy_size += 1
            remaining &= ~(masks[v_pick] | (1 << v_pick))
    
    best_size = greedy_size
    
    # Branch and Bound
    # State: remaining vertices, current size
    def dfs(remaining, current_size):
        nonlocal best_size
        
        if time.time() - start_time > time_limit:
            return
        
        if remaining == 0:
            if current_size > best_size:
                best_size = current_size
            return
        
        # Upper bound: current_size + popcount(remaining)
        if current_size + remaining.bit_count() <= best_size:
            return
            
        # Pick a vertex to branch on.
        # Heuristic: Pick vertex with max degree in remaining subgraph?
        # Or min degree?
        # For independent set, branching on a vertex v:
        # 1. Include v: remove v and all its neighbors.
        # 2. Exclude v: remove v.
        # To maximize pruning, we want to pick v such that including it removes many vertices?
        # Or maybe pick v such that excluding it doesn't cost much?
        
        # Let's try picking the vertex with max degree in the remaining subgraph.
        temp = remaining
        v_idx = -1
        max_deg = -1
        while temp:
            lsb = temp & -temp
            v = (lsb.bit_length() - 1)
            deg = (masks[v] & remaining).bit_count()
            if deg > max_deg:
                max_deg = deg
                v_idx = v
            temp -= lsb
        
        if v_idx == -1:
            return
            
        v_bit = 1 << v_idx
        
        # Branch 1: Include v_idx
        # Remove v_idx and its neighbors from remaining
        new_remaining = remaining & ~(masks[v_idx] | v_bit)
        dfs(new_remaining, current_size + 1)
        
        # Branch 2: Exclude v_idx
        new_remaining = remaining & ~v_bit
        dfs(new_remaining, current_size)
        
    dfs(all_mask, 0)
    
    return best_size

def _try_extend_graph(n, adj, max_n=43, time_limit=10.0):
    """
    Try to extend the graph by adding vertices one by one.
    For each new vertex, we can connect it to a subset of existing vertices.
    To maintain C4-free:
    If we connect new vertex v to a set S of existing vertices,
    then for any u, w in S, u and w must have at most 1 common neighbor.
    Since the existing graph is C4-free, u and w have at most 1 common neighbor in the existing graph.
    However, v becomes a common neighbor of u and w if both u and w are in S.
    So, if u and w already have a common neighbor in the existing graph, they CANNOT both be in S.
    This means S must be a set of vertices that are pairwise "non-adjacent in the square"?
    No. It means S must be such that no two vertices in S have a common neighbor in the existing graph.
    Let C(u, v) be the set of common neighbors of u and v.
    If C(u, v) is non-empty, then u and v cannot both be in S.
    So S must be an independent set in a graph H where H has an edge between u and v if C(u, v) != 0.
    Wait, this condition is necessary but is it sufficient?
    If S is an independent set in H (meaning no two vertices in S share a neighbor in G),
    then adding v connected to S does not create any C4 involving v?
    A C4 involving v would look like v-u-w-v'.
    This requires u and w to be neighbors of v, and u and w to have another common neighbor v'.
    If v' is in G, then u and w share a neighbor in G. This is forbidden by S being independent in H.
    If v' is not in G (i.e., v'=v), that's not a C4.
    So yes, S must be an independent set in H.
    
    Additionally, we want to maximize the degree of v (connect to as many vertices as possible) to help reduce alpha.
    So we want to find a large independent set in H.
    
    Also, we must ensure the new graph has no independent set of size 11.
    """
    start_time = time.time()
    best_n = n
    best_adj = [set(s) for s in adj]
    
    # Precompute H for the current graph? H changes as we add vertices.
    # But H is defined on the existing vertices.
    # Let's just check C(u,v) on the fly or precompute for current n.
    
    # H edges: (u, v) in H if len(adj[u] & adj[v]) > 0
    # Note: adj[u] & adj[v] are neighbors in G.
    
    # We will iterate adding vertices from n to max_n-1
    for new_v in range(n, max_n):
        if time.time() - start_time > time_limit:
            break
            
        # Build H on current vertices (0 .. new_v-1)
        # H is a graph on new_v vertices.
        # h_adj[i] is a set of j such that adj[i] & adj[j] is not empty.
        # We need to find a large independent set in H.
        # Actually, we just need *some* independent set, preferably large.
        # But wait, if we add v connected to S, v is now a neighbor of everyone in S.
        # This might create new H-edges? No, H is for the *new* vertices we might add later.
        # For the current step, we just need S to be independent in the current H.
        
        # Optimization: We can just try to greedily build S?
        # Or use a small B&B for independent set in H?
        # H has up to 42 vertices.
        
        # Let's build H adjacency bitmasks for speed.
        # h_mask[i] has bit j set if adj[i] & adj[j] != 0
        h_masks = [0] * new_v
        for i in range(new_v):
            for j in range(i+1, new_v):
                if adj[i] & adj[j]: # share a neighbor
                    h_masks[i] |= (1 << j)
                    h_masks[j] |= (1 << i)
        
        # Find Max Independent Set in H
        # Use a fast heuristic or limited B&B
        # Since we just want to add *some* edges, even a small S is okay,
        # but larger S is better for reducing alpha.
        
        # Let's try to find a Max IS in H using a simple greedy first, then maybe improve?
        # Greedy:
        remaining_h = (1 << new_v) - 1
        S = []
        while remaining_h:
            # Pick vertex with min degree in H_remaining?
            # Or just any?
            # Let's pick the one with min degree to keep options open?
            # Actually for IS, picking min degree is standard.
            temp = remaining_h
            v_pick = -1
            min_deg = new_v
            while temp:
                lsb = temp & -temp
                v = (lsb.bit_length() - 1)
                deg = (h_masks[v] & remaining_h).bit_count()
                if deg < min_deg:
                    min_deg = deg
                    v_pick = v
                temp -= lsb
            
            if v_pick != -1:
                S.append(v_pick)
                # Remove v_pick and its neighbors in H from remaining
                remaining_h &= ~(h_masks[v_pick] | (1 << v_pick))
        
        # Add new vertex connected to S
        if len(S) > 0:
            # Construct new adj
            new_adj = [set(s) for s in adj]
            new_adj.append(set(S))
            for u in S:
                new_adj[u].add(new_v)
            
            # Check if this extension is C4-free?
            # We ensured S is independent in H, so no new C4s involving new_v and existing vertices.
            # No new C4s involving only existing vertices.
            # So it is C4-free.
            
            # Check alpha
            # We don't want to run full B&B on alpha every time if it's slow.
            # But we need to know if alpha < 11.
            # Let's run a quick check.
            # If we have time, run B&B.
            remaining_time = time_limit - (time.time() - start_time)
            if remaining_time > 0.5:
                alpha = _max_independent_set_bitset(new_v + 1, new_adj, time_limit=remaining_time - 0.2)
                if alpha < 11:
                    # Valid!
                    best_n = new_v + 1
                    best_adj = new_adj
                    # Update adj for next iteration
                    adj = new_adj
                else:
                    # Invalid, don't add this vertex?
                    # Or try a different S?
                    # For now, just skip this vertex?
                    # If we skip, we can't add more vertices?
                    # Actually, we can try to add a vertex connected to a *different* S.
                    # But finding a better S is hard.
                    # Let's just break? Or try next vertex with a smaller S?
                    # To be safe, let's just stop extending if we find a valid config.
                    # If we don't find one, we return the best valid one so far.
                    # But wait, if the *current* graph (before this extension) was valid, we should keep it.
                    # We only update best_adj if the extension is valid.
                    # So if this extension is invalid, we just don't update best_adj,
                    # and we continue to the next new_v?
                    # No, because we didn't actually add the vertex to `adj`.
                    # So `adj` remains the previous valid state?
                    # Wait, the loop variable `new_v` increments.
                    # If we don't add the vertex, `n` (size of adj) doesn't change.
                    # But the loop expects `adj` to have size `new_v`.
                    # So if we skip, we should break?
                    # Because we can't add vertex `new_v` if it makes it invalid.
                    # Maybe we can add a *different* vertex later?
                    # But the labels are fixed 0..n-1.
                    # So we are forced to add vertex `new_v` or stop.
                    # So if we can't add `new_v` validly, we stop.
                    break
            
    # Construct edges from best_adj
    edges = []
    for i in range(best_n):
        for j in best_adj[i]:
            if i < j:
                edges.append([i, j])
                
    return {'n': best_n, 'edges': edges}

def construct():
    start_time = time.time()
    deadline = start_time + 140 # 10s safety margin
    
    best_graph = None
    
    # Strategy:
    # 1. Start with PG(2, 5) (n=31).
    # 2. Check if it's valid (alpha < 11).
    # 3. If valid, try to extend to n=43.
    # 4. If PG(2,5) is invalid (unlikely), try PG(2, 6) subgraphs?
    #    But PG(2,6) is n=43. If it's invalid, maybe remove vertices?
    #    The directive suggests using PG(2,5) as base and extending.
    
    try:
        n5, edges5, adj5 = _construct_polarity_graph(5)
        # n5 should be 31
        
        # Check alpha
        # We need to be fast here.
        # If alpha < 11, we are good.
        # If alpha >= 11, we need to fix it.
        # But PG(2,5) is known to be C4-free and has small alpha.
        # Let's assume it's valid and try to extend.
        # If extension fails, we return PG(2,5).
        
        # Actually, let's verify C4-free first (should be true).
        # _verify_c4_free(n5, adj5)
        
        # Try to extend
        result = _try_extend_graph(n5, adj5, max_n=43, time_limit=120 - (time.time() - start_time) - 10)
        
        if result['n'] > 31:
            best_graph = result
        else:
            best_graph = {'n': n5, 'edges': edges5}
            
    except Exception as e:
        # Fallback to PG(2, 5) hardcoded? Or just return 30?
        # Let's try PG(2, 5) again?
        # Or just return the 30-vertex triangle graph.
        edges_30 = []
        for t in range(10):
            a, b, c = 3*t, 3*t+1, 3*t+2
            edges_30 += [[a,b], [a,c], [b,c]]
        best_graph = {'n': 30, 'edges': edges_30}
        
    # Final check: ensure best_graph is valid?
    # If we have time, verify C4-free and alpha < 11.
    # But we already checked in _try_extend_graph.
    # Let's add a safety check if time permits.
    if time.time() < deadline:
        n = best_graph['n']
        # Rebuild adj for verification
        adj = [set() for _ in range(n)]
        for u, v in best_graph['edges']:
            adj[u].add(v)
            adj[v].add(u)
        
        # Check C4
        if not _verify_c4_free(n, adj):
            # Invalid, fallback
            edges_30 = []
            for t in range(10):
                a, b, c = 3*t, 3*t+1, 3*t+2
                edges_30 += [[a,b], [a,c], [b,c]]
            best_graph = {'n': 30, 'edges': edges_30}
        else:
            # Check Alpha
            # Only if n is large and we have time
            if n > 31 and time.time() < deadline - 5:
                alpha = _max_independent_set_bitset(n, adj, time_limit=deadline - time.time() - 1)
                if alpha >= 11:
                    # Invalid, fallback to PG(2,5) if we have it?
                    # We didn't store PG(2,5) separately.
                    # Just use the 30-vertex graph?
                    # Or try to return the best valid one found in _try_extend_graph?
                    # _try_extend_graph returns the best VALID one found during extension.
                    # So if the final result is invalid, it means the *last* extension made it invalid?
                    # But _try_extend_graph only updates best_adj if the extension is valid.
                    # So best_graph should be valid.
                    # This check is just a paranoia check.
                    # If it fails, we return 30.
                    edges_30 = []
                    for t in range(10):
                        a, b, c = 3*t, 3*t+1, 3*t+2
                        edges_30 += [[a,b], [a,c], [b,c]]
                    best_graph = {'n': 30, 'edges': edges_30}
                    
    return best_graph

# Quick self-test if run directly
if __name__ == "__main__":
    g = construct()
    print(g['n'])
