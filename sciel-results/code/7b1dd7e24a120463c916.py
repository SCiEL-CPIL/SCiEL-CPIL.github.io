import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    N = 16
    all_pairs = [(u, v) for u in range(N) for v in range(u + 1, N)]
    
    # Sort pairs to improve pruning
    # Heuristic: prefer edges that are in the Clebsch graph or connect high-degree vertices in a random graph
    # For now, just sort by index or use a fixed order.
    # A better order: process edges likely to be in the solution first?
    # Or edges that create triangles quickly?
    
    # Clebsch graph edges for initialization
    vectors = [i for i in range(32) if bin(i).count('1') % 2 == 0]
    clebsch_adj = [0] * N
    clebsch_edges = []
    for u_idx in range(N):
        for v_idx in range(u_idx + 1, N):
            if bin(vectors[u_idx] ^ vectors[v_idx]).count('1') == 4:
                clebsch_edges.append((u_idx, v_idx))
                clebsch_adj[u_idx] |= (1 << v_idx)
                clebsch_adj[v_idx] |= (1 << u_idx)
    
    best_edges = [list(e) for e in clebsch_edges]
    best_count = len(clebsch_edges)
    
    # 3-colorability check
    def is_3_colorable(adj_mask):
        order = list(range(N))
        order.sort(key=lambda x: bin(adj_mask[x]).count('1'), reverse=True)
        pos = {v: i for i, v in enumerate(order)}
        later_neighbors = [[] for _ in range(N)]
        for idx in range(N):
            v = order[idx]
            m = adj_mask[v]
            for j in range(idx + 1, N):
                u = order[j]
                if m & (1 << u):
                    later_neighbors[idx].append(j)
        colors = [-1] * N
        forb = [0] * N

        def backtrack(idx):
            if idx == N:
                return True
            avail = 7 & ~forb[idx]
            if avail == 0:
                return False
            c = avail
            while c:
                bit = c & (-c)
                c -= bit
                col = bit.bit_length() - 1
                colors[idx] = col
                touched = []
                for j in later_neighbors[idx]:
                    old = forb[j]
                    new = old | bit
                    forb[j] = new
                    touched.append((j, old))
                if backtrack(idx + 1):
                    return True
                for j, old in touched:
                    forb[j] = old
                colors[idx] = -1
            return False

        return backtrack(0)

    # Branch and Bound
    # State: adj_mask, forbidden_mask, current_edges, edge_count
    # We pass indices into all_pairs
    
    # To make it faster, we can precompute the "conflict" sets.
    # If we add (u,v), which edges become forbidden?
    # (u,w) for w in N(v) and (v,w) for w in N(u).
    
    # We will use a recursive function.
    # To avoid deep recursion limits, we might need to increase recursion limit or use iterative.
    # N=16, depth=120. Recursion limit is usually 1000, so it's fine.
    
    import sys
    sys.setrecursionlimit(5000)
    
    # Order of pairs: 
    # We want to find a 4-chromatic graph.
    # Let's try a random order but biased towards Clebsch edges?
    # Or just a fixed order.
    
    # Let's try multiple random orderings in a loop if time permits.
    
    def search_branch(idx, adj_mask, forbidden_mask, current_edges, edge_count):
        nonlocal best_edges, best_count
        
        if time.time() > deadline:
            return
            
        if idx == len(all_pairs):
            # Check if 4-chromatic
            if not is_3_colorable(adj_mask):
                if edge_count > best_count:
                    best_count = edge_count
                    best_edges = [list(e) for e in current_edges]
            return
            
        # Pruning
        # Max possible edges = current + remaining
        if edge_count + (len(all_pairs) - idx) <= best_count:
            return
            
        u, v = all_pairs[idx]
        bit_u = 1 << u
        bit_v = 1 << v
        
        # Check if (u,v) is forbidden
        if forbidden_mask[u] & bit_v:
            # Must skip
            search_branch(idx + 1, adj_mask, forbidden_mask, current_edges, edge_count)
            return
            
        # Check if adding (u,v) creates a triangle
        # Triangle exists if (adj_mask[u] & adj_mask[v]) != 0
        if adj_mask[u] & adj_mask[v]:
            # Cannot add
            search_branch(idx + 1, adj_mask, forbidden_mask, current_edges, edge_count)
            return
            
        # Branch 1: Add (u,v)
        # Save state
        old_adj_u = adj_mask[u]
        old_adj_v = adj_mask[v]
        old_forb = [m for m in forbidden_mask] # Copy is expensive, use in-place with undo
        
        # Actually, copying 16 ints is fast.
        new_forb = forbidden_mask[:]
        
        # Add edge
        adj_mask[u] |= bit_v
        adj_mask[v] |= bit_u
        current_edges.append((u, v))
        
        # Update forbidden mask
        # For any w in N(u), (v,w) is forbidden
        # For any w in N(v), (u,w) is forbidden
        # N(u) is adj_mask[u] (including v now)
        # We need to iterate through bits
        
        # Efficient update:
        # new_forb[v] |= adj_mask[u] # All neighbors of u are forbidden to connect to v
        # new_forb[u] |= adj_mask[v] # All neighbors of v are forbidden to connect to u
        # But we must not forbid the edge (u,v) itself if it's not in the pair list? 
        # (u,v) is the current edge, so it's handled.
        
        # Note: adj_mask[u] includes v. So new_forb[v] will have bit u set.
        # That's fine.
        
        new_forb[v] |= adj_mask[u]
        new_forb[u] |= adj_mask[v]
        
        search_branch(idx + 1, adj_mask, new_forb, current_edges, edge_count + 1)
        
        # Undo
        adj_mask[u] = old_adj_u
        adj_mask[v] = old_adj_v
        current_edges.pop()
        
        # Branch 2: Skip (u,v)
        search_branch(idx + 1, adj_mask, forbidden_mask, current_edges, edge_count)

    # Run search with a specific order
    # To improve chances, we can shuffle the pairs slightly or use a heuristic order.
    # Heuristic: Edges in Clebsch graph first?
    # If we put Clebsch edges first, we might quickly reach a 4-chromatic state and prune.
    
    clebsch_set = set(clebsch_edges)
    # Sort: Clebsch edges first, then others
    # But we want to explore non-Clebsch edges to find better solutions.
    # Maybe random shuffle is better to escape local optima.
    
    # Let's run the B&B with a random shuffle of pairs.
    # We can run it multiple times with different shuffles if time permits.
    
    iterations = 0
    while time.time() < deadline:
        iterations += 1
        # Shuffle pairs
        random.shuffle(all_pairs)
        
        # Reset state
        adj_mask = [0] * N
        forbidden_mask = [0] * N
        current_edges = []
        
        # Start search
        search_branch(0, adj_mask, forbidden_mask, current_edges, 0)
        
        # If we found a solution with 64 edges (max), we can stop
        if best_count >= 64:
            break
            
        # Optional: If time is low, break
        if time.time() > deadline - 5.0:
            break

    return best_edges
