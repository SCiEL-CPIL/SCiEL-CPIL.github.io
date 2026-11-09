import random
import time

N = 20
C = 20

def _check_conflict(adj_mask, colour_of, u, v, c):
    """
    Check whether adding edge (u,v) with colour c creates a non-rainbow 4-cycle.
    Returns True if conflict exists (invalid), False if valid.
    """
    nu = adj_mask[u]
    nv = adj_mask[v]
    
    # Iterate over neighbors x of v
    # We need to check 4-cycles u-v-x-y-u
    # x must be in N(v)
    # y must be in N(u) and N(x)
    # Vertices u, v, x, y must be distinct.
    
    x_iter = nv
    while x_iter:
        lsb = x_iter & (-x_iter)
        x = lsb.bit_length() - 1
        x_iter -= lsb
        
        # x != v is guaranteed (no self-loops)
        # x != u is guaranteed because u is not yet in N(v)
        
        cvx = colour_of[v][x]
        
        if cvx == c:
            # If cvx == c, any 4-cycle u-v-x-y-u would have c appearing twice (on uv and vx)
            # So we must ensure no such y exists.
            # y must be in N(u) and N(x).
            # Check if N(u) & N(x) is non-empty.
            if (nu & adj_mask[x]):
                return True
            continue
        
        # For each y in N(u) & N(x), check colors
        # y must be distinct from u, v, x.
        # y in N(u) => y != u
        # y in N(x) => y != x
        # y != v?
        # If y == v, then v is in N(x). So (v,x) is an edge.
        # The cycle would be u-v-x-v-u, which is not a simple cycle of length 4 (repeats v).
        # So we must exclude y == v.
        
        # Also, if x is in N(u), then (u,x) is an edge.
        # The cycle is u-v-x-y-u.
        # Vertices: u, v, x, y.
        # If x is in N(u), is it a valid 4-cycle?
        # Yes, a 4-cycle is defined by the sequence of vertices.
        # The existence of chord (u,x) does not invalidate the cycle u-v-x-y-u as a subgraph cycle.
        # However, we must ensure y != v.
        
        # Let's refine the mask for y.
        # y must be in N(u) & N(x).
        # y must not be v.
        
        common = nu & adj_mask[x]
        # Exclude v from common if v is in common
        # v is in N(x) iff x is in N(v). x IS in N(v). So v is in N(x).
        # v is in N(u)? No, u and v are not adjacent yet. So v is NOT in N(u).
        # Therefore v is NOT in (N(u) & N(x)).
        # So we don't need to explicitly exclude v.
        
        # What about u?
        # u is in N(u)? No.
        # So u is not in common.
        
        # What about x?
        # x is in N(x)? No.
        # So x is not in common.
        
        # So y_iter = nu & adj_mask[x] gives all valid candidates for y.
        
        y_iter = common
        while y_iter:
            lsb_y = y_iter & (-y_iter)
            y = lsb_y.bit_length() - 1
            y_iter -= lsb_y
            
            cxy = colour_of[x][y]
            cyu = colour_of[y][u]
            
            # Check distinctness: c, cvx, cxy, cyu
            if c == cxy or c == cyu:
                return True
            if cvx == cxy or cvx == cyu:
                return True
            if cxy == cyu:
                return True
                
    return False

def _greedy_construct(seed, deadline):
    """Run a single greedy construction pass with random edge and color ordering."""
    adj_mask = [0] * N
    colour_of = [[-1] * N for _ in range(N)]
    placed = []
    
    potential_edges = [(i, j) for i in range(N) for j in range(i + 1, N)]
    rng = random.Random(seed)
    rng.shuffle(potential_edges)
    
    for (u, v) in potential_edges:
        if time.time() > deadline:
            break
        
        cols = list(range(C))
        rng.shuffle(cols)
        
        for c in cols:
            if not _check_conflict(adj_mask, colour_of, u, v, c):
                # Add edge
                adj_mask[u] |= (1 << v)
                adj_mask[v] |= (1 << u)
                colour_of[u][v] = c
                colour_of[v][u] = c
                placed.append([u, v, c])
                break
    return placed

def _reconstruct_from_edges(edges):
    """Rebuild adj_mask and colour_of from a list of edges."""
    adj_mask = [0] * N
    colour_of = [[-1] * N for _ in range(N)]
    for u, v, c in edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
        colour_of[u][v] = c
        colour_of[v][u] = c
    return adj_mask, colour_of

def _local_search(edges, deadline):
    """
    Structural local search:
    1. Start with current best edges.
    2. Repeatedly remove a random edge and try to re-add it and other missing edges.
    3. Accept if the number of edges increases.
    """
    if not edges:
        return edges
    
    current_edges = [e[:] for e in edges]
    adj_mask, colour_of = _reconstruct_from_edges(current_edges)
    
    current_count = len(current_edges)
    all_edges_set = set((min(u, v), max(u, v)) for u, v, c in current_edges)
    all_possible = [(i, j) for i in range(N) for j in range(i + 1, N)]
    all_possible_set = set((i, j) for i, j in all_possible)
    
    rng = random.Random(12345)
    
    while time.time() < deadline:
        # Pick a random edge to remove
        if not current_edges:
            break
            
        idx = rng.randrange(len(current_edges))
        u, v, c = current_edges[idx]
        
        # Remove it
        current_edges.pop(idx)
        adj_mask[u] &= ~(1 << v)
        adj_mask[v] &= ~(1 << u)
        colour_of[u][v] = -1
        colour_of[v][u] = -1
        all_edges_set.discard((min(u, v), max(u, v)))
        
        # Candidates to add back: the removed edge + all missing edges
        missing = [e for e in all_possible if (min(e[0], e[1]), max(e[0], e[1])) not in all_edges_set]
        candidates = [(u, v)] + missing
        rng.shuffle(candidates)
        
        added_any = False
        new_edges_added = []
        
        for a, b in candidates:
            if time.time() > deadline:
                break
                
            # Skip if already present (shouldn't happen for missing, but just in case)
            if adj_mask[a] & (1 << b):
                continue
                
            cols = list(range(C))
            rng.shuffle(cols)
            
            for col in cols:
                if not _check_conflict(adj_mask, colour_of, a, b, col):
                    # Add it
                    adj_mask[a] |= (1 << b)
                    adj_mask[b] |= (1 << a)
                    colour_of[a][b] = col
                    colour_of[b][a] = col
                    new_edges_added.append([a, b, col])
                    all_edges_set.add((min(a, b), max(a, b)))
                    added_any = True
                    break
        
        current_edges.extend(new_edges_added)
        
        # If we added more than we removed (which is 1), we improved.
        # Since we removed 1, if we added >= 1, count is same or better.
        # Actually, we want to strictly improve or keep if equal?
        # If we removed 1 and added 1, count is same. We might be stuck in a loop.
        # Let's accept if count increased. If count is same, we can accept with some probability?
        # Or just accept if count >= previous count?
        # If count is same, we are just swapping edges. This can help escape local minima.
        # Let's accept if len(current_edges) >= current_count.
        
        if len(current_edges) >= current_count:
            current_count = len(current_edges)
            # Update best if improved
            # We are returning the current state, so the caller will check improvement.
            # But we should track the best within this function?
            # The prompt says "Maximize the number of edges".
            # Let's just return the current state. The main loop will compare.
            
        # If we didn't add anything back, we are stuck.
        # If we added exactly 1, we are at same size.
        # If we added > 1, we improved.
        
        # To avoid infinite loops on same-size swaps, we could add a small penalty or random restart.
        # But for now, let's just continue.
        
    return current_edges

def construct():
    start_time = time.time()
    # 150s limit. Use 140s to be safe.
    deadline = start_time + 140.0
    
    best_edges = []
    
    # Phase 1: Greedy Random Restarts
    # Run until we have used up 50% of the time or found a good solution.
    phase1_deadline = start_time + 70.0
    
    seed = 0
    while time.time() < phase1_deadline:
        edges = _greedy_construct(seed, phase1_deadline)
        if len(edges) > len(best_edges):
            best_edges = edges
        seed += 1
        
    # Phase 2: Local Search
    # If we have a good start, try to improve it.
    if best_edges:
        # Run local search for the remaining time
        # We might want to run multiple local searches?
        # Or just one long one?
        # Local search is sequential.
        # Let's run it.
        improved = _local_search(best_edges, deadline - 5.0)
        if len(improved) > len(best_edges):
            best_edges = improved
            
    # Phase 3: More Greedy Restarts if time permits and we are not dense?
    # Or maybe run more local searches on different random perturbations?
    
    # Let's try a hybrid: if local search didn't improve much, do more restarts.
    # Or just spend remaining time on more restarts if we are under some threshold.
    
    # Actually, local search can be slow. Let's ensure we return best found.
    
    return best_edges

