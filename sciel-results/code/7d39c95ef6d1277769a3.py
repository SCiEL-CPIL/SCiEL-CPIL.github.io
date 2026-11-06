import time
import random
from itertools import combinations

def generate_paley(q):
    """Generate the Paley graph on q vertices."""
    if q % 4 != 1:
        return None
    squares = set()
    for i in range(1, q):
        squares.add((i * i) % q)
    
    edges = []
    for i in range(q):
        for j in range(i + 1, q):
            if (j - i) % q in squares:
                edges.append([i, j])
    return {'n': q, 'edges': edges}

def build_adjacency(n, edges):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def get_bitmasks(adj, n):
    masks = [0] * n
    for u in range(n):
        mask = 0
        for v in adj[u]:
            mask |= (1 << v)
        masks[u] = mask
    return masks

def has_k4(adj, n):
    """Check if graph contains a K4 using bitmask intersection."""
    masks = [0] * n
    for u in range(n):
        m = 0
        for v in adj[u]:
            m |= (1 << v)
        masks[u] = m
    for u in range(n):
        neighbors = list(adj[u])
        for i in range(len(neighbors)):
            v = neighbors[i]
            for j in range(i + 1, len(neighbors)):
                w = neighbors[j]
                if (masks[v] >> w) & 1:
                    # u, v, w form a triangle
                    # Check for 4th vertex: common neighbors of v and w that are also neighbors of u
                    common = masks[v] & masks[w] & masks[u]
                    # Remove u, v, w themselves
                    common &= ~(1 << u) & ~(1 << v) & ~(1 << w)
                    if common:
                        return True
    return False

def has_independent_set(adj, n, k):
    """Check if graph contains an independent set of size k using bitmask backtracking."""
    if n < k:
        return False
    masks = get_bitmasks(adj, n)
    
    def backtrack(start, current_mask, size):
        if size == k:
            return True
        if start >= n:
            return False
        # Pruning: even taking all remaining can't reach k
        if (n - start) < (k - size):
            return False
        # Try vertices in order of fewest neighbors (more likely to be independent)
        candidates = []
        for v in range(start, n):
            if not (current_mask & (1 << v)):
                candidates.append(v)
        # Sort by degree (ascending) for better pruning
        candidates.sort(key=lambda v: len(adj[v]))
        for v in candidates:
            if backtrack(v + 1, current_mask | masks[v], size + 1):
                return True
        return False
    
    return backtrack(0, 0, 0)

def get_independent_sets_size_k(adj, n, k):
    """Return list of independent sets of size k using bitmasks for speed."""
    if n < k:
        return []
    
    masks = get_bitmasks(adj, n)
    sets = []
    
    def backtrack(start, current_set, neighbor_mask):
        if len(current_set) == k:
            sets.append(tuple(current_set))
            return
        remaining = k - len(current_set)
        if n - start < remaining:
            return
        
        for v in range(start, n):
            if not (neighbor_mask & (1 << v)):
                current_set.append(v)
                backtrack(v + 1, current_set, neighbor_mask | masks[v])
                current_set.pop()
                
    backtrack(0, [], 0)
    return sets

def try_add_vertex(adj, n, deadline, masks):
    """
    Tries to find a valid set of neighbors N for a new vertex v_new.
    Returns the set N if found, else None.
    """
    # Step 1: Find all independent sets of size 5 in G
    ind_sets_5 = get_independent_sets_size_k(adj, n, 5)
    
    if not ind_sets_5:
        # No independent sets of size 5, so constraint 2 is always satisfied.
        # We just need a triangle-free set N.
        N = set()
        remaining = list(range(n))
        random.shuffle(remaining)
        for v in remaining:
            # Check if adding v to N creates a triangle
            neigh_in_N = [u for u in N if u in adj[v]]
            triangle_formed = False
            for i in range(len(neigh_in_N)):
                for j in range(i+1, len(neigh_in_N)):
                    if neigh_in_N[j] in adj[neigh_in_N[i]]:
                        triangle_formed = True
                        break
                if triangle_formed:
                    break
            if not triangle_formed:
                N.add(v)
        return N

    # Step 2: Backtracking search for N
    active_sets = [set(s) for s in ind_sets_5]
    
    def backtrack(N, active_sets_copy):
        if not active_sets_copy:
            return N.copy()
        
        if time.time() > deadline:
            return None
            
        # Heuristic: Pick the active set with the smallest number of candidates
        # that can be added to N without forming a triangle.
        
        best_set_idx = -1
        best_candidates = None
        min_candidates = float('inf')
        
        indices = list(range(len(active_sets_copy)))
        random.shuffle(indices)
        
        for idx in indices:
            target_set = active_sets_copy[idx]
            candidates = []
            for v in target_set:
                # Check if v can be added to N
                valid = True
                neigh_v_in_N = [u for u in N if u in adj[v]]
                for i in range(len(neigh_v_in_N)):
                    for j in range(i+1, len(neigh_v_in_N)):
                        if neigh_v_in_N[j] in adj[neigh_v_in_N[i]]:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    candidates.append(v)
            
            if len(candidates) < min_candidates:
                min_candidates = len(candidates)
                best_set_idx = idx
                best_candidates = candidates
            
            if min_candidates == 0:
                break
        
        if min_candidates == 0:
            return None
            
        random.shuffle(best_candidates)
        
        for v in best_candidates:
            new_N = N | {v}
            
            # Update active sets: remove any set that contains v
            new_active_sets = []
            for s in active_sets_copy:
                if v not in s:
                    new_active_sets.append(s)
            
            result = backtrack(new_N, new_active_sets)
            if result is not None:
                return result
                
        return None

    # Try multiple times with different random shuffles
    for _ in range(50):
        res = backtrack(set(), [set(s) for s in ind_sets_5])
        if res is not None:
            return res
            
    return None

def local_search_polish(adj, n, masks, deadline):
    """
    Try edge swaps (add/remove edges) to enable further vertex additions.
    Returns (improved, new_n) if the graph was modified.
    """
    improved = False
    
    for iteration in range(100):
        if time.time() > deadline:
            break
        
        # Try random edge swaps
        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        if u == v:
            continue
        
        is_edge = v in adj[u]
        
        # Tentatively apply the swap
        if is_edge:
            adj[u].remove(v)
            adj[v].remove(u)
            masks[u] &= ~(1 << v)
            masks[v] &= ~(1 << u)
        else:
            adj[u].add(v)
            adj[v].add(u)
            masks[u] |= (1 << v)
            masks[v] |= (1 << u)
        
        # Check validity
        if has_k4(adj, n) or has_independent_set(adj, n, 6):
            # Revert
            if is_edge:
                adj[u].add(v)
                adj[v].add(u)
                masks[u] |= (1 << v)
                masks[v] |= (1 << u)
            else:
                adj[u].remove(v)
                adj[v].remove(u)
                masks[u] &= ~(1 << v)
                masks[v] &= ~(1 << u)
        else:
            # Valid! Try to add a vertex
            new_N = try_add_vertex(adj, n, deadline, masks)
            if new_N is not None:
                # Add new vertex
                adj.append(set(new_N))
                for u2 in new_N:
                    adj[u2].add(n)
                new_mask = 0
                for u2 in new_N:
                    new_mask |= (1 << u2)
                masks.append(new_mask)
                for u2 in new_N:
                    masks[u2] |= (1 << n)
                n += 1
                improved = True
                break  # Stop after successful addition
            else:
                # Revert the edge swap
                if is_edge:
                    adj[u].add(v)
                    adj[v].add(u)
                    masks[u] |= (1 << v)
                    masks[v] |= (1 << u)
                else:
                    adj[u].remove(v)
                    adj[v].remove(u)
                    masks[u] &= ~(1 << v)
                    masks[v] &= ~(1 << u)
    
    return improved, n

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    # Start with Paley 17
    g17 = generate_paley(17)
    if not g17:
        return {'n': 1, 'edges': []}
        
    adj = build_adjacency(17, g17['edges'])
    n = 17
    masks = get_bitmasks(adj, n)
    
    # Verify initial graph
    if has_k4(adj, n) or has_independent_set(adj, n, 6):
        # Fallback to smaller if P17 is invalid (it shouldn't be)
        return {'n': 1, 'edges': []}
    
    best_n = n
    best_adj = [set(s) for s in adj]
    best_masks = list(masks)
    
    # Main loop: try to add vertices
    while n < 35 and time.time() < deadline:
        added = False
        if time.time() > deadline - 5:
            break
            
        new_N = try_add_vertex(adj, n, deadline, masks)
        
        if new_N is not None:
            # Add new vertex n with neighbors new_N
            # Update adj
            adj.append(set(new_N))
            for u in new_N:
                adj[u].add(n)
            
            # Update masks
            # New mask for vertex n
            new_mask = 0
            for u in new_N:
                new_mask |= (1 << u)
            masks.append(new_mask)
            
            # Update existing masks to include n
            for u in new_N:
                masks[u] |= (1 << n)
                
            n += 1
            added = True
            best_n = n
            best_adj = [set(s) for s in adj]
            best_masks = list(masks)
            
        if not added:
            # Stuck. Try local search polish to enable further additions
            if time.time() < deadline - 10:
                improved, n = local_search_polish(adj, n, masks, deadline)
                if improved:
                    best_n = n
                    best_adj = [set(s) for s in adj]
                    best_masks = list(masks)
                    continue
            
            # If still stuck, try a few more random restarts
            success = False
            for _ in range(5):
                if time.time() > deadline - 5:
                    break
                new_N = try_add_vertex(adj, n, deadline, masks)
                if new_N is not None:
                    adj.append(set(new_N))
                    for u in new_N:
                        adj[u].add(n)
                    new_mask = 0
                    for u in new_N:
                        new_mask |= (1 << u)
                    masks.append(new_mask)
                    for u in new_N:
                        masks[u] |= (1 << n)
                    n += 1
                    success = True
                    best_n = n
                    best_adj = [set(s) for s in adj]
                    best_masks = list(masks)
                    break
            
            if not success:
                break
    
    # Use best result
    n = best_n
    adj = best_adj
    masks = best_masks
    
    # Final verification
    if has_k4(adj, n) or has_independent_set(adj, n, 6):
        # Shouldn't happen, but fallback
        return {'n': 1, 'edges': []}
    
    # Build edge list
    edges = []
    for u in range(n):
        for v in adj[u]:
            if u < v:
                edges.append([u, v])
                
    return {'n': n, 'edges': edges}
