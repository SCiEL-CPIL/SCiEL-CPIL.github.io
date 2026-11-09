import time
import random
import math

def _is_3_colorable(adj_masks):
    """Check if a triangle-free graph is 3-colorable using backtracking with MRV heuristic."""
    n = 16
    # Precompute degrees for ordering
    degrees = [bin(mask).count('1') for mask in adj_masks]
    # Order by degree descending (greedy heuristic)
    order = sorted(range(n), key=lambda v: degrees[v], reverse=True)
    
    color = [-1] * n
    
    def backtrack(idx):
        if idx == n:
            return True
        v = order[idx]
        forbidden = 0
        neighbors = adj_masks[v]
        while neighbors:
            u_bit = neighbors & -neighbors
            u = u_bit.bit_length() - 1
            neighbors ^= u_bit
            if color[u] != -1:
                forbidden |= (1 << color[u])
        
        for c in range(3):
            if not (forbidden & (1 << c)):
                color[v] = c
                if backtrack(idx + 1):
                    return True
                color[v] = -1
        return False
    
    return backtrack(0)

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    n = 16
    
    # Start with a strong seed: Mycielski of C5 (Grötzsch graph) on vertices 0..10
    # Vertices 11..15 are initially isolated but will be connected via search
    adj = [0] * n
    
    # C5: 0-1-2-3-4-0
    for u, v in [(0,1),(1,2),(2,3),(3,4),(4,0)]:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    # Copies 5-9
    for u, v in [(5,1),(5,4),(6,0),(6,2),(7,1),(7,3),(8,2),(8,4),(9,3),(9,0)]:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    # Apex 10
    for v in range(5, 10):
        adj[10] |= (1 << v)
        adj[v] |= (1 << 10)
    
    current_adj = [m for m in adj]
    current_edges = sum(bin(m).count('1') for m in current_adj) // 2
    best_adj = [m for m in current_adj]
    best_edges = current_edges
    
    def get_edge_count(adj):
        return sum(bin(m).count('1') for m in adj) // 2
    
    def get_forbidden_pairs(adj):
        """Return set of (u,v) pairs that would create a triangle if added."""
        forbidden = set()
        for u in range(n):
            for v in range(u+1, n):
                if adj[u] & (1 << v):
                    continue
                if adj[u] & adj[v]:
                    forbidden.add((u, v))
        return forbidden
    
    # Phase 1: Greedy edge addition
    for iteration in range(300):
        if time.time() > deadline:
            break
        
        added_any = True
        while added_any and time.time() < deadline:
            added_any = False
            forbidden = get_forbidden_pairs(current_adj)
            candidates = []
            for u in range(n):
                for v in range(u+1, n):
                    if (u, v) in forbidden:
                        continue
                    if current_adj[u] & (1 << v):
                        continue
                    candidates.append((u, v))
            
            random.shuffle(candidates)
            for u, v in candidates:
                if time.time() > deadline:
                    break
                new_adj = [m for m in current_adj]
                new_adj[u] |= (1 << v)
                new_adj[v] |= (1 << u)
                
                if _is_3_colorable(new_adj):
                    continue
                
                current_adj = new_adj
                current_edges += 1
                added_any = True
                if current_edges > best_edges:
                    best_edges = current_edges
                    best_adj = [m for m in current_adj]
                break
    
    # Phase 2: Local search with edge swaps and additions
    for step in range(2000):
        if time.time() > deadline:
            break
        
        move_type = random.random()
        
        if move_type < 0.5:
            # Add edge
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u == v or (current_adj[u] & (1 << v)):
                continue
            if current_adj[u] & current_adj[v]:
                continue
            
            new_adj = [m for m in current_adj]
            new_adj[u] |= (1 << v)
            new_adj[v] |= (1 << u)
            
            if _is_3_colorable(new_adj):
                continue
            
            current_adj = new_adj
            current_edges += 1
            if current_edges > best_edges:
                best_edges = current_edges
                best_adj = [m for m in current_adj]
                
        elif move_type < 0.7:
            # Remove edge
            edges = []
            for u in range(n):
                v_mask = current_adj[u]
                while v_mask:
                    v_bit = v_mask & -v_mask
                    v = v_bit.bit_length() - 1
                    v_mask ^= v_bit
                    if u < v:
                        edges.append((u, v))
            if not edges:
                continue
            u, v = random.choice(edges)
            new_adj = [m for m in current_adj]
            new_adj[u] &= ~(1 << v)
            new_adj[v] &= ~(1 << u)
            
            if _is_3_colorable(new_adj):
                continue
            
            current_adj = new_adj
            current_edges -= 1
            
        else:
            # Swap: remove one edge, add another
            edges = []
            for u in range(n):
                v_mask = current_adj[u]
                while v_mask:
                    v_bit = v_mask & -v_mask
                    v = v_bit.bit_length() - 1
                    v_mask ^= v_bit
                    if u < v:
                        edges.append((u, v))
            if not edges:
                continue
            u1, v1 = random.choice(edges)
            
            new_adj = [m for m in current_adj]
            new_adj[u1] &= ~(1 << v1)
            new_adj[v1] &= ~(1 << u1)
            
            u2 = random.randint(0, n-1)
            v2 = random.randint(0, n-1)
            if u2 == v2 or (new_adj[u2] & (1 << v2)):
                continue
            if new_adj[u2] & new_adj[v2]:
                continue
            
            new_adj[u2] |= (1 << v2)
            new_adj[v2] |= (1 << u2)
            
            if _is_3_colorable(new_adj):
                continue
            
            current_adj = new_adj
    
    # Final polish: try to add any remaining possible edges
    forbidden = get_forbidden_pairs(best_adj)
    for u in range(n):
        for v in range(u+1, n):
            if (u, v) in forbidden:
                continue
            if best_adj[u] & (1 << v):
                continue
            new_adj = [m for m in best_adj]
            new_adj[u] |= (1 << v)
            new_adj[v] |= (1 << u)
            if not _is_3_colorable(new_adj):
                best_adj = new_adj
                best_edges += 1
    
    # Convert to edge list
    edges = []
    for u in range(n):
        for v in range(u+1, n):
            if best_adj[u] & (1 << v):
                edges.append([u, v])
                
    return edges
