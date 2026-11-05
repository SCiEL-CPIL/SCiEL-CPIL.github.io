import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n = 16
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    # Fast 3-colorability check using bitmasks and backtracking
    def is_3_colorable(adj):
        colors = [-1] * n
        # Order vertices by degree descending
        degrees = [bin(adj[i]).count('1') for i in range(n)]
        order = sorted(range(n), key=lambda x: -degrees[x])
        
        def backtrack(idx):
            if idx == n:
                return True
            v = order[idx]
            for c in range(3):
                conflict = False
                mask = adj[v]
                while mask:
                    lsb = mask & -mask
                    u = lsb.bit_length() - 1
                    if colors[u] == c:
                        conflict = True
                        break
                    mask -= lsb
                if not conflict:
                    colors[v] = c
                    if backtrack(idx + 1):
                        return True
                    colors[v] = -1
            return False
        
        return backtrack(0)
    
    def can_add_edge(adj, u, v):
        return (adj[u] & adj[v]) == 0
    
    # Local search: try to add edges to a 4-chromatic graph
    def local_search_add(adj, current_edges, current_count):
        nonlocal best_edges, best_count
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try all possible edge additions
            for u in range(n):
                for v in range(u + 1, n):
                    if time.time() > deadline:
                        return
                    if adj[u] & (1 << v):
                        continue
                    if can_add_edge(adj, u, v):
                        # Try adding edge
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        if not is_3_colorable(adj):
                            # Still 4-chromatic, keep the edge
                            current_edges.append([u, v])
                            current_count += 1
                            if current_count > best_count:
                                best_count = current_count
                                best_edges = current_edges[:]
                            improved = True
                        else:
                            # Revert
                            adj[u] &= ~(1 << v)
                            adj[v] &= ~(1 << u)
        return current_edges, current_count
    
    # Try to improve a 4-chromatic graph by removing and re-adding edges
    def local_search_refine(adj, current_edges, current_count):
        nonlocal best_edges, best_count
        for iteration in range(50):
            if time.time() > deadline:
                break
            # Try removing a random edge and see if we can add more
            if not current_edges:
                break
            idx = random.randint(0, len(current_edges) - 1)
            u, v = current_edges[idx]
            # Remove edge
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            current_edges.pop(idx)
            current_count -= 1
            
            # Try to add edges greedily
            for eu in range(n):
                for ev in range(eu + 1, n):
                    if time.time() > deadline:
                        break
                    if adj[eu] & (1 << ev):
                        continue
                    if can_add_edge(adj, eu, ev):
                        adj[eu] |= (1 << ev)
                        adj[ev] |= (1 << eu)
                        if not is_3_colorable(adj):
                            current_edges.append([eu, ev])
                            current_count += 1
                        else:
                            adj[eu] &= ~(1 << ev)
                            adj[ev] &= ~(1 << eu)
            
            if current_count > best_count:
                best_count = current_count
                best_edges = current_edges[:]
        
        return current_edges, current_count
    
    # Construct Clebsch graph as baseline
    def build_clebsch():
        vectors = []
        for i in range(32):
            if bin(i).count('1') % 2 == 0:
                vectors.append(i)
        
        adj = [0] * n
        edges = []
        for u_idx in range(n):
            for v_idx in range(u_idx + 1, n):
                xor = vectors[u_idx] ^ vectors[v_idx]
                if bin(xor).count('1') == 4:
                    edges.append([u_idx, v_idx])
                    adj[u_idx] |= (1 << v_idx)
                    adj[v_idx] |= (1 << u_idx)
        return adj, edges
    
    # Initialize with Clebsch graph
    best_edges, best_count = [], 0
    clebsch_adj, clebsch_edges = build_clebsch()
    
    if not is_3_colorable(clebsch_adj):
        best_edges = clebsch_edges[:]
        best_count = len(clebsch_edges)
    
    # Phase 1: Random greedy constructions
    while time.time() < deadline - 20:
        adj = [0] * n
        edges = []
        
        # Shuffle edges
        shuffled = all_edges[:]
        random.shuffle(shuffled)
        
        for u, v in shuffled:
            if can_add_edge(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edges.append([u, v])
        
        # Check if 4-chromatic
        if not is_3_colorable(adj):
            if len(edges) > best_count:
                best_count = len(edges)
                best_edges = edges[:]
            
            # Local search to add more edges
            new_edges, new_count = local_search_add(adj[:], edges[:], len(edges))
            if new_count > best_count:
                best_count = new_count
                best_edges = new_edges[:]
    
    # Phase 2: Local search refinement from best found
    if best_count > 0 and time.time() < deadline - 10:
        # Reconstruct adj from best_edges
        adj = [0] * n
        for u, v in best_edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        
        # Try multiple refinement passes
        for _ in range(20):
            if time.time() > deadline - 5:
                break
            new_edges, new_count = local_search_refine(adj[:], best_edges[:], best_count)
            if new_count > best_count:
                best_count = new_count
                best_edges = new_edges[:]
                for u, v in best_edges:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
    
    # Phase 3: Final random greedy attempts with time remaining
    while time.time() < deadline - 2:
        adj = [0] * n
        edges = []
        
        shuffled = all_edges[:]
        random.shuffle(shuffled)
        
        for u, v in shuffled:
            if can_add_edge(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edges.append([u, v])
        
        if not is_3_colorable(adj):
            if len(edges) > best_count:
                best_count = len(edges)
                best_edges = edges[:]
    
    return best_edges
