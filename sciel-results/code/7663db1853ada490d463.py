import random
import time

def construct():
    n = 20
    best_edges = []
    best_count = 0
    start_time = time.time()
    timeout = 110  # Leave some buffer for 120s limit
    
    # Bit operations for popcount
    def popcount(x):
        return bin(x).count('1')
        
    def can_add_edge(u, v, masks):
        # Check if adding (u, v) violates the condition
        # Condition: No 3 vertices have 3 common neighbours.
        
        # Case 1: Triples {u, a, b}
        # v becomes a common neighbor of u, a, b if v in N(a) and v in N(b).
        # If so, new common neighbors count = old_count + 1.
        # old_count = |N(u) & N(a) & N(b)|
        # We need old_count + 1 <= 2 => old_count <= 1.
        
        # Iterate a
        for a in range(n):
            if a == u or a == v:
                continue
            if not (masks[a] & (1 << v)):
                continue
            # v is in N(a)
            for b in range(a + 1, n):
                if b == u or b == v:
                    continue
                if not (masks[b] & (1 << v)):
                    continue
                # v is in N(a) and N(b)
                inter = masks[u] & masks[a] & masks[b]
                if popcount(inter) > 1:
                    return False
        
        # Case 2: Triples {v, a, b}
        # u becomes a common neighbor of v, a, b if u in N(a) and u in N(b).
        # If so, new common neighbors count = old_count + 1.
        # old_count = |N(v) & N(a) & N(b)|
        # We need old_count <= 1.
        
        for a in range(n):
            if a == u or a == v:
                continue
            if not (masks[a] & (1 << u)):
                continue
            for b in range(a + 1, n):
                if b == u or b == v:
                    continue
                if not (masks[b] & (1 << u)):
                    continue
                inter = masks[v] & masks[a] & masks[b]
                if popcount(inter) > 1:
                    return False
        
        return True

    def get_edges(masks):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if masks[i] & (1 << j):
                    edges.append([i, j])
        return edges

    # Local Search
    for trial in range(1000):
        if time.time() - start_time > timeout:
            break
            
        random.seed(trial)
        masks = [0] * n
        edges = []
        
        # 1. Greedy Initialization
        all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        random.shuffle(all_edges)
        
        for u, v in all_edges:
            if can_add_edge(u, v, masks):
                masks[u] |= (1 << v)
                masks[v] |= (1 << u)
                edges.append([u, v])
                
        current_count = len(edges)
        
        # 2. Local Optimization
        # We will try to improve by removing an edge and re-adding others.
        # To make it efficient, we maintain a list of non-edges.
        # But updating non-edges is tricky. 
        # Instead, we just try random moves.
        
        improved = True
        max_moves = 200 # Limit moves per trial to prevent timeout
        moves_made = 0
        
        while improved and moves_made < max_moves and time.time() - start_time < timeout:
            improved = False
            moves_made += 1
            
            # Try to add a random non-edge
            # Pick random u, v
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            if u > v:
                u, v = v, u
                
            if not (masks[u] & (1 << v)):
                if can_add_edge(u, v, masks):
                    masks[u] |= (1 << v)
                    masks[v] |= (1 << u)
                    edges.append([u, v])
                    current_count += 1
                    improved = True
                    continue
            
            # If no edge added, try a "swap" or "remove and refill"
            # Pick a random edge to remove
            if edges:
                idx = random.randint(0, len(edges) - 1)
                rem_u, rem_v = edges[idx]
                # Remove
                masks[rem_u] &= ~(1 << rem_v)
                masks[rem_v] &= ~(1 << rem_u)
                edges[idx] = edges[-1]
                edges.pop()
                current_count -= 1
                
                # Try to add edges from a shuffled list of all possible non-edges
                # We can just try a few random additions
                for _ in range(20):
                    au = random.randint(0, n - 1)
                    av = random.randint(0, n - 1)
                    if au == av:
                        continue
                    if au > av:
                        au, av = av, au
                    if not (masks[au] & (1 << av)):
                        if can_add_edge(au, av, masks):
                            masks[au] |= (1 << av)
                            masks[av] |= (1 << au)
                            edges.append([au, av])
                            current_count += 1
                            
                if current_count > best_count:
                    best_count = current_count
                    best_edges = [e[:] for e in edges]
        
        if current_count > best_count:
            best_count = current_count
            best_edges = [e[:] for e in edges]
            
    return best_edges
