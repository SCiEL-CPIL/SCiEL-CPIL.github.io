import time
import random
import itertools

def construct():
    start_time = time.time()
    time_limit = 115.0
    
    n = 16
    adj = [0] * n
    
    # Initialize with K16
    for i in range(n):
        for j in range(i+1, n):
            adj[i] |= (1 << j)
            adj[j] |= (1 << i)
            
    subsets_8 = list(itertools.combinations(range(n), 8))
    
    def is_q3(vertices):
        local_adj = [0] * 8
        v_list = list(vertices)
        for i in range(8):
            u = v_list[i]
            mask = adj[u]
            # Extract bits for the 8 vertices
            for j in range(8):
                if i != j and (mask >> v_list[j]) & 1:
                    local_adj[i] |= (1 << j)
                    
        # Degree check
        for i in range(8):
            if local_adj[i].bit_count() != 3:
                return False
                
        # Bipartition check
        colors = [-1] * 8
        for start in range(8):
            if colors[start] != -1: continue
            colors[start] = 0
            stack = [start]
            while stack:
                u = stack.pop()
                mask = local_adj[u]
                for v in range(8):
                    if (mask >> v) & 1:
                        if colors[v] == -1:
                            colors[v] = 1 - colors[u]
                            stack.append(v)
                        elif colors[v] == colors[u]:
                            return False
                            
        if colors.count(0) != 4: return False
        
        # Common neighbors check
        for i in range(8):
            for j in range(i+1, 8):
                common = local_adj[i] & local_adj[j]
                c = common.bit_count()
                if colors[i] == colors[j]:
                    if c != 2: return False
                else:
                    if c != 0: return False
                    
        return True

    def get_q3_edges(vertices):
        edges = []
        v_list = list(vertices)
        for i in range(8):
            u = v_list[i]
            for j in range(i+1, 8):
                if (adj[u] >> v_list[j]) & 1:
                    edges.append((u, v_list[j]))
        return edges

    iterations = 0
    stuck_counter = 0
    
    while time.time() - start_time < time_limit:
        iterations += 1
        
        # Stuck detection
        if stuck_counter > 2000:
            # Kick
            for _ in range(5):
                i = random.randint(0, n-1)
                j = random.randint(0, n-1)
                if i != j and (adj[i] >> j) & 1:
                    adj[i] &= ~(1 << j)
                    adj[j] &= ~(1 << i)
            stuck_counter = 0
            continue

        # Pick random 8-subset
        idx = random.randint(0, len(subsets_8) - 1)
        verts = subsets_8[idx]
        
        if is_q3(verts):
            edges = get_q3_edges(verts)
            if edges:
                e = random.choice(edges)
                u, v = e
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                stuck_counter = 0
        else:
            # Try add
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u == v or ((adj[u] >> v) & 1):
                continue
                
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
            # Limited check: sample 50 subsets containing u,v
            # Generate candidates
            # We can just pick random subsets from subsets_8 and filter?
            # Or generate 6-combos from remaining?
            # Generating 6-combos is faster if we precompute?
            # Let's just pick 50 random 8-subsets from the full list and check if they contain u,v
            # This is biased but fast.
            # Better: pick 50 random 6-subsets from the 14 others.
            others = [x for x in range(n) if x != u and x != v]
            bad = False
            for _ in range(50):
                s = random.sample(others, 6)
                if is_q3(tuple(list(s) + [u, v])):
                    bad = True
                    break
                    
            if bad:
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
            else:
                stuck_counter = 0

    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if (adj[i] >> j) & 1:
                edges.append([i, j])
    return edges
