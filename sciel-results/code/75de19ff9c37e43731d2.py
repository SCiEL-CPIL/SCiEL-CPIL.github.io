import time
import random

def construct():
    n = 20
    deadline = time.time() + 110  # Safety margin
    
    def popcount(x):
        return bin(x).count("1")
        
    def get_neighbors(mask):
        neighbors = []
        i = 0
        while mask:
            if mask & 1:
                neighbors.append(i)
            mask >>= 1
            i += 1
        return neighbors

    best_edges = []
    best_count = 0
    
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    while time.time() < deadline:
        # Strategy: Random dense init, clean, fill
        # Try different densities
        p = random.uniform(0.3, 0.7)
        
        # adj[i] is bitmask of neighbors of i
        adj = [0] * n
        edge_set = set()
        
        # 1. Initialize
        for u in range(n):
            for v in range(u + 1, n):
                if random.random() < p:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    edge_set.add((u, v))
        
        # 2. Clean phase: Remove edges causing K2,3
        # We iterate until no violations or max steps
        max_clean_steps = 1000
        steps = 0
        while steps < max_clean_steps:
            steps += 1
            violation_found = False
            # Find a violating pair
            # We can just scan all pairs
            for u in range(n):
                for v in range(u + 1, n):
                    common = adj[u] & adj[v]
                    if popcount(common) > 2:
                        violation_found = True
                        # Pick a random common neighbor to remove edge to
                        common_list = get_neighbors(common)
                        w = random.choice(common_list)
                        # Remove edge (u, w) or (v, w)
                        if random.random() < 0.5:
                            # Remove (u, w)
                            adj[u] &= ~(1 << w)
                            adj[w] &= ~(1 << u)
                            edge_set.discard((min(u, w), max(u, w)))
                        else:
                            # Remove (v, w)
                            adj[v] &= ~(1 << w)
                            adj[w] &= ~(1 << v)
                            edge_set.discard((min(v, w), max(v, w)))
                        break
                if violation_found:
                    break
            if not violation_found:
                break
        
        # 3. Fill phase: Add missing edges if possible
        # Shuffle missing edges
        missing_edges = [e for e in all_edges if e not in edge_set]
        random.shuffle(missing_edges)
        
        for u, v in missing_edges:
            # Check if adding (u, v) is valid
            # 1. common neighbors of u and v < 3
            if popcount(adj[u] & adj[v]) >= 3:
                continue
            
            # 2. For each b in N(v), common(u, b) < 2 (so <= 1)
            #    Actually, if we add (u,v), then v becomes a neighbor of u.
            #    The condition is: for all x, y in V, |N(x) \cap N(y)| <= 2.
            #    Adding (u,v) affects pairs involving u or v.
            #    Pairs (u, b) for b in N(u) \cup {v}.
            #    Pairs (v, a) for a in N(v) \cup {u}.
            #    Pairs (x, y) where x,y not u,v are unaffected.
            
            valid = True
            
            # Check pairs (u, b) where b is a neighbor of u or b=v
            # If b=v, common(u,v) is already checked.
            # If b is in N(u), common(u,b) must be <= 1 AFTER adding (u,v).
            # Wait, adding (u,v) adds v to N(u) and u to N(v).
            # It does NOT change N(b) for b != u, v.
            # So N(u) \cap N(b) becomes (N(u) \cup {v}) \cap N(b) = (N(u) \cap N(b)) \cup ({v} \cap N(b)).
            # So if v is in N(b), the count increases by 1.
            # We need new count <= 2.
            # Old count + (1 if v in N(b) else 0) <= 2.
            
            # So for all b in N(u):
            # popcount(adj[u] & adj[b]) + (1 if (adj[b] & (1<<v)) else 0) <= 2
            
            mask_u = adj[u]
            i = 0
            while mask_u:
                if mask_u & 1:
                    b = i
                    # b is a neighbor of u
                    # Check if v is a neighbor of b
                    inc = 1 if (adj[b] & (1 << v)) else 0
                    if popcount(adj[u] & adj[b]) + inc > 2:
                        valid = False
                        break
                mask_u >>= 1
                i += 1
            
            if not valid:
                continue
                
            # Check pairs (v, a) where a is a neighbor of v or a=u
            mask_v = adj[v]
            i = 0
            while mask_v:
                if mask_v & 1:
                    a = i
                    # a is a neighbor of v
                    # Check if u is a neighbor of a
                    inc = 1 if (adj[a] & (1 << u)) else 0
                    if popcount(adj[v] & adj[a]) + inc > 2:
                        valid = False
                        break
                mask_v >>= 1
                i += 1
            
            if not valid:
                continue
            
            # Also need to check if u and v themselves have < 3 common neighbors?
            # Yes, checked at start.
            
            # Add edge
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            edge_set.add((u, v))
            
        current_count = len(edge_set)
        if current_count > best_count:
            best_count = current_count
            best_edges = list(edge_set)
            
    return [[u, v] for u, v in best_edges]
