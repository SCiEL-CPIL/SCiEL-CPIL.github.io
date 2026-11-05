import time
import random
import sys

def construct():
    n = 20
    start_time = time.time()
    deadline = start_time + 110.0
    
    best_edges = 0
    best_adj_bits = None
    best_edge_list = []
    
    def check_validity(adj_bits):
        for u in range(n):
            for v in range(u+1, n):
                if (adj_bits[u] >> v) & 1:
                    common = (adj_bits[u] & adj_bits[v]).bit_count()
                    if common > 2:
                        return False
        return True

    def count_edges(adj_bits):
        total = 0
        for i in range(n):
            total += adj_bits[i].bit_count()
        return total // 2

    def get_edge_list(adj_bits):
        edges = []
        for u in range(n):
            for v in range(u+1, n):
                if (adj_bits[u] >> v) & 1:
                    edges.append([u, v])
        return edges

    def can_add_edge(adj_bits, u, v):
        # 1. Check pair (u, v)
        common_uv = (adj_bits[u] & adj_bits[v]).bit_count()
        if common_uv > 2:
            return False
        
        # 2. Check pairs (u, w) where w in N(v)
        # w is neighbor of v.
        # If we add (u,v), v becomes common neighbor of u and w.
        # So common(u,w) increases by 1.
        # We need common(u,w) + 1 <= 2 => common(u,w) <= 1.
        # Iterate neighbors of v
        mask_v = adj_bits[v]
        # Exclude u
        mask_v_ex_u = mask_v & ~(1 << u)
        w = 0
        while mask_v_ex_u:
            # Get lowest set bit
            lsb = mask_v_ex_u & -mask_v_ex_u
            w = lsb.bit_length() - 1
            mask_v_ex_u ^= lsb
            
            common_uw = (adj_bits[u] & adj_bits[w]).bit_count()
            if common_uw > 1:
                return False

        # 3. Check pairs (v, w) where w in N(u)
        mask_u = adj_bits[u]
        mask_u_ex_v = mask_u & ~(1 << v)
        w = 0
        while mask_u_ex_v:
            lsb = mask_u_ex_v & -mask_u_ex_v
            w = lsb.bit_length() - 1
            mask_u_ex_v ^= lsb
            
            common_vw = (adj_bits[v] & adj_bits[w]).bit_count()
            if common_vw > 1:
                return False

        return True

    def try_add_all(adj_bits):
        modified = True
        while modified:
            modified = False
            for u in range(n):
                for v in range(u+1, n):
                    if not ((adj_bits[u] >> v) & 1):
                        if can_add_edge(adj_bits, u, v):
                            adj_bits[u] |= (1 << v)
                            adj_bits[v] |= (1 << u)
                            modified = True
                            # Early exit if deadline? No, just finish this pass.

    def local_search(adj_bits):
        # Greedy fill
        try_add_all(adj_bits)
        
        # Hill climbing
        # We will try to add edges. If we can't, we remove a random edge and try again.
        # To be more effective, we can try to remove an edge that is "blocking" many additions.
        
        for _ in range(500): # More iterations
            if time.time() > deadline:
                break
            
            # Try to add a random missing edge
            added = False
            # Try a few random candidates
            for _ in range(20):
                u = random.randint(0, n-1)
                v = random.randint(0, n-1)
                if u == v: continue
                if u > v: u, v = v, u
                if not ((adj_bits[u] >> v) & 1):
                    if can_add_edge(adj_bits, u, v):
                        adj_bits[u] |= (1 << v)
                        adj_bits[v] |= (1 << u)
                        added = True
                        break
            
            if not added:
                # Stuck. Remove a random edge to unlock.
                # Pick a random existing edge
                existing_edges = []
                for i in range(n):
                    mask = adj_bits[i]
                    while mask:
                        lsb = mask & -mask
                        j = lsb.bit_length() - 1
                        if j > i:
                            existing_edges.append((i, j))
                        mask ^= lsb
                
                if existing_edges:
                    u, v = random.choice(existing_edges)
                    adj_bits[u] &= ~(1 << v)
                    adj_bits[v] &= ~(1 << u)
                    # After removal, try to fill again immediately
                    try_add_all(adj_bits)

    num_restarts = 100
    for r in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Random initialization
        adj_bits = [0] * n
        # Add random edges
        # Bias towards higher density? 
        # p=0.2 is sparse. Maybe p=0.3 or 0.4?
        # Or start with a random structure.
        for u in range(n):
            for v in range(u+1, n):
                if random.random() < 0.3:
                    adj_bits[u] |= (1 << v)
                    adj_bits[v] |= (1 << u)
        
        # Clean up invalid K2,3s initially? 
        # The local search will handle it, but starting valid is better.
        # Simple cleanup: if K2,3 found, remove random edge from it.
        # For simplicity, just run local search. It will add valid edges.
        # But if initial graph has K2,3, it's invalid.
        # We should ensure initial graph is valid or clean it.
        
        # Cleanup phase
        valid = False
        while not valid:
            valid = True
            for u in range(n):
                for v in range(u+1, n):
                    if (adj_bits[u] >> v) & 1:
                        common = (adj_bits[u] & adj_bits[v]).bit_count()
                        if common > 2:
                            # Remove a random common neighbor edge?
                            # Or just remove (u,v)?
                            # Removing (u,v) is safe.
                            adj_bits[u] &= ~(1 << v)
                            adj_bits[v] &= ~(1 << u)
                            valid = False
                            break
                if not valid: break
            if not valid: break

        local_search(adj_bits)
        
        # Final fill
        try_add_all(adj_bits)
        
        # Verify
        if check_validity(adj_bits):
            e_count = count_edges(adj_bits)
            if e_count > best_edges:
                best_edges = e_count
                best_adj_bits = adj_bits[:]
                best_edge_list = get_edge_list(adj_bits)

    if best_adj_bits is None:
        # Fallback
        return []

    return best_edge_list
