import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    best_edges = []
    best_count = 0
    
    n = 16
    
    # Precompute neighbors as bitsets
    # adj[i] is a bitmask of neighbors of i
    
    def get_edges(adj):
        edges = []
        for i in range(n):
            mask = adj[i]
            while mask:
                lsb = mask & -mask
                j = lsb.bit_length() - 1
                if i < j:
                    edges.append([i, j])
                mask ^= lsb
        return edges

    def has_q3(adj, u, v):
        # Check if adding edge (u,v) creates a Q3
        # Assume (u,v) maps to (0,1) in Q3
        # Q3 vertices 0..7
        # Edges: (i, i^1), (i, i^2), (i, i^4) for i in 0..7
        # We fix phi(0)=u, phi(1)=v
        
        # Candidates for phi(2) and phi(4) must be neighbors of u
        # and not v
        
        mask_u = adj[u]
        # Remove v from candidates
        candidates = mask_u & ~(1 << v)
        
        # Iterate over pairs of distinct candidates for phi(2) and phi(4)
        # We need phi(2) != phi(4)
        
        # To be efficient, we can iterate over set bits
        # But number of candidates is small (degree of u)
        
        # Let's collect candidates in a list
        cands = []
        m = candidates
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            cands.append(idx)
            m ^= lsb
            
        for i in range(len(cands)):
            p2 = cands[i]
            for j in range(i+1, len(cands)):
                p4 = cands[j]
                
                # p2, p4 are distinct neighbors of u, not v
                
                # phi(3) in N(v) \cap N(p2)
                # phi(5) in N(v) \cap N(p4)
                
                mask_v = adj[v]
                mask_p2 = adj[p2]
                mask_p4 = adj[p4]
                
                c3_mask = mask_v & mask_p2
                c5_mask = mask_v & mask_p4
                
                # Exclude used vertices: u, v, p2, p4
                used_mask = (1 << u) | (1 << v) | (1 << p2) | (1 << p4)
                
                c3_mask &= ~used_mask
                c5_mask &= ~used_mask
                
                # Get lists for p3 and p5
                c3_list = []
                m3 = c3_mask
                while m3:
                    lsb = m3 & -m3
                    idx = lsb.bit_length() - 1
                    c3_list.append(idx)
                    m3 ^= lsb
                    
                c5_list = []
                m5 = c5_mask
                while m5:
                    lsb = m5 & -m5
                    idx = lsb.bit_length() - 1
                    c5_list.append(idx)
                    m5 ^= lsb
                    
                for p3 in c3_list:
                    for p5 in c5_list:
                        if p3 == p5:
                            continue
                        
                        # phi(6) in N(p2) \cap N(p4)
                        c6_mask = mask_p2 & mask_p4
                        
                        # Exclude u, v, p2, p4, p3, p5
                        used_mask2 = used_mask | (1 << p3) | (1 << p5)
                        c6_mask &= ~used_mask2
                        
                        c6_list = []
                        m6 = c6_mask
                        while m6:
                            lsb = m6 & -m6
                            idx = lsb.bit_length() - 1
                            c6_list.append(idx)
                            m6 ^= lsb
                            
                        for p6 in c6_list:
                            # phi(7) in N(p3) \cap N(p5) \cap N(p6)
                            c7_mask = adj[p3] & adj[p5] & adj[p6]
                            
                            # Exclude all 7 others
                            used_mask3 = used_mask2 | (1 << p6)
                            c7_mask &= ~used_mask3
                            
                            if c7_mask != 0:
                                return True
        return False

    # Greedy search with random restarts
    while time.time() < deadline:
        adj = [0] * n
        edges = []
        
        # Random edge order
        all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_edges)
        
        for u, v in all_edges:
            if not has_q3(adj, u, v):
                # Add edge
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edges.append([u, v])
                
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges[:]
            
        # If we found a very good solution, we might want to break?
        # But let's keep searching for a bit longer to see if we can improve.
        
    return best_edges
