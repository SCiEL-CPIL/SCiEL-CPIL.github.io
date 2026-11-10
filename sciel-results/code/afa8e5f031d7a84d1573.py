import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 115s to be safe
    
    best_edges = []
    best_count = 0
    
    # Pre-compute all possible edges
    all_edges = [(i, j) for i in range(16) for j in range(i+1, 16)]
    
    # Shuffle and try
    # We will run multiple random restarts
    while time.time() < deadline:
        random.shuffle(all_edges)
        adj = [0] * 16  # Bitmasks
        current_edges = []
        
        for u, v in all_edges:
            # Check if adding (u, v) creates a Q3
            if not creates_q3(u, v, adj):
                # Add edge
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                current_edges.append([u, v])
                
        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_edges = current_edges.copy()
            
    return best_edges

def creates_q3(u, v, adj):
    # Check if there exists a Q3 containing (u, v) in the graph defined by adj + (u, v)
    # We assume (u, v) is present.
    # We need to find 6 other vertices forming the rest of the cube.
    
    # Slots:
    # 0: u (000)
    # 1: v (001)
    # 2: s1 (010)
    # 3: s2 (011)
    # 4: s3 (100)
    # 5: s4 (101)
    # 6: s5 (110)
    # 7: s6 (111)
    
    # Edges to check (excluding u-v):
    # (u, s1), (u, s3)
    # (v, s2), (v, s4)
    # (s1, s2), (s1, s5)
    # (s3, s4), (s3, s5)
    # (s2, s6), (s4, s6)
    # (s5, s6)
    
    # Candidates for s1, s3: neighbors of u (excluding v)
    # Candidates for s2, s4: neighbors of v (excluding u)
    
    # Get neighbors
    n_u = [x for x in range(16) if x != v and (adj[u] >> x) & 1]
    n_v = [x for x in range(16) if x != u and (adj[v] >> x) & 1]
    
    # We need s1, s3 from n_u, distinct.
    # We need s2, s4 from n_v, distinct.
    # s5, s6 from remaining vertices.
    
    # To speed up, we can iterate s1, s3 from n_u
    # and s2, s4 from n_v
    # Then check if there exist s5, s6.
    
    for s1 in n_u:
        for s3 in n_u:
            if s1 == s3: continue
            
            # Check edges (u, s1) and (u, s3) are already satisfied by selection
            # Now check (s1, s2) and (s3, s4) later?
            # Let's pick s2, s4 from n_v
            
            for s2 in n_v:
                for s4 in n_v:
                    if s2 == s4: continue
                    if s2 == s1 or s2 == s3: continue
                    if s4 == s1 or s4 == s3: continue
                    
                    # Check edges (s1, s2) and (s3, s4)
                    if not ((adj[s1] >> s2) & 1): continue
                    if not ((adj[s3] >> s4) & 1): continue
                    
                    # Now we need s5, s6
                    # s5 must be connected to s1, s3, s6
                    # s6 must be connected to s2, s4, s5
                    
                    # Candidates for s5: neighbors of s1 and s3, not in {u, v, s1, s2, s3, s4}
                    # Candidates for s6: neighbors of s2 and s4, not in {u, v, s1, s2, s3, s4}
                    
                    used = {u, v, s1, s2, s3, s4}
                    cand_s5 = [x for x in range(16) if x not in used and (adj[s1] >> x) & 1 and (adj[s3] >> x) & 1]
                    cand_s6 = [x for x in range(16) if x not in used and (adj[s2] >> x) & 1 and (adj[s4] >> x) & 1]
                    
                    for s5 in cand_s5:
                        for s6 in cand_s6:
                            if s5 == s6: continue
                            # Check (s1, s5), (s3, s5) already checked for s5 in cand_s5
                            # Check (s2, s6), (s4, s6) already checked for s6 in cand_s6
                            # Check (s5, s6)
                            if (adj[s5] >> s6) & 1:
                                return True
    return False
