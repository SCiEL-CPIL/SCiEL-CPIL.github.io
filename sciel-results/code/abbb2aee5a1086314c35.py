import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s limit
    
    n = 16
    # Edge list in lexicographic order
    edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    num_edges = len(edges)
    
    # Build edge index lookup
    edge_idx = {}
    for idx, (i, j) in enumerate(edges):
        edge_idx[(i, j)] = idx
    
    # Precompute all K4s and their edge indices
    k4s = []
    for a in range(n-3):
        for b in range(a+1, n-2):
            for c in range(b+1, n-1):
                for d in range(c+1, n):
                    e_idx = [
                        edge_idx[(a,b)],
                        edge_idx[(a,c)],
                        edge_idx[(a,d)],
                        edge_idx[(b,c)],
                        edge_idx[(b,d)],
                        edge_idx[(c,d)]
                    ]
                    k4s.append(e_idx)
    
    num_k4s = len(k4s)
    
    # Precompute for each edge, which K4s it belongs to
    edge_to_k4s = [[] for _ in range(num_edges)]
    for k4_idx, e_idx in enumerate(k4s):
        for e in e_idx:
            edge_to_k4s[e].append(k4_idx)
    
    # Helper: count number of K4s with < 5 distinct colors
    def count_bad_k4s(col):
        bad = 0
        for e_idx in k4s:
            c0, c1, c2, c3, c4, c5 = col[e_idx[0]], col[e_idx[1]], col[e_idx[2]], col[e_idx[3]], col[e_idx[4]], col[e_idx[5]]
            if len({c0, c1, c2, c3, c4, c5}) < 5:
                bad += 1
        return bad
    
    # Helper: check if a K4 has at least 5 distinct colors
    def k4_is_good(col, e_idx):
        c0, c1, c2, c3, c4, c5 = col[e_idx[0]], col[e_idx[1]], col[e_idx[2]], col[e_idx[3]], col[e_idx[4]], col[e_idx[5]]
        return len({c0, c1, c2, c3, c4, c5}) >= 5
    
    # Main search: try to find a valid coloring with fewer colors
    random.seed(42)
    best_colors = None
    best_max = 17
    
    # Phase 1: Try structured initializations and local search for each target
    for target_max in range(15, 12, -1):
        if time.time() > deadline:
            break
        
        # Try multiple structured initializations
        for init_type in range(5):
            if time.time() > deadline:
                break
            
            if init_type == 0:
                # (i + j) % (target_max + 1)
                col = [(i + j) % (target_max + 1) for i, j in edges]
            elif init_type == 1:
                # (i * 3 + j * 7) % (target_max + 1)
                col = [(i * 3 + j * 7) % (target_max + 1) for i, j in edges]
            elif init_type == 2:
                # (i * 5 + j * 11) % (target_max + 1)
                col = [(i * 5 + j * 11) % (target_max + 1) for i, j in edges]
            elif init_type == 3:
                # Random
                col = [random.randint(0, target_max) for _ in range(num_edges)]
            else:
                # (i * 2 + j) % (target_max + 1)
                col = [(i * 2 + j) % (target_max + 1) for i, j in edges]
            
            # Local search to fix violations
            improved = True
            iterations = 0
            while improved and time.time() < deadline:
                improved = False
                iterations += 1
                
                # Find a bad K4
                bad_k4_idx = -1
                for k4_idx in range(num_k4s):
                    e_idx = k4s[k4_idx]
                    if not k4_is_good(col, e_idx):
                        bad_k4_idx = k4_idx
                        break
                
                if bad_k4_idx == -1:
                    break  # No bad K4s, we have a valid coloring!
                
                # Pick a random edge from this bad K4 and try to change its color
                e_idx = k4s[bad_k4_idx]
                random.shuffle(e_idx)
                
                for edge_idx in e_idx:
                    old_color = col[edge_idx]
                    # Try a new color
                    new_color = random.randint(0, target_max)
                    if new_color == old_color:
                        continue
                    
                    col[edge_idx] = new_color
                    
                    # Check if this fixes the bad K4
                    if k4_is_good(col, e_idx):
                        # Check if we created new bad K4s
                        created_bad = False
                        for check_k4 in edge_to_k4s[edge_idx]:
                            ce = k4s[check_k4]
                            if not k4_is_good(col, ce):
                                created_bad = True
                                break
                        
                        if not created_bad:
                            improved = True
                            break
                        else:
                            col[edge_idx] = old_color
                
                if time.time() > deadline:
                    break
            
            # Check if we found a valid coloring
            if count_bad_k4s(col) == 0:
                best_colors = col[:]
                best_max = target_max
                break  # Found valid coloring with this target_max, try lower
    
    # If no valid coloring found, use the initial 17-color solution
    if best_colors is None:
        best_colors = [(i + j) % 17 for i, j in edges]
        best_max = 17
    
    return best_colors
