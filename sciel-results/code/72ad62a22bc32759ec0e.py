import math
import random

def is_square(n):
    if n < 0:
        return False
    r = int(math.isqrt(n))
    return r * r == n

def construct():
    # 1. Partition vertices by mod 4
    R0 = [i for i in range(1, 257) if i % 4 == 0]
    R1 = [i for i in range(1, 257) if i % 4 == 1]
    R2 = [i for i in range(1, 257) if i % 4 == 2]
    R3 = [i for i in range(1, 257) if i % 4 == 3]
    
    # Helper: Check if x is safe for diagonal (2x not square)
    def is_diag_safe(x):
        return not is_square(2 * x)

    # Helper: Build adjacency for a list of vertices
    def build_adj(vertices):
        adj = {v: set() for v in vertices}
        v_list = list(vertices)
        for i in range(len(v_list)):
            u = v_list[i]
            for j in range(i + 1, len(v_list)):
                v = v_list[j]
                if is_square(u + v):
                    adj[u].add(v)
                    adj[v].add(u)
        return adj

    # Helper: Find Max Independent Set using Greedy + Local Search
    def find_mis(vertices, adj):
        if not vertices:
            return set()
        
        # 1. Greedy Min-Degree
        U = set(vertices)
        S = set()
        # Sort by degree in initial graph for tie-breaking? 
        # Greedy Min-Degree: pick node with min degree in remaining graph
        while U:
            best_v = None
            min_deg = float('inf')
            for v in U:
                deg = len(adj[v] & U)
                if deg < min_deg:
                    min_deg = deg
                    best_v = v
            S.add(best_v)
            U -= (adj[best_v] | {best_v})
        
        # 2. Local Search: Try to add any vertex not in S that doesn't conflict
        improved = True
        while improved:
            improved = False
            for v in list(vertices):
                if v in S:
                    continue
                # Check if v can be added
                conflict = False
                for u in S:
                    if u in adj[v]:
                        conflict = True
                        break
                if not conflict:
                    S.add(v)
                    improved = True
        
        # 3. Local Search: Try swaps
        # If a vertex v not in S has exactly 1 conflict u in S, 
        # and removing u allows v to be added, we can swap.
        # This doesn't increase size, but might allow future additions.
        # However, for simplicity and speed, we stick to Add-only if it's fast enough.
        # Given the small size, we can try a few random restarts of the greedy phase.
        
        best_S = S
        for _ in range(20):
            # Randomize tie-breaking in Greedy
            U = set(vertices)
            S_rand = set()
            while U:
                candidates = []
                min_deg = float('inf')
                for v in U:
                    deg = len(adj[v] & U)
                    if deg < min_deg:
                        min_deg = deg
                        candidates = [v]
                    elif deg == min_deg:
                        candidates.append(v)
                best_v = random.choice(candidates)
                S_rand.add(best_v)
                U -= (adj[best_v] | {best_v})
            
            # Add phase for S_rand
            improved = True
            while improved:
                improved = False
                for v in list(vertices):
                    if v in S_rand:
                        continue
                    conflict = False
                    for u in S_rand:
                        if u in adj[v]:
                            conflict = True
                            break
                    if not conflict:
                        S_rand.add(v)
                        improved = True
            
            if len(S_rand) > len(best_S):
                best_S = S_rand
        
        return best_S

    def build_candidate(safe_class, conflict_class):
        # Filter diagonal conflicts
        valid_safe = [x for x in safe_class if is_diag_safe(x)]
        valid_conflict = [x for x in conflict_class if is_diag_safe(x)]
        
        # Build adjacency for conflict class
        adj_conflict = build_adj(valid_conflict)
        
        # Find MIS for conflict class
        mis_conflict = find_mis(valid_conflict, adj_conflict)
        
        # Combine
        S = set(valid_safe)
        S.update(mis_conflict)
        
        # Global Local Search: Try to add any remaining vertex from 1..256
        all_vertices = list(range(1, 257))
        improved = True
        while improved:
            improved = False
            for x in all_vertices:
                if x in S:
                    continue
                if not is_diag_safe(x):
                    continue
                # Check conflicts with S
                conflict = False
                for y in S:
                    if is_square(x + y):
                        conflict = True
                        break
                if not conflict:
                    S.add(x)
                    improved = True
        
        return S

    cand1 = build_candidate(R1, R2)
    cand2 = build_candidate(R3, R0)
    
    # Also try a candidate based on R2 and R0 (compatible)
    # R2 and R0 have no cross edges.
    # But they don't have a "safe" class to anchor them.
    # We can just find MIS of R2_safe U R0_safe?
    # Since no edges between R2 and R0, MIS(R2 U R0) = MIS(R2) U MIS(R0).
    # But we might be able to add R1 or R3 if they don't conflict?
    # R1 conflicts with R0. R3 conflicts with R2.
    # So if we take MIS(R2) U MIS(R0), we can only add R1 nodes that don't conflict with R0, 
    # and R3 nodes that don't conflict with R2.
    
    valid_R0 = [x for x in R0 if is_diag_safe(x)]
    valid_R2 = [x for x in R2 if is_diag_safe(x)]
    valid_R1 = [x for x in R1 if is_diag_safe(x)]
    valid_R3 = [x for x in R3 if is_diag_safe(x)]
    
    adj_R0 = build_adj(valid_R0)
    adj_R2 = build_adj(valid_R2)
    
    mis_R0 = find_mis(valid_R0, adj_R0)
    mis_R2 = find_mis(valid_R2, adj_R2)
    
    S3 = set(mis_R0)
    S3.update(mis_R2)
    
    # Try to add R1 nodes compatible with S3 (specifically R0 part)
    # R1 conflicts with R0? Yes. R1 conflicts with R2? No.
    # So R1 nodes can be added if they don't conflict with mis_R0.
    # R3 nodes can be added if they don't conflict with mis_R2.
    
    # We need to be careful: R1 and R3 conflict with each other.
    # So we can't just add all compatible R1 and R3.
    
    # Greedy add R1 and R3 to S3
    improved = True
    while improved:
        improved = False
        for x in valid_R1 + valid_R3:
            if x in S3:
                continue
            conflict = False
            for y in S3:
                if is_square(x + y):
                    conflict = True
                    break
            if not conflict:
                S3.add(x)
                improved = True
                
    # Try to add any other node
    for x in range(1, 257):
        if x in S3:
            continue
        if not is_diag_safe(x):
            continue
        conflict = False
        for y in S3:
            if is_square(x + y):
                conflict = True
                break
        if not conflict:
            S3.add(x)
            
    candidates = [cand1, cand2, S3]
    best = max(candidates, key=len)
    return list(best)
