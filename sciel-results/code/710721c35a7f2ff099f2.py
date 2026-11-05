import math

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
    
    # Strategy: Combine R1 (internally safe) with a subset of R2 (no cross conflicts with R1)
    # Also try R3 (internally safe) with a subset of R0 (no cross conflicts with R3)
    # We will evaluate both and pick the best.
    
    def build_candidate(safe_class, conflict_class):
        # safe_class has no internal conflicts (mod 4 sum != 0,1)
        # conflict_class has internal conflicts
        # Cross conflicts between safe and conflict:
        # R1 (1) + R2 (2) = 3 mod 4 (safe)
        # R3 (3) + R0 (0) = 3 mod 4 (safe)
        
        S = set(safe_class)
        
        # Filter diagonal conflicts in conflict_class
        # 2*x must not be a square
        valid_conflict = [x for x in conflict_class if not is_square(2 * x)]
        
        # Greedy add from valid_conflict to S
        # Since no cross conflicts, we just need MIS of valid_conflict
        # We use a greedy min-degree removal on the subgraph induced by valid_conflict
        
        # Build adjacency for valid_conflict
        adj = {v: set() for v in valid_conflict}
        v_list = valid_conflict
        for i in range(len(v_list)):
            u = v_list[i]
            for j in range(i + 1, len(v_list)):
                v = v_list[j]
                if is_square(u + v):
                    adj[u].add(v)
                    adj[v].add(u)
        
        # Greedy Min-Degree Add for the conflict_class part
        U = set(valid_conflict)
        selected_conflict = []
        while U:
            best_v = None
            min_deg = float('inf')
            for v in U:
                deg = len(adj[v] & U)
                if deg < min_deg:
                    min_deg = deg
                    best_v = v
            selected_conflict.append(best_v)
            U -= (adj[best_v] | {best_v})
        
        S.update(selected_conflict)
        
        # Local Search: Try to add any remaining vertices from R0, R2, R3, R1 that are not in S
        # and don't conflict with S
        remaining = [x for x in range(1, 257) if x not in S]
        added = []
        for x in remaining:
            if is_square(2 * x):
                continue
            # Check conflicts with current S
            conflict = False
            for y in S:
                if is_square(x + y):
                    conflict = True
                    break
            if not conflict:
                S.add(x)
                added.append(x)
                
        return list(S)
        
    cand1 = build_candidate(R1, R2)
    cand2 = build_candidate(R3, R0)
    
    # Also try a mixed approach if one is significantly better?
    # For now, pick the largest
    
    if len(cand1) >= len(cand2):
        return cand1
    else:
        return cand2
