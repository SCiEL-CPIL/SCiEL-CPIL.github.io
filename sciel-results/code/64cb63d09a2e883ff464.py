import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s limit to be safe
    
    N = 256
    numbers = list(range(1, N + 1))
    
    # Precompute conflicts using the efficient divisor method
    # conflicts[i] is a bitset of indices j such that numbers[i] and numbers[j] conflict
    # Index i corresponds to number i+1
    conflicts = [0] * N
    
    for a in range(1, N + 1):
        a_sq = a * a
        limit = int(math.isqrt(a_sq))
        for k in range(1, limit + 1):
            if a_sq % k == 0:
                d1 = k
                d2 = a_sq // k
                for d in (d1, d2):
                    if a < d <= a + N:
                        b = d - a
                        if 1 <= b <= N and b != a:
                            # Mark conflict between a-1 and b-1
                            conflicts[a - 1] |= (1 << (b - 1))
                            conflicts[b - 1] |= (1 << (a - 1))
    
    # Precompute vertex bits
    vertex_bit = [1 << i for i in range(N)]
    all_vertices = (1 << N) - 1
    
    # Compatibility: no conflict and not self
    # compat[i] is a bitset of j such that i != j and no conflict
    compat = [0] * N
    for i in range(N):
        # All bits set except i and its conflicts
        compat[i] = all_vertices ^ vertex_bit[i] ^ conflicts[i]
    
    # 1. Greedy Initialization
    # We want a large Independent Set.
    # Heuristic: Pick vertex with min degree in the current conflict subgraph?
    # Or max degree in compatibility?
    # Let's try greedy MIS (Max Independent Set) by picking min degree in conflict graph.
    
    def greedy_mis():
        C = all_vertices
        S = []
        while C:
            best_v = -1
            min_deg = float('inf')
            temp = C
            while temp:
                lsb = temp & -temp
                v = lsb.bit_length() - 1
                # Degree of v in conflict graph induced by C
                deg = (conflicts[v] & C).bit_count()
                if deg < min_deg:
                    min_deg = deg
                    best_v = v
                temp ^= lsb
            
            if best_v == -1:
                break
                
            S.append(best_v)
            # Remove best_v and its neighbors from C
            C &= ~(conflicts[best_v] | vertex_bit[best_v])
            
        return S

    best_set = greedy_mis()
    best_size = len(best_set)
    
    # 2. Branch and Bound for Maximum Independent Set
    # We will use a recursive search.
    # State: (C, current_size, current_set)
    # C: bitset of available vertices
    # We branch on a vertex v in C.
    # Include v: new C = C & ~conflicts[v] & ~vertex_bit[v]
    # Exclude v: new C = C & ~vertex_bit[v]
    
    # To make it efficient, we need good pruning.
    # Bound: current_size + upper_bound(C) <= best_size -> prune
    # Upper bound on independent set size in C:
    # Simple bound: |C|
    # Better bound: Greedy coloring of the conflict graph induced by C?
    # Or just |C| is often enough if we order vertices well.
    
    # Let's implement a simple B&B with |C| bound first, and see if it's fast enough.
    # If not, we can add a greedy coloring bound.
    
    # To improve pruning, we can order the branching vertices.
    # Pick a vertex v in C.
    # Strategy: Pick v with max degree in conflict graph induced by C?
    # This tends to reduce the "include" branch significantly.
    
    def expand(C, current_size, current_set):
        nonlocal best_size, best_set
        
        if time.time() > deadline:
            return False
            
        if C == 0:
            if current_size > best_size:
                best_size = current_size
                best_set = current_set[:]
            return True
            
        # Pruning:
        if current_size + C.bit_count() <= best_size:
            return True
            
        # Select vertex to branch on
        # Heuristic: Max degree in conflict graph induced by C
        best_v = -1
        max_deg = -1
        temp = C
        while temp:
            lsb = temp & -temp
            v = lsb.bit_length() - 1
            deg = (conflicts[v] & C).bit_count()
            if deg > max_deg:
                max_deg = deg
                best_v = v
            temp ^= lsb
            
        if best_v == -1:
            return True
            
        v_bit = vertex_bit[best_v]
        
        # Branch 1: Include best_v
        # New C: remove best_v and its neighbors
        new_C_include = C & ~(conflicts[best_v] | v_bit)
        current_set.append(best_v)
        
        # Check time
        if time.time() > deadline:
            current_set.pop()
            return False
            
        expand(new_C_include, current_size + 1, current_set)
        current_set.pop()
        
        if time.time() > deadline:
            return False
            
        # Branch 2: Exclude best_v
        new_C_exclude = C & ~v_bit
        expand(new_C_exclude, current_size, current_set)
        
        return True

    # Start B&B
    # We start with the greedy solution as the initial lower bound.
    # We can also try to improve the initial solution with local search?
    # For now, just run B&B.
    
    # To avoid deep recursion stack issues, we might need to increase recursion limit
    # or implement iteratively. N=256 is manageable for recursion in Python if optimized,
    # but let's be careful.
    import sys
    sys.setrecursionlimit(10000)
    
    # We pass the initial greedy set as the starting point?
    # No, B&B starts from scratch usually, but uses the best_size found by greedy for pruning.
    # So we just call expand(all_vertices, 0, []).
    
    # However, to make it faster, we can order the vertices in the initial call?
    # The B&B handles ordering dynamically.
    
    # One optimization: If the greedy solution is already very good, B&B prunes heavily.
    
    # Let's run B&B
    expand(all_vertices, 0, [])
    
    # If B&B didn't finish or didn't improve, we still have the greedy set.
    # We can also try a few random restarts of greedy if time permits?
    # Or just return the best found.
    
    # Convert indices to values
    result = [numbers[i] for i in best_set]
    
    # Verify
    for i in range(len(result)):
        for j in range(i+1, len(result)):
            a = result[i]
            b = result[j]
            if (a * b) % (a + b) == 0:
                # Fallback to odd numbers if validation fails (should not happen)
                return list(range(1, 257, 2))
        
    return result
