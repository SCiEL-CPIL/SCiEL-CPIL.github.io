import time
import math
import sys

def construct():
    start_time = time.time()
    DEADLINE = start_time + 85
    
    LIMIT = 512
    N = LIMIT
    
    # 1. Build Compatibility Graph
    # compat[i] is a bitmask of nodes j (1..N) such that lcm(i, j) <= LIMIT
    compat = [0] * (N + 1)
    
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            g = math.gcd(i, j)
            l = (i // g) * j
            if l <= LIMIT:
                compat[i] |= (1 << (j - 1))
                compat[j] |= (1 << (i - 1))
    
    # Precompute popcounts for speed? No, int.bit_count() is fast.
    
    # 2. Tomita's Algorithm with Bitmasks
    best_clique = []
    
    # Greedy Lower Bound: Pick nodes with high degree
    def greedy_clique():
        # Sort nodes by degree descending
        nodes = list(range(1, N + 1))
        nodes.sort(key=lambda x: -compat[x].bit_count())
        
        clique = []
        for node in nodes:
            if time.time() > DEADLINE:
                break
            ok = True
            for c in clique:
                if not (compat[node] & (1 << (c - 1))):
                    ok = False
                    break
            if ok:
                clique.append(node)
        return clique

    init_clique = greedy_clique()
    if init_clique:
        best_clique = init_clique
        
    # Sort all vertices by degree descending for initial ordering
    # This helps in the first few levels of recursion
    all_vertices = list(range(1, N + 1))
    all_vertices.sort(key=lambda x: -compat[x].bit_count())
    
    sys.setrecursionlimit(10000)
    
    def expand(P_mask, C, depth):
        nonlocal best_clique
        
        if time.time() > DEADLINE:
            return
            
        if P_mask == 0:
            if depth > len(best_clique):
                best_clique = C[:]
            return
        
        if depth + P_mask.bit_count() <= len(best_clique):
            return
            
        # Extract vertices from P_mask
        P = []
        temp = P_mask
        while temp:
            lsb = temp & -temp
            idx = lsb.bit_length() - 1
            P.append(idx + 1)
            temp ^= lsb
            
        # Greedy Coloring
        # We sort P by degree within P? Or static degree?
        # Static degree is precomputed, so it's faster.
        P.sort(key=lambda v: -compat[v].bit_count())
        
        color_list = []
        vertex_color = {}
        
        for v in P:
            assigned = False
            for k in range(len(color_list)):
                # Check if v is compatible with any node in color k
                # If yes, conflict.
                # We can optimize this by storing the mask of nodes in each color
                # but for small N=512, list check is okay.
                conflict = False
                for u in color_list[k]:
                    if compat[v] & (1 << (u - 1)):
                        conflict = True
                        break
                if not conflict:
                    color_list[k].append(v)
                    vertex_color[v] = k
                    assigned = True
                    break
            
            if not assigned:
                color_list.append([v])
                vertex_color[v] = len(color_list) - 1
        
        max_color = len(color_list)
        
        if depth + max_color <= len(best_clique):
            return
        
        # Branching
        # Iterate colors in reverse (highest color first)
        # Vertices in higher colors are "more constrained" or "later" in the greedy order
        # Tomita suggests iterating from highest color to lowest.
        
        for k in range(max_color - 1, -1, -1):
            # Iterate vertices in this color
            # We need to be careful with removal from P_mask
            for v in color_list[k]:
                if not (P_mask & (1 << (v - 1))):
                    continue
                
                # Include v
                C.append(v)
                
                # New P: neighbors of v in P_mask \ {v}
                new_P_mask = P_mask & compat[v] & ~(1 << (v - 1))
                
                expand(new_P_mask, C, depth + 1)
                
                C.pop()
                
                # Exclude v
                P_mask &= ~(1 << (v - 1))
                
                if time.time() > DEADLINE:
                    return

    initial_P_mask = (1 << N) - 1
    expand(initial_P_mask, [], 0)
    
    if not best_clique:
        return [1]
        
    return best_clique

if __name__ == "__main__":
    result = construct()
    print(f"Found clique of size {len(result)}")
    print(result)
