import time
import math
import sys
from collections import defaultdict

def construct():
    start_time = time.time()
    DEADLINE = start_time + 85  # 85s safety margin for 90s limit
    
    LIMIT = 512
    N = LIMIT
    
    # 1. Build Compatibility Graph
    # compat[i] is a bitmask of nodes j (1..N) such that lcm(i, j) <= LIMIT
    # We use 0-indexed bits for nodes 1..N, so bit j-1 corresponds to node j.
    compat = [0] * (N + 1)
    
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            g = math.gcd(i, j)
            l = (i // g) * j
            if l <= LIMIT:
                compat[i] |= (1 << (j - 1))
                compat[j] |= (1 << (i - 1))
    
    # Precompute degrees for fast sorting
    degrees = [0] * (N + 1)
    for i in range(1, N + 1):
        degrees[i] = bin(compat[i]).count('1')
    
    best_clique = []
    
    # Greedy Lower Bound with multiple strategies
    def greedy_clique(strategy=0):
        if strategy == 0:
            # Sort by degree descending
            nodes = sorted(range(1, N+1), key=lambda x: -degrees[x])
        elif strategy == 1:
            # Sort by value ascending (smaller numbers tend to have lower LCMs)
            nodes = list(range(1, N+1))
        else:
            # Sort by degree ascending (try low-degree nodes first)
            nodes = sorted(range(1, N+1), key=lambda x: degrees[x])
        
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

    # Try multiple greedy strategies to get a good initial lower bound
    for strat in range(3):
        init_clique = greedy_clique(strat)
        if len(init_clique) > len(best_clique):
            best_clique = init_clique
    
    # Tomita's Algorithm with optimized coloring
    sys.setrecursionlimit(10000)
    
    def expand(P, C, depth):
        nonlocal best_clique
        
        if time.time() > DEADLINE:
            return
            
        if not P:
            if depth > len(best_clique):
                best_clique = C[:]
            return
        
        if depth + len(P) <= len(best_clique):
            return
            
        # Fast greedy coloring using bitmask operations
        # Sort P by static degree descending
        P_sorted = sorted(P, key=lambda v: -degrees[v])
        
        # Color assignment: for each vertex, find the smallest color k
        # such that no neighbor of v in already-colored vertices has color k.
        # We maintain color_masks[k] as a bitmask of vertices with color k.
        
        color_masks = []  # color_masks[k] = bitmask of vertices assigned color k
        vertex_color = {}
        
        for v in P_sorted:
            mask_v = compat[v]
            assigned = False
            for k in range(len(color_masks)):
                # Check if v has any neighbor in color_masks[k]
                # If mask_v & color_masks[k] != 0, then v conflicts with some vertex of color k
                if mask_v & color_masks[k]:
                    continue
                else:
                    color_masks[k] |= (1 << (v - 1))
                    vertex_color[v] = k
                    assigned = True
                    break
            
            if not assigned:
                color_masks.append(1 << (v - 1))
                vertex_color[v] = len(color_masks) - 1
        
        max_color = len(color_masks)
        
        if depth + max_color <= len(best_clique):
            return
        
        # Branching: process vertices in reverse color order
        # For each color class from max_color-1 down to 0,
        # iterate through vertices in that class.
        
        P_set = set(P)
        
        for k in range(max_color - 1, -1, -1):
            # Get vertices with color k from P_set
            # We can extract them from color_masks[k]
            mask_k = color_masks[k]
            # Iterate through bits in mask_k that are also in P_set
            remaining = mask_k
            while remaining:
                # Get lowest set bit
                v_bit = remaining & (-remaining)
                v = v_bit.bit_length()  # This gives position, but we need 1-indexed
                # Actually, bit_length() gives index+1 for the highest set bit
                # For a single bit at position p (0-indexed), bit_length() = p+1
                # So v = v_bit.bit_length() gives us the 1-indexed vertex number
                # But we need to check if v is still in P_set
                if v in P_set:
                    # Bound check: remaining vertices in this and higher color classes
                    # Simple bound: if depth + (max_color - k) <= best, we could prune
                    # But let's use the tighter: count of vertices in P_set with color >= k
                    # For now, use the simple bound
                    if depth + max_color <= len(best_clique):
                        return
                    
                    # Include v
                    C.append(v)
                    
                    # New P: neighbors of v in P_set \ {v}
                    mask_v = compat[v]
                    new_P = []
                    for u in P_set:
                        if u != v and (mask_v & (1 << (u - 1))):
                            new_P.append(u)
                    
                    expand(new_P, C, depth + 1)
                    
                    C.pop()
                    
                    # Exclude v
                    P_set.discard(v)
                    
                    if time.time() > DEADLINE:
                        return
                
                remaining &= remaining - 1  # Clear lowest set bit
        
        # Local search polish: try to add more vertices to best_clique
        if best_clique and time.time() < DEADLINE - 5:
            current = best_clique[:]
            improved = True
            while improved and time.time() < DEADLINE - 2:
                improved = False
                # Try adding each vertex not in current
                for v in range(1, N + 1):
                    if v in current:
                        continue
                    # Check if v is compatible with all in current
                    mask_v = compat[v]
                    ok = True
                    for c in current:
                        if not (mask_v & (1 << (c - 1))):
                            ok = False
                            break
                    if ok:
                        current.append(v)
                        improved = True
                        if len(current) > len(best_clique):
                            best_clique = current[:]
                        break  # Restart the while loop after adding one
    
    # Start the algorithm
    initial_P = list(range(1, N + 1))
    initial_P.sort(key=lambda v: -degrees[v])
    
    expand(initial_P, [], 0)
    
    if not best_clique:
        return [1]
        
    return best_clique

if __name__ == "__main__":
    result = construct()
    print(f"Found clique of size {len(result)}")
    print(result)
