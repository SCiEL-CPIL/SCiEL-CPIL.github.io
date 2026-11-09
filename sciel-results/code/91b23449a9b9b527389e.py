import time
import math
import sys

def construct():
    start_time = time.time()
    DEADLINE = start_time + 85
    
    LIMIT = 512
    N = LIMIT
    
    # Precompute compatibility
    compat = [0] * (N + 1)
    neighbors = [[] for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            g = math.gcd(i, j)
            l = (i // g) * j
            if l <= LIMIT:
                compat[i] |= (1 << (j - 1))
                compat[j] |= (1 << (i - 1))
                if i != j:
                    neighbors[i].append(j)
                    neighbors[j].append(i)
    
    # Precompute neighbor sets for O(1) lookup
    neighbor_sets = [set(neighbors[i]) for i in range(N + 1)]
    degrees = [len(neighbors[i]) for i in range(N + 1)]
    
    # Greedy initial solution
    def greedy_clique():
        S = []
        order = sorted(range(1, N + 1), key=lambda x: -degrees[x])
        for v in order:
            ok = True
            for c in S:
                if v not in neighbor_sets[c]:
                    ok = False
                    break
            if ok:
                S.append(v)
        return S
    
    best_clique = greedy_clique()
    best_size = len(best_clique)
    
    sys.setrecursionlimit(10000)
    
    def expand(P, C, depth):
        nonlocal best_clique, best_size
        
        if time.time() > DEADLINE:
            return
        
        if not P:
            if depth > best_size:
                best_clique = C[:]
                best_size = depth
            return
        
        if depth + len(P) <= best_size:
            return
        
        # Greedy coloring using neighbor sets for fast conflict checks
        P_set = set(P)
        P_sorted = sorted(P, key=lambda v: -len(neighbor_sets[v] & P_set))
        
        color_of = {}
        color_list = []
        
        for v in P_sorted:
            # Find smallest color not used by any neighbor of v in already-colored P
            used_colors = set()
            for u in P_sorted:
                if u == v:
                    continue
                if u in color_of and u in neighbor_sets[v]:
                    used_colors.add(color_of[u])
            
            c = 0
            while c in used_colors:
                c += 1
            color_of[v] = c
            if c >= len(color_list):
                color_list.append([])
            color_list[c].append(v)
        
        max_color = len(color_list)
        
        if depth + max_color <= best_size:
            return
        
        # Branch on vertices in reverse color order
        for k in range(max_color - 1, -1, -1):
            for v in color_list[k]:
                if v not in P_set:
                    continue
                
                C.append(v)
                new_P = [u for u in P_set if u != v and u in neighbor_sets[v]]
                expand(new_P, C, depth + 1)
                C.pop()
                
                P_set.remove(v)
                
                if time.time() > DEADLINE:
                    return
    
    initial_P = sorted(range(1, N + 1), key=lambda x: -degrees[x])
    expand(initial_P, [], 0)
    
    if not best_clique:
        return [1]
    
    return best_clique

if __name__ == "__main__":
    result = construct()
    print(f"Found clique of size {len(result)}")
    print(result)
