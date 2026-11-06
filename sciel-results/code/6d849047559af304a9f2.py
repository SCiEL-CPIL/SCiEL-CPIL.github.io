import time
import sys
import random

sys.setrecursionlimit(10000)

def construct():
    start_time = time.time()
    timeout = 55.0  # Safety margin for 60s limit
    
    n = 256
    
    # 1. Build Graph
    # Edge (a,b) if (a*b) % (a+b) == 0
    adj_mask = [0] * (n + 1)
    
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            if (a * b) % (a + b) == 0:
                adj_mask[a] |= (1 << (b - 1))
                adj_mask[b] |= (1 << (a - 1))
                
    # Complement graph for Max Clique
    full_mask = (1 << n) - 1
    comp_adj_mask = [0] * (n + 1)
    for v in range(1, n + 1):
        comp_adj_mask[v] = full_mask ^ adj_mask[v] ^ (1 << (v - 1))
        
    # Precompute popcount for speed? Python 3.10+ has int.bit_count()
    # Let's assume a modern Python environment.
    
    # 2. Initial greedy solution to get a good lower bound
    # Conflict degree in G (number of neighbors in the "bad" graph)
    conflict_deg = [0] * (n + 1)
    for v in range(1, n + 1):
        conflict_deg[v] = adj_mask[v].bit_count()
    
    # Greedy: sort by conflict degree ascending, add if valid
    # This tends to pick vertices that are "safe" (few conflicts) first
    init_set = []
    init_mask = 0
    for v in sorted(range(1, n + 1), key=lambda x: conflict_deg[x]):
        if adj_mask[v] & init_mask == 0:
            init_set.append(v)
            init_mask |= (1 << (v - 1))
            
    best_size = len(init_set)
    best_set = init_set[:]
    
    # Try a few random restarts of greedy to find a better initial solution
    for _ in range(20):
        random.shuffle(init_set)
        # Re-run greedy with shuffled order? No, just shuffle the candidate list.
        cand = list(range(1, n + 1))
        random.shuffle(cand)
        curr_set = []
        curr_mask = 0
        for v in cand:
            if adj_mask[v] & curr_mask == 0:
                curr_set.append(v)
                curr_mask |= (1 << (v - 1))
        if len(curr_set) > best_size:
            best_size = len(curr_set)
            best_set = curr_set[:]
            
    # Local search to improve initial solution (swap-based)
    # Try to swap one element out for two in, or one in for one out if it allows more later?
    # Simple local search: try to add any vertex.
    in_set = set(best_set)
    improved = True
    while improved:
        improved = False
        for v in range(1, n + 1):
            if v in in_set:
                continue
            if adj_mask[v] & best_mask == 0:
                best_set.append(v)
                in_set.add(v)
                best_mask |= (1 << (v - 1))
                best_size += 1
                improved = True
                
    # 3. Tomita's Branch and Bound for Maximum Clique on complement graph
    # Optimizations:
    # - Use bit_count() instead of bin().count('1')
    # - Efficient coloring
    # - Better vertex ordering (degree in induced subgraph)
    
    def expand(P_list, clique_size, clique):
        nonlocal best_size, best_set
        
        if time.time() - start_time > timeout:
            return
            
        if not P_list:
            if clique_size > best_size:
                best_size = clique_size
                best_set = clique[:]
            return
            
        # Greedy Coloring
        # P_list is a list of vertices.
        # We want to color them such that adjacent vertices in G' have different colors.
        # This gives an upper bound on the clique size.
        
        # To make coloring fast and tight, we can sort P_list by degree in G' descending?
        # The standard Tomita algorithm colors in the order the vertices are presented.
        # If we present them in a "good" order, the coloring is tighter.
        
        # Let's sort P_list by degree in the induced subgraph (G'[P]) descending.
        # Degree of v in G'[P] = popcount(comp_adj_mask[v] & P_mask)
        
        P_mask = 0
        for v in P_list:
            P_mask |= (1 << (v - 1))
            
        # Sort by degree descending
        # This is O(|P|^2) if we compute degree for each, but |P| <= 256.
        # Actually, computing degree is O(1) with bit_count.
        # Sorting is O(|P| log |P|).
        
        P_list.sort(key=lambda v: comp_adj_mask[v].bit_count() & P_mask.bit_count(), reverse=True)
        # Wait, comp_adj_mask[v] & P_mask gives the neighbors of v in P.
        # So degree is (comp_adj_mask[v] & P_mask).bit_count().
        
        P_list.sort(key=lambda v: (comp_adj_mask[v] & P_mask).bit_count(), reverse=True)
        
        color_of = [0] * (n + 1)
        colored_mask = 0
        max_color = 0
        v_colors = []
        
        for v in P_list:
            # Find colors of colored neighbors in G'
            neighbor_mask = comp_adj_mask[v] & colored_mask
            used_colors = set()
            temp = neighbor_mask
            while temp:
                lsb = temp & -temp
                u = lsb.bit_length()
                if color_of[u] > 0:
                    used_colors.add(color_of[u])
                temp ^= lsb
                
            c = 1
            while c in used_colors:
                c += 1
            color_of[v] = c
            if c > max_color:
                max_color = c
            v_colors.append((v, c))
            colored_mask |= (1 << (v - 1))
            
        # Sort by color ascending for branching
        v_colors.sort(key=lambda x: x[1])
        
        # Precompute suffix masks
        m = len(v_colors)
        suffix_masks = [0] * (m + 1)
        for i in range(m - 1, -1, -1):
            v_i = v_colors[i][0]
            suffix_masks[i] = suffix_masks[i+1] | (1 << (v_i - 1))
        
        # Branch
        for idx in range(m):
            if time.time() - start_time > timeout:
                return
            
            v, c = v_colors[idx]
            
            # Bound check
            # Max clique size from remaining vertices is at most max_color - c + 1
            # So total size <= clique_size + 1 + (max_color - c)
            if clique_size + 1 + (max_color - c) <= best_size:
                break
            
            # Branch Include
            # New P is neighbors of v in G' that are in suffix_masks[idx+1]
            new_P_mask = comp_adj_mask[v] & suffix_masks[idx+1]
            
            if new_P_mask:
                # Extract vertices
                new_P_list = []
                temp = new_P_mask
                while temp:
                    lsb = temp & -temp
                    u = lsb.bit_length()
                    new_P_list.append(u)
                    temp ^= lsb
                    
                expand(new_P_list, clique_size + 1, clique + [v])
            else:
                # Leaf
                if clique_size + 1 > best_size:
                    best_size = clique_size + 1
                    best_set = clique + [v]
                
    # Initial P: all vertices
    # We can sort them by degree in G' descending initially
    initial_P_list = list(range(1, n + 1))
    # initial_P_list.sort(key=lambda v: comp_adj_mask[v].bit_count(), reverse=True)
    # The expand function sorts P_list at each level, so initial order doesn't matter much.
    
    expand(initial_P_list, 0, [])
    
    if not best_set:
        best_set = [1]
        
    return list(best_set)

if __name__ == "__main__":
    res = construct()
    print(len(res))
