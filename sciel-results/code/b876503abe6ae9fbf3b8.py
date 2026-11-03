import numpy as np
import time
from itertools import combinations, product
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    D = 11
    indices = list(range(D))
    
    candidates = []
    types = [] # 2 for weight 2, 3 for weight 3
    
    # Weight 2
    for support in combinations(indices, 2):
        for signs in product([1, -1], repeat=2):
            v = np.zeros(D, dtype=np.int8)
            for idx, s in zip(support, signs):
                v[idx] = s
            candidates.append(v)
            types.append(2)
            
    # Weight 3
    for support in combinations(indices, 3):
        for signs in product([1, -1], repeat=3):
            v = np.zeros(D, dtype=np.int8)
            for idx, s in zip(support, signs):
                v[idx] = s
            candidates.append(v)
            types.append(3)
            
    candidates = np.array(candidates, dtype=np.int8)
    N = len(candidates)
    
    # Compute dot products
    dots = candidates @ candidates.T
    
    # Determine compatibility
    # Thresholds:
    # (2,2): dot <= 0
    # (2,3): dot <= 1
    # (3,3): dot <= 1
    
    # Create a mask for types
    type_arr = np.array(types)
    
    # Initialize compatibility matrix with False
    compat = np.zeros((N, N), dtype=bool)
    
    # We can vectorize the thresholds
    # For all pairs, we need to know the types.
    # This is a bit tricky to vectorize fully without broadcasting huge arrays, 
    # but N=1540 is small enough for a loop or block operations.
    
    # Let's just use a loop for clarity and speed is likely fine for 1540x1540
    # Or better, create type matrices.
    # T_i, T_j
    
    # Actually, we can just compute the threshold matrix directly.
    # Threshold[i,j] = 0 if types[i]==2 and types[j]==2 else 1
    
    # But we can just check:
    # If min(type[i], type[j]) == 2 and max(type[i], type[j]) == 2 -> threshold 0
    # Else threshold 1
    
    # Let's build the compatibility matrix
    # compat[i,j] = dots[i,j] <= threshold[i,j]
    
    # To avoid O(N^2) python loop, we can use numpy broadcasting on types
    # But N is small. Let's try a vectorized approach.
    
    # Create a matrix of types for rows and cols
    # T_row[i, j] = type[i]
    # T_col[i, j] = type[j]
    # This is 1540x1540 = 2.3M elements. Manageable.
    
    # Actually, simpler:
    # All pairs have threshold 1, EXCEPT (2,2) pairs which have threshold 0.
    # So:
    # compat = (dots <= 1)
    # Then, for all (2,2) pairs, we need dots <= 0.
    # So we find (2,2) pairs and update them.
    
    compat = (dots <= 1)
    
    # Find indices of weight 2 vectors
    w2_indices = np.where(np.array(types) == 2)[0]
    
    # For these indices, check dots <= 0
    # Submatrix of dots for w2 x w2
    if len(w2_indices) > 0:
        sub_dots = dots[np.ix_(w2_indices, w2_indices)]
        sub_compat = (sub_dots <= 0)
        compat[np.ix_(w2_indices, w2_indices)] = sub_compat
        
    # Ensure diagonal is False (self incompatibility? No, we want distinct vectors)
    # Max Clique finds distinct vertices.
    np.fill_diagonal(compat, False)
    
    # Build neighbor lists
    neighbors = []
    for i in range(N):
        neighbors.append(np.where(compat[i])[0])
        
    # Tomita's Max Clique Algorithm
    best_clique = []
    
    # Order vertices by degree (descending) to help pruning?
    # Tomita usually sorts by color.
    
    def expand(C, P):
        nonlocal best_clique
        if time.time() > deadline:
            return
            
        if len(P) == 0:
            if len(C) > len(best_clique):
                best_clique = C.copy()
            return
            
        # Sort P by degree in the subgraph induced by P? 
        # Or just use the global order.
        # Tomita: sort by color.
        
        # Greedy coloring for upper bound
        # Color the vertices in P
        # We want to assign colors such that adjacent vertices have different colors.
        # The max color is an upper bound on the clique size in P.
        
        # Simple greedy coloring
        colors = {}
        color_class = []
        # Order P by degree in P (descending) for better coloring
        # Calculate degrees in P
        P_set = set(P)
        degs = {v: len(set(neighbors[v]) & P_set) for v in P}
        P_sorted = sorted(P, key=lambda x: degs[x], reverse=True)
        
        max_color = 0
        color_map = {}
        
        for v in P_sorted:
            # Find smallest color not used by neighbors in P_sorted (already colored)
            used_colors = set()
            for u in neighbors[v]:
                if u in color_map:
                    used_colors.add(color_map[u])
            
            # Assign smallest available color
            c = 0
            while c in used_colors:
                c += 1
            color_map[v] = c
            if c > max_color:
                max_color = c
                
        # Upper bound is max_color
        # We iterate through P in reverse order of color (highest color first? No, usually lowest color first in some variants, but we want to prune early)
        # Standard Tomita: process in order of increasing color? 
        # Actually, the bound is: if len(C) + color(v) <= best_size, we can skip v and all with lower color?
        # No, the bound is global for the set P.
        # We sort P by color (descending).
        # If len(C) + color[P[i]] <= best_size, we can break?
        # The coloring gives an upper bound on the clique size in the subgraph of P[i:] (if sorted by color).
        
        # Let's sort P by color descending
        P_by_color = sorted(P, key=lambda x: color_map[x], reverse=True)
        
        for v in P_by_color:
            if time.time() > deadline:
                return
                
            # Upper bound check
            if len(C) + color_map[v] <= len(best_clique):
                break
                
            # Recurse
            new_P = [u for u in P_by_color if u in neighbors[v] and u != v]
            # Note: new_P should be the neighbors of v in P, sorted appropriately.
            # Since P_by_color is sorted, we can just filter.
            # But we need to ensure we only consider vertices that come AFTER v in the original ordering?
            # In Tomita, we usually partition P into N(v) and P\N(v).
            # We recurse on N(v) intersect P.
            
            new_C = C + [v]
            expand(new_C, new_P)
            
            # Remove v from P for the next iteration? 
            # The loop structure in Tomita usually handles this by iterating through a list and removing elements.
            # Here, P_by_color is static for this call. 
            # We must ensure that in the next iteration of the loop, we don't consider v again?
            # No, the loop iterates through candidates for the NEXT vertex in the clique.
            # Once we pick v, we remove it from consideration for the current level?
            # Actually, the standard implementation iterates through P, picks v, recurses, then removes v from P.
            
    # Initial call
    # P = all vertices
    # C = []
    expand([], list(range(N)))
    
    if not best_clique:
        # Fallback to seed-like solution if algo failed or timed out
        # Return a small valid set
        vectors = []
        for i in range(D):
            for j in range(i + 1, D):
                v = np.zeros(D, dtype=np.int8)
                v[i] = 1
                v[j] = 1
                vectors.append(v)
        return np.array(vectors, dtype=np.int8)
        
    return candidates[best_clique]
