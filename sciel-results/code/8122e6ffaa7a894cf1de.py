import numpy as np
import itertools
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    D = 11

    # 1. Generate all supports (combinations of 4 indices)
    supports = list(itertools.combinations(range(D), 4))
    N_S = len(supports)  # 330

    # Bitmasks for fast intersection
    masks = []
    for s in supports:
        m = 0
        for idx in s:
            m |= (1 << idx)
        masks.append(m)

    # 2. Build conflict graph: conflict if intersection size is 3
    adj = [set() for _ in range(N_S)]
    adj_bit = [0] * N_S  # bitmask of neighbors (Python int)

    for i in range(N_S):
        for j in range(i + 1, N_S):
            inter = (masks[i] & masks[j]).bit_count()
            if inter == 3:
                adj[i].add(j)
                adj[j].add(i)
                adj_bit[i] |= (1 << j)
                adj_bit[j] |= (1 << i)

    # 3. Maximum Independent Set via branch-and-bound with coloring bound
    # We'll implement a Tomita-style maximum clique algorithm on the complement graph,
    # which is equivalent to maximum independent set on the original graph.
    
    # Complement graph: edges where NOT conflicting (intersection <= 2)
    # For coloring bound, we need the complement adjacency
    comp_adj = [set() for _ in range(N_S)]
    for i in range(N_S):
        for j in range(N_S):
            if i != j and j not in adj[i]:
                comp_adj[i].add(j)
    
    # Convert to sorted lists for efficiency
    comp_adj_lists = [sorted(comp_adj[i]) for i in range(N_S)]
    
    # Tomita maximum clique algorithm with pivot-based coloring
    best_clique = []
    best_size = 0
    
    def color_sort(P, color):
        """Color the vertices in P and return them sorted by color (descending)."""
        if not P:
            return [], []
        
        # Simple greedy coloring
        color_map = {}
        available = set(P)
        color_idx = 0
        
        while available:
            # Find a maximal independent set in the remaining available set
            # using a simple greedy approach
            current_color_set = set()
            for v in list(available):
                # v can be in this color if none of its neighbors in available are already in current_color_set
                if all((u not in current_color_set) for u in comp_adj[v] & available):
                    current_color_set.add(v)
                    available.discard(v)
            
            for v in current_color_set:
                color_map[v] = color_idx + 1
            
            color_idx += 1
        
        # Sort by color descending
        sorted_P = sorted(P, key=lambda v: -color_map[v])
        return sorted_P, color_map
    
    def expand(R, P, color_map):
        """Recursive expansion for maximum clique."""
        nonlocal best_clique, best_size
        
        if not P:
            if len(R) > best_size:
                best_size = len(R)
                best_clique = R[:]
            return
        
        # If current potential can't beat best, prune
        if len(R) + len(P) <= best_size:
            return
        
        # Color sort P
        sorted_P, color_map = color_sort(P, color_map)
        
        # Process in reverse order (highest color first)
        for i in range(len(sorted_P) - 1, -1, -1):
            v = sorted_P[i]
            
            # Upper bound: current size + color of v
            if len(R) + color_map[v] <= best_size:
                return
            
            # Include v in clique
            R.append(v)
            
            # New P: neighbors of v in P that come after i in sorted order
            new_P = [u for u in sorted_P[i+1:] if u in comp_adj[v]]
            
            expand(R, new_P, color_map)
            
            R.pop()
            
            # Remove v from P for subsequent iterations
            # (This is handled implicitly by the loop structure)
    
    # Run maximum clique on complement graph
    initial_P = list(range(N_S))
    expand([], initial_P, {})
    
    # The clique in complement graph is an independent set in original graph
    independent_set = set(best_clique)
    
    # 4. Local search: 1-2 swaps to improve
    def local_search_1_2_swap(iset):
        """Try to improve independent set using 1-2 swaps."""
        iset = set(iset)
        improved = True
        
        while improved and time.time() < deadline - 5.0:
            improved = False
            
            # Try removing each vertex and adding two compatible vertices
            for v in list(iset):
                if time.time() > deadline - 5.0:
                    break
                
                # Find all vertices compatible with iset \ {v}
                potential_adds = []
                for x in range(N_S):
                    if x in iset or x == v:
                        continue
                    # Check if x is compatible with all y in iset \ {v}
                    if not (adj[x] & (iset - {v})):
                        potential_adds.append(x)
                
                # Try to find two compatible vertices from potential_adds
                found = False
                for i in range(len(potential_adds)):
                    u1 = potential_adds[i]
                    for j in range(i + 1, len(potential_adds)):
                        u2 = potential_adds[j]
                        # Check if u1 and u2 are compatible
                        if u2 not in adj[u1]:
                            iset.remove(v)
                            iset.add(u1)
                            iset.add(u2)
                            improved = True
                            found = True
                            break
                    if found:
                        break
            
            if improved:
                continue
        
        return iset
    
    # Apply local search
    independent_set = local_search_1_2_swap(independent_set)
    
    # 5. Construct vectors: all 16 sign combinations for each support in independent set
    vectors = []
    for s_idx in independent_set:
        support = supports[s_idx]
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int8)
            for idx, sign in zip(support, signs):
                v[idx] = sign
            vectors.append(v)
    
    if not vectors:
        return np.array([], dtype=np.int8).reshape(0, D)
        
    final_vectors = np.array(vectors, dtype=np.int8)
    
    return final_vectors
