import numpy as np
import time
from itertools import combinations, product
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    D = 11
    
    # Generate all weight-4 supports
    supports = list(combinations(range(D), 4))
    N_sup = len(supports)  # 330
    
    # Precompute support masks for fast intersection
    sup_masks = []
    for s in supports:
        mask = 0
        for i in s:
            mask |= (1 << i)
        sup_masks.append(mask)
    
    # Build incompatibility: two supports conflict if |intersection| >= 3
    incompat = np.zeros((N_sup, N_sup), dtype=np.uint8)
    for i in range(N_sup):
        if time.time() > deadline - 30.0:
            break
        for j in range(i + 1, N_sup):
            inter = sup_masks[i] & sup_masks[j]
            if inter.bit_count() >= 3:
                incompat[i, j] = 1
                incompat[j, i] = 1
    
    # Build neighbor lists
    neighbors = [set(np.where(incompat[i] == 1)[0]) for i in range(N_sup)]
    incompat_bool = incompat.astype(bool)
    
    def greedy_mis(available, rng):
        """Greedy MIS using minimum degree heuristic."""
        selected = set()
        avail = set(available)
        
        while avail:
            if time.time() > deadline - 15.0:
                break
            
            if len(avail) > 200:
                avail_arr = np.array(list(avail), dtype=np.int32)
                deg_arr = incompat_bool[avail_arr][:, avail_arr].sum(axis=1)
                min_deg = deg_arr.min()
                min_mask = deg_arr == min_deg
                candidates = avail_arr[min_mask]
                u = int(rng.choice(candidates))
            else:
                min_deg = float('inf')
                candidates = []
                for u in avail:
                    d = len(neighbors[u] & avail)
                    if d < min_deg:
                        min_deg = d
                        candidates = [u]
                    elif d == min_deg:
                        candidates.append(u)
                u = rng.choice(candidates)
            
            selected.add(u)
            to_remove = {u} | (neighbors[u] & avail)
            avail -= to_remove
        
        return selected
    
    def local_search_add(selected):
        """Add all compatible vertices."""
        selected = set(selected)
        if not selected:
            return selected
        
        sel_arr = np.array(list(selected), dtype=np.int32)
        compatible_mask = ~incompat_bool[:, sel_arr].any(axis=1)
        compatible_mask[sel_arr] = False
        
        addable = np.where(compatible_mask)[0]
        for x in addable:
            if time.time() > deadline - 10.0:
                break
            selected.add(int(x))
        
        return selected
    
    def local_search_1_2_swap(selected, rng):
        """Improve MIS using 1-2 swaps."""
        selected = set(selected)
        improved = True
        
        while improved and time.time() < deadline - 10.0:
            improved = False
            
            for v in list(selected):
                if time.time() > deadline - 10.0:
                    break
                
                potential_adds = []
                for x in range(N_sup):
                    if x in selected or x == v:
                        continue
                    if not (neighbors[x] & (selected - {v})):
                        potential_adds.append(x)
                
                n_pot = len(potential_adds)
                for i in range(n_pot):
                    u1 = potential_adds[i]
                    for j in range(i + 1, n_pot):
                        u2 = potential_adds[j]
                        if incompat[u1, u2] == 0:
                            selected.remove(v)
                            selected.add(u1)
                            selected.add(u2)
                            improved = True
                            break
                    if improved:
                        break
                
                if improved:
                    break
        
        return selected
    
    def branch_and_bound_mis():
        """Exact MIS using branch and bound with coloring upper bound."""
        # Use a simple branch and bound
        best_set = set()
        best_size = 0
        
        def color_bound(vertices):
            """Greedy coloring to get upper bound on clique size (lower bound on MIS complement)."""
            # For MIS, we need upper bound on independent set size
            # Use greedy coloring on the induced subgraph
            if not vertices:
                return 0
            vlist = list(vertices)
            colors = {}
            color_count = 0
            for v in vlist:
                used_colors = set()
                for u in neighbors[v]:
                    if u in colors:
                        used_colors.add(colors[u])
                for c in range(color_count + 1):
                    if c not in used_colors:
                        colors[v] = c
                        break
                else:
                    colors[v] = color_count
                    color_count += 1
            return color_count
        
        def bbs(current_set, available, depth):
            nonlocal best_set, best_size
            
            if time.time() > deadline - 20.0:
                return
            
            if not available:
                if len(current_set) > best_size:
                    best_size = len(current_set)
                    best_set = set(current_set)
                return
            
            # Upper bound: current size + size of available
            if len(current_set) + len(available) <= best_size:
                return
            
            # Pick vertex with max degree in available
            max_deg = -1
            pivot = None
            for v in available:
                d = len(neighbors[v] & available)
                if d > max_deg:
                    max_deg = d
                    pivot = v
            
            # Branch 1: include pivot
            new_set = current_set | {pivot}
            new_avail = available - {pivot} - neighbors[pivot]
            bbs(new_set, new_avail, depth + 1)
            
            # Branch 2: exclude pivot
            bbs(current_set, available - {pivot}, depth + 1)
        
        bbs(set(), set(range(N_sup)), 0)
        return best_set
    
    # Strategy 1: Try to find a good MIS using greedy + local search
    rng = np.random.default_rng(42)
    best_iset = set()
    best_size = 0
    
    n_restarts = 50
    for r in range(n_restarts):
        if time.time() > deadline - 25.0:
            break
        
        if r == 0:
            iset = greedy_mis(set(range(N_sup)), rng)
        elif r < 20:
            iset = greedy_mis(set(range(N_sup)), rng)
            iset = local_search_add(iset)
            iset = local_search_1_2_swap(iset, rng)
        else:
            # Perturbed greedy
            avail = set(range(N_sup))
            n_remove = rng.integers(5, 40)
            to_remove = set(rng.choice(N_sup, size=n_remove, replace=False))
            avail -= to_remove
            iset = greedy_mis(avail, rng)
            iset = local_search_add(iset)
            iset = local_search_1_2_swap(iset, rng)
        
        size = len(iset)
        if size > best_size:
            best_size = size
            best_iset = set(iset)
    
    # Final improvement
    if time.time() < deadline - 10.0:
        best_iset = local_search_add(best_iset)
        best_iset = local_search_1_2_swap(best_iset, rng)
    
    # Generate vectors from MIS: all 16 sign patterns per support
    vectors = []
    for i in best_iset:
        sup = supports[i]
        for signs in product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int8)
            for idx, s in zip(sup, signs):
                v[idx] = s
            vectors.append(v)
    
    # Add Type A vectors: ±2 * e_i for i in range(11)
    for i in range(D):
        for sign in [1, -1]:
            v = np.zeros(D, dtype=np.int8)
            v[i] = 2 * sign
            vectors.append(v)
    
    # Check if we have enough vectors; if not, use lattice fallback
    if len(vectors) < 252:
        # Use E8 x D3 lattice
        vectors = []
        # E8 minimal vectors: 240 vectors
        # Type 1: ±2 on two coordinates
        for i in range(8):
            for j in range(i + 1, 8):
                for si in [1, -1]:
                    for sj in [1, -1]:
                        v = np.zeros(11, dtype=np.int8)
                        v[i] = 2 * si
                        v[j] = 2 * sj
                        vectors.append(v)
        # Type 2: ±1 on all 8 coordinates with even number of -1s
        from itertools import product as prod
        for signs in prod([1, -1], repeat=8):
            if sum(1 for s in signs if s == -1) % 2 == 0:
                v = np.zeros(11, dtype=np.int8)
                for i, s in enumerate(signs):
                    v[i] = s
                vectors.append(v)
        
        # D3 minimal vectors: 12 vectors in coordinates 8,9,10
        # D3 = {x in Z^3 : sum(x_i) even}
        # Minimal vectors of D3: ±e_i ± e_j for i<j, and ±(1,1,1) patterns with even sum
        # Actually D3 minimal vectors: 12 vectors
        # They are: (±1,±1,0) and permutations (6 vectors × 2 signs = 12? No)
        # D3 = {x : x_i integers, sum x_i even}
        # Minimal norm vectors have norm sqrt(2): (±1,±1,0) and permutations
        # There are 3 choices for zero position × 2 × 2 = 12
        for zero_pos in range(3):
            for i in range(3):
                for j in range(3):
                    if i == j or i == zero_pos or j == zero_pos:
                        continue
                    for si in [1, -1]:
                        for sj in [1, -1]:
                            v = np.zeros(11, dtype=np.int8)
                            v[8 + i] = si
                            v[8 + j] = sj
                            vectors.append(v)
        
        # Deduplicate
        unique_vectors = []
        seen = set()
        for v in vectors:
            key = tuple(v)
            if key not in seen:
                seen.add(key)
                unique_vectors.append(v)
        
        vectors = unique_vectors
    
    if not vectors:
        vectors = []
        for i in range(D):
            for j in range(i + 1, D):
                v = np.zeros(D, dtype=np.int8)
                v[i] = 1
                v[j] = 1
                vectors.append(v)
    
    return np.array(vectors, dtype=np.int8)
