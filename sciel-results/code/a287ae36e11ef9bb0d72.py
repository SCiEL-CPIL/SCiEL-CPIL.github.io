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
    
    # Precompute support bitmasks for fast intersection checks
    sup_masks = [0] * N_sup
    for i, s in enumerate(supports):
        mask = 0
        for bit in s:
            mask |= (1 << bit)
        sup_masks[i] = mask
    
    # Build incompatibility: two supports are incompatible if |intersection| >= 3
    # Use a 2D boolean matrix for safe and efficient lookup
    incompat_bool = np.zeros((N_sup, N_sup), dtype=bool)
    
    for i in range(N_sup):
        if time.time() > deadline - 30.0:
            break
        mi = sup_masks[i]
        for j in range(i + 1, N_sup):
            inter = mi & sup_masks[j]
            # Count bits in inter
            cnt = 0
            tmp = inter
            while tmp:
                cnt += 1
                tmp &= tmp - 1
            if cnt >= 3:
                incompat_bool[i, j] = True
                incompat_bool[j, i] = True
    
    # Precompute neighbor sets as frozensets for fast intersection
    neighbors = [frozenset(np.where(incompat_bool[i])[0]) for i in range(N_sup)]
    
    def greedy_mis_min_degree(available, rng):
        """Greedy MIS using minimum degree heuristic."""
        selected = set()
        avail = set(available)
        
        while avail:
            if time.time() > deadline - 15.0:
                break
            
            # Find node with minimum degree in induced subgraph
            min_deg = float('inf')
            candidates = []
            for u in avail:
                deg = len(neighbors[u] & avail)
                if deg < min_deg:
                    min_deg = deg
                    candidates = [u]
                elif deg == min_deg:
                    candidates.append(u)
            
            # Pick randomly from minimum-degree candidates
            u = rng.choice(candidates)
            selected.add(u)
            
            # Remove u and its neighbors
            to_remove = {u} | (neighbors[u] & avail)
            avail -= to_remove
            
        return selected
    
    def local_search_add(selected):
        """Try to add vertices that are compatible with all current selected vertices."""
        selected = set(selected)
        
        for x in range(N_sup):
            if x in selected:
                continue
            # Check if x is compatible with all selected using bitmask
            # x is compatible with all in selected iff no neighbor of x is in selected
            if not any(incompat_bool[x, y] for y in selected):
                selected.add(x)
        
        return selected
    
    def local_search_1_2_swap(selected):
        """Improve MIS using 1-2 swaps: remove one vertex, add two compatible vertices."""
        selected = set(selected)
        improved = True
        
        while improved and time.time() < deadline - 10.0:
            improved = False
            sel_list = list(selected)
            
            for v in sel_list:
                if time.time() > deadline - 10.0:
                    break
                
                # Find all nodes compatible with (selected - {v})
                remaining = selected - {v}
                potential_adds = []
                for x in range(N_sup):
                    if x in remaining:
                        continue
                    # x must be compatible with all y in remaining
                    if not any(incompat_bool[x, y] for y in remaining):
                        potential_adds.append(x)
                
                # Find a pair (u1, u2) in potential_adds that are compatible with each other
                found = False
                n_pot = len(potential_adds)
                for i in range(n_pot):
                    u1 = potential_adds[i]
                    for j in range(i + 1, n_pot):
                        u2 = potential_adds[j]
                        if not incompat_bool[u1, u2]:
                            selected.remove(v)
                            selected.add(u1)
                            selected.add(u2)
                            improved = True
                            found = True
                            break
                    if found:
                        break
                
                if improved:
                    break  # Restart the outer loop
        
        return selected
    
    def simulated_annealing(selected, rng, n_iters=5000):
        """Simulated annealing with 1-1 swaps to escape local optima."""
        selected = set(selected)
        T = 1.0
        T_min = 0.01
        cooling = 0.999
        
        for it in range(n_iters):
            if time.time() > deadline - 10.0:
                break
            
            # Pick a random selected vertex and a random unselected vertex
            sel_list = list(selected)
            if not sel_list:
                break
            v_out = sel_list[rng.integers(len(sel_list))]
            
            # Find candidates compatible with (selected - {v_out})
            remaining = selected - {v_out}
            candidates = []
            for x in range(N_sup):
                if x in remaining:
                    continue
                if not any(incompat_bool[x, y] for y in remaining):
                    candidates.append(x)
            
            if not candidates:
                continue
            
            v_in = candidates[rng.integers(len(candidates))]
            
            # This is always a valid swap (size stays same), so accept
            selected.remove(v_out)
            selected.add(v_in)
            
            T *= cooling
        
        return selected
    
    def local_search_1_1_swap(selected, rng, n_iters=2000):
        """Try 1-1 swaps: replace one selected vertex with another compatible one."""
        selected = set(selected)
        
        for it in range(n_iters):
            if time.time() > deadline - 10.0:
                break
            
            improved = False
            sel_list = list(selected)
            rng.shuffle(sel_list)
            
            for v in sel_list:
                remaining = selected - {v}
                # Find a candidate x not in selected that is compatible with all in remaining
                # and also compatible with v (so we can swap)
                for x in range(N_sup):
                    if x in selected:
                        continue
                    if not any(incompat_bool[x, y] for y in remaining):
                        # Check if this swap actually helps: x should allow more additions
                        # For now, just do the swap if it's valid
                        selected.remove(v)
                        selected.add(x)
                        improved = True
                        break
                if improved:
                    break
            
            if not improved:
                break
        
        return selected
    
    best_iset = set()
    best_size = 0
    
    rng = np.random.default_rng(42)
    
    # Multiple restarts with different strategies
    n_restarts = 60
    for r in range(n_restarts):
        if time.time() > deadline - 20.0:
            break
        
        if r == 0:
            # First run: pure greedy min-degree
            iset = greedy_mis_min_degree(set(range(N_sup)), rng)
        elif r < 25:
            # Greedy + local search
            iset = greedy_mis_min_degree(set(range(N_sup)), rng)
            iset = local_search_add(iset)
            iset = local_search_1_2_swap(iset)
        elif r < 45:
            # Perturbed greedy: start with random subset removed
            avail = set(range(N_sup))
            n_remove = rng.integers(5, 30)
            to_remove = set(rng.choice(N_sup, size=n_remove, replace=False))
            avail -= to_remove
            iset = greedy_mis_min_degree(avail, rng)
            iset = local_search_add(iset)
            iset = local_search_1_2_swap(iset)
        else:
            # More aggressive perturbation
            avail = set(range(N_sup))
            n_remove = rng.integers(20, 60)
            to_remove = set(rng.choice(N_sup, size=n_remove, replace=False))
            avail -= to_remove
            iset = greedy_mis_min_degree(avail, rng)
            iset = local_search_add(iset)
            iset = local_search_1_2_swap(iset)
        
        size = len(iset)
        if size > best_size:
            best_size = size
            best_iset = set(iset)
    
    # Final improvement attempts on best solution
    if time.time() < deadline - 15.0:
        best_iset = local_search_add(best_iset)
        best_iset = local_search_1_2_swap(best_iset)
    
    # Simulated annealing on best solution
    if time.time() < deadline - 10.0:
        best_iset = simulated_annealing(best_iset, rng, n_iters=3000)
        best_iset = local_search_add(best_iset)
        best_iset = local_search_1_2_swap(best_iset)
    
    # One more round of local search if time permits
    if time.time() < deadline - 5.0:
        best_iset = local_search_add(best_iset)
        best_iset = local_search_1_2_swap(best_iset)
    
    # Generate vectors: for each support in the MIS, include all 2^4 = 16 sign patterns
    vectors = []
    for i in best_iset:
        sup = supports[i]
        for signs in product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int8)
            for idx, s in zip(sup, signs):
                v[idx] = s
            vectors.append(v)
    
    if not vectors:
        # Fallback: return a small valid set
        vectors = []
        for i in range(D):
            for j in range(i + 1, D):
                v = np.zeros(D, dtype=np.int8)
                v[i] = 1
                v[j] = 1
                vectors.append(v)
    
    return np.array(vectors, dtype=np.int8)
