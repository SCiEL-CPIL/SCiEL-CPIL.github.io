import numpy as np
import time
from itertools import combinations, product

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    D = 11
    
    # Generate all weight-4 supports
    supports = list(combinations(range(D), 4))
    N_sup = len(supports)
    
    # Build incompatibility graph
    # incompat_sup[i, j] = 1 if |sup_i ∩ sup_j| >= 3
    sup_sets = [set(s) for s in supports]
    incompat_sup = np.zeros((N_sup, N_sup), dtype=np.int8)
    
    for i in range(N_sup):
        if time.time() > deadline - 10.0:
            break
        for j in range(i + 1, N_sup):
            if len(sup_sets[i] & sup_sets[j]) >= 3:
                incompat_sup[i, j] = 1
                incompat_sup[j, i] = 1
                
    # Precompute neighbors for faster access
    neighbors = [set(np.where(incompat_sup[i] == 1)[0]) for i in range(N_sup)]

    def greedy_min_degree(available, rng):
        """
        Greedy MIS using minimum degree heuristic.
        available: set of node indices
        rng: random generator for tie-breaking
        """
        selected = set()
        avail = set(available)
        
        while avail:
            if time.time() > deadline - 5.0:
                break
            
            # Find node in avail with minimum degree in the induced subgraph on avail
            min_deg = float('inf')
            candidates = []
            for u in avail:
                deg = len(neighbors[u] & avail)
                if deg < min_deg:
                    min_deg = deg
                    candidates = [u]
                elif deg == min_deg:
                    candidates.append(u)
            
            # Pick random from candidates to break symmetry
            u = rng.choice(candidates)
            selected.add(u)
            
            # Remove u and its neighbors from avail
            to_remove = {u} | (neighbors[u] & avail)
            avail -= to_remove
            
        return selected

    def local_search(selected, rng):
        """
        Try to improve the independent set using 1-2 swaps.
        """
        selected = set(selected)
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            for v in list(selected):
                if time.time() > deadline - 5.0:
                    break
                potential_adds = []
                for x in range(N_sup):
                    if x in selected or x == v:
                        continue
                    if not (neighbors[x] & (selected - {v})):
                        potential_adds.append(x)
                        
                found = False
                for i in range(len(potential_adds)):
                    u1 = potential_adds[i]
                    for j in range(i + 1, len(potential_adds)):
                        u2 = potential_adds[j]
                        if incompat_sup[u1, u2] == 0:
                            selected.remove(v)
                            selected.add(u1)
                            selected.add(u2)
                            improved = True
                            found = True
                            break
                    if found:
                        break
            if improved:
                continue
        return selected

    def walk_sat(selected, rng, max_iters=5000):
        """
        WalkSAT-style local search.
        Objective: Minimize conflicts (pairs of selected nodes that are incompatible).
        If conflicts == 0, we have a valid independent set.
        We try to increase the size of the set while maintaining 0 conflicts,
        or reduce conflicts if stuck.
        
        Strategy:
        1. If current set is valid (0 conflicts), try to add a random unselected node.
           If it creates conflicts, we might keep it if it allows adding more later?
           No, let's stick to: 
           - If valid: try to add a node that has 0 conflicts with current set.
           - If invalid: try to remove a node involved in the most conflicts.
           - Also try flips: remove one, add one.
        """
        selected = set(selected)
        # Precompute conflict count for each node in selected?
        # Let's just compute total conflicts.
        
        def count_conflicts(s):
            c = 0
            lst = list(s)
            for i in range(len(lst)):
                u = lst[i]
                for j in range(i+1, len(lst)):
                    v = lst[j]
                    if incompat_sup[u, v]:
                        c += 1
            return c

        current_conflicts = count_conflicts(selected)
        best_selected = set(selected)
        best_conflicts = current_conflicts
        
        for _ in range(max_iters):
            if time.time() > deadline - 5.0:
                break
                
            # 1. Try to add a random node not in selected
            # Pick a random node from [0, N_sup)
            u = rng.integers(N_sup)
            
            if u not in selected:
                # Check conflicts of u with selected
                new_conflicts = current_conflicts
                # Count conflicts between u and existing selected nodes
                added_conflicts = 0
                for v in selected:
                    if incompat_sup[u, v]:
                        added_conflicts += 1
                
                if added_conflicts == 0:
                    # Safe to add
                    selected.add(u)
                    current_conflicts = new_conflicts
                    if len(selected) > len(best_selected):
                        best_selected = set(selected)
                        best_conflicts = current_conflicts
                else:
                    # Adding u creates conflicts.
                    # WalkSAT heuristic: With some probability, add it anyway to escape local optima?
                    # Or try to remove one of its conflicting neighbors.
                    # Let's try removing a random conflicting neighbor.
                    confs = [v for v in selected if incompat_sup[u, v]]
                    if confs:
                        v_to_remove = confs[rng.integers(len(confs))]
                        # Calculate delta conflicts
                        # Remove v_to_remove:
                        # - Removes conflicts between v_to_remove and (selected \ {v_to_remove})
                        # - Adds conflicts between u and (selected \ {v_to_remove})
                        # This is getting complex. Let's just simulate.
                        new_selected = set(selected)
                        new_selected.remove(v_to_remove)
                        new_selected.add(u)
                        new_conf = count_conflicts(new_selected)
                        
                        # Accept if better or equal with some probability
                        if new_conf <= current_conflicts or (rng.random() < 0.1 and new_conf <= current_conflicts + 2):
                            selected = new_selected
                            current_conflicts = new_conf
                            if len(selected) > len(best_selected) and current_conflicts == 0:
                                best_selected = set(selected)
                                best_conflicts = current_conflicts
            else:
                # u is in selected. Try to remove it?
                # Or just skip.
                pass
            
            # 2. If we have 0 conflicts, we might be stuck in a local optimum of size.
            # Try to find a node to add.
            if current_conflicts == 0:
                # Try to add a random node
                u = rng.integers(N_sup)
                if u not in selected:
                    added_conflicts = 0
                    for v in selected:
                        if incompat_sup[u, v]:
                            added_conflicts += 1
                    if added_conflicts == 0:
                        selected.add(u)
                        current_conflicts = 0
                        if len(selected) > len(best_selected):
                            best_selected = set(selected)
                    else:
                        # Try to swap: remove a neighbor of u, add u
                        confs = [v for v in selected if incompat_sup[u, v]]
                        if confs:
                            v_to_remove = confs[rng.integers(len(confs))]
                            new_selected = set(selected)
                            new_selected.remove(v_to_remove)
                            new_selected.add(u)
                            # Check if new_selected is valid
                            # It should be if u was only conflicting with v_to_remove?
                            # No, u might conflict with others.
                            # We checked added_conflicts > 0.
                            # If we remove v_to_remove, does u still conflict with others?
                            # Yes, if len(confs) > 1.
                            # So this swap only works if u conflicts with exactly one node in selected.
                            if len(confs) == 1:
                                new_conf = 0
                                selected = new_selected
                                current_conflicts = 0
                                if len(selected) > len(best_selected):
                                    best_selected = set(selected)
                            else:
                                # Try to remove all confs? No, that's too aggressive.
                                # Just try to remove one and see if we can fix the rest?
                                # For now, skip.
                                pass
            
        return best_selected

    best_iset = None
    best_size = 0
    
    rng = np.random.default_rng(42)
    
    # Run multiple restarts
    n_restarts = 50
    for r in range(n_restarts):
        if time.time() > deadline - 10.0:
            break
            
        avail = set(range(N_sup))
        
        # 1. Greedy
        iset = greedy_min_degree(avail, rng)
        
        # 2. Local Search (1-2 swap)
        iset = local_search(iset, rng)
        
        # 3. WalkSAT (stochastic)
        # Pass a copy
        iset_w = walk_sat(iset, rng, max_iters=2000)
        
        # Use the best of the two
        if len(iset_w) > len(iset):
            iset = iset_w
            
        size = len(iset)
        if size > best_size:
            best_size = size
            best_iset = iset.copy()
            
    if best_iset is None:
        return np.array([[]], dtype=np.int8)
        
    # Generate vectors
    vectors = []
    for i in best_iset:
        sup = supports[i]
        for signs in product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int8)
            for idx, s in zip(sup, signs):
                v[idx] = s
            vectors.append(v)
            
    if not vectors:
        return np.array([[]], dtype=np.int8)

    return np.array(vectors, dtype=np.int8)
