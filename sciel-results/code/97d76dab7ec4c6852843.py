"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random as _random
import numpy as np

def _generate_graph(n, p, seed):
    """Generate graph edges using the same RNG sequence as the verifier."""
    rng = _random.Random(seed)
    adj = [set() for _ in range(n)]
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                adj[u].add(v)
                adj[v].add(u)
    return adj

def _bron_kerbosch(adj, n):
    """Enumerate all maximal cliques of size >= 2 using Bron-Kerbosch with pivot."""
    cliques = []
    
    def bk(R, P, X):
        if not P and not X:
            if len(R) >= 2:
                cliques.append(frozenset(R))
            return
        # Choose pivot to minimize branching
        U = P | X
        if not U:
            return
        # Pick pivot with maximum degree in U
        pivot = max(U, key=lambda v: len(adj[v] & U))
        # Branch on vertices in P \ N(pivot)
        candidates = P - adj[pivot]
        for v in list(candidates):
            R_new = R | {v}
            P_new = P & adj[v]
            X_new = X & adj[v]
            bk(R_new, P_new, X_new)
            P.remove(v)
            X.add(v)
    
    bk(frozenset(), set(range(n)), set())
    return cliques

def _to_bitmasks(cliques, n):
    """Convert list of cliques (frozensets) to list of integer bitmasks."""
    masks = []
    for c in cliques:
        mask = 0
        for v in c:
            mask |= (1 << v)
        masks.append(mask)
    return masks

def _greedy_hitting_set_bitmask(clique_masks, n):
    """Greedy set cover using bitmasks."""
    if not clique_masks:
        return set()
    
    # Map vertex to bitmask of cliques it covers? 
    # No, easier: for each vertex, list of clique indices.
    # But with bitmasks, we can check coverage quickly.
    
    uncovered = list(range(len(clique_masks)))
    selected = set()
    
    # Precompute for each vertex, the bitmask of cliques it hits?
    # A vertex v hits clique i if (1 << v) & clique_masks[i] != 0.
    
    while uncovered:
        best_v = -1
        best_count = 0
        for v in range(n):
            if v in selected:
                continue
            count = 0
            for i in uncovered:
                if (1 << v) & clique_masks[i]:
                    count += 1
            if count > best_count:
                best_count = count
                best_v = v
        
        if best_v == -1 or best_count == 0:
            break
            
        selected.add(best_v)
        
        # Remove covered cliques
        new_uncovered = []
        v_bit = 1 << best_v
        for i in uncovered:
            if not (v_bit & clique_masks[i]):
                new_uncovered.append(i)
        uncovered = new_uncovered
        
    return selected

def _local_search_bitmask(selected, clique_masks, n, time_limit):
    """Local search using bitmasks for fast coverage checks."""
    start_time = time.time()
    selected = set(selected)
    
    def is_covered(S):
        """Check if S covers all cliques. Returns True if valid."""
        S_mask = 0
        for v in S:
            S_mask |= (1 << v)
        for cm in clique_masks:
            if not (S_mask & cm):
                return False
        return True

    def get_uncovered_indices(S):
        """Return list of indices of cliques not covered by S."""
        S_mask = 0
        for v in S:
            S_mask |= (1 << v)
        unc = []
        for i, cm in enumerate(clique_masks):
            if not (S_mask & cm):
                unc.append(i)
        return unc

    # 1. Deletion Pass
    changed = True
    while changed:
        changed = False
        if time.time() - start_time > time_limit:
            break
        for v in list(selected):
            S_new = selected - {v}
            if is_covered(S_new):
                selected = S_new
                changed = True
                if time.time() - start_time > time_limit:
                    break

    # 2. 2-for-1 Swap Pass
    changed = True
    while changed:
        changed = False
        if time.time() - start_time > time_limit:
            break
        selected_list = list(selected)
        for i in range(len(selected_list)):
            for j in range(i + 1, len(selected_list)):
                if time.time() - start_time > time_limit:
                    break
                v1, v2 = selected_list[i], selected_list[j]
                S_base = selected - {v1, v2}
                
                # Find cliques not covered by S_base
                unc_idx = get_uncovered_indices(S_base)
                
                if not unc_idx:
                    # We can just remove v1 and v2
                    selected = S_base
                    changed = True
                    break
                    
                # We need a vertex w not in selected that covers all cliques in unc_idx
                # Intersection of these cliques must contain w
                inter_mask = (1 << n) - 1
                for idx in unc_idx:
                    inter_mask &= clique_masks[idx]
                
                # Check if any w in inter_mask is not in selected
                # Also w must not be v1 or v2 (they are removed, but if they are in inter, 
                # we can't use them because we are removing them. 
                # Actually, if w == v1, then S_base U {v1} = selected - {v2}. 
                # This is a 1-for-1 swap effectively, but we are trying to reduce size by 1.
                # So w must be in inter_mask AND w not in selected.
                
                # Note: v1 and v2 are not in S_base.
                # If w is v1, then new set is selected - {v2}. Size reduces by 1.
                # This is valid!
                # So we just need w in inter_mask and w not in S_base.
                # Since S_base = selected - {v1, v2}, w can be v1 or v2 or something else.
                
                # Iterate through bits in inter_mask
                # But we must ensure w is a valid vertex (0..n-1)
                # And w not in S_base.
                
                # Efficiently find such w:
                # candidates_mask = inter_mask & ~S_base_mask
                S_base_mask = 0
                for v in S_base:
                    S_base_mask |= (1 << v)
                
                candidates_mask = inter_mask & ~S_base_mask
                
                if candidates_mask:
                    # Pick the first set bit
                    # Find index of lowest set bit
                    w = (candidates_mask & -candidates_mask).bit_length() - 1
                    # Verify w < n
                    if w < n:
                        selected = S_base | {w}
                        changed = True
                        break
            if changed:
                break

    # 3. 3-for-2 Swap Pass (Optional, if time permits)
    # This is more complex. Might skip for now to keep code clean and fast.
    # 2-for-1 is usually very effective.

    return selected

def _milp_solve(cliques, n, time_limit):
    """Try to solve the hitting set problem using MILP via scipy."""
    m = len(cliques)
    if m == 0:
        return set()
    
    # Build constraint matrix
    A = np.zeros((m, n), dtype=np.int8)
    for i, clique in enumerate(cliques):
        for v in clique:
            A[i, v] = 1
    
    c = np.ones(n, dtype=np.float64)
    
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        
        constraints = LinearConstraint(A, 1, np.inf)
        bounds_obj = Bounds(0, 1)
        
        res = milp(c=c, constraints=constraints, integrality=np.ones(n, dtype=np.int8), 
                   bounds=bounds_obj, 
                   options={"time_limit": time_limit, "presolve": True})
        
        if res.x is not None:
            sol = set(i for i in range(n) if res.x[i] > 0.5)
            # Verify validity
            valid = True
            for c_clique in cliques:
                if not any(v in c_clique for v in sol):
                    valid = False
                    break
            if valid:
                return sol
    except Exception:
        pass
    
    return None

def construct():
    deadline = time.time() + 140.0  # Safety margin before 150s kill
    instances = [(30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)]
    results = []
    
    for n, p, seed in instances:
        if time.time() > deadline - 30.0:
            # Not enough time left; return trivial solution for remaining
            adj = _generate_graph(n, p, seed)
            touched = set()
            for u in range(n):
                for v in adj[u]:
                    touched.add(u)
                    touched.add(v)
            results.append(sorted(touched))
            continue
        
        adj = _generate_graph(n, p, seed)
        
        # Enumerate maximal cliques
        cliques = _bron_kerbosch(adj, n)
        
        if not cliques:
            results.append([])
            continue
        
        # Convert to bitmasks for heuristics
        clique_masks = _to_bitmasks(cliques, n)
        
        best_solution = None
        
        # Try MILP if number of cliques is reasonable
        # MILP is exact, so if it finishes, it's the best possible.
        time_left = deadline - time.time() - 10.0
        milp_budget = min(15.0, time_left * 0.4)
        
        if len(cliques) < 500: # Threshold to avoid huge matrices
            selected_milp = _milp_solve(cliques, n, milp_budget)
            if selected_milp is not None:
                best_solution = selected_milp
        
        # Run Heuristic
        time_left = deadline - time.time() - 10.0
        if time_left > 1.0:
            # Get initial solution
            if best_solution is None:
                selected_init = _greedy_hitting_set_bitmask(clique_masks, n)
            else:
                selected_init = best_solution
            
            # Polish with local search
            selected_ls = _local_search_bitmask(selected_init, clique_masks, n, time_left)
            
            if best_solution is None or len(selected_ls) < len(best_solution):
                best_solution = selected_ls
        
        if best_solution is None:
            # Fallback: greedy only
            best_solution = _greedy_hitting_set_bitmask(clique_masks, n)
            
        results.append(sorted(best_solution))
    
    return results
# EVOLVE-BLOCK-END
