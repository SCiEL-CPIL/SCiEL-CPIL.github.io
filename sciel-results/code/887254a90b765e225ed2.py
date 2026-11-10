# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def furstenberg_sarkozy_205() -> np.ndarray:
    """
    Find a large subset of Z/205Z with no two elements differing by a nonzero square modulo 205.
    
    Strategy:
    1. Compute forbidden differences (nonzero quadratic residues mod 205).
    2. Build conflict bitmask: conflict[i] has bit j set if i and j cannot both be in set.
    3. Use multiple starting heuristics:
       - Random Greedy
       - Dynamic Min-Degree Greedy
    4. Refine using systematic local search (1-add, 1-swap, 2-swap).
    5. Return the best solution found within time budget.
    """
    m = 205
    deadline = time.time() + 620  # 10s safety margin before 630s kill
    
    # Compute forbidden differences: nonzero squares mod 205
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, 2, m))
    forbidden.discard(0)
    
    # Precompute: for each i, conflict[i] is a bitmask of all j that conflict with i
    conflict = [0] * m
    for i in range(m):
        for j in range(m):
            if i == j:
                continue
            d1 = (i - j) % m
            d2 = (j - i) % m
            if d1 in forbidden or d2 in forbidden:
                conflict[i] |= (1 << j)
                
    def can_add(cand, selected_mask):
        """Check if cand can be added to the set represented by selected_mask."""
        return (conflict[cand] & selected_mask) == 0

    def greedy_random(seed):
        """Random greedy construction using bitmask conflict checks."""
        rng = random.Random(seed)
        candidates = list(range(m))
        rng.shuffle(candidates)
        selected_mask = 0
        selected_list = []
        for cand in candidates:
            if can_add(cand, selected_mask):
                selected_mask |= (1 << cand)
                selected_list.append(cand)
        return selected_mask, selected_list

    def greedy_dynamic_min_degree(seed):
        """
        Greedy MIS using dynamic min-degree heuristic.
        At each step, pick the vertex with the minimum degree in the remaining graph.
        """
        rng = random.Random(seed)
        remaining_mask = (1 << m) - 1
        selected_mask = 0
        selected_list = []
        
        while remaining_mask:
            best_v = -1
            best_deg = m + 1
            best_rand = 1.0
            temp_mask = remaining_mask
            while temp_mask:
                lsb = temp_mask & -temp_mask
                v = lsb.bit_length() - 1
                d = bin(conflict[v] & remaining_mask).count('1')
                r = rng.random()
                if d < best_deg or (d == best_deg and r < best_rand):
                    best_deg = d
                    best_rand = r
                    best_v = v
                temp_mask ^= lsb
                
            if best_v == -1:
                break
                
            selected_mask |= (1 << best_v)
            selected_list.append(best_v)
            to_remove = (1 << best_v) | (conflict[best_v] & remaining_mask)
            remaining_mask &= ~to_remove
            
        return selected_mask, selected_list

    def local_search(sel_mask, sel_list, rng, time_budget=5.0):
        """
        Systematic local search: 1-add, 1-swap (full enumeration), 2-swap (sampled).
        Returns (mask, list, size).
        """
        b_mask = sel_mask
        b_list = list(sel_list)
        b_size = len(b_list)
        start_time = time.time()
        
        while time.time() - start_time < time_budget:
            improved = False
            
            # 1. Try to add nodes - pick the one with minimum conflict degree among addable
            addable = [c for c in range(m) if not (b_mask >> c) & 1 and can_add(c, b_mask)]
            if addable:
                # Pick the addable node with minimum degree in the current selected set's complement
                # Actually, just pick the one that conflicts with the fewest other addable nodes
                # Simpler: pick the one with minimum total conflict degree
                best_c = addable[0]
                best_deg = bin(conflict[best_c]).count('1')
                for c in addable[1:]:
                    deg = bin(conflict[c]).count('1')
                    if deg < best_deg:
                        best_deg = deg
                        best_c = c
                b_mask |= (1 << best_c)
                b_list.append(best_c)
                b_size += 1
                improved = True
                continue
            
            # 2. Try 1-Swaps: for each u in S, for each v not in S, check if swap is valid and enables an add
            swap_found = False
            for u_idx, u in enumerate(b_list):
                temp_mask = b_mask & ~(1 << u)
                # Find all valid v's for this u
                for v in range(m):
                    if (b_mask >> v) & 1:
                        continue
                    if not can_add(v, temp_mask):
                        continue
                    new_mask = temp_mask | (1 << v)
                    # Check if any w can be added to new_mask
                    w_found = False
                    for w in range(m):
                        if (new_mask >> w) & 1:
                            continue
                        if can_add(w, new_mask):
                            w_found = True
                            break
                    if w_found:
                        b_mask = new_mask
                        b_list[u_idx] = v
                        improved = True
                        swap_found = True
                        break
                if swap_found:
                    break
            
            if improved:
                continue
            
            # 3. Try 2-Swaps (sampled): remove u1, u2 from S; add v1, v2 not in S
            # This is expensive, so sample a limited number
            two_swap_found = False
            if len(b_list) >= 2:
                # Sample up to 50 random 2-swap attempts
                for _ in range(50):
                    u1_idx = rng.randint(0, len(b_list) - 1)
                    u2_idx = rng.randint(0, len(b_list) - 2)
                    if u2_idx >= u1_idx:
                        u2_idx += 1
                    u1 = b_list[u1_idx]
                    u2 = b_list[u2_idx]
                    temp_mask = b_mask & ~(1 << u1) & ~(1 << u2)
                    
                    # Find valid v1, v2
                    v1_candidates = [v for v in range(m) if not (b_mask >> v) & 1 and can_add(v, temp_mask)]
                    if len(v1_candidates) < 2:
                        continue
                    rng.shuffle(v1_candidates)
                    v1 = v1_candidates[0]
                    temp_mask2 = temp_mask | (1 << v1)
                    v2_candidates = [v for v in range(m) if not (temp_mask2 >> v) & 1 and can_add(v, temp_mask2)]
                    if not v2_candidates:
                        continue
                    v2 = rng.choice(v2_candidates)
                    new_mask = temp_mask2 | (1 << v2)
                    
                    # Check if this enables an add
                    w_found = False
                    for w in range(m):
                        if (new_mask >> w) & 1:
                            continue
                        if can_add(w, new_mask):
                            w_found = True
                            break
                    if w_found:
                        b_mask = new_mask
                        b_list[u1_idx] = v1
                        b_list[u2_idx] = v2
                        improved = True
                        two_swap_found = True
                        break
            
            if not improved:
                break
        
        return b_mask, b_list, b_size

    # Main search loop
    best_mask = 0
    best_list = []
    best_size = 0
    
    # Start with deterministic greedy (in order)
    sel = []
    sel_mask = 0
    for cand in range(m):
        if can_add(cand, sel_mask):
            sel.append(cand)
            sel_mask |= (1 << cand)
    best_list = sel
    best_size = len(sel)
    best_mask = sel_mask
    
    # Try multiple random restarts and dynamic greedy
    seed = 1
    while time.time() < deadline:
        # Random greedy start
        start_mask, start_list = greedy_random(seed)
        res_mask, res_list, res_size = local_search(start_mask, start_list, random.Random(seed), time_budget=3.0)
        if res_size > best_size:
            best_mask = res_mask
            best_list = res_list
            best_size = res_size
            
        # Dynamic min-degree greedy
        dyn_mask, dyn_list = greedy_dynamic_min_degree(seed)
        res_mask, res_list, res_size = local_search(dyn_mask, dyn_list, random.Random(seed + 1000), time_budget=3.0)
        if res_size > best_size:
            best_mask = res_mask
            best_list = res_list
            best_size = res_size
            
        # Also try local search on the current best to see if we can improve further
        if time.time() < deadline - 10:
            res_mask, res_list, res_size = local_search(best_mask, best_list, random.Random(seed + 2000), time_budget=3.0)
            if res_size > best_size:
                best_mask = res_mask
                best_list = res_list
                best_size = res_size
            
        seed += 1
        
    return np.array(best_list, dtype=np.int64)

# EVOLVE-BLOCK-END
