import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 118  # safety margin before 120s
    
    points = list(range(16))
    all_triples = list(itertools.combinations(points, 3))
    n_triples = len(all_triples)
    
    # Precompute masks for each triple
    triple_masks = [0] * n_triples
    for i, t in enumerate(all_triples):
        m = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
        triple_masks[i] = m
    
    # Precompute point masks
    point_masks = [1 << p for p in range(16)]
    
    # Precompute all subsets by size as bitmasks for fast alpha checking
    # For each size k, store list of (mask, frozenset) - but we just need masks
    subsets_by_size = {}
    for k in range(1, 17):
        subsets_by_size[k] = []
        for combo in itertools.combinations(range(16), k):
            m = 0
            for p in combo:
                m |= (1 << p)
            subsets_by_size[k].append(m)
    
    def is_linear(T_indices):
        """Check if T_indices forms a linear triple system (any two share at most 1 point)"""
        for i in range(len(T_indices)):
            for j in range(i + 1, len(T_indices)):
                inter = triple_masks[T_indices[i]] & triple_masks[T_indices[j]]
                if inter & (inter - 1):  # more than 1 bit set
                    return False
        return True
    
    def get_alpha_fast(T_indices):
        """
        Fast exact alpha computation using bitmask operations.
        For each size k from 16 down to 1, check if any subset of size k
        contains no triple from T.
        """
        if not T_indices:
            return 16
        
        # Build a set of triple masks for fast lookup
        T_mask_set = set(triple_masks[i] for i in T_indices)
        
        for k in range(16, 0, -1):
            for mask in subsets_by_size[k]:
                # Check if mask contains any triple from T
                # A triple is contained in mask if (triple_mask & mask) == triple_mask
                contains = False
                for tm in T_mask_set:
                    if (tm & mask) == tm:
                        contains = True
                        break
                if not contains:
                    return k
        return 1
    
    def get_alpha_bitwise(T_indices):
        """
        Alternative fast alpha: for each subset mask of size k,
        check if it's independent by verifying no triple is a subset.
        Uses the precomputed T_mask_set.
        """
        if not T_indices:
            return 16
        
        T_mask_list = [triple_masks[i] for i in T_indices]
        
        for k in range(16, 0, -1):
            for mask in subsets_by_size[k]:
                # Check if any triple is fully contained in mask
                independent = True
                for tm in T_mask_list:
                    if (tm & mask) == tm:
                        independent = False
                        break
                if independent:
                    return k
        return 1
    
    def find_max_independent_set(T_indices):
        """Find a maximum independent set for T_indices"""
        if not T_indices:
            return set(range(16))
        
        T_mask_list = [triple_masks[i] for i in T_indices]
        
        for k in range(16, 0, -1):
            for mask in subsets_by_size[k]:
                independent = True
                for tm in T_mask_list:
                    if (tm & mask) == tm:
                        independent = False
                        break
                if independent:
                    # Convert mask to set of points
                    s = set()
                    for p in range(16):
                        if mask & (1 << p):
                            s.add(p)
                    return s
        return set(range(16))
    
    def greedy_construct():
        """
        Greedy construction: start with empty system, repeatedly add the triple
        that most reduces the independence number (or breaks the current max independent set).
        """
        T = []
        
        while time.time() < deadline:
            # Find current max independent set
            mis = find_max_independent_set(T)
            if len(mis) <= 5:
                break
            
            # Find all triples compatible with T
            best_triple = None
            best_score = -1
            
            for idx in range(n_triples):
                # Check compatibility with all in T
                ok = True
                for t_idx in T:
                    inter = triple_masks[idx] & triple_masks[t_idx]
                    if inter & (inter - 1):  # more than 1 bit
                        ok = False
                        break
                if not ok:
                    continue
                
                # Score: how many points from MIS does this triple cover?
                mask = triple_masks[idx]
                score = bin(mask & (sum(1 << p for p in mis))).count('1')
                
                # Tie-break: prefer triples with higher "density" (more points covered in general)
                if score > best_score:
                    best_score = score
                    best_triple = idx
            
            if best_triple is None:
                break
            
            T.append(best_triple)
        
        return T
    
    def local_search_improve(T):
        """
        Local search: try swaps and additions to reduce alpha.
        """
        current_alpha = get_alpha_bitwise(T)
        
        while time.time() < deadline:
            improved = False
            
            # Try additions
            for idx in range(n_triples):
                if idx in T:
                    continue
                ok = True
                for t_idx in T:
                    inter = triple_masks[idx] & triple_masks[t_idx]
                    if inter & (inter - 1):
                        ok = False
                        break
                if not ok:
                    continue
                
                new_T = T + [idx]
                new_alpha = get_alpha_bitwise(new_T)
                if new_alpha < current_alpha:
                    T = new_T
                    current_alpha = new_alpha
                    improved = True
                    break
            
            if improved or time.time() > deadline:
                continue
            
            # Try swaps
            if len(T) > 0:
                for remove_idx in T:
                    for add_idx in range(n_triples):
                        if add_idx == remove_idx or add_idx in T:
                            continue
                        ok = True
                        for t_idx in T:
                            if t_idx == remove_idx:
                                continue
                            inter = triple_masks[add_idx] & triple_masks[t_idx]
                            if inter & (inter - 1):
                                ok = False
                                break
                        if not ok:
                            continue
                        
                        new_T = [t for t in T if t != remove_idx]
                        new_T.append(add_idx)
                        new_alpha = get_alpha_bitwise(new_T)
                        if new_alpha < current_alpha:
                            T = new_T
                            current_alpha = new_alpha
                            improved = True
                            break
                    if improved:
                        break
            
            if not improved:
                # Try 2-swap: remove two, add two
                if len(T) >= 2:
                    found = False
                    for i in range(len(T)):
                        for j in range(i + 1, len(T)):
                            for add1 in range(n_triples):
                                if add1 in T:
                                    continue
                                for add2 in range(n_triples):
                                    if add2 in T or add2 == add1:
                                        continue
                                    
                                    # Check add1 compatible with T minus {T[i], T[j]}
                                    ok1 = True
                                    for t_idx in T:
                                        if t_idx == T[i] or t_idx == T[j]:
                                            continue
                                        inter = triple_masks[add1] & triple_masks[t_idx]
                                        if inter & (inter - 1):
                                            ok1 = False
                                            break
                                    if not ok1:
                                        continue
                                    
                                    # Check add2 compatible with T minus {T[i], T[j]} and add1
                                    ok2 = True
                                    for t_idx in T:
                                        if t_idx == T[i] or t_idx == T[j]:
                                            continue
                                        inter = triple_masks[add2] & triple_masks[t_idx]
                                        if inter & (inter - 1):
                                            ok2 = False
                                            break
                                    if not ok2:
                                        continue
                                    
                                    # Check add1 and add2 compatible with each other
                                    inter = triple_masks[add1] & triple_masks[add2]
                                    if inter & (inter - 1):
                                        continue
                                    
                                    new_T = [t for t in T if t != T[i] and t != T[j]]
                                    new_T.append(add1)
                                    new_T.append(add2)
                                    new_alpha = get_alpha_bitwise(new_T)
                                    if new_alpha < current_alpha:
                                        T = new_T
                                        current_alpha = new_alpha
                                        found = True
                                        break
                                if found:
                                    break
                            if found:
                                break
                        if found:
                            break
                    if found:
                        continue
            
            # If no improvement after full sweep, break to avoid infinite loop
            # Actually, let's do one more random perturbation
            if random.random() < 0.1:
                # Random restart component: remove a random triple
                if T:
                    T.pop(random.randint(0, len(T) - 1))
        
        return T, current_alpha
    
    # Main search strategy
    best_T = []
    best_alpha = 17
    
    # Phase 1: Greedy construction
    T1 = greedy_construct()
    a1 = get_alpha_bitwise(T1)
    if a1 < best_alpha:
        best_alpha = a1
        best_T = T1[:]
    
    # Phase 2: Local search from greedy result
    if time.time() < deadline:
        T2, a2 = local_search_improve(T1[:])
        if a2 < best_alpha:
            best_alpha = a2
            best_T = T2[:]
    
    # Phase 3: Multiple random restarts with greedy + local search
    attempts = 0
    while time.time() < deadline - 5:
        attempts += 1
        
        # Random initial packing
        T = []
        available = list(range(n_triples))
        random.shuffle(available)
        
        for idx in available:
            ok = True
            for t_idx in T:
                inter = triple_masks[idx] & triple_masks[t_idx]
                if inter & (inter - 1):
                    ok = False
                    break
            if ok:
                T.append(idx)
        
        a = get_alpha_bitwise(T)
        if a < best_alpha:
            best_alpha = a
            best_T = T[:]
        
        # Local search
        if time.time() < deadline - 5:
            T_ls, a_ls = local_search_improve(T[:])
            if a_ls < best_alpha:
                best_alpha = a_ls
                best_T = T_ls[:]
        
        # If we found alpha=5, we can stop early (likely optimal)
        if best_alpha <= 5:
            break
    
    # Convert indices back to triples
    result = [list(all_triples[i]) for i in best_T]
    return result
