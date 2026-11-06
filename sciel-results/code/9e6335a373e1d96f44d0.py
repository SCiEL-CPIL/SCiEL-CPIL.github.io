import time
import itertools
import random
import math

def _count_independent_ksets(triples, k, vertices=None):
    """Count independent k-sets in the hypergraph defined by triples."""
    if vertices is None:
        vertices = list(range(16))
    n = len(vertices)
    if k > n:
        return 0
    if k <= 0:
        return 1
    
    # Build a set of triple bitmasks for fast checking
    triple_set = set()
    for t in triples:
        m = 0
        for p in t:
            m |= (1 << p)
        triple_set.add(m)
    
    # Count independent k-sets
    count = 0
    for combo in itertools.combinations(range(16), k):
        s_mask = 0
        for j in combo:
            s_mask |= (1 << j)
        # Check if any triple is fully contained
        independent = True
        for m in triple_set:
            if (s_mask & m) == m:
                independent = False
                break
        if independent:
            count += 1
    return count

def _find_independent_kset(triples, k):
    """Find one independent k-set, or return None."""
    triple_set = set()
    for t in triples:
        m = 0
        for p in t:
            m |= (1 << p)
        triple_set.add(m)
    
    for combo in itertools.combinations(range(16), k):
        s_mask = 0
        for j in combo:
            s_mask |= (1 << j)
        independent = True
        for m in triple_set:
            if (s_mask & m) == m:
                independent = False
                break
        if independent:
            return list(combo)
    return None

def _greedy_construct(seed):
    random.seed(seed)
    V = list(range(16))
    all_triples = list(itertools.combinations(V, 3))
    random.shuffle(all_triples)
    
    used_pairs = set()
    triples = []
    
    for t in all_triples:
        ok = True
        for a, b in itertools.combinations(t, 2):
            pair = (min(a, b), max(a, b))
            if pair in used_pairs:
                ok = False
                break
        if ok:
            triples.append(list(t))
            for a, b in itertools.combinations(t, 2):
                used_pairs.add((min(a, b), max(a, b)))
                
    return triples

def _local_search(sys_in, deadline):
    """Local search to minimize the number of independent 5-sets."""
    sys_cur = [list(t) for t in sys_in]
    n_triples = len(sys_cur)
    
    # Build triple bitmasks
    tri_masks = []
    for t in sys_cur:
        m = 0
        for p in t:
            m |= (1 << p)
        tri_masks.append(m)
    
    # Count independent 5-sets
    def count_ind5():
        cnt = 0
        for combo in itertools.combinations(range(16), 5):
            s_mask = 0
            for j in combo:
                s_mask |= (1 << j)
            independent = True
            for m in tri_masks:
                if (s_mask & m) == m:
                    independent = False
                    break
            if independent:
                cnt += 1
        return cnt
    
    best_count = count_ind5()
    best_sys = [list(t) for t in sys_cur]
    if best_count == 0:
        return best_sys, 0
    
    # Simulated annealing
    T = 2.0
    cooling = 0.995
    current_count = best_count
    
    while time.time() < deadline:
        # Pick a random triple to replace
        idx = random.randint(0, n_triples - 1)
        old_t = sys_cur[idx]
        
        # Generate a candidate replacement triple
        # Must be linear with all other triples
        used_pairs = set()
        for j, t in enumerate(sys_cur):
            if j == idx:
                continue
            for a, b in itertools.combinations(t, 2):
                used_pairs.add((min(a, b), max(a, b)))
        
        # Find all valid triples (not using any used pair)
        valid_triples = []
        for t in itertools.combinations(range(16), 3):
            ok = True
            for a, b in itertools.combinations(t, 2):
                pair = (min(a, b), max(a, b))
                if pair in used_pairs:
                    ok = False
                    break
            if ok:
                valid_triples.append(t)
        
        if not valid_triples:
            continue
        
        new_t = random.choice(valid_triples)
        
        # Update triple_masks
        old_mask = tri_masks[idx]
        new_mask = 0
        for p in new_t:
            new_mask |= (1 << p)
        tri_masks[idx] = new_mask
        
        new_count = count_ind5()
        
        if new_count <= current_count or random.random() < min(1.0, math.exp(-abs(new_count - current_count) / T)):
            # Accept
            sys_cur[idx] = list(new_t)
            current_count = new_count
            if new_count < best_count:
                best_count = new_count
                best_sys = [list(t) for t in sys_cur]
                if best_count == 0:
                    return best_sys, 0
        else:
            # Revert
            tri_masks[idx] = old_mask
        
        T *= cooling
    
    return best_sys, best_count

def construct():
    start = time.time()
    DEADLINE = 115.0  # Safety margin
    
    best_alpha = 16
    best_sys = []
    
    seed = 0
    while time.time() - start < DEADLINE:
        sys_candidate = _greedy_construct(seed)
        
        # Check if already optimal (alpha <= 4)
        # We check if there are any independent 5-sets.
        # If 0, alpha <= 4. Since alpha >= 4, alpha = 4.
        count5 = _count_independent_ksets(sys_candidate, 5)
        
        if count5 == 0:
            return sys_candidate
        
        # Local Search
        remaining = DEADLINE - (time.time() - start)
        ls_budget = min(5.0, remaining / 2.0)  # Max 5s per LS, or half remaining
        if ls_budget < 1.0:
            break
            
        ls_sys, ls_count = _local_search(sys_candidate, time.time() + ls_budget)
        
        if ls_count == 0:
            return ls_sys
        
        seed += 1
        
    # Fallback if no system found (should not happen)
    if not best_sys:
        best_sys = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]
        
    return best_sys
