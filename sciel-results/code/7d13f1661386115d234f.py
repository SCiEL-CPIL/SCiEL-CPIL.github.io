"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
from math import gcd
from itertools import combinations

def _lcm(a, b):
    return a * b // gcd(a, b)

def _precompute_bad_triples(max_n):
    """
    Precompute all bad triples {a,b,c} with a<b<c<=max_n
    where lcm(a,b)==lcm(a,c)==lcm(b,c).
    """
    n = max_n
    lcm_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            lcm_table[i][j] = _lcm(i, j)
            if i != j:
                lcm_table[j][i] = lcm_table[i][j]
    
    bad_triples = []
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            lab = lcm_table[a][b]
            for c in range(b + 1, n + 1):
                if lcm_table[a][c] == lab and lcm_table[b][c] == lab:
                    bad_triples.append((a, b, c))
    return bad_triples

def _build_triple_index(triples):
    """
    Build index: element -> list of triple indices containing it.
    Also returns the list of triples.
    """
    n_triples = len(triples)
    elem_to_triples = {}
    for i, t in enumerate(triples):
        for x in t:
            if x not in elem_to_triples:
                elem_to_triples[x] = []
            elem_to_triples[x].append(i)
    return triples, elem_to_triples

def _count_conflicts_fast(selected_set, triples):
    """
    Count number of bad triples fully contained in selected_set.
    """
    count = 0
    for t in triples:
        if t[0] in selected_set and t[1] in selected_set and t[2] in selected_set:
            count += 1
    return count

def _greedy_hit_set(triples, elem_to_triples, seed=0):
    """
    Greedy hitting set: repeatedly remove the element that hits the most remaining bad triples.
    Uses efficient active-triple tracking.
    """
    rng = random.Random(seed)
    n_triples = len(triples)
    # active_triple[i] = True if triple i is still active (not yet hit)
    active = [True] * n_triples
    removed = set()
    # Count how many elements of each triple are still in the "unremoved" pool
    # Actually, we track which triples are hit
    
    while True:
        # Find which element hits the most active triples
        freq = {}
        for ti in range(n_triples):
            if not active[ti]:
                continue
            t = triples[ti]
            for x in t:
                freq[x] = freq.get(x, 0) + 1
        
        if not freq:
            break
        
        max_freq = max(freq.values())
        candidates = [x for x, f in freq.items() if f == max_freq]
        x = rng.choice(candidates)
        
        removed.add(x)
        
        # Deactivate all active triples containing x
        for ti in elem_to_triples.get(x, []):
            if active[ti]:
                active[ti] = False
        
    return removed

def _sa_with_incremental_conflicts(initial_removed, triples, elem_to_triples, max_n, rng, time_limit):
    """
    Simulated annealing with incremental conflict counting.
    State: removed set.
    We track conflict_count incrementally.
    
    Key insight: when we add element x to removed (remove from selected),
    conflicts that included x are no longer conflicts.
    When we remove element x from removed (add to selected),
    new conflicts may form where the other two elements of a triple are already in selected.
    """
    start = time.time()
    current_removed = set(initial_removed)
    current_selected = set(range(1, max_n + 1)) - current_removed
    
    # Initial conflict count
    current_conflicts = _count_conflicts_fast(current_selected, triples)
    current_cost = len(current_removed) + 100 * current_conflicts
    
    best_removed = set(current_removed)
    best_conflicts = current_conflicts
    best_cost = current_cost
    
    T_start = 5.0
    T_end = 0.01
    cooling = 0.998
    
    T = T_start
    step = 0
    max_steps = 200000  # Higher cap since steps are now faster
    
    all_elements = list(range(1, max_n + 1))
    
    while T > T_end and (time.time() - start) < time_limit and step < max_steps:
        step += 1
        
        move_type = rng.random()
        
        if move_type < 0.45 and current_removed:
            # Remove from removed (add to selected) - may create new conflicts
            x = rng.choice(list(current_removed))
            new_removed = current_removed.copy()
            new_removed.discard(x)
            new_selected = current_selected.copy()
            new_selected.add(x)
            
            # Count new conflicts: triples containing x where other two are in new_selected
            delta_conflicts = 0
            for ti in elem_to_triples.get(x, []):
                t = triples[ti]
                # Check if the other two elements are in new_selected
                other1, other2 = None, None
                for elem in t:
                    if elem != x:
                        if other1 is None:
                            other1 = elem
                        else:
                            other2 = elem
                if other1 in new_selected and other2 in new_selected:
                    delta_conflicts += 1
            
            new_conflicts = current_conflicts + delta_conflicts
            new_cost = len(new_removed) + 100 * new_conflicts
            
            if delta_conflicts <= 0 or rng.random() < math.exp(-(new_cost - current_cost) / T):
                current_removed = new_removed
                current_selected = new_selected
                current_conflicts = new_conflicts
                current_cost = new_cost
                
                if new_conflicts == 0 and len(new_removed) < len(best_removed):
                    best_removed = set(new_removed)
                    best_conflicts = 0
                    best_cost = len(new_removed)
                    
        elif move_type < 0.75:
            # Add to removed (remove from selected) - reduces or maintains conflicts
            candidates = [x for x in all_elements if x not in current_removed]
            if not candidates:
                continue
            x = rng.choice(candidates)
            new_removed = current_removed.copy()
            new_removed.add(x)
            new_selected = current_selected.copy()
            new_selected.discard(x)
            
            # Count conflicts removed: triples containing x where other two were in current_selected
            delta_conflicts = 0
            for ti in elem_to_triples.get(x, []):
                t = triples[ti]
                other1, other2 = None, None
                for elem in t:
                    if elem != x:
                        if other1 is None:
                            other1 = elem
                        else:
                            other2 = elem
                if other1 in current_selected and other2 in current_selected:
                    delta_conflicts += 1
            
            new_conflicts = current_conflicts - delta_conflicts
            new_cost = len(new_removed) + 100 * new_conflicts
            
            if new_cost <= current_cost or rng.random() < math.exp(-(new_cost - current_cost) / T):
                current_removed = new_removed
                current_selected = new_selected
                current_conflicts = new_conflicts
                current_cost = new_cost
                
                if new_conflicts == 0 and len(new_removed) < len(best_removed):
                    best_removed = set(new_removed)
                    best_conflicts = 0
                    best_cost = len(new_removed)
                    
        else:
            # Swap: remove x from removed, add y to removed
            if current_removed and len(current_removed) < max_n:
                candidates_remove = list(current_removed)
                candidates_add = [x for x in all_elements if x not in current_removed]
                if not candidates_add:
                    continue
                x_out = rng.choice(candidates_remove)
                x_in = rng.choice(candidates_add)
                
                new_removed = current_removed.copy()
                new_removed.discard(x_out)
                new_removed.add(x_in)
                new_selected = current_selected.copy()
                new_selected.add(x_out)
                new_selected.discard(x_in)
                
                # Compute new conflict count from scratch for swap (simpler and still fast enough)
                new_conflicts = _count_conflicts_fast(new_selected, triples)
                new_cost = len(new_removed) + 100 * new_conflicts
                
                if new_cost <= current_cost or rng.random() < math.exp(-(new_cost - current_cost) / T):
                    current_removed = new_removed
                    current_selected = new_selected
                    current_conflicts = new_conflicts
                    current_cost = new_cost
                    
                    if new_conflicts == 0 and len(new_removed) < len(best_removed):
                        best_removed = set(new_removed)
                        best_conflicts = 0
                        best_cost = len(new_removed)
        
        T *= cooling
    
    # If best has conflicts, fix them
    if best_conflicts > 0:
        fixed_removed = set(best_removed)
        fixed_selected = set(range(1, max_n + 1)) - fixed_removed
        active_conflicts = []
        for t in triples:
            if t[0] in fixed_selected and t[1] in fixed_selected and t[2] in fixed_selected:
                active_conflicts.append(t)
        
        while active_conflicts:
            freq = {}
            for t in active_conflicts:
                for x in t:
                    freq[x] = freq.get(x, 0) + 1
            max_f = max(freq.values())
            candidates = [x for x, f in freq.items() if f == max_f]
            x = rng.choice(candidates)
            fixed_removed.add(x)
            fixed_selected.discard(x)
            new_active = []
            for t in active_conflicts:
                if x not in t:
                    new_active.append(t)
            active_conflicts = new_active
        
        best_removed = fixed_removed
    
    return best_removed

def construct():
    start_time = time.time()
    deadline = start_time + 58  # 58s limit to be safe (90s kill, leave margin)
    
    max_n = 128
    bad_triples = _precompute_bad_triples(max_n)
    
    if not bad_triples:
        return list(range(1, max_n + 1))
    
    triples, elem_to_triples = _build_triple_index(bad_triples)
    
    best_removed = None
    best_size = -1
    
    # Phase 1: Greedy hitting set with multiple seeds
    num_greedy = 80
    for i in range(num_greedy):
        if time.time() > deadline - 15:
            break
        removed = _greedy_hit_set(triples, elem_to_triples, seed=i)
        size = max_n - len(removed)
        if size > best_size:
            best_size = size
            best_removed = removed
    
    if best_removed is None:
        # Fallback
        selected = []
        for x in range(1, max_n + 1):
            if all(not (_lcm(a,b)==_lcm(a,x)==_lcm(b,x)) for a,b in combinations(selected,2)):
                selected.append(x)
        return selected
    
    # Phase 2: Simulated Annealing with incremental conflicts to polish
    if time.time() < deadline - 2:
        remaining_time = deadline - time.time() - 2
        num_sa = 5
        sa_time_each = remaining_time / num_sa
        
        for i in range(num_sa):
            if time.time() > deadline - 2:
                break
            rng = random.Random(2000 + i)
            improved_removed = _sa_with_incremental_conflicts(best_removed, triples, elem_to_triples, max_n, rng, sa_time_each)
            size = max_n - len(improved_removed)
            if size > best_size:
                best_size = size
                best_removed = improved_removed
    
    return [x for x in range(1, max_n + 1) if x not in best_removed]

# EVOLVE-BLOCK-END
