# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

P = {'n': 12, 'm': 40}

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s kill
    
    n = P['n']
    m = P['m']
    total_masks = 1 << n  # 4096
    
    # Precompute comparability bitsets
    # comp_bitset[i] has bit j set iff i and j are comparable (one is strict subset of other)
    all_ones = total_masks - 1  # 0xFFF
    comp_bitset = [0] * total_masks
    
    for i in range(total_masks):
        ci = 0
        # Enumerate all proper submasks of i
        sub = (i - 1) & i
        while True:
            ci |= (1 << sub)
            if sub == 0:
                break
            sub = (sub - 1) & i
        # Enumerate all proper supermasks of i
        comp = all_ones ^ i
        if comp != 0:
            sub = (comp - 1) & comp
            while True:
                s = i | sub
                if s != i:
                    ci |= (1 << s)
                if sub == 0:
                    break
                sub = (sub - 1) & comp
        comp_bitset[i] = ci
    
    def count_pairs(selected):
        """Count comparable pairs in a list of masks."""
        sel_set = set(selected)
        count = 0
        for i in range(len(selected)):
            for j in range(i + 1, len(selected)):
                a, b = selected[i], selected[j]
                if (a & b) == a or (a & b) == b:
                    count += 1
        return count
    
    def greedy_construct(initial=None):
        """Greedy construction with optional initial selection."""
        if initial is not None:
            selected = list(initial)
        else:
            selected = []
        
        selected_mask = 0
        for s in selected:
            selected_mask |= (1 << s)
        
        # Add empty set if not already there
        if 0 not in selected:
            selected.append(0)
            selected_mask |= 1
        
        while len(selected) < m:
            best_mask = -1
            best_score = -1
            for k in range(total_masks):
                if (selected_mask >> k) & 1:
                    continue
                score = (comp_bitset[k] & selected_mask).bit_count()
                if score > best_score:
                    best_score = score
                    best_mask = k
            selected.append(best_mask)
            selected_mask |= (1 << best_mask)
        
        return selected
    
    def local_search(selected, max_iterations=500):
        """Single-swap local search."""
        selected = list(selected)
        selected_mask = 0
        for s in selected:
            selected_mask |= (1 << s)
        
        current_score = count_pairs(selected)
        best_score = current_score
        best_selected = list(selected)
        
        improved = True
        iterations = 0
        while improved and iterations < max_iterations and time.time() < deadline:
            improved = False
            iterations += 1
            for idx in range(m):
                if time.time() >= deadline:
                    break
                old_mask = selected[idx]
                sel_without_old = selected_mask & ~(1 << old_mask)
                loss = (comp_bitset[old_mask] & sel_without_old).bit_count()
                
                best_new = -1
                best_delta = 0
                for new_mask in range(total_masks):
                    if (selected_mask >> new_mask) & 1:
                        continue
                    gain = (comp_bitset[new_mask] & sel_without_old).bit_count()
                    delta = gain - loss
                    if delta > best_delta:
                        best_delta = delta
                        best_new = new_mask
                
                if best_new >= 0 and best_delta > 0:
                    selected[idx] = best_new
                    selected_mask = (selected_mask & ~(1 << old_mask)) | (1 << best_new)
                    current_score += best_delta
                    if current_score > best_score:
                        best_score = current_score
                        best_selected = list(selected)
                    improved = True
        
        return best_selected, best_score
    
    def simulated_annealing(selected, max_iterations=2000):
        """Simulated annealing with single-swap moves."""
        selected = list(selected)
        selected_mask = 0
        for s in selected:
            selected_mask |= (1 << s)
        
        current_score = count_pairs(selected)
        best_score = current_score
        best_selected = list(selected)
        
        T = 5.0
        T_min = 0.1
        cooling = 0.995
        
        iterations = 0
        while iterations < max_iterations and time.time() < deadline:
            iterations += 1
            idx = random.randint(0, m - 1)
            old_mask = selected[idx]
            sel_without_old = selected_mask & ~(1 << old_mask)
            loss = (comp_bitset[old_mask] & sel_without_old).bit_count()
            
            # Pick a random new mask
            new_mask = random.randint(0, total_masks - 1)
            while (selected_mask >> new_mask) & 1:
                new_mask = random.randint(0, total_masks - 1)
            
            gain = (comp_bitset[new_mask] & sel_without_old).bit_count()
            delta = gain - loss
            
            if delta >= 0 or random.random() < pow(2.718281828, delta / T):
                selected[idx] = new_mask
                selected_mask = (selected_mask & ~(1 << old_mask)) | (1 << new_mask)
                current_score += delta
                if current_score > best_score:
                    best_score = current_score
                    best_selected = list(selected)
            
            T *= cooling
            if T < T_min:
                T = T_min
        
        return best_selected, best_score
    
    # Strategy 1: Pure greedy from scratch
    sel1 = greedy_construct()
    sel1, score1 = local_search(sel1, max_iterations=300)
    
    # Strategy 2: Start with a dense structure (all subsets of a small set)
    # Take all subsets of {0,1,2,3,4} = 32 sets, then add 8 more
    small_set = [0, 1, 2, 3, 4]
    small_mask = 0
    for x in small_set:
        small_mask |= (1 << x)
    
    # All subsets of small_set
    all_subsets = []
    sub = small_mask
    while True:
        all_subsets.append(sub)
        if sub == 0:
            break
        sub = (sub - 1) & small_mask
    
    # Add some supersets to make 40 total
    initial2 = list(all_subsets[:32])
    # Add 8 more masks that extend the structure
    for k in range(total_masks):
        if len(initial2) >= m:
            break
        if k not in initial2 and (k & small_mask) == small_mask:
            initial2.append(k)
    
    # Fill remaining with greedy
    sel2 = greedy_construct(initial2 if len(initial2) >= m else None)
    if len(sel2) > m:
        sel2 = sel2[:m]
    sel2, score2 = local_search(sel2, max_iterations=300)
    
    # Strategy 3: Random restarts with greedy + local search
    best_score = max(score1, score2)
    best_selected = sel1 if score1 >= score2 else sel2
    
    random.seed(42)
    restarts = 0
    while time.time() < deadline - 5.0:
        restarts += 1
        # Random initial selection
        random_sel = random.sample(range(total_masks), min(m, 20))
        sel3 = greedy_construct(random_sel)
        if len(sel3) > m:
            sel3 = sel3[:m]
        sel3, score3 = local_search(sel3, max_iterations=150)
        
        if score3 > best_score:
            best_score = score3
            best_selected = sel3
        
        # Also try simulated annealing from best
        if time.time() < deadline - 10.0:
            sel4, score4 = simulated_annealing(best_selected, max_iterations=500)
            if score4 > best_score:
                best_score = score4
                best_selected = sel4
    
    # Final polish with more local search iterations
    best_selected, final_score = local_search(best_selected, max_iterations=500)
    
    # Convert to lists of integers
    result = []
    for mask in best_selected:
        result.append([b for b in range(n) if (mask >> b) & 1])
    
    # Ensure proper types
    final_result = []
    for sl in result:
        final_result.append([int(x) for x in sl])
    
    return final_result
# EVOLVE-BLOCK-END
