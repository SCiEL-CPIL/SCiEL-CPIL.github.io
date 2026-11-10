import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    n = 12
    m = 40
    num_masks = 1 << n  # 4096
    
    # Precompute comparability: comp[a][b] = 1 if a is strict subset of b
    # We'll use bitsets for efficiency
    # For each mask, compute its supersets and subsets masks
    
    # Precompute subset/superset relationships using bit manipulation
    # For mask m, its subsets are all masks s where s & m == s and s != m
    # Its supersets are all masks s where s & m == m and s != m
    
    # Precompute for each mask, the list of masks it's comparable with (strict subset relation)
    # comp[i] = set of j such that i is strict subset of j OR j is strict subset of i
    # But we need it for degree calculation in the bipartite graph sense
    
    # Actually, let's think of it differently:
    # The objective is sum over all unordered pairs {a,b} of [a subset b or b subset a]
    # = sum over a of (number of b != a such that a subset b or b subset a)
    # But each pair is counted once, so objective = (1/2) * sum over a of degree(a)
    # where degree(a) = number of b != a such that a subset b or b subset a
    
    # So we want to maximize sum of degrees of selected nodes.
    
    # Precompute degree contribution: for each pair (a,b), if comparable, contributes 1 to total
    # When we select node a, it gains connections to all selected nodes b that are comparable with a.
    
    # Let's precompute for each mask, which other masks are comparable with it
    # comp_mask[a] = bitmask of all b where a and b are comparable (a subset b or b subset a)
    
    # For efficiency, precompute using bit operations
    # Two masks a, b are comparable iff (a & b) == a or (a & b) == b
    
    # Precompute comparable sets for all 4096 masks
    # This is expensive: 4096 * 4096 = 16M comparisons, but each is fast
    
    comp = [0] * num_masks  # comp[a] will be a bitmask indicating which b are comparable with a
    
    for a in range(num_masks):
        # Find all b where (a & b) == a or (a & b) == b
        # This is: b is a superset of a, OR b is a subset of a
        # Supersets of a: all b where b & a == a
        # Subsets of a: all b where b & a == b
        
        # We can enumerate subsets and supersets
        # Subsets of a: iterate over all submasks of a
        # Supersets of a: iterate over all supermasks of a
        
        # For a given a, number of subsets = 2^(popcount(a)), supersets = 2^(n-popcount(a))
        # Total work per a: 2^(popcount(a)) + 2^(n-popcount(a)) - 1 (exclude a itself)
        
        mask_bits = a
        # Enumerate all subsets of a
        sub = a
        while True:
            if sub != a:
                comp[a] |= (1 << sub)
            if sub == 0:
                break
            sub = (sub - 1) & a
        
        # Enumerate all supersets of a
        # Supersets of a: a | t where t is any subset of (~a) & max_mask
        remaining = ((1 << n) - 1) ^ a
        sup_extra = remaining
        while True:
            sup = a | sup_extra
            if sup != a:
                comp[a] |= (1 << sup)
            if sup_extra == 0:
                break
            sup_extra = (sup_extra - 1) & remaining
    
    # Now comp[a] is a bitmask where bit b is set iff a and b are comparable (and a != b)
    # Note: comp[a] includes b if a subset b or b subset a, a != b
    
    # Objective: select m=40 distinct masks to maximize sum over selected pairs of comparability
    # = (1/2) * sum over selected a of popcount(comp[a] & selected_mask)
    # where selected_mask is the bitmask of all selected nodes
    
    # Greedy approach with dynamic degree updates:
    # Start with empty selection. At each step, add the mask that maximizes its degree to current selection.
    # Degree of candidate c to current selection S = popcount(comp[c] & S)
    
    # But greedy might not be optimal. Let's use greedy initialization then local search.
    
    # Greedy initialization
    selected = [False] * num_masks
    selected_list = []
    selected_bitmask = 0
    
    for _ in range(m):
        best_cand = -1
        best_deg = -1
        for c in range(num_masks):
            if selected[c]:
                continue
            # Degree to current selection
            deg = bin(comp[c] & selected_bitmask).count('1')
            if deg > best_deg:
                best_deg = deg
                best_cand = c
        selected[best_cand] = True
        selected_list.append(best_cand)
        selected_bitmask |= (1 << best_cand)
    
    # Calculate current score
    def calc_score_from_list(sel_list):
        score = 0
        for i in range(len(sel_list)):
            for j in range(i + 1, len(sel_list)):
                a = sel_list[i]
                b = sel_list[j]
                if (a & b) == a or (a & b) == b:
                    score += 1
        return score
    
    current_score = calc_score_from_list(selected_list)
    best_score = current_score
    best_list = selected_list[:]
    
    # Local search: try swapping each selected node with an unselected one
    # For speed, precompute degrees
    
    # For local search, we need to efficiently compute the change in score when swapping
    # When we remove node x and add node y:
    # Change = (degree of y to S\{x}) - (degree of x to S\{x})
    # where S is the current selection
    
    # Let's do hill climbing with random restarts
    
    random.seed(42)
    
    polish_deadline = deadline - 2.0
    
    improved = True
    iteration = 0
    while improved and time.time() < polish_deadline:
        improved = False
        sel_set = set(selected_list)
        sel_bitmask = 0
        for s in selected_list:
            sel_bitmask |= (1 << s)
        
        for i in range(m):
            old = selected_list[i]
            # Compute degree of old to rest of selection
            rest_bitmask = sel_bitmask ^ (1 << old)
            old_deg = bin(comp[old] & rest_bitmask).count('1')
            
            best_new = -1
            best_delta = 0
            for new in range(num_masks):
                if selected[new]:
                    continue
                # Degree of new to rest of selection
                new_deg = bin(comp[new] & rest_bitmask).count('1')
                delta = new_deg - old_deg
                if delta > best_delta:
                    best_delta = delta
                    best_new = new
            
            if best_new != -1 and best_delta > 0:
                # Swap
                selected[old] = False
                selected[best_new] = True
                selected_list[i] = best_new
                current_score += best_delta
                improved = True
                if current_score > best_score:
                    best_score = current_score
                    best_list = selected_list[:]
    
    # Final conversion
    result = []
    for mask in best_list:
        s = []
        for i in range(n):
            if (mask >> i) & 1:
                s.append(i)
        result.append(s)
        
    return result
