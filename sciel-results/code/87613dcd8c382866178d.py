import time
import random
import sys

def construct():
    n = 12
    m = 40
    total_masks = 1 << n
    
    # Precompute comparability
    # comp[i][j] is 1 if i is a subset of j or j is a subset of i (and i != j)
    # We can use bit operations: (i | j) == i means i subset j.
    # (i | j) == j means j subset i.
    # So comp[i][j] = 1 if (i | j) in [i, j] and i != j.
    
    comp = [[0] * total_masks for _ in range(total_masks)]
    for i in range(total_masks):
        for j in range(i + 1, total_masks):
            if (i | j) == i or (i | j) == j:
                comp[i][j] = 1
                comp[j][i] = 1

    def get_score(family):
        score = 0
        for i in range(len(family)):
            for j in range(i + 1, len(family)):
                score += comp[family[i]][family[j]]
        return score

    def greedy_init(seed):
        random.seed(seed)
        family = []
        used = set()
        
        # Start with empty set or random
        # A good heuristic is to start with a chain
        cur = 0
        family.append(cur)
        used.add(cur)
        for i in range(n):
            cur |= (1 << i)
            family.append(cur)
            used.add(cur)
            
        # Greedily add best set
        while len(family) < m:
            best_gain = -1
            best_mask = None
            for s in range(total_masks):
                if s in used:
                    continue
                gain = 0
                for f in family:
                    gain += comp[s][f]
                if gain > best_gain:
                    best_gain = gain
                    best_mask = s
            
            if best_mask is None:
                break
            family.append(best_mask)
            used.add(best_mask)
            
        # If we still don't have enough, add random
        while len(family) < m:
            s = random.randint(0, total_masks - 1)
            if s not in used:
                family.append(s)
                used.add(s)
                
        return family

    def local_search(family, time_limit):
        fam = family[:]
        in_fam = [False] * total_masks
        for mask in fam:
            in_fam[mask] = True
            
        # sum_comp[s] = number of sets in fam comparable to s
        sum_comp = [0] * total_masks
        for s in range(total_masks):
            sc = 0
            for x in fam:
                sc += comp[s][x]
            sum_comp[s] = sc
            
        # contrib[i] = number of sets in fam comparable to fam[i] (excluding self)
        # This is effectively sum_comp[fam[i]] since comp[i][i]=0
        contribs = [sum_comp[fam[i]] for i in range(m)]
        
        current_score = sum(contribs) // 2
        best_score = current_score
        best_fam = fam[:]
        
        T = 5.0
        cooling = 0.999
        T_min = 0.1
        no_improve = 0
        
        start_time = time.time()
        
        while time.time() - start_time < time_limit:
            # 1. Find best 1-swap
            best_1_swap_delta = -1000
            best_1_swap_i = -1
            best_1_swap_s = -1
            
            for i in range(m):
                old_val = fam[i]
                # We want to replace old_val with s
                # Delta = (sum_comp[s] - comp[s][old_val]) - contribs[i]
                # Note: sum_comp[s] includes comp[s][old_val] if s is comparable to old_val?
                # sum_comp[s] is calculated based on current fam.
                # If we remove old_val, the comparable count for s becomes sum_comp[s] - comp[s][old_val].
                # The comparable count for old_val in the rest of the family is contribs[i].
                # So new contribution from s is (sum_comp[s] - comp[s][old_val]).
                # Old contribution from old_val is contribs[i].
                # Delta = (sum_comp[s] - comp[s][old_val]) - contribs[i]
                
                # Optimization: Iterate only over unused sets?
                # Iterating all 4096 is fast enough.
                for s in range(total_masks):
                    if in_fam[s]:
                        continue
                    val = sum_comp[s] - comp[s][old_val]
                    delta = val - contribs[i]
                    if delta > best_1_swap_delta:
                        best_1_swap_delta = delta
                        best_1_swap_i = i
                        best_1_swap_s = s

            # 2. Find best 2-swap
            # To save time, we can restrict to a subset of candidates or just do random pairs
            # Given m=40, pairs are 780.
            # For each pair (i, j), we look for best s, t.
            # This is O(m^2 * N^2) which is too slow (40^2 * 4096^2).
            # We can use top-k candidates for 1-swap to guide 2-swap.
            
            # Let's collect top 5 candidates for each i
            top_candidates = [[] for _ in range(m)]
            for i in range(m):
                old_val = fam[i]
                cands = []
                for s in range(total_masks):
                    if in_fam[s]:
                        continue
                    val = sum_comp[s] - comp[s][old_val]
                    cands.append((val, s))
                cands.sort(reverse=True)
                top_candidates[i] = cands[:5]
            
            best_2_swap_delta = -1000
            best_2_swap_i1 = -1
            best_2_swap_i2 = -1
            best_2_swap_s1 = -1
            best_2_swap_s2 = -1
            
            for i1 in range(m):
                if not top_candidates[i1]:
                    continue
                for i2 in range(i1 + 1, m):
                    if not top_candidates[i2]:
                        continue
                    
                    old1 = fam[i1]
                    old2 = fam[i2]
                    
                    # Try combinations of top candidates
                    for delta1, s1 in top_candidates[i1]:
                        # delta1 here is val, not delta.
                        # val1 = sum_comp[s1] - comp[s1][old1]
                        # We need to account for removing old2 as well.
                        # When removing old1 and old2:
                        # New comparable count for s1:
                        # sum_comp[s1] - comp[s1][old1] - comp[s1][old2]
                        # (Assuming s1 is not old2, which is true since s1 is unused and old2 is used)
                        
                        for delta2, s2 in top_candidates[i2]:
                            # val2 = sum_comp[s2] - comp[s2][old2]
                            # New comparable count for s2:
                            # sum_comp[s2] - comp[s2][old2] - comp[s2][old1]
                            
                            # New Score Contribution for s1, s2:
                            new_c1 = sum_comp[s1] - comp[s1][old1] - comp[s1][old2]
                            new_c2 = sum_comp[s2] - comp[s2][old2] - comp[s2][old1]
                            
                            # Old Score Contribution for old1, old2:
                            old_c1 = contribs[i1]
                            old_c2 = contribs[i2]
                            
                            # Note: contribs[i1] includes comp[old1][old2].
                            # contribs[i2] includes comp[old2][old1].
                            # So the total old contribution of the pair to the sum is old_c1 + old_c2.
                            # The total new contribution is new_c1 + new_c2.
                            # Delta = new_c1 + new_c2 - old_c1 - old_c2
                            
                            delta = new_c1 + new_c2 - old_c1 - old_c2
                            
                            if delta > best_2_swap_delta:
                                best_2_swap_delta = delta
                                best_2_swap_i1 = i1
                                best_2_swap_i2 = i2
                                best_2_swap_s1 = s1
                                best_2_swap_s2 = s2

            # Decide move
            accept = False
            move_type = 0
            move_i1 = -1
            move_i2 = -1
            move_s1 = -1
            move_s2 = -1
            move_delta = 0
            
            # Compare best 1-swap and 2-swap
            # Note: best_1_swap_delta is the delta for the best 1-swap.
            # best_2_swap_delta is the delta for the best 2-swap.
            
            # We should prefer 2-swap if it's better, but 2-swap is more expensive to execute.
            # However, we already found the best.
            
            # Let's just pick the one with higher delta if positive, or accept negative with SA.
            
            if best_2_swap_delta > best_1_swap_delta:
                if best_2_swap_delta > 0:
                    accept = True
                    move_type = 2
                    move_i1 = best_2_swap_i1
                    move_i2 = best_2_swap_i2
                    move_s1 = best_2_swap_s1
                    move_s2 = best_2_swap_s2
                    move_delta = best_2_swap_delta
                else:
                    if random.random() < pow(2.718281828, best_2_swap_delta / T):
                        accept = True
                        move_type = 2
                        move_i1 = best_2_swap_i1
                        move_i2 = best_2_swap_i2
                        move_s1 = best_2_swap_s1
                        move_s2 = best_2_swap_s2
                        move_delta = best_2_swap_delta
            else:
                if best_1_swap_delta > 0:
                    accept = True
                    move_type = 1
                    move_i1 = best_1_swap_i
                    move_s1 = best_1_swap_s
                    move_delta = best_1_swap_delta
                else:
                    if random.random() < pow(2.718281828, best_1_swap_delta / T):
                        accept = True
                        move_type = 1
                        move_i1 = best_1_swap_i
                        move_s1 = best_1_swap_s
                        move_delta = best_1_swap_delta

            if accept:
                if move_type == 1:
                    old_val = fam[move_i1]
                    new_val = move_s1
                    
                    fam[move_i1] = new_val
                    in_fam[old_val] = False
                    in_fam[new_val] = True
                    
                    # Update sum_comp
                    # For all s, sum_comp[s] changes by comp[s][new_val] - comp[s][old_val]
                    for s in range(total_masks):
                        sum_comp[s] += comp[s][new_val] - comp[s][old_val]
                        
                    # Update contribs
                    # contribs[move_i1] = sum_comp[new_val]
                    # For other k, contribs[k] changes by comp[fam[k]][new_val] - comp[fam[k]][old_val]
                    contribs[move_i1] = sum_comp[new_val]
                    for k in range(m):
                        if k == move_i1:
                            continue
                        contribs[k] += comp[fam[k]][new_val] - comp[fam[k]][old_val]

                elif move_type == 2:
                    old1 = fam[move_i1]
                    old2 = fam[move_i2]
                    new1 = move_s1
                    new2 = move_s2
                    
                    fam[move_i1] = new1
                    fam[move_i2] = new2
                    
                    in_fam[old1] = False
                    in_fam[old2] = False
                    in_fam[new1] = True
                    in_fam[new2] = True
                    
                    # Update sum_comp
                    for s in range(total_masks):
                        sum_comp[s] += comp[s][new1] + comp[s][new2] - comp[s][old1] - comp[s][old2]
                        
                    # Update contribs
                    contribs[move_i1] = sum_comp[new1]
                    contribs[move_i2] = sum_comp[new2]
                    for k in range(m):
                        if k == move_i1 or k == move_i2:
                            continue
                        contribs[k] += comp[fam[k]][new1] + comp[fam[k]][new2] - comp[fam[k]][old1] - comp[fam[k]][old2]

                current_score = sum(contribs) // 2
                
                if current_score > best_score:
                    best_score = current_score
                    best_fam = fam[:]
                    no_improve = 0
                else:
                    no_improve += 1
                    
                T *= cooling
                if T < T_min:
                    T = T_min
                    
                if no_improve > 50:
                    # Restart from best
                    fam = best_fam[:]
                    in_fam = [False] * total_masks
                    for mask in fam:
                        in_fam[mask] = True
                    for s in range(total_masks):
                        sc = 0
                        for x in fam:
                            sc += comp[s][x]
                        sum_comp[s] = sc
                    contribs = [sum_comp[fam[i]] for i in range(m)]
                    current_score = best_score
                    T = T_min * 2
                    no_improve = 0
            else:
                no_improve += 1
                T *= cooling
                if T < T_min:
                    T = T_min

        return best_fam, best_score

    # Main loop with multiple restarts
    global_deadline = time.time() + 85
    best_global_score = -1
    best_global_fam = []
    
    # First run a long SA
    init_fam = greedy_init(42)
    fam, score = local_search(init_fam, time.time() + 30)
    if score > best_global_score:
        best_global_score = score
        best_global_fam = fam[:]
        
    # Use remaining time for more restarts
    while time.time() < global_deadline:
        seed = random.randint(1, 1000000)
        init_fam = greedy_init(seed)
        # Shorter search for each restart
        remaining = global_deadline - time.time()
        if remaining < 2:
            break
        time_per_restart = min(5.0, remaining / 2)
        fam, score = local_search(init_fam, time_per_restart)
        if score > best_global_score:
            best_global_score = score
            best_global_fam = fam[:]
            # If we found a very good one, maybe stop?
            # Max possible is C(40, 2) = 780.
            # Current best is around 2.59 * 40? No, score is count.
            # Score 2.59 is the ratio? No, "raw value is the count".
            # "search score is value/seed".
            # So raw value is the count of pairs.
            # Let's check the magnitude.
            # If we have a chain of 12, pairs are 66.
            # 40 sets can have much more.
            # Max pairs in a poset of size 40 from 12 elements.
            
    result = []
    for mask in best_global_fam:
        s = []
        for b in range(n):
            if (mask >> b) & 1:
                s.append(b)
        result.append(s)
        
    return result

if __name__ == "__main__":
    res = construct()
    print(len(res))
