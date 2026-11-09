import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    n = 12
    m = 40
    total_sets = 1 << n  # 4096
    
    # Helper to count comparable pairs
    def count_comparables(family):
        count = 0
        k = len(family)
        for i in range(k):
            fi = family[i]
            for j in range(i + 1, k):
                fj = family[j]
                if (fi & fj) == fi or (fi & fj) == fj:
                    count += 1
        return count
    
    # Precompute for each set s, the count of comparable sets among all 4096
    # This helps with greedy and local search
    comp_count_all = [0] * total_sets
    for s in range(total_sets):
        cnt = 0
        for t in range(total_sets):
            if t != s and ((s & t) == s or (s & t) == t):
                cnt += 1
        comp_count_all[s] = cnt

    # Greedy initialization
    def greedy_init():
        family = []
        family_set = set()
        family.append(0)
        family_set.add(0)
        while len(family) < m:
            best_candidate = -1
            best_gain = -1
            for s in range(total_sets):
                if s in family_set:
                    continue
                gain = 0
                for f in family:
                    if (f & s) == f or (f & s) == s:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_candidate = s
            if best_candidate != -1:
                family.append(best_candidate)
                family_set.add(best_candidate)
        return family

    family = greedy_init()
    current_score = count_comparables(family)
    best_family = family[:]
    best_score = current_score

    # Local search polish
    def local_search_polish(fam, max_rounds=10, time_limit=None):
        """Full-scan local search: for each position, try all possible replacements."""
        fam = fam[:]
        fam_set = set(fam)
        score = count_comparables(fam)
        improved = True
        rounds = 0
        while improved and rounds < max_rounds:
            improved = False
            rounds += 1
            if time_limit and time.time() > time_limit:
                break
            for i in range(m):
                old_s = fam[i]
                best_new = old_s
                best_delta = 0
                # Try all possible replacements
                for s in range(total_sets):
                    if s == old_s:
                        continue
                    # Check if s is already in fam (excluding position i)
                    if s in fam_set:
                        # s is in fam; if it's at position i, that's fine, but we're replacing i
                        # so we need to check if s appears elsewhere
                        # Since fam_set contains all elements, and we're removing old_s and adding s,
                        # if s == old_s we skip. If s != old_s and s in fam_set, it's a duplicate.
                        continue
                    # Calculate delta
                    count_new = 0
                    count_old = 0
                    for j, f in enumerate(fam):
                        if j == i:
                            continue
                        if (f & s) == f or (f & s) == s:
                            count_new += 1
                        if (f & old_s) == f or (f & old_s) == old_s:
                            count_old += 1
                    delta = count_new - count_old
                    if delta > best_delta:
                        best_delta = delta
                        best_new = s
                if best_delta > 0:
                    fam_set.discard(old_s)
                    fam[i] = best_new
                    fam_set.add(best_new)
                    score += best_delta
                    improved = True
        return fam, score
    
    # Polish the greedy solution
    polished_family, polished_score = local_search_polish(family, max_rounds=5, time_limit=deadline - 10)
    if polished_score > best_score:
        best_score = polished_score
        best_family = polished_family[:]
    
    # SA with biased candidate generation
    T = 5.0
    T_min = 0.01
    alpha = 0.9998
    max_iter = 100000
    
    for restart in range(3):
        if time.time() > deadline - 5:
            break
        
        # Start from best found so far (or random for diversity)
        if restart == 0:
            fam = best_family[:]
        else:
            fam = best_family[:]
            # Perturb: swap a few random elements
            for _ in range(restart * 2):
                i = random.randint(0, m - 1)
                j = random.randint(0, m - 1)
                if i != j:
                    fam[i], fam[j] = fam[j], fam[i]
        
        cur_score = count_comparables(fam)
        fam_set = set(fam)
        
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            i = random.randint(0, m - 1)
            old_s = fam[i]
            
            # Biased candidate: pick from sets that are comparable to many existing sets
            # With probability 0.7, pick a comparable set; with 0.3, pick random
            if random.random() < 0.7:
                # Pick a random set that is comparable to at least one family member
                found = False
                for _ in range(10):  # Try up to 10 random picks
                    new_s = random.randint(0, total_sets - 1)
                    if new_s == old_s:
                        continue
                    # Check if comparable to at least one family member
                    for f in fam:
                        if f != old_s and ((f & new_s) == f or (f & new_s) == new_s):
                            found = True
                            break
                    if found:
                        break
                
                if not found:
                    new_s = random.randint(0, total_sets - 1)
            else:
                new_s = random.randint(0, total_sets - 1)
            
            if new_s == old_s:
                continue
            
            # Check duplicate
            if new_s in fam_set:
                continue
            
            # Calculate delta
            count_new = 0
            count_old = 0
            for j, f in enumerate(fam):
                if j == i:
                    continue
                if (f & new_s) == f or (f & new_s) == new_s:
                    count_new += 1
                if (f & old_s) == f or (f & old_s) == old_s:
                    count_old += 1
            
            delta = count_new - count_old
            
            if delta >= 0 or random.random() < math.exp(delta / T):
                fam_set.discard(old_s)
                fam[i] = new_s
                fam_set.add(new_s)
                cur_score += delta
                if cur_score > best_score:
                    best_score = cur_score
                    best_family = fam[:]
            
            T *= alpha
            if T < T_min:
                T = T_min
    
    # Final polish
    polished_family, polished_score = local_search_polish(best_family, max_rounds=3, time_limit=deadline - 2)
    if polished_score > best_score:
        best_score = polished_score
        best_family = polished_family[:]
    
    family = best_family

    # Convert bitmasks to lists
    result = []
    for s in family:
        subset = []
        for i in range(n):
            if (s >> i) & 1:
                subset.append(i)
        result.append(subset)
    
    return result
