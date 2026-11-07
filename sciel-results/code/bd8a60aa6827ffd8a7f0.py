import time
import random

def construct():
    n = 12
    m = 40
    total_masks = 1 << n
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin before 90s limit

    def time_left():
        return deadline - time.time()

    # Precompute subset/superset relationships using bit tricks
    # For each mask, precompute all its subsets and supersets
    subsets_of = [[] for _ in range(total_masks)]
    supersets_of = [[] for _ in range(total_masks)]
    
    # For each mask, find all its subsets
    for a in range(total_masks):
        # All subsets of a: iterate through submasks
        sub = a
        while True:
            subsets_of[a].append(sub)
            if sub == 0:
                break
            sub = (sub - 1) & a
    
    # For each mask, find all its supersets
    for a in range(total_masks):
        # All supersets of a: masks s where (a & ~s) == 0
        # Equivalent: s has all bits of a
        for s in range(total_masks):
            if (a & ~s) == 0:
                supersets_of[a].append(s)

    # Build comparability: comp[a] is a set of masks comparable to a
    # Two masks a, b are comparable if a is subset of b or b is subset of a (and a != b)
    comp_table = [set() for _ in range(total_masks)]
    for a in range(total_masks):
        # a is comparable to all its proper subsets and proper supersets
        for s in subsets_of[a]:
            if s != a:
                comp_table[a].add(s)
        for s in supersets_of[a]:
            if s != a:
                comp_table[a].add(s)

    def calc_score(family_list):
        """Calculate number of comparable pairs."""
        score = 0
        L = len(family_list)
        for i in range(L):
            for j in range(i + 1, L):
                if family_list[j] in comp_table[family_list[i]]:
                    score += 1
        return score

    def greedy_construct(start_chain=None):
        """Build family greedily."""
        if start_chain is None:
            # Default: full chain
            start_chain = []
            cur = 0
            start_chain.append(cur)
            for i in range(n):
                cur |= (1 << i)
                start_chain.append(cur)
        
        family = list(start_chain)
        used = set(family)
        
        while len(family) < m and time_left() > 3:
            best_gain = -1
            best_mask = None
            for s in range(total_masks):
                if s in used:
                    continue
                gain = 0
                for f in family:
                    if s in comp_table[f]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_mask = s
            if best_mask is None:
                break
            family.append(best_mask)
            used.add(best_mask)
        
        return family

    def hill_climb(start_family, max_iterations=500):
        """Local search with swap moves."""
        family_set = set(start_family)
        fam_list = list(family_set)
        current_score = calc_score(fam_list)
        
        improved = True
        iterations = 0
        while improved and time_left() > 2 and iterations < max_iterations:
            improved = False
            iterations += 1
            
            best_delta = 0
            best_a = None
            best_b = None
            
            # Precompute comp counts
            comp_counts = {}
            for a in fam_list:
                count = 0
                for b in fam_list:
                    if a != b and b in comp_table[a]:
                        count += 1
                comp_counts[a] = count
            
            for a in fam_list:
                comp_a = comp_counts[a]
                for b in range(total_masks):
                    if b in family_set:
                        continue
                    # If we remove a and add b, what's the delta?
                    # Remove a: lose comp_a comparable pairs
                    # Add b: gain comparable pairs with remaining family
                    new_count = 0
                    for f in fam_list:
                        if f != a and b in comp_table[f]:
                            new_count += 1
                    # Also check if b is comparable to a (it doesn't matter since a is removed)
                    delta = new_count - comp_a
                    if delta > best_delta:
                        best_delta = delta
                        best_a = a
                        best_b = b
            
            if best_delta > 0:
                family_set.remove(best_a)
                family_set.add(best_b)
                fam_list = list(family_set)
                current_score += best_delta
                improved = True
        
        return list(family_set), current_score

    def two_opt(family_list):
        """Try swapping pairs of elements."""
        family_set = set(family_list)
        fam_list = list(family_set)
        current_score = calc_score(fam_list)
        
        improved = True
        while improved and time_left() > 2:
            improved = False
            best_delta = 0
            best_a1 = None
            best_a2 = None
            best_b1 = None
            best_b2 = None
            
            for i in range(len(fam_list)):
                for j in range(i + 1, len(fam_list)):
                    a1 = fam_list[i]
                    a2 = fam_list[j]
                    for b1 in range(total_masks):
                        if b1 in family_set:
                            continue
                        for b2 in range(total_masks):
                            if b2 in family_set or b2 == b1:
                                continue
                            # Calculate delta for swapping (a1,a2) with (b1,b2)
                            new_family = [x for x in fam_list if x != a1 and x != a2] + [b1, b2]
                            new_score = calc_score(new_family)
                            delta = new_score - current_score
                            if delta > best_delta:
                                best_delta = delta
                                best_a1 = a1
                                best_a2 = a2
                                best_b1 = b1
                                best_b2 = b2
            
            if best_delta > 0:
                family_set.remove(best_a1)
                family_set.remove(best_a2)
                family_set.add(best_b1)
                family_set.add(best_b2)
                fam_list = list(family_set)
                current_score += best_delta
                improved = True
        
        return list(family_set), current_score

    # Generate multiple diverse starts
    starts = []
    
    # Start 1: Full chain + greedy fill
    starts.append(greedy_construct())
    
    # Start 2: Multiple short chains
    family2 = []
    used2 = set()
    # Chain A: 0, 1, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047, 4095
    chain_a = [0, 1, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047, 4095]
    for c in chain_a:
        if c < total_masks and c not in used2:
            family2.append(c)
            used2.add(c)
    # Chain B: 2, 6, 14, 30, 62, 126, 254, 510, 1022, 2046
    chain_b = [2, 6, 14, 30, 62, 126, 254, 510, 1022, 2046]
    for c in chain_b:
        if c < total_masks and c not in used2:
            family2.append(c)
            used2.add(c)
    # Chain C: 4, 12, 28, 60, 124, 252, 508, 1020
    chain_c = [4, 12, 28, 60, 124, 252, 508, 1020]
    for c in chain_c:
        if c < total_masks and c not in used2:
            family2.append(c)
            used2.add(c)
    # Fill rest greedily
    while len(family2) < m and time_left() > 3:
        best_gain = -1
        best_mask = None
        for s in range(total_masks):
            if s in used2:
                continue
            gain = 0
            for f in family2:
                if s in comp_table[f]:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_mask = s
        if best_mask is None:
            break
        family2.append(best_mask)
        used2.add(best_mask)
    starts.append(family2)
    
    # Start 3: Random perturbation of greedy
    base = greedy_construct()
    random_start = base[:]
    random.seed(42)
    for _ in range(5):
        i = random.randint(0, m - 1)
        candidates = [x for x in range(total_masks) if x not in set(random_start)]
        if candidates:
            random_start[i] = random.choice(candidates)
    starts.append(random_start)

    # Start 4: Chain with different structure
    family4 = []
    used4 = set()
    # Use all sets of size <= 3, then fill with larger ones
    for size in range(4):
        from itertools import combinations
        for combo in combinations(range(n), size):
            mask = 0
            for v in combo:
                mask |= (1 << v)
            if mask not in used4:
                family4.append(mask)
                used4.add(mask)
            if len(family4) >= m:
                break
        if len(family4) >= m:
            break
    # Fill rest greedily
    while len(family4) < m and time_left() > 3:
        best_gain = -1
        best_mask = None
        for s in range(total_masks):
            if s in used4:
                continue
            gain = 0
            for f in family4:
                if s in comp_table[f]:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_mask = s
        if best_mask is None:
            break
        family4.append(best_mask)
        used4.add(best_mask)
    starts.append(family4)

    best_family = None
    best_score = -1

    for idx, start in enumerate(starts):
        if time_left() < 5:
            break
        fam, score = hill_climb(start)
        if score > best_score:
            best_score = score
            best_family = fam

    if best_family is None:
        best_family = greedy_construct()

    # Convert to lists of vertices
    result = []
    for mask in best_family:
        result.append([b for b in range(n) if (mask >> b) & 1])
    
    return result
