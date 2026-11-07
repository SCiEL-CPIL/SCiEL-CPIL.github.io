import time
import random

def construct():
    n = 12
    m = 40
    total_masks = 1 << n
    deadline = time.time() + 85  # 5s safety margin for 90s limit

    # Precompute bit counts and maybe comparability properties if needed
    # For speed, we rely on bit operations: a & ~b == 0 means a is subset of b
    
    def is_comparable(a, b):
        if a == b:
            return False
        return (a & ~b) == 0 or (b & ~a) == 0

    def calc_score(family_list):
        score = 0
        L = len(family_list)
        for i in range(L):
            ai = family_list[i]
            for j in range(i + 1, L):
                bj = family_list[j]
                if is_comparable(ai, bj):
                    score += 1
        return score

    def greedy_construct():
        """
        Build family greedily.
        Start with high-degree nodes (sets with many comparable sets).
        """
        family = []
        used = set()
        
        # Calculate degree of each mask
        # Degree of mask m is number of masks x such that x is comparable to m
        # This is expensive to compute for all 4096 masks if done naively.
        # But we can approximate or just use size-based heuristic.
        # Let's just use a chain as a strong baseline, or random high-degree.
        
        # Let's try starting with the 40 masks with the highest "potential"
        # Potential of mask m: sum of 1 for all x in 0..2^n-1 if comparable(m,x)
        # Actually, let's just use the greedy method from the previous version 
        # but ensure we pick sets that connect well.
        
        # Start with empty set and full set?
        # Let's stick to the chain start, it's robust.
        cur = 0
        family.append(0)
        used.add(0)
        for i in range(n):
            cur |= (1 << i)
            family.append(cur)
            used.add(cur)
        
        all_masks = list(range(total_masks))
        random.shuffle(all_masks)
        
        # Greedily add remaining sets
        while len(family) < m:
            best_gain = -1
            best_mask = None
            for s in all_masks:
                if s in used:
                    continue
                gain = 0
                for f in family:
                    if is_comparable(s, f):
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_mask = s
            if best_mask is None:
                break
            family.append(best_mask)
            used.add(best_mask)
        
        return family

    def hill_climb(start_family):
        """
        Local search with single and double swaps.
        """
        family_set = set(start_family)
        current_score = calc_score(list(family_set))
        
        # Precompute comparability counts for all masks against current family
        # comp_counts[mask] = number of sets in family comparable to mask
        comp_counts = [0] * total_masks
        for a in family_set:
            for mask in range(total_masks):
                if is_comparable(a, mask):
                    comp_counts[mask] += 1
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Single swap
            best_delta = 0
            best_a = None
            best_b = None
            
            for a in family_set:
                # comp_counts[a] includes a itself? No, is_comparable(a,a) is False.
                # So comp_counts[a] is the number of OTHER sets in family comparable to a.
                # Wait, the definition above: 
                # for a in family_set: for mask... if comparable(a,mask) count++.
                # So comp_counts[a] is the number of sets in family (excluding a? no, including if comparable, but a not comparable to a) 
                # So comp_counts[a] is the degree of a within the family.
                
                for b in range(total_masks):
                    if b in family_set:
                        continue
                    # If we remove a and add b:
                    # New score = current_score - (comparables of a in S\{a}) + (comparables of b in S\{a})
                    # Comparables of a in S\{a} is comp_counts[a] (since a not comparable to itself)
                    # Comparables of b in S\{a} is:
                    #   comp_counts[b] (comparables in S) 
                    #   minus 1 if b comparable to a (because a is removed)
                    #   Note: b is not in S, so comp_counts[b] does not include b itself.
                    
                    is_comp_ab = 1 if is_comparable(a, b) else 0
                    # If b is comparable to a, then a was counted in comp_counts[b].
                    # So comparables of b in S\{a} is comp_counts[b] - is_comp_ab.
                    
                    delta = (comp_counts[b] - is_comp_ab) - comp_counts[a]
                    if delta > best_delta:
                        best_delta = delta
                        best_a = a
                        best_b = b
            
            if best_delta > 0:
                family_set.remove(best_a)
                family_set.add(best_b)
                current_score += best_delta
                
                # Update comp_counts
                # Remove best_a: decrement counts for all masks comparable to best_a
                for mask in range(total_masks):
                    if is_comparable(best_a, mask):
                        comp_counts[mask] -= 1
                # Add best_b: increment counts for all masks comparable to best_b
                for mask in range(total_masks):
                    if is_comparable(best_b, mask):
                        comp_counts[mask] += 1
                
                improved = True
                continue # Restart search after a move

            # If no single swap improvement, try 2-swap
            # This is expensive O(N^4) if done naively.
            # We can sample or limit.
            # Given time constraints, maybe skip 2-swap or do it rarely.
            # Let's try a simple 2-swap: remove 2 random, add 2 best?
            # Or just stick to single swap with random restarts if stuck.
            
            # If stuck, do a random perturbation
            if random.random() < 0.1:
                # Swap a random element with a random candidate
                a = random.choice(list(family_set))
                candidates = [x for x in range(total_masks) if x not in family_set]
                if candidates:
                    b = random.choice(candidates)
                    # Calculate delta
                    is_comp_ab = 1 if is_comparable(a, b) else 0
                    delta = (comp_counts[b] - is_comp_ab) - comp_counts[a]
                    # Accept if delta > -threshold (simulated annealing style)
                    # For simplicity, accept any move if we are stuck?
                    # Or just accept to escape.
                    family_set.remove(a)
                    family_set.add(b)
                    current_score += delta
                    
                    for mask in range(total_masks):
                        if is_comparable(a, mask):
                            comp_counts[mask] -= 1
                    for mask in range(total_masks):
                        if is_comparable(b, mask):
                            comp_counts[mask] += 1
                    improved = True

        return list(family_set), current_score

    # Generate diverse starts
    starts = []
    starts.append(greedy_construct())
    
    # Random start
    all_masks = list(range(total_masks))
    random_start = random.sample(all_masks, m)
    starts.append(random_start)
    
    # Another random start
    random_start2 = random.sample(all_masks, m)
    starts.append(random_start2)

    best_family = None
    best_score = -1

    for idx, start in enumerate(starts):
        if time.time() >= deadline - 5:
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
