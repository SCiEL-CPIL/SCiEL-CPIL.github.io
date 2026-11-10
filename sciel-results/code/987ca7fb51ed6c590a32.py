import time
import random
import sys

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    N = 180
    
    # Precompute forbidden triples (a, b, c) with a < b < c and 1/a = 1/b + 1/c
    # a = bc / (b+c), condition: (b+c) divides (b*c)
    triples = []
    for b in range(1, N + 1):
        for c in range(b + 1, N + 1):
            s = b + c
            prod = b * c
            if prod % s == 0:
                a = prod // s
                if 1 <= a < b:
                    triples.append((a, b, c))
    
    # Build conflict structure: for each element, which pairs of other elements conflict with it
    # conflicts[x] = list of (y, z) such that {x, y, z} forms a forbidden triple
    conflicts = {i: [] for i in range(1, N + 1)}
    for a, b, c in triples:
        conflicts[a].append((b, c))
        conflicts[b].append((a, c))
        conflicts[c].append((a, b))
    
    # Degree: number of triples each element participates in
    degree = [0] * (N + 1)
    for a, b, c in triples:
        degree[a] += 1
        degree[b] += 1
        degree[c] += 1
    
    # Pair conflict: two elements x, y conflict if there exists z such that {x,y,z} is a triple
    # This is useful for understanding the structure
    pair_conflicts = {}
    for a, b, c in triples:
        pair_conflicts[(a, b)] = c
        pair_conflicts[(b, a)] = c
        pair_conflicts[(a, c)] = b
        pair_conflicts[(c, a)] = b
        pair_conflicts[(b, c)] = a
        pair_conflicts[(c, b)] = a
    
    def can_add(x, S):
        """Check if x can be added to set S without creating a forbidden triple."""
        for (y, z) in conflicts[x]:
            if y in S and z in S:
                return False
        return True

    def is_valid(S):
        """Verify S has no forbidden triple."""
        for x in S:
            for (y, z) in conflicts[x]:
                if y in S and z in S:
                    return False
        return True

    def count_violations(x, S):
        """Count how many triples involving x are violated if x is in S."""
        count = 0
        for (y, z) in conflicts[x]:
            if y in S and z in S:
                count += 1
        return count

    def maximize(S, order):
        """Greedy addition until no more elements can be added."""
        improved = True
        while improved:
            if time.time() > deadline:
                return S
            improved = False
            for x in order:
                if x not in S:
                    if can_add(x, S):
                        S.add(x)
                        improved = True
        return S

    def greedy_with_order(order):
        """Construct a set using a specific order."""
        S = set()
        for x in order:
            if can_add(x, S):
                S.add(x)
        return S

    def greedy_min_degree():
        """Subtractive greedy: start with all elements, remove those causing most conflicts."""
        S = set(range(1, N + 1))
        # Repeatedly remove the element that causes the most violations
        while True:
            if time.time() > deadline:
                break
            worst = -1
            worst_count = 0
            for x in S:
                c = count_violations(x, S)
                if c > worst_count:
                    worst_count = c
                    worst = x
            if worst_count == 0:
                break
            S.remove(worst)
        return S

    def swap_local_search(S, max_iterations=100):
        """
        Local search: try swaps that improve the set.
        """
        if time.time() > deadline - 1:
            return S
        
        all_elements = list(range(1, N + 1))
        rng_local = random.Random(hash(frozenset(S)) % (2**32))
        
        for iteration in range(max_iterations):
            if time.time() > deadline - 1:
                return S
            
            best_swap = None
            best_delta = 0  # We want to increase size while maintaining validity
            
            s_elements = list(S)
            if len(s_elements) > 25:
                s_elements = rng_local.sample(s_elements, 25)
            
            not_in_S = [x for x in all_elements if x not in S]
            if len(not_in_S) > 35:
                not_in_S = rng_local.sample(not_in_S, 35)
            
            for r in s_elements:
                for a in not_in_S:
                    S_temp = S - {r}
                    if can_add(a, S_temp):
                        # This swap is valid; accept it (same size, but might enable future additions)
                        best_swap = (r, a)
                        break
                if best_swap is not None:
                    break
            
            if best_swap is None:
                break
            
            r, a = best_swap
            S.remove(r)
            S.add(a)
            
            # After swap, try to add more elements
            order = sorted(all_elements, key=lambda x: (degree[x], rng_local.random()))
            maximize(S, order)
        
        return S

    def combined_ils(S, rng, max_rounds=80):
        """Combined ILS: perturb with removals, then maximize with greedy + swap search."""
        best_S = set(S)
        best_size = len(best_S)
        
        for rnd in range(max_rounds):
            if time.time() > deadline - 2:
                break
            
            # Perturb: remove k random elements
            kick_size = rng.randint(1, 5)
            S_perturbed = set(S)
            if len(S_perturbed) > kick_size:
                elements_to_remove = rng.sample(list(S_perturbed), kick_size)
                for e in elements_to_remove:
                    S_perturbed.remove(e)
            
            # Maximize with greedy using min-degree order
            order_ils = list(range(1, N + 1))
            order_ils.sort(key=lambda x: (degree[x], rng.random()))
            maximize(S_perturbed, order_ils)
            
            # Then apply swap local search
            S_perturbed = swap_local_search(S_perturbed)
            
            if len(S_perturbed) > best_size:
                best_S = set(S_perturbed)
                best_size = len(best_S)
                S = set(S_perturbed)
        
        return best_S

    best_set = set()
    best_size = 0
    rng = random.Random(42)
    
    all_elements = list(range(1, N + 1))
    
    # Strategy 1: Subtractive greedy (min vertex cover approximation)
    S_sub = greedy_min_degree()
    if len(S_sub) > best_size:
        best_set = set(S_sub)
        best_size = len(S_sub)
    
    # Strategy 2: Multiple additive greedy orderings
    orderings = [
        sorted(all_elements, key=lambda x: degree[x]),           # degree asc
        sorted(all_elements, key=lambda x: -degree[x]),          # degree desc
        all_elements,                                             # ascending
        sorted(all_elements, reverse=True),                       # descending
    ]
    
    for order in orderings:
        if time.time() > deadline - 20:
            break
        S = greedy_with_order(order)
        maximize(S, order)
        S = swap_local_search(S)
        if len(S) > best_size:
            best_set = set(S)
            best_size = len(S)
    
    # Strategy 3: Random orderings
    for _ in range(8):
        if time.time() > deadline - 15:
            break
        order_rand = list(all_elements)
        rng.shuffle(order_rand)
        Sr = greedy_with_order(order_rand)
        maximize(Sr, order_rand)
        Sr = swap_local_search(Sr)
        if len(Sr) > best_size:
            best_set = set(Sr)
            best_size = len(Sr)
    
    # Strategy 4: ILS
    S = set(best_set)
    S = combined_ils(S, rng, max_rounds=150)
    
    if len(S) > best_size:
        best_set = set(S)
        best_size = len(S)
    
    # Strategy 5: Additional ILS with different perturbation
    for kick_range in [(2, 6), (3, 8), (1, 4)]:
        if time.time() > deadline - 5:
            break
        S = set(best_set)
        for _ in range(40):
            if time.time() > deadline - 3:
                break
            kick_size = rng.randint(kick_range[0], kick_range[1])
            S_perturbed = set(S)
            if len(S_perturbed) > kick_size:
                elements_to_remove = rng.sample(list(S_perturbed), kick_size)
                for e in elements_to_remove:
                    S_perturbed.remove(e)
            
            order_ils = list(range(1, N + 1))
            order_ils.sort(key=lambda x: (degree[x], rng.random()))
            maximize(S_perturbed, order_ils)
            S_perturbed = swap_local_search(S_perturbed)
            
            if len(S_perturbed) > best_size:
                best_set = set(S_perturbed)
                best_size = len(S_perturbed)
                S = set(S_perturbed)
    
    # Final verification
    if not is_valid(best_set):
        # Fallback: use subtractive greedy result
        S_fallback = greedy_min_degree()
        if is_valid(S_fallback) and len(S_fallback) > 0:
            best_set = S_fallback
        else:
            best_set = set(range(91, N + 1))
            if not is_valid(best_set):
                best_set = {180}
    
    return list(best_set)

# EVOLVE-BLOCK-END
