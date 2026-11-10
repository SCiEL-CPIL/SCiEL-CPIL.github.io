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
    
    # Build unified conflict list: conflicts[x] = list of (y, z) pairs
    conflicts = {i: [] for i in range(1, N + 1)}
    for a, b, c in triples:
        conflicts[a].append((b, c))
        conflicts[b].append((a, c))
        conflicts[c].append((a, b))
    
    # Degree
    degree = [0] * (N + 1)
    for a, b, c in triples:
        degree[a] += 1
        degree[b] += 1
        degree[c] += 1
    
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

    def swap_local_search(S, max_iterations=200):
        """
        Max-SAT style local search: try swaps that reduce violations.
        A swap removes element r from S and adds element a not in S.
        We accept if the net change in violations is negative (improvement)
        or if it maintains the same number of violations but increases |S|.
        """
        if time.time() > deadline - 1:
            return S
        
        all_elements = list(range(1, N + 1))
        rng_local = random.Random(hash(frozenset(S)) % (2**32))
        
        for iteration in range(max_iterations):
            if time.time() > deadline - 1:
                return S
            
            # Try to find a beneficial swap
            best_swap = None
            best_delta = 0  # We want delta < 0 (fewer violations) or delta == 0 with size gain
            
            # For each element r in S, try swapping with elements a not in S
            # To keep it fast, sample a subset
            s_elements = list(S)
            if len(s_elements) > 30:
                s_elements = rng_local.sample(s_elements, 30)
            
            not_in_S = [x for x in all_elements if x not in S]
            if len(not_in_S) > 40:
                not_in_S = rng_local.sample(not_in_S, 40)
            
            for r in s_elements:
                # Compute violations caused by r in S
                viol_r = count_violations(r, S)
                if viol_r == 0:
                    continue  # r doesn't cause violations, no point removing
                
                for a in not_in_S:
                    # If we remove r and add a, what's the change in violations?
                    # New violations from a: count_violations(a, S - {r})
                    # Old violations from r: viol_r
                    # But also, removing r might fix violations of OTHER elements
                    # Simplified: just check if a can be added to S-{r} without new violations
                    # and if removing r reduces total violations
                    
                    # Quick check: can a be added to S without r?
                    S_temp = S - {r}
                    if not can_add(a, S_temp):
                        continue
                    
                    # Estimate: removing r removes viol_r violations
                    # Adding a adds 0 violations (since can_add passed)
                    # Net change: -viol_r (improvement)
                    # But we need to be more careful: some of r's violations might involve a
                    # Let's just accept if viol_r > 0
                    
                    delta = -viol_r  # Negative is good
                    if delta < best_delta:
                        best_delta = delta
                        best_swap = (r, a)
            
            if best_swap is None:
                # No improving swap found; try a neutral swap (same violations, same size)
                # or a slightly worsening swap with size gain
                for r in s_elements:
                    viol_r = count_violations(r, S)
                    if viol_r > 1:  # Only consider if r causes multiple violations
                        continue
                    
                    for a in not_in_S:
                        S_temp = S - {r}
                        if not can_add(a, S_temp):
                            continue
                        # This swap maintains or improves
                        best_swap = (r, a)
                        break
                    if best_swap is not None:
                        break
            
            if best_swap is not None:
                r, a = best_swap
                S.remove(r)
                S.add(a)
            else:
                break
        
        return S

    def combined_ils(S, rng, max_rounds=100):
        """Combined ILS: perturb with removals, then maximize with greedy + swap search."""
        best_S = set(S)
        best_size = len(best_S)
        
        for rnd in range(max_rounds):
            if time.time() > deadline - 2:
                break
            
            # Perturb: remove k random elements
            kick_size = rng.randint(1, 4)
            S_perturbed = set(S)
            if len(S_perturbed) > kick_size:
                elements_to_remove = rng.sample(list(S_perturbed), kick_size)
                for e in elements_to_remove:
                    S_perturbed.remove(e)
            elif len(S_perturbed) > 0:
                S_perturbed.clear()
            
            # Maximize with greedy
            order_ils = list(range(1, N + 1))
            order_ils.sort(key=lambda x: (degree[x], rng.random()))
            maximize(S_perturbed, order_ils)
            
            # Then apply swap local search
            S_perturbed = swap_local_search(S_perturbed)
            
            if len(S_perturbed) > best_size:
                best_S = set(S_perturbed)
                best_size = len(best_S)
                S = set(S_perturbed)  # Update current for next iteration
        
        return best_S

    best_set = set()
    best_size = 0
    rng = random.Random(42)
    
    all_elements = list(range(1, N + 1))
    
    # Multiple diverse initial orderings
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
    
    # Random orderings
    for _ in range(6):
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
    
    # ILS with combined perturb + greedy + swap
    S = set(best_set)
    S = combined_ils(S, rng, max_rounds=200)
    
    if len(S) > best_size:
        best_set = set(S)
        best_size = len(S)
    
    # Additional ILS rounds with different perturbation strengths
    for kick_range in [(2, 5), (3, 7), (1, 3)]:
        if time.time() > deadline - 5:
            break
        S = set(best_set)
        for _ in range(50):
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
                best_size = len(best_size)
                S = set(S_perturbed)
    
    # Final verification
    if not is_valid(best_set):
        # Fallback
        best_set = set(range(91, N + 1))
        if not is_valid(best_set):
            best_set = {180}
    
    return list(best_set)

# EVOLVE-BLOCK-END
