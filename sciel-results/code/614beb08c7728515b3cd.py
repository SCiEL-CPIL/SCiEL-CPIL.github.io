"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd, lcm
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    N = 128
    all_elements = list(range(1, N + 1))
    
    # Pre-compute LCM table for all pairs
    lcm_table = {}
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            lcm_table[(i, j)] = lcm(i, j)
    
    # Find all bad triples: (a, b, c) with a < b < c where lcm(a,b) == lcm(a,c) == lcm(b,c)
    bad_triples = []
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            lab = lcm_table[(a, b)]
            for c in range(b + 1, N + 1):
                if lcm_table[(a, c)] == lab and lcm_table[(b, c)] == lab:
                    bad_triples.append((a, b, c))
    
    if time.time() > deadline:
        return list(range(1, N + 1))
    
    # Build conflict map: for each element v, the set of pairs (u, w) such that {v, u, w} is bad
    conflict_map = defaultdict(set)
    degree = defaultdict(int)
    for a, b, c in bad_triples:
        conflict_map[a].add((b, c))
        conflict_map[b].add((a, c))
        conflict_map[c].add((a, b))
        degree[a] += 1
        degree[b] += 1
        degree[c] += 1
    
    # Helper: check if adding element v to selected_set creates a bad triple
    def can_add(v, selected_set):
        for u, w in conflict_map[v]:
            if u in selected_set and w in selected_set:
                return False
        return True
    
    # Helper: check if selected_set is valid (no bad triples)
    def is_valid(selected_set):
        for v in selected_set:
            for u, w in conflict_map[v]:
                if u in selected_set and w in selected_set:
                    # Check if v is the smallest to avoid redundant checks, 
                    # but actually any violation is a violation.
                    # We just need to ensure no triple exists.
                    return False
        return True

    # Strategy: Greedy Independent Set on Conflict Hypergraph
    # Sort elements by degree (ascending) - least constrained first
    sorted_elements = sorted(all_elements, key=lambda x: degree[x])
    
    selected_set = set()
    
    for v in sorted_elements:
        if time.time() > deadline:
            break
        if can_add(v, selected_set):
            selected_set.add(v)
            
    # Local Search: 1-swap improvement
    # Try to replace an element in selected_set with an element outside to increase size?
    # No, 1-swap keeps size same. We want to maximize cardinality.
    # The greedy approach already maximizes cardinality for that specific order.
    # To improve, we can try random restarts or different orderings.
    
    best_set = set(selected_set)
    best_size = len(best_set)
    
    # Try a few random restarts with different ordering perturbations if time permits
    import random
    random.seed(42)
    
    for _ in range(50):
        if time.time() > deadline:
            break
        
        # Perturb the order: sort by degree + random noise
        def noisy_key(x):
            return degree[x] + random.random()
            
        perturbed_elements = sorted(all_elements, key=noisy_key)
        
        current_set = set()
        for v in perturbed_elements:
            if can_add(v, current_set):
                current_set.add(v)
                
        if len(current_set) > best_size:
            best_set = current_set
            best_size = len(best_set)
            
    # One more pass: try to add any remaining elements if possible (greedy might have skipped due to order)
    # Actually, the greedy loop adds if possible. So if it skipped, it wasn't possible *at that time*.
    # But maybe removing one element allows adding two? That's a 1-for-2 swap.
    
    # Improved local search: 1-for-2 swaps and 2-for-3 swaps
    if time.time() < deadline - 5.0:
        current_best = set(best_set)
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            
            # 1-for-2 swap: remove one element, try to add two
            for v in list(current_best):
                if time.time() > deadline - 2.0:
                    break
                test_set = current_best - {v}
                # Find candidates that can be added to test_set
                candidates = []
                for w in all_elements:
                    if w in test_set:
                        continue
                    if can_add(w, test_set):
                        candidates.append(w)
                        
                # If we can add 2 distinct candidates, we improve size by 1
                if len(candidates) >= 2:
                    # Try all pairs of candidates (not just first valid)
                    found_pair = False
                    for i in range(len(candidates)):
                        if found_pair:
                            break
                        for j in range(i + 1, len(candidates)):
                            w1, w2 = candidates[i], candidates[j]
                            if can_add(w1, test_set) and can_add(w2, test_set | {w1}):
                                new_set = test_set | {w1, w2}
                                if len(new_set) > len(current_best):
                                    current_best = new_set
                                    improved = True
                                    found_pair = True
                                    break
                                    
            # 2-for-3 swap: remove two elements, try to add three
            if time.time() < deadline - 2.0:
                for i, v1 in enumerate(list(current_best)):
                    if time.time() > deadline - 2.0:
                        break
                    for v2 in list(current_best):
                        if v2 <= v1:
                            continue
                        test_set = current_best - {v1, v2}
                        # Find candidates that can be added to test_set
                        candidates = []
                        for w in all_elements:
                            if w in test_set:
                                continue
                            if can_add(w, test_set):
                                candidates.append(w)
                        
                        # If we can add 3 distinct candidates, we improve size by 1
                        if len(candidates) >= 3:
                            found_triple = False
                            for a in range(len(candidates)):
                                if found_triple:
                                    break
                                for b in range(a + 1, len(candidates)):
                                    if found_triple:
                                        break
                                    for c in range(b + 1, len(candidates)):
                                        w1, w2, w3 = candidates[a], candidates[b], candidates[c]
                                        if (can_add(w1, test_set) and 
                                            can_add(w2, test_set | {w1}) and 
                                            can_add(w3, test_set | {w1, w2})):
                                            new_set = test_set | {w1, w2, w3}
                                            if len(new_set) > len(current_best):
                                                current_best = new_set
                                                improved = True
                                                found_triple = True
                                                break
                                                
        if len(current_best) > len(best_set):
            best_set = current_best
            best_size = len(best_set)

    # Ensure non-empty
    if not best_set:
        best_set = {1}
        
    return list(best_set)

# EVOLVE-BLOCK-END
