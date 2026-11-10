"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill
    
    n = 128
    elements = list(range(1, n + 1))
    
    # Precompute LCM table
    def lcm(a, b):
        return a * b // math.gcd(a, b)
    
    # Build hyperedges: sets of 3 distinct elements {a,b,c} such that
    # lcm(a,b) == lcm(a,c) == lcm(b,c)
    # We'll store for each pair (a,b) the set of c's that form a bad triple
    # Actually, let's build a conflict graph approach:
    # Two elements "conflict" in the sense that they can be part of a bad triple
    # with some third element. But actually, we need to avoid any triple where
    # all three pairwise LCMs are equal.
    
    # Let's precompute all bad triples
    bad_triples = []
    triple_set = set()
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            for c in range(b + 1, n + 1):
                lab = lcm(a, b)
                lac = lcm(a, c)
                lbc = lcm(b, c)
                if lab == lac == lbc:
                    bad_triples.append((a, b, c))
                    triple_set.add((a, b, c))
    
    # For each element, store the set of other elements that form a bad triple with it
    # Actually, we need a different approach. Let's use a greedy + local search approach.
    # A set S is valid if no three elements in S form a bad triple.
    
    def is_valid_set(S):
        S_set = set(S)
        for (a, b, c) in bad_triples:
            if a in S_set and b in S_set and c in S_set:
                return False
        return True
    
    # Greedy construction with multiple orderings
    def greedy_construct(order):
        selected = []
        selected_set = set()
        for x in order:
            # Check if adding x creates any bad triple
            can_add = True
            for (a, b, c) in bad_triples:
                # Check if x is in this triple and the other two are in selected
                if x == a:
                    if b in selected_set and c in selected_set:
                        can_add = False
                        break
                elif x == b:
                    if a in selected_set and c in selected_set:
                        can_add = False
                        break
                elif x == c:
                    if a in selected_set and b in selected_set:
                        can_add = False
                        break
            if can_add:
                selected.append(x)
                selected_set.add(x)
        return selected
    
    # Local search: try to swap elements to improve the set
    def local_search(S, max_iter=200):
        S_set = set(S)
        best_S = list(S)
        best_len = len(S)
        
        for iteration in range(max_iter):
            if time.time() > deadline:
                break
            
            improved = False
            # Try removing one element and adding another
            for i in range(len(S)):
                if time.time() > deadline:
                    break
                for j in range(len(S)):
                    if i == j:
                        continue
                    # Try swapping S[i] with S[j]
                    new_S = list(S)
                    new_S[i] = S[j]
                    new_S_set = set(new_S)
                    # Check validity
                    valid = True
                    for (a, b, c) in bad_triples:
                        if a in new_S_set and b in new_S_set and c in new_S_set:
                            valid = False
                            break
                    if valid and len(new_S) > best_len:
                        best_S = new_S
                        best_len = len(new_S)
                        S = new_S
                        S_set = new_S_set
                        improved = True
                        break
                if improved:
                    break
            
            if not improved:
                break
        
        return best_S
    
    # Try multiple random orderings and keep the best
    best_result = []
    
    # First, try natural order
    result = greedy_construct(elements)
    if len(result) > len(best_result):
        best_result = result
    
    # Try reverse order
    result = greedy_construct(list(reversed(elements)))
    if len(result) > len(best_result):
        best_result = result
    
    # Try sorted by number of conflicts (elements involved in more bad triples first)
    conflict_count = {x: 0 for x in elements}
    for (a, b, c) in bad_triples:
        conflict_count[a] += 1
        conflict_count[b] += 1
        conflict_count[c] += 1
    
    # Sort by conflict count descending
    order_by_conflict_desc = sorted(elements, key=lambda x: conflict_count[x], reverse=True)
    result = greedy_construct(order_by_conflict_desc)
    if len(result) > len(best_result):
        best_result = result
    
    # Sort by conflict count ascending
    order_by_conflict_asc = sorted(elements, key=lambda x: conflict_count[x])
    result = greedy_construct(order_by_conflict_asc)
    if len(result) > len(best_result):
        best_result = result
    
    # Random restarts
    random.seed(42)
    for restart in range(50):
        if time.time() > deadline:
            break
        
        random_order = elements[:]
        random.shuffle(random_order)
        result = greedy_construct(random_order)
        if len(result) > len(best_result):
            best_result = result
    
    # Apply local search to the best result
    if time.time() < deadline:
        improved = local_search(best_result)
        if len(improved) > len(best_result):
            best_result = improved
    
    # Try local search on other good solutions
    if time.time() < deadline - 5:
        for restart in range(20):
            if time.time() > deadline:
                break
            random_order = elements[:]
            random.shuffle(random_order)
            result = greedy_construct(random_order)
            if len(result) >= len(best_result) - 2:
                improved = local_search(result, max_iter=100)
                if len(improved) > len(best_result):
                    best_result = improved
    
    return best_result
# EVOLVE-BLOCK-END
