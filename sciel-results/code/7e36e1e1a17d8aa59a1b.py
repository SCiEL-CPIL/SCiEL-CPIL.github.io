import time
import random
from collections import defaultdict

def construct():
    deadline = time.time() + 85  # safety margin before 90s kill
    
    N = 180
    
    # Precompute all forbidden triples: (a, b, c) with 1/a = 1/b + 1/c, all distinct
    # 1/a = 1/b + 1/c => c = ab/(b-a), requiring b > a, (b-a) | ab
    # Also need a < c (since 1/a > 1/c means a < c)
    # And b != c (distinct)
    
    # For each a from 1 to N, for each b from a+1 to N:
    #   d = b - a
    #   if ab % d == 0: c = ab // d
    #   if 1 <= c <= N and c != a and c != b: record triple (a, min(b,c), max(b,c))
    
    # Store conflicts: for each element, which other elements form a triple with it
    # triple_conflicts[i] = set of (j, k) such that {i, j, k} is a forbidden triple
    # Actually, let's store: for each pair (i, j), which k values would complete a triple
    
    # Better: store all triples as frozensets, and for each element, the set of elements it conflicts with
    # An element x conflicts with y if there exists z such that {x,y,z} is a forbidden triple
    
    # Actually for the constraint: we need NO three distinct elements forming a triple.
    # So if we include x and y, we cannot include any z that completes a triple with x and y.
    
    # Let's precompute: triples_list = list of (a, b, c) sorted
    # And conflict_map: dict mapping element -> set of elements that form a triple with it (with some third element)
    
    triples = []
    # For each a, b with b > a:
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            d = b - a
            if (a * b) % d == 0:
                c = (a * b) // d
                if 1 <= c <= N and c != a and c != b:
                    # Sort the triple for consistent storage
                    t = tuple(sorted([a, b, c]))
                    triples.append(t)
    
    # Remove duplicates
    triples = list(set(triples))
    
    # Build conflict structure
    # For each element, the set of other elements that appear in a triple with it
    elem_conflicts = defaultdict(set)
    for (a, b, c) in triples:
        elem_conflicts[a].add(b)
        elem_conflicts[a].add(c)
        elem_conflicts[b].add(a)
        elem_conflicts[b].add(c)
        elem_conflicts[c].add(a)
        elem_conflicts[c].add(b)
    
    # Also build: for each pair (x, y), the set of z such that {x,y,z} is a triple
    pair_to_third = defaultdict(set)
    for (a, b, c) in triples:
        pair_to_third[(a, b)].add(c)
        pair_to_third[(b, a)].add(c)
        pair_to_third[(a, c)].add(b)
        pair_to_third[(c, a)].add(b)
        pair_to_third[(b, c)].add(a)
        pair_to_third[(c, b)].add(a)
    
    def is_valid(S):
        """Check if set S has no forbidden triple."""
        S_set = set(S)
        for (a, b, c) in triples:
            if a in S_set and b in S_set and c in S_set:
                return False
        return True
    
    def count_triples_in(S):
        """Count number of forbidden triples fully contained in S."""
        S_set = set(S)
        count = 0
        for (a, b, c) in triples:
            if a in S_set and b in S_set and c in S_set:
                count += 1
        return count
    
    # Greedy approach: start with all elements, remove those in most triples
    # Or: build up greedily
    
    # Strategy 1: Greedy removal
    # Start with all 180 elements, repeatedly remove the element that is in the most remaining triples
    
    def greedy_remove():
        S = set(range(1, N + 1))
        # Count how many triples each element is in
        elem_triple_count = defaultdict(int)
        for (a, b, c) in triples:
            elem_triple_count[a] += 1
            elem_triple_count[b] += 1
            elem_triple_count[c] += 1
        
        while True:
            if time.time() > deadline:
                break
            # Find element with max triple count
            if not S:
                break
            max_count = -1
            max_elem = None
            for e in S:
                if elem_triple_count[e] > max_count:
                    max_count = elem_triple_count[e]
                    max_elem = e
            if max_count == 0:
                break
            # Remove max_elem
            S.remove(max_elem)
            # Update counts for all triples containing max_elem
            for (a, b, c) in triples:
                if max_elem in (a, b, c):
                    for e in (a, b, c):
                        if e != max_elem and e in S:
                            elem_triple_count[e] -= 1
                    elem_triple_count[max_elem] = 0  # already removed
        
        return S
    
    # Strategy 2: Greedy build-up
    # Process elements in some order, add if it doesn't create a triple
    
    def greedy_build(order):
        S = set()
        for e in order:
            # Check if adding e creates any triple
            conflict = False
            for (a, b, c) in triples:
                if e in (a, b, c):
                    others = [x for x in (a, b, c) if x != e]
                    if all(o in S for o in others):
                        conflict = True
                        break
            if not conflict:
                S.add(e)
        return S
    
    # Strategy 3: Local search from a seed
    def local_search(S, max_iters=50000):
        S = set(S)
        best = S
        best_size = len(S)
        
        # Precompute for each element not in S, which elements in S would conflict
        # For adding element x: check if there's any triple (x, y, z) with y, z in S
        # For removing element x: always safe
        # For swapping: remove x from S, add y not in S
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            
            improved = False
            
            # Try adding elements
            for x in range(1, N + 1):
                if x not in S:
                    # Check if adding x creates a triple
                    can_add = True
                    for (a, b, c) in triples:
                        if x in (a, b, c):
                            others = [o for o in (a, b, c) if o != x]
                            if all(o in S for o in others):
                                can_add = False
                                break
                    if can_add:
                        S.add(x)
                        improved = True
                        if len(S) > best_size:
                            best = set(S)
                            best_size = len(S)
            
            # Try removing elements that are in many triples, then re-adding others
            # Actually, let's try: for each x in S, temporarily remove, see if we can add more
            # This is expensive. Let's do simple swaps.
            
            # Try swaps: remove x from S, add y not in S
            # Only do this if it might help (x is in many triples, y is in few)
            
            if not improved and len(S) < N:
                # Try random swaps
                in_S = list(S)
                not_in_S = [x for x in range(1, N + 1) if x not in S]
                if in_S and not_in_S:
                    for _ in range(10):  # try a few random swaps
                        if time.time() > deadline:
                            break
                        x = random.choice(in_S)
                        y = random.choice(not_in_S)
                        S_test = S - {x} | {y}
                        if is_valid(S_test):
                            S = S_test
                            in_S = list(S)
                            not_in_S = [z for z in range(1, N + 1) if z not in S]
                            if len(S) > best_size:
                                best = set(S)
                                best_size = len(S)
                            improved = True
                            break
        
        return best
    
    # Run greedy removal first
    result = greedy_remove()
    best_result = set(result)
    best_size = len(result)
    
    # Try local search from this result
    if time.time() < deadline - 10:
        ls_result = local_search(result, max_iters=20000)
        if len(ls_result) > best_size:
            best_result = ls_result
            best_size = len(ls_result)
    
    # Try greedy build with different orders
    if time.time() < deadline - 15:
        # Order by increasing degree (fewest conflicts first)
        degree = {e: len(elem_conflicts[e]) for e in range(1, N + 1)}
        order_by_degree = sorted(range(1, N + 1), key=lambda e: degree[e])
        gb_result = greedy_build(order_by_degree)
        if len(gb_result) > best_size:
            best_result = gb_result
            best_size = len(gb_result)
        
        # Order by decreasing degree
        order_by_degree_desc = sorted(range(1, N + 1), key=lambda e: -degree[e])
        gb_result2 = greedy_build(order_by_degree_desc)
        if len(gb_result2) > best_size:
            best_result = gb_result2
            best_size = len(gb_result2)
        
        # Random orders
        for _ in range(5):
            if time.time() > deadline - 5:
                break
            random_order = list(range(1, N + 1))
            random.shuffle(random_order)
            gb_result3 = greedy_build(random_order)
            if len(gb_result3) > best_size:
                best_result = gb_result3
                best_size = len(gb_result3)
    
    # Final local search on best result
    if time.time() < deadline - 5:
        final_result = local_search(best_result, max_iters=30000)
        if len(final_result) > best_size:
            best_result = final_result
            best_size = len(final_result)
    
    return list(best_result)
