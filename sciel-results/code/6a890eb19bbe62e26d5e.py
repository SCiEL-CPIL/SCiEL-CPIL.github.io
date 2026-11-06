import time
import random
from collections import defaultdict
import numpy as np
from scipy.optimize import linprog

def construct():
    deadline = time.time() + 55  # Safety margin for 60s limit
    
    N = 180
    
    # 1. Precompute all forbidden triples (a, b, c) distinct in [1, N] with 1/a = 1/b + 1/c
    # 1/a = 1/b + 1/c => c = ab / (b - a) for b > a
    # We need b > a. Also c must be distinct from a and b.
    # Note: The equation is symmetric in b and c.
    # So we can just iterate a, b and compute c.
    triples = []
    seen = set()
    
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            d = b - a
            if (a * b) % d == 0:
                c = (a * b) // d
                if 1 <= c <= N and c != a and c != b:
                    t = tuple(sorted([a, b, c]))
                    if t not in seen:
                        seen.add(t)
                        triples.append(t)
    
    # Build adjacency for fast conflict checking
    # element_triples[x] = list of triple indices containing x
    element_triples = defaultdict(list)
    for i, t in enumerate(triples):
        for x in t:
            element_triples[x].append(i)
    
    def is_valid(S):
        S_set = set(S)
        for t in triples:
            if all(x in S_set for x in t):
                return False
        return True
        
    def count_violations(S):
        S_set = set(S)
        count = 0
        for t in triples:
            if all(x in S_set for x in t):
                count += 1
        return count

    def repair_set(S):
        """Remove elements to make S valid. Prioritize removing elements with high 'conflict weight'."""
        S = set(S)
        # Repeatedly find a violated triple and remove the element in it that is in the most violated triples
        while True:
            violated = []
            for t in triples:
                if all(x in S for x in t):
                    violated.append(t)
            if not violated:
                break
            
            # Count how many violated triples each element in S is in
            elem_viol = defaultdict(int)
            for t in violated:
                for x in t:
                    if x in S:
                        elem_viol[x] += 1
            
            if not elem_viol:
                break
                
            # Remove the element with the highest count
            # Tie-breaker: remove smaller index? Or larger? 
            # Usually removing "central" elements (high degree) is good.
            best_elem = max(elem_viol.keys(), key=lambda x: (elem_viol[x], -x))
            S.discard(best_elem)
            
        return S

    def local_search(S, max_iters=2000):
        """
        Simulated Annealing / Hill Climbing.
        Moves:
        1. Remove one element, try to add any compatible removed element.
        2. Remove two elements, try to add two compatible removed elements.
        """
        S = set(S)
        best_set = set(S)
        best_size = len(S)
        current_temp = 1.0
        cooling_rate = 0.99
        
        for i in range(max_iters):
            if time.time() > deadline - 5:
                break
                
            # Generate a neighbor
            move_type = random.random()
            neighbor = set(S)
            
            if move_type < 0.7:
                # Single removal, try to add one
                if not S:
                    continue
                rem_elem = random.choice(list(S))
                neighbor.discard(rem_elem)
                
                # Try to add any element from the "removed" pool
                removed_pool = set(range(1, N + 1)) - neighbor
                if removed_pool:
                    # Shuffle for diversity
                    candidates = list(removed_pool)
                    random.shuffle(candidates)
                    for add_elem in candidates:
                        ok = True
                        for tri_idx in element_triples[add_elem]:
                            t = triples[tri_idx]
                            if all(y in neighbor for y in t if y != add_elem):
                                ok = False
                                break
                        if ok:
                            neighbor.add(add_elem)
                            break
            else:
                # Double removal, try to add two
                if len(S) < 2:
                    continue
                rem_elems = random.sample(list(S), 2)
                for r in rem_elems:
                    neighbor.discard(r)
                
                removed_pool = set(range(1, N + 1)) - neighbor
                if len(removed_pool) >= 2:
                    candidates = list(removed_pool)
                    random.shuffle(candidates)
                    added = 0
                    for add_elem in candidates:
                        if added == 2:
                            break
                        ok = True
                        for tri_idx in element_triples[add_elem]:
                            t = triples[tri_idx]
                            if all(y in neighbor for y in t if y != add_elem):
                                ok = False
                                break
                        if ok:
                            neighbor.add(add_elem)
                            added += 1

            new_size = len(neighbor)
            if new_size > best_size:
                best_set = set(neighbor)
                best_size = new_size
                S = neighbor
            elif new_size == best_size:
                # Accept equal size to escape plateaus?
                if random.random() < 0.5:
                    S = neighbor
            else:
                # Accept worse solution with probability exp(-delta/T)
                delta = best_size - new_size
                if delta == 0:
                    S = neighbor
                else:
                    prob = np.exp(-delta / current_temp)
                    if random.random() < prob:
                        S = neighbor
            
            current_temp *= cooling_rate
            
        return best_set

    # Strategy 1: Greedy Removal (from full set)
    S_greedy_remove = set(range(1, N + 1))
    S_greedy_remove = repair_set(S_greedy_remove)
    best_set = set(S_greedy_remove)
    best_size = len(S_greedy_remove)

    # Strategy 2: LP Relaxation
    try:
        # Variables x_1 ... x_N
        # Maximize sum(x)
        # Constraints: sum(x_i for i in T) <= 2 for each triple T
        # 0 <= x_i <= 1
        
        c = -np.ones(N) # Minimize -sum(x)
        
        A_ub = []
        b_ub = []
        for t in triples:
            row = np.zeros(N)
            for idx in t:
                row[idx-1] = 1.0
            A_ub.append(row)
            b_ub.append(2.0)
            
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        
        bounds = [(0, 1) for _ in range(N)]
        
        # Solve LP
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if res.success:
            x_vals = res.x
            # Round: x_i > 0.5
            S_lp = set(i+1 for i in range(N) if x_vals[i] > 0.5)
            S_lp_repaired = repair_set(S_lp)
            
            if len(S_lp_repaired) > best_size:
                best_set = S_lp_repaired
                best_size = len(S_lp_repaired)
                
            # Also try rounding at 0.4, 0.6 etc?
            for thresh in [0.4, 0.6, 0.3]:
                S_lp_t = set(i+1 for i in range(N) if x_vals[i] > thresh)
                S_lp_t_repaired = repair_set(S_lp_t)
                if len(S_lp_t_repaired) > best_size:
                    best_set = S_lp_t_repaired
                    best_size = len(S_lp_t_repaired)
    except Exception:
        pass

    # Strategy 3: Weighted Greedy Construction
    # Sort by triple count (degree)
    triple_count = [0] * (N + 1)
    for t in triples:
        for x in t:
            triple_count[x] += 1
            
    # Try a few random restarts of greedy construction
    random.seed(42)
    for restart in range(20):
        if time.time() > deadline - 15:
            break
            
        S_greedy = set()
        # Order: prefer low degree, with random tie-breaking
        candidates = list(range(1, N + 1))
        # Shuffle first to break ties randomly in sort
        random.shuffle(candidates)
        candidates.sort(key=lambda x: (triple_count[x], random.random()))
        
        for x in candidates:
            if time.time() > deadline - 15:
                break
            ok = True
            for tri_idx in element_triples[x]:
                t = triples[tri_idx]
                if all(y in S_greedy for y in t if y != x):
                    ok = False
                    break
            if ok:
                S_greedy.add(x)
                
        # Try to add back removed elements
        removed = set(range(1, N + 1)) - S_greedy
        improved = True
        while improved and time.time() < deadline - 15:
            improved = False
            for x in list(removed):
                if time.time() > deadline - 15:
                    break
                ok = True
                for tri_idx in element_triples[x]:
                    t = triples[tri_idx]
                    if all(y in S_greedy for y in t if y != x):
                        ok = False
                        break
                if ok:
                    S_greedy.add(x)
                    removed.discard(x)
                    improved = True
                    
        if len(S_greedy) > best_size:
            best_set = set(S_greedy)
            best_size = len(S_greedy)

    # Strategy 4: Local Search from Best
    # Run local search from the current best
    if time.time() < deadline - 10:
        polished = local_search(best_set, max_iters=3000)
        if len(polished) > best_size:
            best_set = polished
            best_size = len(polished)
            
    # Run local search from a few other good candidates if time permits
    if time.time() < deadline - 5:
        # Restart local search with different random seeds
        random.seed(99)
        polished2 = local_search(best_set, max_iters=2000)
        if len(polished2) > best_size:
            best_set = polished2
            best_size = len(polished2)

    return list(best_set)
