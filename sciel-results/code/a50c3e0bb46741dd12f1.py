"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill
    
    N = 180
    elements = list(range(1, N+1))
    
    # Precompute all conflict triples: (a, b, c) where 1/a = 1/b + 1/c
    # a = k*m*n, b = k*m*(m+n), c = k*n*(m+n), gcd(m,n)=1
    triples = set()
    for k in range(1, N+1):
        for m in range(1, N+1):
            for n in range(1, N+1):
                if m >= n:
                    continue
                if gcd(m, n) != 1:
                    continue
                a = k * m * n
                b = k * m * (m + n)
                c = k * n * (m + n)
                if a <= N and b <= N and c <= N:
                    if len({a, b, c}) == 3:
                        triples.add((min(a,b,c), min(max(a,b), max(b,c)), max(a,b,c)))
    
    # Build conflict structures
    element_conflicts = {i: set() for i in range(1, N+1)}
    element_triples = {i: [] for i in range(1, N+1)}
    for (a, b, c) in triples:
        element_conflicts[a].update([b, c])
        element_conflicts[b].update([a, c])
        element_conflicts[c].update([a, b])
        element_triples[a].append((a, b, c))
        element_triples[b].append((a, b, c))
        element_triples[c].append((a, b, c))
    
    def try_add(s, x):
        """Try to add x to set s, return True if valid"""
        for (a, b, c) in element_triples[x]:
            if x == a:
                if b in s and c in s:
                    return False
            elif x == b:
                if a in s and c in s:
                    return False
            elif x == c:
                if a in s and b in s:
                    return False
        return True
    
    def greedy_build_fast(order):
        """Faster greedy using precomputed triples"""
        result = set()
        for x in order:
            can_add = True
            for (a, b, c) in element_triples[x]:
                if x == a:
                    if b in result and c in result:
                        can_add = False
                        break
                elif x == b:
                    if a in result and c in result:
                        can_add = False
                        break
                elif x == c:
                    if a in result and b in result:
                        can_add = False
                        break
            if can_add:
                result.add(x)
        return result
    
    def min_degree_greedy(order):
        """Minimum-degree greedy: always pick element with fewest conflicts with selected set"""
        remaining = list(order)
        result = set()
        while remaining:
            if time.time() - start_time > DEADLINE:
                break
            # Find element with minimum conflicts with current result
            best_x = None
            best_conf = float('inf')
            for x in remaining:
                conf = len(element_conflicts[x] & result)
                if conf < best_conf:
                    best_conf = conf
                    best_x = x
                    if conf == 0:
                        break
            if best_x is None:
                break
            if try_add(result, best_x):
                result.add(best_x)
            remaining.remove(best_x)
        return result
    
    def local_search_add(s):
        """Repeatedly try to add elements that can be added"""
        improved = True
        while improved:
            if time.time() - start_time > DEADLINE:
                break
            improved = False
            for x in range(1, N+1):
                if x not in s and try_add(s, x):
                    s.add(x)
                    improved = True
        return s
    
    def local_search_swap(s):
        """Try swapping elements to improve set size"""
        improved = True
        while improved:
            if time.time() - start_time > DEADLINE:
                break
            improved = False
            for x in list(s):
                test_set = s - {x}
                added = 0
                for y in range(1, N+1):
                    if y not in test_set and try_add(test_set, y):
                        test_set.add(y)
                        added += 1
                if len(test_set) > len(s):
                    s = test_set
                    improved = True
                    break
        return s
    
    def max_sat_score(s):
        """Score = number of elements - penalty for violated triples (Max-SAT objective)"""
        violated = 0
        for (a, b, c) in triples:
            if a in s and b in s and c in s:
                violated += 1
        return len(s) - violated * 0.1
    
    def simulated_annealing(start_set, iterations=50000):
        """Simulated annealing on Max-SAT landscape: flip elements to maximize score"""
        if time.time() - start_time > DEADLINE:
            return start_set
        
        current = set(start_set)
        current_score = max_sat_score(current)
        best = set(current)
        best_score = current_score
        
        T = 1.0
        T_min = 0.01
        cooling = 0.9999
        
        all_elements = list(range(1, N+1))
        rng = random.Random(12345)
        
        for it in range(iterations):
            if time.time() - start_time > DEADLINE:
                break
            if T < T_min:
                break
            
            # Randomly pick an element to flip
            x = rng.choice(all_elements)
            
            # Compute score change
            old_score = current_score
            if x in current:
                # Removing x
                new_set = current - {x}
                new_score = max_sat_score(new_set)
            else:
                # Adding x
                new_set = current | {x}
                new_score = max_sat_score(new_set)
            
            delta = new_score - old_score
            
            if delta >= 0 or rng.random() < __import__('math').exp(delta / T):
                if x in current:
                    current.discard(x)
                else:
                    current.add(x)
                current_score = new_score
                
                if len(current) > len(best):
                    best = set(current)
                    best_score = current_score
            
            T *= cooling
        
        return best
    
    def multi_swap_local_search(s):
        """More aggressive local search: try removing 1-2 elements and re-adding"""
        improved = True
        while improved:
            if time.time() - start_time > DEADLINE:
                break
            improved = False
            # Try removing 1 element
            for x in list(s):
                test_set = s - {x}
                for y in range(1, N+1):
                    if y not in test_set and try_add(test_set, y):
                        test_set.add(y)
                if len(test_set) > len(s):
                    s = test_set
                    improved = True
                    break
            if improved:
                continue
            # Try removing 2 elements
            s_list = list(s)
            for i in range(len(s_list)):
                if time.time() - start_time > DEADLINE:
                    break
                for j in range(i+1, len(s_list)):
                    test_set = s - {s_list[i], s_list[j]}
                    for y in range(1, N+1):
                        if y not in test_set and try_add(test_set, y):
                            test_set.add(y)
                    if len(test_set) > len(s):
                        s = test_set
                        improved = True
                        break
                if improved:
                    break
        return s
    
    random.seed(42)
    
    best_set = set()
    best_size = 0
    
    # Strategy 1: Minimum-degree greedy with different seed orderings
    seed_orderings = [
        list(range(1, N+1)),
        list(range(N, 0, -1)),
        list(range(91, N+1)) + list(range(1, 91)),
    ]
    
    for _ in range(30):
        if time.time() - start_time > DEADLINE:
            break
        order = list(range(1, N+1))
        random.shuffle(order)
        seed_orderings.append(order)
    
    for order in seed_orderings:
        if time.time() - start_time > DEADLINE:
            break
        s = min_degree_greedy(order)
        s = local_search_add(s)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s
    
    # Strategy 2: Standard greedy with different orderings
    for order in seed_orderings[:10]:
        if time.time() - start_time > DEADLINE:
            break
        s = greedy_build_fast(order)
        s = local_search_add(s)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s
    
    # Strategy 3: Local search with swaps
    if time.time() - start_time < DEADLINE:
        best_set = local_search_swap(best_set)
        best_size = len(best_set)
    
    # Strategy 4: Multi-swap local search
    if time.time() - start_time < DEADLINE:
        best_set = multi_swap_local_search(best_set)
        best_size = len(best_set)
    
    # Strategy 5: Simulated annealing from multiple starting points
    if time.time() - start_time < DEADLINE:
        sa_starts = [best_set]
        # Add a few other greedy solutions as starting points
        for order in seed_orderings[:5]:
            if time.time() - start_time > DEADLINE:
                break
            s = min_degree_greedy(order)
            s = local_search_add(s)
            sa_starts.append(s)
        
        for start in sa_starts:
            if time.time() - start_time > DEADLINE:
                break
            sa_result = simulated_annealing(start, iterations=30000)
            sa_result = local_search_add(sa_result)
            if len(sa_result) > best_size:
                best_size = len(sa_result)
                best_set = sa_result
    
    # Final polish: try all single additions
    if time.time() - start_time < DEADLINE:
        best_set = local_search_add(best_set)
        best_size = len(best_set)
    
    return list(best_set)

# EVOLVE-BLOCK-END
