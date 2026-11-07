"""Suite task: large integer sets with no two-other-product divisibility."""
import time
import random
from math import isqrt

NMAX = 256

def _precompute_conflicts():
    """
    Precompute conflict data for greedy removal and local search.
    conflict_pairs[x]: list of pairs (b, c) with b < c, b,c in 2..NMAX, 
                       such that x divides b*c.
    contribution[x][y]: number of pairs in conflict_pairs[y] that contain x.
    """
    conflict_pairs = {x: [] for x in range(2, NMAX + 1)}
    
    # Generate all pairs (b, c) with 2 <= b < c <= NMAX
    pairs = []
    for b in range(2, NMAX + 1):
        for c in range(b + 1, NMAX + 1):
            pairs.append((b, c))
            
    # For each x, find pairs (b, c) such that x | b*c
    for x in range(2, NMAX + 1):
        for b, c in pairs:
            if (b * c) % x == 0:
                conflict_pairs[x].append((b, c))
                
    # Precompute contribution matrix
    # contribution[x][y] = count of pairs in conflict_pairs[y] that include x
    contribution = [[0] * (NMAX + 1) for _ in range(NMAX + 1)]
    for y in range(2, NMAX + 1):
        for b, c in conflict_pairs[y]:
            contribution[b][y] += 1
            contribution[c][y] += 1
            
    return conflict_pairs, contribution

# Global precomputation
CONFLICT_PAIRS, CONTRIBUTION = _precompute_conflicts()

def _valid_addition_fast(x, S_list):
    """
    Check if x can be added to S_list.
    S_list is a list of ints.
    Constraint: for all distinct a,b,c in S_list + {x}, a does not divide b*c.
    New conflicts involving x:
    1. x | b*c for b,c in S_list, b!=c
    2. b | x*c for b,c in S_list, b!=c
    3. c | x*b for b,c in S_list, b!=c
    Note: b and c are distinct.
    """
    n = len(S_list)
    for i in range(n):
        b = S_list[i]
        for j in range(n):
            if i == j:
                continue
            c = S_list[j]
            if (b * c) % x == 0:
                return False
            if (x * c) % b == 0:
                return False
            if (x * b) % c == 0:
                return False
    return True

def _greedy_remove(deadline):
    """
    Greedy removal based on conflict degree.
    Start with all numbers, remove the one causing the most conflicts.
    """
    S = set(range(2, NMAX + 1))
    # degree[x] = number of pairs (b,c) in S such that x | b*c
    # Initially, all pairs are in S.
    degree = {x: len(CONFLICT_PAIRS[x]) for x in range(2, NMAX + 1)}
    
    while S:
        if time.time() > deadline:
            break
        
        # Find x in S with max degree
        max_deg = -1
        max_x = None
        for x in S:
            if degree[x] > max_deg:
                max_deg = degree[x]
                max_x = x
                
        if max_deg == 0:
            break
            
        # Remove max_x
        S.remove(max_x)
        
        # Update degrees for remaining elements
        # degree[y] -= contribution[max_x][y]
        for y in S:
            degree[y] -= CONTRIBUTION[max_x][y]
            
    return list(S)

def _local_search(S, deadline, max_iters=10000):
    """
    Local search to improve the set.
    Try adding elements, removing elements, swapping.
    """
    S = list(S)
    S_set = set(S)
    best = S[:]
    best_size = len(S)
    rng = random.Random(12345)
    
    for _ in range(max_iters):
        if time.time() > deadline:
            break
            
        op = rng.random()
        
        if op < 0.4:
            # Try to add a random element
            candidates = [x for x in range(2, NMAX + 1) if x not in S_set]
            if candidates:
                x = rng.choice(candidates)
                if _valid_addition_fast(x, S):
                    S.append(x)
                    S_set.add(x)
                    if len(S) > best_size:
                        best = S[:]
                        best_size = len(S)
        elif op < 0.6 and S:
            # Try to remove a random element (to allow adding others later?)
            # This is usually bad for size, but might help escape local minima
            # Let's stick to swaps and adds primarily.
            pass
        else:
            # Try swap: remove one, add another
            if S:
                i = rng.randrange(len(S))
                removed = S[i]
                S_temp = S[:i] + S[i+1:]
                S_temp_set = S_set - {removed}
                
                candidates = [x for x in range(2, NMAX + 1) if x not in S_temp_set]
                if candidates:
                    # Try a few random candidates
                    rng.shuffle(candidates)
                    improved = False
                    for x in candidates[:10]:
                        if _valid_addition_fast(x, S_temp):
                            S = S_temp + [x]
                            S_set = S_temp_set | {x}
                            if len(S) > best_size:
                                best = S[:]
                                best_size = len(S)
                            improved = True
                            break
                    if not improved:
                        S = S_temp + [removed]
                        S_set = S_temp_set | {removed}
                        
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    best = []
    best_size = 0
    
    # Phase 1: Greedy Removal
    # This is very fast (O(N^2))
    S_greedy = _greedy_remove(deadline - 60.0)
    if len(S_greedy) > best_size:
        best = S_greedy
        best_size = len(S_greedy)
    
    # Phase 2: Local Search from Greedy Result
    if time.time() < deadline - 10.0:
        S_ls = _local_search(S_greedy, deadline - 10.0, max_iters=50000)
        if len(S_ls) > best_size:
            best = S_ls
            best_size = len(S_ls)
            
    # Phase 3: Multi-start Local Search
    # Try different random perturbations of the greedy result
    rng = random.Random(999)
    for _ in range(20):
        if time.time() > deadline - 5.0:
            break
            
        # Perturb
        S_perturbed = best[:]
        n_remove = rng.randint(1, min(5, len(S_perturbed)))
        indices = rng.sample(range(len(S_perturbed)), n_remove)
        indices.sort(reverse=True)
        for idx in indices:
            S_perturbed.pop(idx)
            
        S_ls = _local_search(S_perturbed, deadline - 5.0, max_iters=20000)
        if len(S_ls) > best_size:
            best = S_ls
            best_size = len(S_ls)
            
    # Phase 4: Try Greedy Addition from Scratch (different order)
    # Just in case removal was too aggressive
    S_add = []
    S_add_set = set()
    candidates = list(range(NMAX, 1, -1))
    rng.shuffle(candidates)
    for x in candidates:
        if time.time() > deadline - 2.0:
            break
        if x not in S_add_set:
            if _valid_addition_fast(x, S_add):
                S_add.append(x)
                S_add_set.add(x)
                
    if len(S_add) > best_size:
        best = S_add
        best_size = len(S_add)
        
    # Final Polish
    if time.time() < deadline - 1.0:
        S_final = _local_search(best, deadline - 1.0, max_iters=50000)
        if len(S_final) > best_size:
            best = S_final
            best_size = len(S_final)
            
    return [int(x) for x in best]
