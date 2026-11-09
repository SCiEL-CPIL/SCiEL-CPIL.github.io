"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd

def _lcm(a, b):
    return a * b // gcd(a, b)

def _precompute(max_n):
    """
    Precompute LCM table and conflict structures.
    """
    n = max_n
    lcm_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            lcm_table[i][j] = _lcm(i, j)
            if i != j:
                lcm_table[j][i] = lcm_table[i][j]
    
    # Find all bad triples
    bad_triples = []
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            lab = lcm_table[a][b]
            for c in range(b + 1, n + 1):
                if lcm_table[a][c] == lab and lcm_table[b][c] == lab:
                    bad_triples.append((a, b, c))
                    
    # Precompute conflict pairs for each element
    # conflict_pairs[x] is a list of (a, b) such that {x, a, b} is a bad triple
    # We store (min(a,b), max(a,b)) to keep consistent
    conflict_pairs = {i: [] for i in range(1, n + 1)}
    for a, b, c in bad_triples:
        # Pairs involving a: (b, c)
        conflict_pairs[a].append((b, c))
        # Pairs involving b: (a, c)
        conflict_pairs[b].append((a, c))
        # Pairs involving c: (a, b)
        conflict_pairs[c].append((a, b))
        
    return lcm_table, bad_triples, conflict_pairs

def _can_add(x, S_mask, conflict_pairs):
    """
    Check if x can be added to the set represented by S_mask.
    Returns True if no conflict.
    """
    for a, b in conflict_pairs[x]:
        # Check if both a and b are in S_mask
        if (S_mask & (1 << (a - 1))) and (S_mask & (1 << (b - 1))):
            return False
    return True

def _greedy_construct(order, conflict_pairs, max_n):
    """
    Greedily add elements from order to S.
    """
    S = []
    S_mask = 0
    
    for x in order:
        if _can_add(x, S_mask, conflict_pairs):
            S.append(x)
            S_mask |= (1 << (x - 1))
            
    return S, S_mask

def _local_search(S, S_mask, conflict_pairs, max_n, deadline):
    """
    Improve S by adding elements or swapping.
    """
    S_set = set(S)
    full_set = set(range(1, max_n + 1))
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Phase 1: Try to add any element not in S
        for x in full_set - S_set:
            if _can_add(x, S_mask, conflict_pairs):
                S.append(x)
                S_set.add(x)
                S_mask |= (1 << (x - 1))
                improved = True
                break
        
        if improved:
            continue
        
        # Phase 2: Try 1-swaps (remove one, add one)
        # We try to remove an element 'a' and add 'x' if possible.
        # To be efficient, we can prioritize removing elements that are "restrictive".
        # For now, iterate through S.
        
        found_swap = False
        # Create a list to iterate over to avoid modification issues
        S_list = list(S)
        
        for a in S_list:
            # Temporarily remove a
            S_mask_no_a = S_mask & ~(1 << (a - 1))
            S_set_no_a = S_set - {a}
            
            # Try to add any x not in S_set_no_a
            # Optimization: Only try x that are "conflicting" with a or generally high degree?
            # For now, try all.
            for x in full_set - S_set_no_a:
                if _can_add(x, S_mask_no_a, conflict_pairs):
                    # Accept swap
                    S.remove(a)
                    S.append(x)
                    S_set.discard(a)
                    S_set.add(x)
                    S_mask = S_mask_no_a | (1 << (x - 1))
                    found_swap = True
                    improved = True
                    break
            
            if found_swap:
                break
            if time.time() > deadline:
                break
            
        if not found_swap:
            # Try 2-swaps? Or random restart?
            # Given time, break.
            break
            
    return S, S_mask

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 85s limit to be safe (evaluator kills at 90s)
    
    max_n = 128
    lcm_table, bad_triples, conflict_pairs = _precompute(max_n)
    
    if not bad_triples:
        return list(range(1, max_n + 1))
        
    best_set = []
    best_size = 0
    best_mask = 0
    
    all_elements = list(range(1, max_n + 1))
    
    # Generate diverse orderings
    orderings = []
    
    # 1. Natural order
    orderings.append(all_elements)
    
    # 2. Reverse order
    orderings.append(list(reversed(all_elements)))
    
    # 3. Order by degree (number of conflict pairs)
    degrees = {x: len(conflict_pairs[x]) for x in all_elements}
    orderings.append(sorted(all_elements, key=lambda x: degrees[x]))
    orderings.append(sorted(all_elements, key=lambda x: -degrees[x]))
    
    # 4. Random orderings
    random.seed(42)
    for _ in range(100):
        orderings.append(random.sample(all_elements, max_n))
        
    # Run Greedy
    for i, order in enumerate(orderings):
        if time.time() > deadline - 15:
            break
            
        S, S_mask = _greedy_construct(order, conflict_pairs, max_n)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S[:]
            best_mask = S_mask
            
    # Local Search on the best set
    if time.time() < deadline - 5:
        improved_set, improved_mask = _local_search(best_set, best_mask, conflict_pairs, max_n, deadline)
        if len(improved_set) > best_size:
            best_set = improved_set
            best_size = len(best_set)
            best_mask = improved_mask
            
    # Final Validation
    S_set = set(best_set)
    valid = True
    for a, b, c in bad_triples:
        if a in S_set and b in S_set and c in S_set:
            valid = False
            break
            
    if not valid:
        # Fallback to a known valid set (e.g., all primes)
        primes = [x for x in range(2, max_n + 1) if all(x % i != 0 for i in range(2, int(x**0.5) + 1))]
        if len(primes) > len(best_set):
            best_set = primes
            
    if not best_set:
        best_set = [1]
        
    return best_set

# EVOLVE-BLOCK-END
