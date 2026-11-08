import time
import math
from itertools import combinations

def score_set(U):
    """Compute the score for a set U."""
    if not U or 0 not in U:
        return -1
    U = sorted(set(U))
    n = len(U)
    if n < 2:
        return -1
    
    max_u = max(U)
    if max_u == 0:
        return -1
        
    # Compute sumset and difference set
    # Using sets for simplicity, n is small in exhaustive search
    sumset = set()
    diffset = set()
    for i in range(n):
        ui = U[i]
        for j in range(n):
            sumset.add(ui + U[j])
            diffset.add(ui - U[j])
            
    ratio = len(diffset) / len(sumset)
    if ratio <= 1:
        return 1.0
        
    denom = math.log(2 * max_u + 1)
    if denom <= 0:
        return -1
        
    return 1.0 + math.log(ratio) / denom

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    best_set = [0, 1, 3, 7]
    best_score = score_set(best_set)
    
    # Exhaustive search for small sets
    # We search for subsets of size k from 1..M_max
    # U = {0} + subset
    
    max_m = 20
    max_n = 8
    
    for n in range(3, max_n + 1):
        if time.time() > deadline - 10:
            break
            
        # Max possible M for this n to keep search space reasonable
        # If n=8, M=20, combos = 125970. OK.
        # If n=9, M=20, combos = 167960. OK.
        # Let's adjust M_max based on n
        m_limit = min(max_m, 25)
        
        for m in range(n - 1, m_limit + 1):
            if time.time() > deadline - 10:
                break
                
            # We need to pick n-1 elements from 1..m
            # But max element must be <= m
            # Actually, we are iterating m as the max element?
            # No, we are iterating subsets of 1..m_limit.
            # The max element of the subset will be <= m_limit.
            # We don't need to iterate m separately if we just iterate all subsets of 1..m_limit
            # and take the max.
            
            pass # This loop structure is inefficient. Let's just iterate subsets of 1..m_limit
            
    # Better approach: Iterate all subsets of 1..M_global
    M_global = 20
    range_vals = range(1, M_global + 1)
    
    for n in range(3, 9):
        if time.time() > deadline - 10:
            break
        for comb in combinations(range_vals, n - 1):
            U = [0] + list(comb)
            s = score_set(U)
            if s > best_score:
                best_score = s
                best_set = U
            if time.time() > deadline - 10:
                break
                
    # Local search from best_set
    import random
    random.seed(42)
    current_set = list(best_set)
    current_score = best_score
    
    max_iter = 20000
    for i in range(max_iter):
        if time.time() > deadline - 5:
            break
            
        new_set = set(current_set)
        mutation_type = random.choice(['add', 'remove', 'replace'])
        
        if mutation_type == 'add' and len(new_set) < 15:
            max_elem = max(new_set)
            new_elem = random.randint(1, max_elem + 10)
            if new_elem not in new_set:
                new_set.add(new_elem)
        elif mutation_type == 'remove' and len(new_set) > 2:
            elems = [x for x in new_set if x != 0]
            if elems:
                new_set.remove(random.choice(elems))
        elif mutation_type == 'replace' and len(new_set) > 2:
            elems = [x for x in new_set if x != 0]
            if elems:
                old = random.choice(elems)
                new_set.remove(old)
                max_elem = max(new_set) if new_set else 1
                new_elem = random.randint(1, max_elem + 10)
                if new_elem not in new_set:
                    new_set.add(new_elem)
        
        if len(new_set) < 2:
            continue
        
        new_score = score_set(list(new_set))
        if new_score > current_score:
            current_set = list(new_set)
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_set = list(new_set)
                
    return best_set
