import time
import random
import math

def construct():
    n = 256
    start_time = time.time()
    deadline = 45.0  # Safety margin
    
    # Precompute 3-APs for fast checking
    # For each v, precompute which pairs (a, b) in [1, n] would form an AP with v
    # Actually, let's precompute for each v the set of "forbidden pairs"
    # A pair (a, b) is forbidden with v if {a, b, v} forms a 3-AP
    # Cases:
    # 1. v is middle: a + b = 2v, a != b, a != v, b != v
    # 2. v is end: a + v = 2b (v is larger end) or v + b = 2a (v is smaller end)
    #    In both cases, the other end is 2b - v or 2a - v
    
    # For fast checking when adding v to S:
    # v conflicts if there exist a, b in S such that {a, b, v} is a 3-AP
    # We can check:
    # - For each a in S: check if 2v - a is in S (v is middle, a is one end)
    # - For each a in S: check if (a + v) is even and (a + v) // 2 is in S (v is an end)
    
    def is_valid_addition(S, v):
        for a in S:
            # v is middle: a, v, c where c = 2v - a
            c = 2 * v - a
            if c != v and c in S:
                return False
            # v is end: a, b, v where b = (a + v) / 2
            if (a + v) % 2 == 0:
                b = (a + v) // 2
                if b != v and b in S:
                    return False
        return True
    
    def get_score(S):
        return sum(1.0 / x for x in S)
    
    def greedy_fill(S):
        """Add elements in order 1, 2, ..., n if valid"""
        for x in range(1, n + 1):
            if x not in S:
                if is_valid_addition(S, x):
                    S.add(x)
    
    # 1. Greedy Initialization
    S = set()
    greedy_fill(S)
    
    best_S = set(S)
    best_score = get_score(best_S)
    
    # 2. Local Search with 1-swap and 2-swap
    current_S = set(S)
    current_score = get_score(current_S)
    
    no_improvement_count = 0
    
    while time.time() - start_time < deadline:
        improved = False
        
        # Get lists
        in_S = sorted(current_S)
        not_in_S = sorted([x for x in range(1, n + 1) if x not in current_S])
        
        # Try 1-swaps: replace u in S with v not in S where v < u
        for u in in_S:
            for v in not_in_S:
                if v >= u:
                    break  # Since not_in_S is sorted, no more v < u
                
                # Check if swap is valid
                test_S = current_S - {u}
                if is_valid_addition(test_S, v):
                    new_score = current_score - 1.0 / u + 1.0 / v
                    if new_score > current_score + 1e-9:
                        current_S = test_S
                        current_S.add(v)
                        current_score = new_score
                        improved = True
                        break
            if improved:
                break
        
        # If no 1-swap improvement, try 2-swaps
        if not improved:
            in_S = sorted(current_S)
            not_in_S = sorted([x for x in range(1, n + 1) if x not in current_S])
            
            # Try replacing two elements u1, u2 in S with v1, v2 not in S
            # where 1/v1 + 1/v2 > 1/u1 + 1/u2
            # To keep it manageable, only try pairs where v1 < u1 and v2 < u2 (roughly)
            # Actually, just try all pairs but with a limit
            found_2swap = False
            for i, u1 in enumerate(in_S):
                if found_2swap:
                    break
                for u2 in in_S[i+1:]:
                    if found_2swap:
                        break
                    # Skip if removing u1, u2 would make it too small
                    if len(current_S) <= 3:
                        continue
                    
                    test_S = current_S - {u1, u2}
                    
                    # Try to add v1, v2 from not_in_S
                    # Sort not_in_S by weight descending (smaller first)
                    for v1 in not_in_S:
                        if found_2swap:
                            break
                        if is_valid_addition(test_S, v1):
                            test_S2 = test_S | {v1}
                            for v2 in not_in_S:
                                if v2 == v1:
                                    continue
                                if not is_valid_addition(test_S2, v2):
                                    continue
                                
                                new_score = current_score - 1.0 / u1 - 1.0 / u2 + 1.0 / v1 + 1.0 / v2
                                if new_score > current_score + 1e-9:
                                    current_S = test_S2 | {v2}
                                    current_score = new_score
                                    improved = True
                                    found_2swap = True
                                    break
        if improved:
            if current_score > best_score:
                best_score = current_score
                best_S = set(current_S)
        else:
            no_improvement_count += 1
            if no_improvement_count > 3:
                # Perturbation: remove 1-2 random elements and re-greedy
                perturbed_S = set(current_S)
                if len(perturbed_S) > 3:
                    elems = list(perturbed_S)
                    random.shuffle(elems)
                    removed = elems[:2]
                    for r in removed:
                        perturbed_S.remove(r)
                    greedy_fill(perturbed_S)
                    if get_score(perturbed_S) > current_score:
                        current_S = perturbed_S
                        current_score = get_score(current_S)
                        no_improvement_count = 0
                    else:
                        no_improvement_count += 5

    return list(best_S)
