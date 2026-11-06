import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s limit
    
    # Precompute all possible triples
    all_triples = list(combinations(range(16), 3))
    n_all = len(all_triples)  # 560
    
    # For efficient violation checking, precompute for each triple the list of
    # "conflict partners" - triples that would complete a K(2,2,2) with it.
    # Actually, let's think about this differently.
    
    # A K(2,2,2) consists of 6 points {x,x',y,y',z,z'} where x!=x', etc., all 6 distinct,
    # and the 8 transversal triples (x or x', y or y', z or z').
    # 
    # When we add triple T=(a,b,c), a violation occurs if there exist a',b',c'
    # such that {a,a',b,b',c,c'} are all 6 distinct and all 8 transversals are present.
    # Since T=(a,b,c) was just added, we need the other 7 to be present.
    #
    # For a given triple (a,b,c), the number of ways to choose a',b',c' with all 6 distinct
    # is: a' can be any of 13 remaining (not a,b,c), b' can be any of 12 remaining,
    # c' can be any of 11 remaining. Total = 13*12*11 = 1716 potential completions.
    
    # Strategy: Use a smarter greedy approach with backtracking or local search.
    # Key insight: We want to maximize the number of triples while avoiding K(2,2,2).
    
    # Let's use a different approach: 
    # 1. Try multiple random greedy constructions with different heuristics
    # 2. Use local search to improve
    
    # Precompute: for each triple, which other triples "conflict" with it in the sense
    # that they participate in a potential K(2,2,2) with it. This is complex, so let's
    # use a direct check but make it fast.
    
    def check_violation_fast(triple, system_set):
        a, b, c = triple
        # For each choice of a' != a,b,c
        for a_prime in range(16):
            if a_prime == a or a_prime == b or a_prime == c:
                continue
            for b_prime in range(16):
                if b_prime == a or b_prime == b or b_prime == c or b_prime == a_prime:
                    continue
                for c_prime in range(16):
                    if (c_prime == a or c_prime == b or c_prime == c or 
                        c_prime == a_prime or c_prime == b_prime):
                        continue
                    
                    # Check the other 7 transversals
                    t1 = tuple(sorted((a_prime, b, c)))
                    if t1 not in system_set:
                        continue
                    t2 = tuple(sorted((a, b_prime, c)))
                    if t2 not in system_set:
                        continue
                    t3 = tuple(sorted((a, b, c_prime)))
                    if t3 not in system_set:
                        continue
                    t4 = tuple(sorted((a_prime, b_prime, c)))
                    if t4 not in system_set:
                        continue
                    t5 = tuple(sorted((a_prime, b, c_prime)))
                    if t5 not in system_set:
                        continue
                    t6 = tuple(sorted((a, b_prime, c_prime)))
                    if t6 not in system_set:
                        continue
                    t7 = tuple(sorted((a_prime, b_prime, c_prime)))
                    if t7 not in system_set:
                        continue
                    
                    # All 7 present, violation!
                    return True
        return False
    
    best_system = None
    best_size = 0
    
    # Strategy 1: Multiple random greedy with different orderings
    for attempt in range(30):
        if time.time() > deadline:
            break
        
        order = list(all_triples)
        random.shuffle(order)
        
        system = []
        system_set = set()
        
        for triple in order:
            if time.time() > deadline:
                break
            if not check_violation_fast(triple, system_set):
                system.append(list(triple))
                system_set.add(triple)
        
        if len(system) > best_size:
            best_size = len(system)
            best_system = [t[:] for t in system]
    
    # Strategy 2: Try to improve via local search - add rejected triples
    if best_system is not None and time.time() < deadline - 10:
        current_system = [list(t) for t in best_system]
        current_set = set(tuple(t) for t in current_system)
        
        # Find triples not in the system, try to add them
        remaining = [t for t in all_triples if t not in current_set]
        random.shuffle(remaining)
        
        for triple in remaining:
            if time.time() > deadline:
                break
            if not check_violation_fast(triple, current_set):
                current_system.append(list(triple))
                current_set.add(triple)
        
        if len(current_system) > best_size:
            best_size = len(current_system)
            best_system = current_system
    
    # Strategy 3: Try a structured approach - start with a known good base
    if time.time() < deadline - 20:
        # Try: all triples containing 0, plus some others
        star = [tuple(t) for t in all_triples if 0 in t]
        star_set = set(star)
        star_system = [list(t) for t in star]
        
        candidates = [t for t in all_triples if 0 not in t]
        random.shuffle(candidates)
        
        for triple in candidates:
            if time.time() > deadline:
                break
            if not check_violation_fast(triple, star_set):
                star_system.append(list(triple))
                star_set.add(triple)
        
        if len(star_system) > best_size:
            best_size = len(star_system)
            best_system = star_system
    
    # Final attempt: try to add any remaining triples to best_system
    if best_system is not None and time.time() < deadline:
        final_set = set(tuple(t) for t in best_system)
        remaining = [t for t in all_triples if t not in final_set]
        random.shuffle(remaining)
        
        for triple in remaining:
            if time.time() > deadline:
                break
            if not check_violation_fast(triple, final_set):
                best_system.append(list(triple))
                final_set.add(triple)
        
        best_size = len(best_system)
    
    # Fallback: ensure we have something valid
    if best_system is None or best_size == 0:
        # Return a small valid system: all triples containing 0
        star_system = [[0, a, b] for a in range(1, 16) for b in range(a + 1, 16)]
        return star_system
    
    return best_system

# EVOLVE-BLOCK-START
# The construct function is defined above.
# EVOLVE-BLOCK-END
