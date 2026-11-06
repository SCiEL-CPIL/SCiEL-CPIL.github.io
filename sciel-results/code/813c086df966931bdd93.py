import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n = 16
    all_triples = list(combinations(range(n), 3))
    
    def check_add(triple_set, a, b, c):
        """Return True if adding {a,b,c} is safe (no forbidden K(2,2,2) created)."""
        others = [x for x in range(n) if x not in (a, b, c)]
        for a_prime in others:
            for b_prime in others:
                if b_prime == a_prime:
                    continue
                for c_prime in others:
                    if c_prime == a_prime or c_prime == b_prime:
                        continue
                    # Check all 7 other transversals
                    t1 = frozenset((a, b, c_prime))
                    t2 = frozenset((a, b_prime, c))
                    t3 = frozenset((a, b_prime, c_prime))
                    t4 = frozenset((a_prime, b, c))
                    t5 = frozenset((a_prime, b, c_prime))
                    t6 = frozenset((a_prime, b_prime, c))
                    t7 = frozenset((a_prime, b_prime, c_prime))
                    if t1 in triple_set and t2 in triple_set and t3 in triple_set and \
                       t4 in triple_set and t5 in triple_set and t6 in triple_set and t7 in triple_set:
                        return False
        return True
    
    best_count = 0
    best_set = set()
    
    # Try multiple random orderings
    random.seed(42)
    attempts = 0
    while time.time() < deadline:
        random.shuffle(all_triples)
        current_set = set()
        for triple in all_triples:
            a, b, c = triple
            if check_add(current_set, a, b, c):
                current_set.add(frozenset(triple))
        
        if len(current_set) > best_count:
            best_count = len(current_set)
            best_set = current_set
        attempts += 1
        if best_count >= 500:  # good enough?
            break
    
    return [list(t) for t in best_set]
