import random
from itertools import combinations

def construct():
    n = 16
    all_triples = list(combinations(range(n), 3))
    random.seed(42)
    random.shuffle(all_triples)
    
    present = set()
    for t in all_triples:
        a, b, c = t
        # Check if adding {a,b,c} creates a K(2,2,2)
        # For each choice of a', b', c' (all distinct, not in {a,b,c}),
        # check if all 8 transversals are present (with {a,b,c} being the new one)
        valid = True
        others = [x for x in range(n) if x not in (a,b,c)]
        for a_prime in others:
            for b_prime in others:
                if b_prime == a_prime:
                    continue
                for c_prime in others:
                    if c_prime == a_prime or c_prime == b_prime:
                        continue
                    # Check if the other 7 transversals are all present
                    # The 8 transversals are:
                    # (a,b,c), (a,b,c'), (a',b,c), (a',b,c'), (a,b',c), (a,b',c'), (a',b',c), (a',b',c')
                    # We're adding (a,b,c), so check the other 7
                    t1 = (a, b, c_prime) if (a, b, c_prime) in present else None
                    t2 = (a_prime, b, c) if (a_prime, b, c) in present else None
                    t3 = (a_prime, b, c_prime) if (a_prime, b, c_prime) in present else None
                    t4 = (a, b_prime, c) if (a, b_prime, c) in present else None
                    t5 = (a, b_prime, c_prime) if (a, b_prime, c_prime) in present else None
                    t6 = (a_prime, b_prime, c) if (a_prime, b_prime, c) in present else None
                    t7 = (a_prime, b_prime, c_prime) if (a_prime, b_prime, c_prime) in present else None
                    # Need to check if all 7 are present (as sets)
                    if (frozenset((a,b,c_prime)) in present and 
                        frozenset((a_prime,b,c)) in present and
                        frozenset((a_prime,b,c_prime)) in present and
                        frozenset((a,b_prime,c)) in present and
                        frozenset((a,b_prime,c_prime)) in present and
                        frozenset((a_prime,b_prime,c)) in present and
                        frozenset((a_prime,b_prime,c_prime)) in present):
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        if valid:
            present.add(frozenset(t))
    
    return [list(t) for t in present]
