import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    
    n = 16
    # Precompute all possible triples
    all_triples = list(itertools.combinations(range(n), 3))
    triple_index = {}
    for i, t in enumerate(all_triples):
        triple_index[t] = i
    
    # For efficient conflict checking, precompute pair->thirds mapping
    # pair_to_thirds[(a,b)] = set of c such that (min(a,b), mid, max) is a triple
    pair_to_thirds = {}
    for a in range(n):
        for b in range(a+1, n):
            pair_to_thirds[(a,b)] = set()
    for t in all_triples:
        a, b, c = t
        pair_to_thirds[(min(a,b), max(a,b))].add(c)
        pair_to_thirds[(min(a,c), max(a,c))].add(b)
        pair_to_thirds[(min(b,c), max(b,c))].add(a)
    
    def check_conflict(triple, present):
        """Check if adding `triple` to `present` creates a K(2,2,2)."""
        a, b, c = triple
        # The new triple {a,b,c} could be a transversal triple in a K(2,2,2)
        # with pairs {a,a'}, {b,b'}, {c,c'}
        # Need all 8 transversal triples present (one is the new one, 7 must be in present)
        for a_prime in range(n):
            if a_prime == a:
                continue
            for b_prime in range(n):
                if b_prime == b or b_prime == a or b_prime == a_prime:
                    continue
                for c_prime in range(n):
                    if c_prime == c or c_prime == a or c_prime == b or c_prime == a_prime or c_prime == b_prime:
                        continue
                    # Check all 7 other transversal triples
                    t1 = (min(a_prime, b), max(a_prime, b))
                    t2 = (min(a, b_prime), max(a, b_prime))
                    t3 = (min(a, c_prime), max(a, c_prime))
                    t4 = (min(a_prime, b_prime), max(a_prime, b_prime))
                    t5 = (min(a_prime, c_prime), max(a_prime, c_prime))
                    t6 = (min(b, c_prime), max(b, c_prime))
                    t7 = (min(b_prime, c_prime), max(b_prime, c_prime))
                    
                    if (c in pair_to_thirds.get(t1, set()) and
                        c_prime in pair_to_thirds.get(t2, set()) and
                        b_prime in pair_to_thirds.get(t3, set()) and
                        c in pair_to_thirds.get(t4, set()) and
                        b in pair_to_thirds.get(t5, set()) and
                        a_prime in pair_to_thirds.get(t6, set()) and
                        a in pair_to_thirds.get(t7, set())):
                        # Need to verify these are actually in present
                        # t1 with c: {a_prime, b, c}
                        # t2 with c_prime: {a, b_prime, c_prime}
                        # t3 with b_prime: {a, c_prime, b_prime}
                        # t4 with c: {a_prime, b_prime, c}
                        # t5 with b: {a_prime, c_prime, b}
                        # t6 with a_prime: {b, c_prime, a_prime}
                        # t7 with a: {b_prime, c_prime, a}
                        
                        triples_to_check = [
                            (min(a_prime, b, c), sorted([a_prime, b, c])[1], max(a_prime, b, c)),
                            (min(a, b_prime, c_prime), sorted([a, b_prime, c_prime])[1], max(a, b_prime, c_prime)),
                            (min(a, c_prime, b_prime), sorted([a, c_prime, b_prime])[1], max(a, c_prime, b_prime)),
                            (min(a_prime, b_prime, c), sorted([a_prime, b_prime, c])[1], max(a_prime, b_prime, c)),
                            (min(a_prime, c_prime, b), sorted([a_prime, c_prime, b])[1], max(a_prime, c_prime, b)),
                            (min(b, c_prime, a_prime), sorted([b, c_prime, a_prime])[1], max(b, c_prime, a_prime)),
                            (min(b_prime, c_prime, a), sorted([b_prime, c_prime, a])[1], max(b_prime, c_prime, a)),
                        ]
                        for tt in triples_to_check:
                            if tt not in present:
                                break
                        else:
                            return True
        return False
    
    best_result = None
    best_count = 0
    
    # Multiple random restarts
    rng = random.Random(42)
    
    for restart in range(50):
        if time.time() > deadline:
            break
        
        # Random order of triples
        order = all_triples[:]
        rng.shuffle(order)
        
        present = set()
        for triple in order:
            if time.time() > deadline:
                break
            if check_conflict(triple, present):
                continue
            present.add(triple)
        
        if len(present) > best_count:
            best_count = len(present)
            best_result = [list(t) for t in present]
        
        if best_count >= 200:  # good enough, stop early
            break
    
    if best_result is None:
        # Fallback to seed
        best_result = [[0, a, b] for a in range(1, 16) for b in range(a + 1, 16)]
    
    return best_result
