"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time
import sys

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit

    n = 16
    # Precompute all C(16,3) = 560 triples
    all_triples = []
    for a in range(n):
        for b in range(a + 1, n):
            for c in range(b + 1, n):
                all_triples.append((a, b, c))
    
    # For each triple, precompute the set of "forbidden" triples
    # A triple T=(a,b,c) is in a K(2,2,2) with three disjoint pairs
    # Actually, K(2,2,2) in the 3-partite sense: we need 3 disjoint pairs
    # (x1,y1), (x2,y2), (x3,y3) such that all 8 transversals are present.
    # The 8 transversals are all triples picking one from each pair.
    # So if we have pairs P1={p1a,p1b}, P2={p2a,p2b}, P3={p3a,p3b},
    # the 8 triples are all (pi) where pi is chosen from Pi.
    
    # For a given triple T, it participates in a K(2,2,2) if there exist
    # two other points such that T plus 7 others form the complete 3-partite.
    
    # Let me think about this differently. A K(2,2,2) is defined by 6 points
    # partitioned into 3 pairs. The 8 transversal triples are all present.
    # So for a set of triples to be K(2,2,2)-free, no 6-point subset partitioned
    # into 3 pairs should have all 8 corresponding triples present.
    
    # Precompute: for each triple, which other triples would complete a K(2,2,2)
    # with it? Actually, let's think about it as: given a triple T=(a,b,c),
    # it is one of the 8 transversals for some partition of 6 points into 3 pairs.
    # T picks one from each pair. So the other 7 transversals pick the other
    # element from at least one pair.
    
    # For T=(a,b,c), the possible pairings are:
    # - a is paired with some a', b with b', c with c'
    # The 6 points are {a, a', b, b', c, c'}
    # The 8 transversals are all choices of one from each pair.
    
    # So for a triple T=(a,b,c), to find all K(2,2,2)s containing T,
    # we need to find a', b', c' (distinct from a,b,c and each other)
    # such that all 8 triples from the 3 pairs are in our system.
    
    # This is complex. Let me use a different approach:
    # For each potential K(2,2,2) (defined by 6 points and a partition into 3 pairs),
    # check if all 8 transversals are present.
    
    # Number of ways to choose 6 points from 16: C(16,6) = 8008
    # Number of ways to partition 6 points into 3 pairs: 15
    # Total K(2,2,2) candidates: 8008 * 15 = 120120
    
    # This is manageable. Let me precompute all K(2,2,2) patterns.
    
    # Actually, let me use a greedy approach with local search as suggested by the knowledge.
    
    def is_k222_free(triple_set):
        """Check if the set of triples is K(2,2,2)-free."""
        # Convert to set of frozensets for fast lookup
        triple_set_frozen = set(frozenset(t) for t in triple_set)
        
        # For each combination of 6 points, check all 15 pairings
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    for l in range(k+1, n):
                        for m in range(l+1, n):
                            for o in range(m+1, n):
                                pts = [i, j, k, l, m, o]
                                # 15 ways to partition 6 elements into 3 pairs
                                # Pairings: (0,1)(2,3)(4,5), (0,1)(2,4)(3,5), etc.
                                # Let's generate them
                                pairs_list = [
                                    [(0,1),(2,3),(4,5)],
                                    [(0,1),(2,4),(3,5)],
                                    [(0,1),(2,5),(3,4)],
                                    [(0,2),(1,3),(4,5)],
                                    [(0,2),(1,4),(3,5)],
                                    [(0,2),(1,5),(3,4)],
                                    [(0,3),(1,2),(4,5)],
                                    [(0,3),(1,4),(2,5)],
                                    [(0,3),(1,5),(2,4)],
                                    [(0,4),(1,2),(3,5)],
                                    [(0,4),(1,3),(2,5)],
                                    [(0,4),(1,5),(2,3)],
                                    [(0,5),(1,2),(3,4)],
                                    [(0,5),(1,3),(2,4)],
                                    [(0,5),(1,4),(2,3)],
                                ]
                                for pairs in pairs_list:
                                    # Check if all 8 transversals are present
                                    all_present = True
                                    for mask in range(8):
                                        triple = tuple(pts[pairs[d][mask >> d & 1]] for d in range(3))
                                        if frozenset(triple) not in triple_set_frozen:
                                            all_present = False
                                            break
                                    if all_present:
                                        return False
        return True
    
    # This is too slow. Let me use a smarter approach.
    # 
    # Key insight: For a triple T=(a,b,c), it can only be part of K(2,2,2)s
    # where a is paired with some a'≠a, b with b'≠b, c with c'≠c.
    # The other 7 triples are determined.
    #
    # So when adding a new triple T, we only need to check if it completes
    # any K(2,2,2). For T=(a,b,c), the possible K(2,2,2)s are those where
    # a is in pair with a', b with b', c with c'.
    # The 8 triples are: (a,b,c), (a',b,c), (a,b',c), (a,b,c'),
    #                    (a',b',c), (a',b,c'), (a,b',c'), (a',b',c')
    #
    # So when we add T=(a,b,c), we need to check for all a'≠a, b'≠b, c'≠c
    # (with a',b',c' distinct from each other and from a,b,c) whether the
    # other 7 triples are all already in our set.
    
    def completes_k222(triple, existing_set):
        """Check if adding this triple completes a K(2,2,2)."""
        a, b, c = triple
        for a_prime in range(n):
            if a_prime == a:
                continue
            for b_prime in range(n):
                if b_prime == b or b_prime == a_prime:
                    continue
                for c_prime in range(n):
                    if c_prime == c or c_prime == a_prime or c_prime == b_prime:
                        continue
                    # Check if the other 7 triples are in existing_set
                    other_triples = [
                        (a_prime, b, c),
                        (a, b_prime, c),
                        (a, b, c_prime),
                        (a_prime, b_prime, c),
                        (a_prime, b, c_prime),
                        (a, b_prime, c_prime),
                        (a_prime, b_prime, c_prime),
                    ]
                    for ot in other_triples:
                        if frozenset(ot) not in existing_set:
                            break
                    else:
                        return True
        return False
    
    # Greedy randomized construction
    best_triples = []
    best_size = 0
    
    for trial in range(50):
        if time.time() > deadline - 5:
            break
            
        random.shuffle(all_triples)
        current_set = set()
        current_triples = []
        
        for triple in all_triples:
            if time.time() > deadline - 10:
                break
            if not completes_k222(triple, current_set):
                current_set.add(frozenset(triple))
                current_triples.append(list(triple))
        
        if len(current_triples) > best_size:
            best_size = len(current_triples)
            best_triples = current_triples[:]
        
        # Local search: try to add more triples
        if time.time() > deadline - 10:
            break
        
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            for triple in all_triples:
                if frozenset(triple) in current_set:
                    continue
                if not completes_k222(triple, current_set):
                    current_set.add(frozenset(triple))
                    current_triples.append(list(triple))
                    improved = True
        
        if len(current_triples) > best_size:
            best_size = len(current_triples)
            best_triples = current_triples[:]
    
    return best_triples
# EVOLVE-BLOCK-END
