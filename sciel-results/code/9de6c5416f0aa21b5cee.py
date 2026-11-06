# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 150s kill
    
    n = 16
    # Generate all C(16,3) = 560 possible triples
    all_triples = []
    for a in range(n):
        for b in range(a+1, n):
            for c in range(b+1, n):
                all_triples.append((a, b, c))
    
    # Precompute conflict structure:
    # For each triple t, find which "pair-triples" (sets of 3 disjoint pairs) 
    # it participates in, and count how many of the 8 transversals are present.
    
    # A "pair-triple" is a set of 3 disjoint pairs = 6 distinct points.
    # There are C(16,6) * 15 ways to partition 6 points into 3 pairs... 
    # Actually, number of ways to partition 6 labeled points into 3 unordered pairs = 15.
    # Total pair-triples = C(16,6) * 15 = 8008 * 15 = 120120. That's a lot.
    
    # Alternative: for each triple, determine which 3-pair configurations it "completes".
    # A triple {x,y,z} is one of the 8 transversals of a pair-triple if the 3 points 
    # can be assigned one from each pair.
    
    # Better approach: track for each pair-triple how many of its 8 transversals are in our set.
    # But 120120 pair-triples is manageable if we're smart.
    
    # Actually, let me think differently. For a given triple {a,b,c}, it conflicts 
    # (contributes to completing) with pair-triples where {a,b,c} is a transversal.
    # {a,b,c} is a transversal of pairs (p1,p2,p3) where each pi is a 2-element subset
    # of {a,b,c}... no, that's not right either.
    
    # Let me reconsider. The 8 transversals of pairs (a,b),(c,d),(e,f) are:
    # {a,c,e},{a,c,f},{a,d,e},{a,d,f},{b,c,e},{b,c,f},{b,d,e},{b,d,f}
    # So a triple {x,y,z} is a transversal iff {x,y,z} intersects each pair in exactly one element.
    
    # For triple {a,b,c} to be a transversal of pairs (p1,p2,p3):
    # p1, p2, p3 are disjoint pairs, and |{a,b,c} ∩ pi| = 1 for each i.
    
    # This means each pair contains exactly one of {a,b,c} and one other point.
    # The "other points" must be distinct from each other and from {a,b,c}.
    
    # So: choose which point from {a,b,c} goes to which pair (3! = 6 ways),
    # then choose 3 distinct "other" points from the remaining 13 points,
    # and assign them to the pairs (but the assignment to specific pairs is determined
    # by which point from {a,b,c} is in that pair).
    
    # Actually: fix the pairing. Say a is in p1, b in p2, c in p3.
    # Then p1 = {a, x}, p2 = {b, y}, p3 = {c, z} where x,y,z are distinct, 
    # not in {a,b,c}, and x≠y≠z.
    # Number of choices: P(13,3) = 13*12*11 = 1716.
    # And 6 ways to assign {a,b,c} to the 3 pairs.
    # Total pair-triples containing {a,b,c} as transversal: 6 * 1716 = 10296.
    
    # That's too many to precompute per triple. Let me use a different strategy.
    
    # Greedy with incremental conflict tracking:
    # Maintain for each pair-triple a count of how many of its 8 transversals are selected.
    # When adding a triple, increment counts for all pair-triples where it's a transversal.
    # If any count reaches 8, we have a conflict (but actually we need to check BEFORE adding).
    
    # The issue is memory: 120120 pair-triples, each needing a count. That's fine.
    # But iterating over 10296 pair-triples per candidate triple is expensive.
    
    # Alternative: use a different conflict check. 
    # When considering adding triple t, check if it would complete any pair-triple.
    # t completes a pair-triple if there exist 2 other points x,y such that... 
    # Hmm, this is getting complex.
    
    # Let me try a simpler approach: greedy with randomized restarts and local search.
    # For conflict checking, use the fact that a conflict occurs iff there exist 
    # three disjoint pairs where all 8 transversals are in our set.
    
    # Efficient check: for a given set S of triples, a conflict exists iff
    # there are 3 disjoint pairs (a,b),(c,d),(e,f) such that all 8 transversals are in S.
    
    # To check if adding triple t creates a conflict: t must be one of the 8 transversals
    # of some pair-triple where the other 7 are already in S.
    
    # So: for triple t = {a,b,c}, check all ways to extend it to a pair-triple.
    # t is a transversal of pairs (p1,p2,p3) where each pi contains exactly one of {a,b,c}.
    # WLOG, assign a→p1, b→p2, c→p3 (we'll handle all 6 permutations).
    # Then p1={a,x}, p2={b,y}, p3={c,z} for distinct x,y,z ∉ {a,b,c}.
    # The other 7 transversals are:
    #   {b,x,e}... wait, let me recompute.
    #   Pairs: (a,x), (b,y), (c,z)
    #   Transversals: {a,b,c}, {a,b,z}, {a,y,c}, {a,y,z}, {x,b,c}, {x,b,z}, {x,y,c}, {x,y,z}
    #   So the other 7 are: {a,b,z}, {a,y,c}, {a,y,z}, {x,b,c}, {x,b,z}, {x,y,c}, {x,y,z}
    
    # For each such extension, check if all 7 other transversals are in S.
    # Number of extensions: 6 * C(13,3) = 6 * 286 = 1716... wait, C(13,3)=286, and we need
    # to assign 3 distinct points to x,y,z, which is P(13,3)=1716, times 6 permutations = 10296.
    
    # That's still a lot per candidate. But with 560 total triples and maybe 300-400 in the solution,
    # we might be okay if we're smart.
    
    # Let me try a different approach: use bitsets for efficiency.
    
    # Represent each triple as a bitmask of 16 bits (3 bits set).
    # For each pair-triple, store a bitmask of which of its 8 transversals are present.
    
    # Actually, let me just do a straightforward greedy with a reasonable time budget,
    # using randomized ordering and local improvement.
    
    def check_conflict(triple, selected_set):
        """Check if adding triple would create a conflict. Return True if conflict."""
        a, b, c = triple
        # For each permutation of assigning a,b,c to the 3 pairs
        for perm in [(a,b,c), (a,c,b), (b,a,c), (b,c,a), (c,a,b), (c,b,a)]:
            p1, p2, p3 = perm
            # p1 is in pair1, p2 in pair2, p3 in pair3
            # Other points x,y,z must be distinct and not in {a,b,c}
            others = [i for i in range(16) if i not in (a,b,c)]
            for i in range(len(others)):
                x = others[i]
                for j in range(i+1, len(others)):
                    y = others[j]
                    for k in range(j+1, len(others)):
                        z = others[k]
                        # Pairs: (p1,x), (p2,y), (p3,z)
                        # Other 7 transversals:
                        # {p1,p2,z}, {p1,y,p3}, {p1,y,z}, {x,p2,p3}, {x,p2,z}, {x,y,p3}, {x,y,z}
                        t1 = frozenset([p1,p2,z])
                        t2 = frozenset([p1,y,p3])
                        t3 = frozenset([p1,y,z])
                        t4 = frozenset([x,p2,p3])
                        t5 = frozenset([x,p2,z])
                        t6 = frozenset([x,y,p3])
                        t7 = frozenset([x,y,z])
                        if all(t in selected_set for t in [t1,t2,t3,t4,t5,t6,t7]):
                            return True
        return False
    
    best_result = []
    
    for attempt in range(50):
        if time.time() > deadline - 5:
            break
            
        random.seed(attempt)
        shuffled = all_triples[:]
        random.shuffle(shuffled)
        
        selected = []
        selected_set = set()
        
        for t in shuffled:
            if time.time() > deadline - 5:
                break
            if not check_conflict(t, selected_set):
                selected.append(t)
                selected_set.add(frozenset(t))
        
        # Local improvement: try to add any remaining triple
        improved = True
        while improved:
            if time.time() > deadline - 5:
                break
            improved = False
            remaining = [t for t in all_triples if frozenset(t) not in selected_set]
            random.shuffle(remaining)
            for t in remaining:
                if time.time() > deadline - 5:
                    break
                if not check_conflict(t, selected_set):
                    selected.append(t)
                    selected_set.add(frozenset(t))
                    improved = True
        
        if len(selected) > len(best_result):
            best_result = selected[:]
    
    # Convert to list of lists
    return [list(t) for t in best_result]
# EVOLVE-BLOCK-END
