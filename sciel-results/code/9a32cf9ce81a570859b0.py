import time
import random
from itertools import combinations

# Points are 0..15
V = list(range(16))
ALL_TRIPLES = list(combinations(V, 3))
# Map triple to index for fast lookup? Or just use set of tuples.
# Set of tuples is fine.

def count_k222(hypergraph_set):
    """
    Counts the number of K(2,2,2) configurations in the hypergraph.
    A K(2,2,2) is defined by 3 disjoint pairs.
    We iterate through all sets of 6 vertices, then all 15 partitions into 3 pairs.
    For each partition, check if all 8 triples are in the hypergraph.
    """
    count = 0
    # Precompute all 6-subsets and their partitions?
    # 120120 configurations total.
    # It is better to iterate and check.
    # However, for cleaning, we just need to find ANY.
    pass

def has_k222(hypergraph_set):
    """
    Returns True if there is at least one K(2,2,2) in the hypergraph.
    """
    # Iterate over all 6-subsets
    for six_verts in combinations(V, 6):
        # Get all 15 perfect matchings of these 6 vertices
        # Let the vertices be v0..v5
        # Matchings:
        # (0,1)(2,3)(4,5)
        # (0,1)(2,4)(3,5)
        # ...
        # There are 15.
        # We can generate them or hardcode the pattern.
        # Let's generate indices.
        # Actually, we can just check all 15.
        # Indices:
        # 0: (0,1),(2,3),(4,5)
        # 1: (0,1),(2,4),(3,5)
        # 2: (0,1),(2,5),(3,4)
        # 3: (0,2),(1,3),(4,5)
        # 4: (0,2),(1,4),(3,5)
        # 5: (0,2),(1,5),(3,4)
        # 6: (0,3),(1,2),(4,5)
        # 7: (0,3),(1,4),(2,5)
        # 8: (0,3),(1,5),(2,4)
        # 9: (0,4),(1,2),(3,5)
        # 10: (0,4),(1,3),(2,5)
        # 11: (0,4),(1,5),(2,3)
        # 12: (0,5),(1,2),(3,4)
        # 13: (0,5),(1,3),(2,4)
        # 14: (0,5),(1,4),(2,3)
        
        matchings = [
            ((0,1),(2,3),(4,5)),
            ((0,1),(2,4),(3,5)),
            ((0,1),(2,5),(3,4)),
            ((0,2),(1,3),(4,5)),
            ((0,2),(1,4),(3,5)),
            ((0,2),(1,5),(3,4)),
            ((0,3),(1,2),(4,5)),
            ((0,3),(1,4),(2,5)),
            ((0,3),(1,5),(2,4)),
            ((0,4),(1,2),(3,5)),
            ((0,4),(1,3),(2,5)),
            ((0,4),(1,5),(2,3)),
            ((0,5),(1,2),(3,4)),
            ((0,5),(1,3),(2,4)),
            ((0,5),(1,4),(2,3))
        ]
        
        for m in matchings:
            p1 = (six_verts[m[0][0]], six_verts[m[0][1]])
            p2 = (six_verts[m[1][0]], six_verts[m[1][1]])
            p3 = (six_verts[m[2][0]], six_verts[m[2][1]])
            
            # The 8 triples are all combinations of one from each pair
            t1 = (p1[0], p2[0], p3[0])
            t2 = (p1[0], p2[0], p3[1])
            t3 = (p1[0], p2[1], p3[0])
            t4 = (p1[0], p2[1], p3[1])
            t5 = (p1[1], p2[0], p3[0])
            t6 = (p1[1], p2[0], p3[1])
            t7 = (p1[1], p2[1], p3[0])
            t8 = (p1[1], p2[1], p3[1])
            
            # Sort them for set lookup
            # Actually, we can just check if they are in the set.
            # But the set contains sorted tuples.
            # So we must sort each triple.
            if (tuple(sorted(t1)) in hypergraph_set and
                tuple(sorted(t2)) in hypergraph_set and
                tuple(sorted(t3)) in hypergraph_set and
                tuple(sorted(t4)) in hypergraph_set and
                tuple(sorted(t5)) in hypergraph_set and
                tuple(sorted(t6)) in hypergraph_set and
                tuple(sorted(t7)) in hypergraph_set and
                tuple(sorted(t8)) in hypergraph_set):
                return True
    return False

def remove_one_k222(hypergraph_set):
    """
    Finds one K(2,2,2) and removes one of its edges.
    Returns the modified set.
    """
    for six_verts in combinations(V, 6):
        matchings = [
            ((0,1),(2,3),(4,5)),
            ((0,1),(2,4),(3,5)),
            ((0,1),(2,5),(3,4)),
            ((0,2),(1,3),(4,5)),
            ((0,2),(1,4),(3,5)),
            ((0,2),(1,5),(3,4)),
            ((0,3),(1,2),(4,5)),
            ((0,3),(1,4),(2,5)),
            ((0,3),(1,5),(2,4)),
            ((0,4),(1,2),(3,5)),
            ((0,4),(1,3),(2,5)),
            ((0,4),(1,5),(2,3)),
            ((0,5),(1,2),(3,4)),
            ((0,5),(1,3),(2,4)),
            ((0,5),(1,4),(2,3))
        ]
        for m in matchings:
            p1 = (six_verts[m[0][0]], six_verts[m[0][1]])
            p2 = (six_verts[m[1][0]], six_verts[m[1][1]])
            p3 = (six_verts[m[2][0]], six_verts[m[2][1]])
            
            triples = [
                tuple(sorted((p1[0], p2[0], p3[0]))),
                tuple(sorted((p1[0], p2[0], p3[1]))),
                tuple(sorted((p1[0], p2[1], p3[0]))),
                tuple(sorted((p1[0], p2[1], p3[1]))),
                tuple(sorted((p1[1], p2[0], p3[0]))),
                tuple(sorted((p1[1], p2[0], p3[1]))),
                tuple(sorted((p1[1], p2[1], p3[0]))),
                tuple(sorted((p1[1], p2[1], p3[1])))
            ]
            
            if all(t in hypergraph_set for t in triples):
                # Remove a random one
                idx = random.randint(0, 7)
                hypergraph_set.remove(triples[idx])
                return hypergraph_set
    return hypergraph_set

def can_add(hypergraph_set, triple):
    """
    Checks if adding 'triple' to 'hypergraph_set' creates a K(2,2,2).
    triple is a tuple of 3 distinct vertices.
    """
    u, v, w = triple
    # u, v, w are distinct.
    # We need to find u', v', w' such that pairs {u,u'}, {v,v'}, {w,w'} are disjoint
    # and the other 7 triples are in hypergraph_set.
    
    # To speed up, we can precompute the set of vertices not in {u,v,w}
    # u' cannot be u, v, w.
    # v' cannot be u, u', v, w.
    # w' cannot be u, u', v, v', w.
    
    # Iterate u'
    for u_prime in V:
        if u_prime == u or u_prime == v or u_prime == w:
            continue
        # Pair 1: {u, u'}
        # Iterate v'
        for v_prime in V:
            if v_prime == u or v_prime == u_prime or v_prime == v or v_prime == w:
                continue
            # Pair 2: {v, v'}
            # Check disjointness with Pair 1:
            # u != v (true), u != v' (true by loop), u' != v (true by loop), u' != v' (true by loop)
            
            # Iterate w'
            for w_prime in V:
                if w_prime == u or w_prime == u_prime or w_prime == v or w_prime == v_prime or w_prime == w:
                    continue
                # Pair 3: {w, w'}
                
                # Check if the 7 other triples are in hypergraph_set
                # The triples are formed by choosing one from each pair.
                # We have {u, u'}, {v, v'}, {w, w'}.
                # The added triple is (u, v, w).
                # The others are:
                t1 = (u, v, w_prime)
                t2 = (u, v_prime, w)
                t3 = (u, v_prime, w_prime)
                t4 = (u_prime, v, w)
                t5 = (u_prime, v, w_prime)
                t6 = (u_prime, v_prime, w)
                t7 = (u_prime, v_prime, w_prime)
                
                # Sort them
                if (tuple(sorted(t1)) in hypergraph_set and
                    tuple(sorted(t2)) in hypergraph_set and
                    tuple(sorted(t3)) in hypergraph_set and
                    tuple(sorted(t4)) in hypergraph_set and
                    tuple(sorted(t5)) in hypergraph_set and
                    tuple(sorted(t6)) in hypergraph_set and
                    tuple(sorted(t7)) in hypergraph_set):
                    return False
    return True

def construct():
    start_time = time.time()
    time_limit = 150.0
    best_size = 0
    best_set = set()
    
    # Precompute sorted tuples for all triples
    all_triples_sorted = [tuple(sorted(t)) for t in ALL_TRIPLES]
    
    while time.time() - start_time < time_limit - 2.0: # 2s safety margin
        random.seed()
        
        # Strategy: Random Start + Greedy Fill
        # Try different probabilities for initial set
        p = random.choice([0.1, 0.15, 0.2, 0.25])
        
        current_set = set()
        for t in all_triples_sorted:
            if random.random() < p:
                current_set.add(t)
                
        # Clean up initial set
        # Remove edges from K(2,2,2)s until none exist
        # To avoid infinite loop, limit iterations or check has_k222
        # has_k222 is expensive (120k checks).
        # Instead, we can just try to remove a few times.
        # Or, better: just run greedy from empty?
        # The seed is 105. Random p=0.2 gives ~112.
        # Cleaning might reduce it to ~100.
        # Then greedy adds more.
        
        # Let's try a simpler cleaning:
        # If we have a K(2,2,2), remove a random edge from it.
        # Repeat a fixed number of times?
        # Actually, if we start with p=0.2, expected # K(2,2,2) is 120120 * 0.2^8 = 120120 * 2.56e-6 = 0.3.
        # So it's likely there are 0 or 1.
        # If p=0.3, 120120 * 0.3^8 = 120120 * 6.5e-5 = 7.8.
        # So for p<=0.2, cleaning is very cheap.
        
        # Let's just run remove_one_k222 until no K(2,2,2) found.
        # But has_k222 is slow.
        # We can just call remove_one_k222 which returns the set (modified or not).
        # If it modified, repeat.
        # But remove_one_k222 scans all 120k configs.
        # So one call is ~120k * 8 lookups = 1M lookups. ~0.1s.
        # So we can do ~10-20 cleanings.
        
        # Optimization: Only clean if we think there are K(2,2,2)s.
        # Or just do a few passes.
        
        # Let's do:
        for _ in range(20):
            if has_k222(current_set):
                remove_one_k222(current_set)
            else:
                break
        
        # Now current_set is valid.
        
        # Greedy Fill
        remaining = [t for t in all_triples_sorted if t not in current_set]
        random.shuffle(remaining)
        
        for t in remaining:
            if can_add(current_set, t):
                current_set.add(t)
                
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = set(current_set)
            
    # Return the best set found
    return [list(t) for t in best_set]

# To ensure we don't crash if best_set is empty (shouldn't happen)
# But construct() must return a list of lists.

# Let's run a quick mental check on performance.
# has_k222: 120,120 iterations. Inside: 8 set lookups.
# 120,120 * 8 = ~1M lookups.
# In Python, 1M lookups is ~0.1s.
# So one has_k222 call is ~0.1s.
# If we call it 20 times, that's 2s.
# can_add: 2730 iterations * 7 lookups = ~20k lookups.
# 560 triples * 20k = 11M lookups. ~1s.
# Total per run: ~3s.
# 150s / 3s = 50 runs.
# This is acceptable.

# We can optimize has_k222 by not checking all 120k if we find one early?
# No, we need to know if ANY exists.
# But remove_one_k222 already does that and removes one.
# So we can just call remove_one_k222 in a loop.
# If it returns without modifying, we are clean.
# But how do we know it didn't modify?
# We can check size before and after.
# Or modify remove_one_k222 to return a boolean.

def remove_one_k222_optimized(hypergraph_set):
    """
    Returns True if a K(2,2,2) was found and an edge removed.
    """
    for six_verts in combinations(V, 6):
        matchings = [
            ((0,1),(2,3),(4,5)),
            ((0,1),(2,4),(3,5)),
            ((0,1),(2,5),(3,4)),
            ((0,2),(1,3),(4,5)),
            ((0,2),(1,4),(3,5)),
            ((0,2),(1,5),(3,4)),
            ((0,3),(1,2),(4,5)),
            ((0,3),(1,4),(2,5)),
            ((0,3),(1,5),(2,4)),
            ((0,4),(1,2),(3,5)),
            ((0,4),(1,3),(2,5)),
            ((0,4),(1,5),(2,3)),
            ((0,5),(1,2),(3,4)),
            ((0,5),(1,3),(2,4)),
            ((0,5),(1,4),(2,3))
        ]
        for m in matchings:
            p1 = (six_verts[m[0][0]], six_verts[m[0][1]])
            p2 = (six_verts[m[1][0]], six_verts[m[1][1]])
            p3 = (six_verts[m[2][0]], six_verts[m[2][1]])
            
            triples = [
                tuple(sorted((p1[0], p2[0], p3[0]))),
                tuple(sorted((p1[0], p2[0], p3[1]))),
                tuple(sorted((p1[0], p2[1], p3[0]))),
                tuple(sorted((p1[0], p2[1], p3[1]))),
                tuple(sorted((p1[1], p2[0], p3[0]))),
                tuple(sorted((p1[1], p2[0], p3[1]))),
                tuple(sorted((p1[1], p2[1], p3[0]))),
                tuple(sorted((p1[1], p2[1], p3[1])))
            ]
            
            if all(t in hypergraph_set for t in triples):
                idx = random.randint(0, 7)
                hypergraph_set.remove(triples[idx])
                return True
    return False

# Update construct to use this.

