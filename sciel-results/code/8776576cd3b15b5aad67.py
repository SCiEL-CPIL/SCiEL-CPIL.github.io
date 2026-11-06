"""Improved construction for ex_3(16, K(2,2,2)) via greedy + local search."""
import random
import time
import itertools
import sys

# Deadline: must finish within 120s wall-clock; use 110s as safety margin
DEADLINE = 110.0

def _all_triples():
    """Generate all C(16,3) = 560 triples."""
    return [list(t) for t in itertools.combinations(range(16), 3)]

def _is_valid(triple_set, triples_list):
    """
    Check that no three disjoint pairs have all eight transversal triples present.
    
    K(2,2,2) in the 3-partite sense: we have 3 parts, each of size 2.
    Points 0..15. A K(2,2,2) configuration is:
      - Three disjoint pairs: {a1,a2}, {b1,b2}, {c1,c2}
      - The 8 transversal triples: {a_i, b_j, c_k} for i,j,k in {1,2}
    
    So we need to check: for any choice of 3 disjoint pairs (6 distinct points),
    are all 8 triples of the form {a_i, b_j, c_k} present in our triple set?
    
    If yes, that's a forbidden K(2,2,2).
    """
    ts = set()
    for t in triple_set:
        ts.add(tuple(sorted(t)))
    
    points = list(range(16))
    # Enumerate all ways to pick 3 disjoint pairs from 16 points
    # This is C(16,6) * 6! / (2^3 * 3!) = 8008 * 60 = 480480 configurations
    # But we can be smarter: enumerate 6 distinct points, then check all 15 ways to partition into 3 pairs
    # Actually, for each 6-subset, there are 15 ways to partition into 3 unordered pairs.
    # For each such partition, we need to check if all 8 transversal triples are present.
    
    for six in itertools.combinations(points, 6):
        # All ways to partition 6 elements into 3 pairs: 15 ways
        # Fix first element, pair it with one of the other 5, then pair remaining 4 into 2 pairs (3 ways)
        a = six[0]
        rest = list(six[1:])
        for i in range(5):
            b = rest[i]
            remaining = rest[:i] + rest[i+1:]
            for j in range(3):
                c = remaining[j]
                rem2 = remaining[:j] + remaining[j+1:]
                d, e = rem2[0], rem2[1]
                # Pairs: {a,b}, {c,d}, {e,f} where f is the last
                f = rem2[1] if rem2[0] != rem2[1] else rem2[0]
                # Actually let's just be explicit
                pair1 = (a, b)
                pair2 = (c, rem2[0] if rem2[0] != c else rem2[1])
                # This is getting messy. Let me use a cleaner approach.
                pass
        # Let me redo this more carefully
        # For 6 elements x0..x5, the 15 perfect matchings:
        # Fix x0. It pairs with x1..x5 (5 choices). Then 4 remain, 3 matchings each.
        for i in range(1, 6):
            p1 = (six[0], six[i])
            rem = [six[j] for j in range(6) if j != 0 and j != i]
            # 3 ways to pair rem[0],rem[1],rem[2],rem[3]
            # (rem[0],rem[1]) & (rem[2],rem[3])
            # (rem[0],rem[2]) & (rem[1],rem[3])
            # (rem[0],rem[3]) & (rem[1],rem[2])
            for k in range(3):
                if k == 0:
                    p2 = (rem[0], rem[1])
                    p3 = (rem[2], rem[3])
                elif k == 1:
                    p2 = (rem[0], rem[2])
                    p3 = (rem[1], rem[3])
                else:
                    p2 = (rem[0], rem[3])
                    p3 = (rem[1], rem[2])
                
                # Check all 8 transversal triples
                a1, a2 = p1
                b1, b2 = p2
                c1, c2 = p3
                all_present = True
                for ai in (a1, a2):
                    for bi in (b1, b2):
                        for ci in (c1, c2):
                            triple = tuple(sorted((ai, bi, ci)))
                            if triple not in ts:
                                all_present = False
                                break
                        if not all_present:
                            break
                    if not all_present:
                        break
                if all_present:
                    return False
    return True

def _is_valid_fast(triple_set):
    """
    Faster validity check using a different approach.
    For each triple, we can track which K(2,2,2) configurations it participates in.
    
    Actually, let's think about it differently. A K(2,2,2) is determined by 6 points
    partitioned into 3 pairs. The 8 transversal triples must all be present.
    
    Let's precompute for each 6-subset whether it forms a K(2,2,2).
    But that's still 8008 * 15 = 120120 checks per validity test, which is too slow
    if we do it many times.
    
    Alternative: incremental check. When adding a triple, only new K(2,2,2)s
    that include this triple can be formed. A triple {a,b,c} can be part of a
    K(2,2,2) in limited ways.
    
    For triple {a,b,c} to be a transversal triple in a K(2,2,2), the 6 points are
    {a, a', b, b', c, c'} where a' != a, b' != b, c' != c, and the pairs are
    {a,a'}, {b,b'}, {c,c'}. The other 7 transversal triples are:
    {a',b,c}, {a,b',c}, {a,b,c'}, {a',b',c}, {a',b,c'}, {a,b',c'}, {a',b',c'}
    
    So when adding triple {a,b,c}, we need to check for each possible a', b', c'
    (a' in 0..15 \ {a}, etc.) whether the other 7 triples are all present.
    That's 15*15*15 = 3375 checks per triple addition, which is manageable.
    
    But we also need to handle the case where the triple could be a transversal
    triple in multiple ways (e.g., {a,b,c} could have a as the "a" point or as
    the "b" point, etc.). Actually no - in a K(2,2,2), each of the 3 pairs is
    distinct, and the triple {a,b,c} has one point from each pair. So if we fix
    which point is from which pair, it's determined.
    
    Let me just use the straightforward approach but optimize it.
    """
    ts = set()
    for t in triple_set:
        ts.add(tuple(sorted(t)))
    
    points = list(range(16))
    for six in itertools.combinations(points, 6):
        for i in range(1, 6):
            p1 = (six[0], six[i])
            rem = [six[j] for j in range(6) if j != 0 and j != i]
            for k in range(3):
                if k == 0:
                    p2 = (rem[0], rem[1])
                    p3 = (rem[2], rem[3])
                elif k == 1:
                    p2 = (rem[0], rem[2])
                    p3 = (rem[1], rem[3])
                else:
                    p2 = (rem[0], rem[3])
                    p3 = (rem[1], rem[2])
                
                a1, a2 = p1
                b1, b2 = p2
                c1, c2 = p3
                all_present = True
                for ai in (a1, a2):
                    for bi in (b1, b2):
                        for ci in (c1, c2):
                            triple = tuple(sorted((ai, bi, ci)))
                            if triple not in ts:
                                all_present = False
                                break
                        if not all_present:
                            break
                    if not all_present:
                        break
                if all_present:
                    return False
    return True

def _count_violations(triple_set):
    """Count the number of K(2,2,2) configurations present."""
    ts = set()
    for t in triple_set:
        ts.add(tuple(sorted(t)))
    
    count = 0
    points = list(range(16))
    for six in itertools.combinations(points, 6):
        for i in range(1, 6):
            p1 = (six[0], six[i])
            rem = [six[j] for j in range(6) if j != 0 and j != i]
            for k in range(3):
                if k == 0:
                    p2 = (rem[0], rem[1])
                    p3 = (rem[2], rem[3])
                elif k == 1:
                    p2 = (rem[0], rem[2])
                    p3 = (rem[1], rem[3])
                else:
                    p2 = (rem[0], rem[3])
                    p3 = (rem[1], rem[2])
                
                a1, a2 = p1
                b1, b2 = p2
                c1, c2 = p3
                all_present = True
                for ai in (a1, a2):
                    for bi in (b1, b2):
                        for ci in (c1, c2):
                            triple = tuple(sorted((ai, bi, ci)))
                            if triple not in ts:
                                all_present = False
                                break
                        if not all_present:
                            break
                    if not all_present:
                        break
                if all_present:
                    count += 1
    return count

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    # Start with all triples through point 0 (the seed): C(15,2) = 105 triples
    # This is guaranteed valid because any K(2,2,2) using point 0 would need
    # 6 points including 0, but all our triples contain 0, so the 8 transversal
    # triples would all need to contain 0, which is impossible for 3 disjoint pairs.
    
    all_triples = _all_triples()
    current_set = set(tuple(sorted(t)) for t in [[0, a, b] for a in range(1, 16) for b in range(a + 1, 16)])
    current_list = [list(t) for t in current_set]
    
    # Greedy: try to add triples one by one, skipping those that create a K(2,2,2)
    # Sort triples by some heuristic - maybe by number of "conflicts" they'd create
    remaining = [t for t in all_triples if tuple(sorted(t)) not in current_set]
    
    # For efficiency, let's precompute for each triple which 6-subsets it could be part of
    # in a K(2,2,2). Actually, let's just try adding and check validity incrementally.
    
    # Incremental check: when adding triple T = {a,b,c}, check if it completes any K(2,2,2).
    # T is a transversal triple if the 6 points are {a,a',b,b',c,c'} with pairs {a,a'},{b,b'},{c,c'}.
    # The other 7 triples must all be in current_set.
    
    def _check_addable(triple, current_set):
        """Check if adding triple to current_set would create a K(2,2,2)."""
        a, b, c = triple
        # For each choice of a', b', c' (distinct from a,b,c respectively, and all 6 distinct)
        for a_prime in range(16):
            if a_prime == a:
                continue
            for b_prime in range(16):
                if b_prime == b or b_prime == a_prime:
                    continue
                for c_prime in range(16):
                    if c_prime == c or c_prime == a_prime or c_prime == b_prime:
                        continue
                    # Check the other 7 transversal triples
                    other_triples = [
                        (a_prime, b, c),
                        (a, b_prime, c),
                        (a, b, c_prime),
                        (a_prime, b_prime, c),
                        (a_prime, b, c_prime),
                        (a, b_prime, c_prime),
                        (a_prime, b_prime, c_prime),
                    ]
                    # Also need to make sure a,b,c are all distinct from a',b',c'
                    # which we already checked
                    all_present = True
                    for ot in other_triples:
                        ot_sorted = tuple(sorted(ot))
                        if ot_sorted not in current_set:
                            all_present = False
                            break
                    if all_present:
                        return False
        return True
    
    # Greedy addition
    # Sort remaining triples by a heuristic: prefer triples with fewer "near-misses"
    # For simplicity, just iterate and add if possible
    added = 0
    for triple in remaining:
        if time.time() > deadline - 5:
            break
        triple_sorted = tuple(sorted(triple))
        if _check_addable(triple_sorted, current_set):
            current_set.add(triple_sorted)
            added += 1
    
    current_list = [list(t) for t in current_set]
    best_list = list(current_list)
    best_size = len(current_list)
    
    # Local search: try swapping triples to improve
    # Or: try removing a triple and adding multiple others
    # For now, let's do a simple local search: try to replace a triple with two others
    
    # Actually, let's try a different approach: start from scratch with a randomized greedy
    # and keep the best result
    
    rng = random.Random(42)
    
    for iteration in range(20):
        if time.time() > deadline - 10:
            break
        
        # Randomized greedy
        trial_set = set(tuple(sorted(t)) for t in [[0, a, b] for a in range(1, 16) for b in range(a + 1, 16)])
        
        # Shuffle remaining triples
        trial_remaining = [t for t in all_triples if tuple(sorted(t)) not in trial_set]
        rng.shuffle(trial_remaining)
        
        for triple in trial_remaining:
            if time.time() > deadline - 10:
                break
            triple_sorted = tuple(sorted(triple))
            if _check_addable(triple_sorted, trial_set):
                trial_set.add(triple_sorted)
        
        if len(trial_set) > best_size:
            best_size = len(trial_set)
            best_list = [list(t) for t in trial_set]
    
    # Final local search: try to remove one triple and add two
    if time.time() < deadline - 20:
        current_set = set(tuple(sorted(t)) for t in best_list)
        
        # Try removing each triple and see if we can add more
        for i in range(len(best_list)):
            if time.time() > deadline - 10:
                break
            
            trial_set = set(current_set)
            removed = best_list[i]
            removed_sorted = tuple(sorted(removed))
            trial_set.discard(removed_sorted)
            
            # Try adding triples that weren't in the original set
            for triple in all_triples:
                triple_sorted = tuple(sorted(triple))
                if triple_sorted in current_set:
                    continue
                if _check_addable(triple_sorted, trial_set):
                    trial_set.add(triple_sorted)
            
            if len(trial_set) > best_size:
                best_size = len(trial_set)
                best_list = [list(t) for t in trial_set]
                current_set = trial_set
    
    return best_list

# Precompute all triples at module level for speed
_ALL_TRIPLES = _all_triples()

# Override the internal function to use the precomputed list
def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    all_triples = _ALL_TRIPLES
    
    # Start with all triples through point 0
    current_set = set(tuple(sorted(t)) for t in [[0, a, b] for a in range(1, 16) for b in range(a + 1, 16)])
    current_list = [list(t) for t in current_set]
    
    def _check_addable(triple, current_set):
        a, b, c = triple
        for a_prime in range(16):
            if a_prime == a:
                continue
            for b_prime in range(16):
                if b_prime == b or b_prime == a_prime:
                    continue
                for c_prime in range(16):
                    if c_prime == c or c_prime == a_prime or c_prime == b_prime:
                        continue
                    other_triples = [
                        (a_prime, b, c),
                        (a, b_prime, c),
                        (a, b, c_prime),
                        (a_prime, b_prime, c),
                        (a_prime, b, c_prime),
                        (a, b_prime, c_prime),
                        (a_prime, b_prime, c_prime),
                    ]
                    all_present = True
                    for ot in other_triples:
                        ot_sorted = tuple(sorted(ot))
                        if ot_sorted not in current_set:
                            all_present = False
                            break
                    if all_present:
                        return False
        return True
    
    # Greedy addition
    remaining = [t for t in all_triples if tuple(sorted(t)) not in current_set]
    for triple in remaining:
        if time.time() > deadline - 5:
            break
        triple_sorted = tuple(sorted(triple))
        if _check_addable(triple_sorted, current_set):
            current_set.add(triple_sorted)
    
    current_list = [list(t) for t in current_set]
    best_list = list(current_list)
    best_size = len(current_list)
    
    # Randomized greedy restarts
    rng = random.Random(42)
    
    for iteration in range(30):
        if time.time() > deadline - 10:
            break
        
        trial_set = set(tuple(sorted(t)) for t in [[0, a, b] for a in range(1, 16) for b in range(a + 1, 16)])
        
        trial_remaining = [t for t in all_triples if tuple(sorted(t)) not in trial_set]
        rng.shuffle(trial_remaining)
        
        for triple in trial_remaining:
            if time.time() > deadline - 10:
                break
            triple_sorted = tuple(sorted(triple))
            if _check_addable(triple_sorted, trial_set):
                trial_set.add(triple_sorted)
        
        if len(trial_set) > best_size:
            best_size = len(trial_set)
            best_list = [list(t) for t in trial_set]
    
    # Local search: remove one, add many
    if time.time() < deadline - 20:
        current_set = set(tuple(sorted(t)) for t in best_list)
        
        for i in range(len(best_list)):
            if time.time() > deadline - 10:
                break
            
            trial_set = set(current_set)
            removed = best_list[i]
            removed_sorted = tuple(sorted(removed))
            trial_set.discard(removed_sorted)
            
            for triple in all_triples:
                triple_sorted = tuple(sorted(triple))
                if triple_sorted in current_set:
                    continue
                if _check_addable(triple_sorted, trial_set):
                    trial_set.add(triple_sorted)
            
            if len(trial_set) > best_size:
                best_size = len(trial_set)
                best_list = [list(t) for t in trial_set]
                current_set = trial_set
    
    return best_list
