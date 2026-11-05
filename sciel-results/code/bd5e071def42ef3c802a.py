"""Max K(2,2,2)-free triple system on 16 points via greedy + local search."""
import time
import random
import itertools

def construct():
    deadline = time.time() + 110  # safety margin before 120s limit
    random.seed(42)

    N = 16
    all_triples = list(itertools.combinations(range(N), 3))
    # Index triples for O(1) lookup
    triple_set = set()

    def check_violation(triple, triples_set):
        """Check if adding this triple creates a K(2,2,2).
        A K(2,2,2) needs 3 disjoint pairs covering 6 points with all 8 transversals.
        For a new triple (x,y,z), check if there exist points a,b,c,d,e,f
        such that {x,y,z} is one of the 8 transversals of pairs (a,b),(c,d),(e,f).
        
        The 8 transversals of pairs (p1,p2),(q1,q2),(r1,r2) are all (pi,qj,rk).
        So (x,y,z) is a transversal iff x is in one pair, y in another, z in third.
        
        For each way to assign x,y,z to three different pairs (6 ways),
        we need to find the "other" points in each pair.
        """
        x, y, z = triple
        # For each permutation of which pair contains which of x,y,z
        # Pair1 contains x, pair2 contains y, pair3 contains z:
        # Need: a in pair1 with x (a!=x), c in pair2 with y (c!=y), e in pair3 with z (e!=z)
        # All 8 transversals must be in triples_set.
        # Transversals: (x,y,z),(x,y,e),(x,c,z),(x,c,e),(a,y,z),(a,y,e),(a,c,z),(a,c,e)
        # We know (x,y,z) is being added (or already in set).
        # Check if all other 7 are in triples_set.
        
        for perm in itertools.permutations([x, y, z]):
            px, py, pz = perm
            # px is in pair1, py in pair2, pz in pair3
            # pair1 = {px, a}, pair2 = {py, c}, pair3 = {pz, e}
            # a, c, e must be distinct and not in {px,py,pz}
            # Check all 8 transversals
            for a in range(N):
                if a == px or a == py or a == pz:
                    continue
                for c in range(N):
                    if c == px or c == py or c == pz or c == a:
                        continue
                    for e in range(N):
                        if e == px or e == py or e == pz or e == a or e == c:
                            continue
                        # Check all 8 transversals
                        t1 = tuple(sorted((px, py, pz)))
                        t2 = tuple(sorted((px, py, e)))
                        t3 = tuple(sorted((px, c, pz)))
                        t4 = tuple(sorted((px, c, e)))
                        t5 = tuple(sorted((a, py, pz)))
                        t6 = tuple(sorted((a, py, e)))
                        t7 = tuple(sorted((a, c, pz)))
                        t8 = tuple(sorted((a, c, e)))
                        # t1 is our triple (being added), check others are in set
                        if (t2 in triples_set and t3 in triples_set and t4 in triples_set and
                            t5 in triples_set and t6 in triples_set and t7 in triples_set and
                            t8 in triples_set):
                            return True
        return False

    def count_violations(triples_list):
        """Count number of K(2,2,2) violations in the triple system."""
        ts = set(map(tuple, triples_list))
        count = 0
        # For each set of 6 points, check if they form a K(2,2,2)
        # There are C(16,6) = 8008 sets of 6 points
        # For each, check all ways to partition into 3 pairs
        for six in itertools.combinations(range(N), 6):
            # Number of ways to partition 6 points into 3 unordered pairs: 15
            # Actually: (6-1)!! = 15
            points = list(six)
            used = [False] * 6
            # Generate all perfect matchings
            def gen_matchings(idx, matching):
                nonlocal used
                if idx == 6:
                    yield matching
                    return
                if used[idx]:
                    yield from gen_matchings(idx + 1, matching)
                    return
                for j in range(idx + 1, 6):
                    if not used[j]:
                        used[idx] = used[j] = True
                        yield from gen_matchings(idx + 1, matching + [(points[idx], points[j])])
                        used[idx] = used[j] = False
            
            for matching in gen_matchings(0, []):
                # Check all 8 transversals
                (a, b), (c, d), (e, f) = matching
                ok = True
                for x in (a, b):
                    for y in (c, d):
                        for z in (e, f):
                            t = tuple(sorted((x, y, z)))
                            if t not in ts:
                                ok = False
                                break
                        if not ok:
                            break
                    if not ok:
                        break
                if ok:
                    count += 1
        return count

    # Greedy construction: add triples in random order if they don't create a violation
    best_triples = []
    best_count = 0

    def greedy_build():
        random.shuffle(all_triples)
        ts = set()
        triples = []
        for t in all_triples:
            if check_violation(t, ts):
                continue
            ts.add(t)
            triples.append(list(t))
        return triples, len(triples)

    # Initial greedy
    triples, cnt = greedy_build()
    best_triples = triples
    best_count = cnt
    best_ts = set(map(tuple, triples))

    # Local search: try to add more triples, or swap
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try adding each missing triple
        ts = set(map(tuple, best_triples))
        for t in all_triples:
            t_tuple = tuple(t)
            if t_tuple in ts:
                continue
            if not check_violation(t_tuple, ts):
                ts.add(t_tuple)
                best_triples.append(list(t_tuple))
                best_count += 1
                improved = True
        # Try swaps: remove one, add two
        if not improved and time.time() < deadline:
            # Try removing a triple and adding two others
            for i in range(len(best_triples)):
                if time.time() > deadline:
                    break
                removed = tuple(best_triples[i])
                ts = set(map(tuple, best_triples))
                ts.discard(removed)
                # Try adding pairs of triples
                for j in range(len(best_triples)):
                    if j == i or time.time() > deadline:
                        continue
                    t1 = tuple(best_triples[j])
                    if t1 in ts:
                        continue
                    if not check_violation(t1, ts):
                        ts_new = ts | {t1}
                        for k in range(len(best_triples)):
                            if k == i or k == j or time.time() > deadline:
                                continue
                            t2 = tuple(best_triples[k])
                            if t2 in ts_new:
                                continue
                            if not check_violation(t2, ts_new):
                                # Swap: remove best_triples[i], add t1 and t2
                                new_triples = [list(x) for x in ts_new]
                                new_count = len(new_triples)
                                if new_count > best_count:
                                    best_triples = new_triples
                                    best_count = new_count
                                    improved = True
                                    break
                        if improved:
                            break
                if improved:
                    break

    return best_triples
