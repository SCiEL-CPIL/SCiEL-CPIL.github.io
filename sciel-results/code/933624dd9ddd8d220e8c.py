import itertools
import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    n = 16
    triples = []
    triple_to_idx = {}
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                t = (i, j, k)
                triple_to_idx[t] = len(triples)
                triples.append(t)
                
    total_triples = len(triples)
    present = [True] * total_triples
    
    # Precompute partitions of 6 indices into 3 pairs
    # We represent a partition as a tuple of 3 pairs of indices (0-5)
    # But we need to map to actual vertices.
    # Let's precompute the structure.
    partitions = []
    # 0 paired with 1..5
    for i in range(1, 6):
        rem = [j for j in range(6) if j != 0 and j != i]
        # rem has 4 elements
        # 3 ways to pair 4 elements
        # (0,1)-(2,3), (0,2)-(1,3), (0,3)-(1,2)
        # rem[0] with rem[1], rem[2], rem[3]
        for j in range(1, 4):
            p1 = (0, i)
            p2 = (rem[0], rem[j])
            # The remaining two
            rem2 = [x for x in rem if x != rem[j]]
            p3 = (rem2[0], rem2[1])
            partitions.append((p1, p2, p3))
            
    # Verify 15 partitions
    # assert len(partitions) == 15

    def is_k222_present(verts, part):
        # verts is tuple of 6
        # part is tuple of 3 pairs of indices into verts
        p1, p2, p3 = part
        v1a, v1b = verts[p1[0]], verts[p1[1]]
        v2a, v2b = verts[p2[0]], verts[p2[1]]
        v3a, v3b = verts[p3[0]], verts[p3[1]]
        
        # Check 8 transversals
        # (v1a, v2a, v3a)
        t = tuple(sorted((v1a, v2a, v3a)))
        if not present[triple_to_idx[t]]: return False
        t = tuple(sorted((v1a, v2a, v3b)))
        if not present[triple_to_idx[t]]: return False
        t = tuple(sorted((v1a, v2b, v3a)))
        if not present[triple_to_idx[t]]: return False
        t = tuple(sorted((v1a, v2b, v3b)))
        if not present[triple_to_idx[t]]: return False
        t = tuple(sorted((v1b, v2a, v3a)))
        if not present[triple_to_idx[t]]: return False
        t = tuple(sorted((v1b, v2a, v3b)))
        if not present[triple_to_idx[t]]: return False
        t = tuple(sorted((v1b, v2b, v3a)))
        if not present[triple_to_idx[t]]: return False
        t = tuple(sorted((v1b, v2b, v3b)))
        if not present[triple_to_idx[t]]: return False
        return True

    def deterministic_cleanup():
        # Find and remove one triple from any K(2,2,2)
        for v in itertools.combinations(range(n), 6):
            if time.time() > deadline: return
            for part in partitions:
                if is_k222_present(v, part):
                    # Remove one random triple from this K
                    p1, p2, p3 = part
                    v1a, v1b = v[p1[0]], v[p1[1]]
                    v2a, v2b = v[p2[0]], v[p2[1]]
                    v3a, v3b = v[p3[0]], v[p3[1]]
                    candidates = [
                        (v1a, v2a, v3a), (v1a, v2a, v3b),
                        (v1a, v2b, v3a), (v1a, v2b, v3b),
                        (v1b, v2a, v3a), (v1b, v2a, v3b),
                        (v1b, v2b, v3a), (v1b, v2b, v3b)
                    ]
                    # Pick random one
                    t = tuple(sorted(random.choice(candidates)))
                    present[triple_to_idx[t]] = False
                    return True
        return False

    def can_add(idx):
        t = triples[idx]
        a, b, c = t
        others = [v for v in range(n) if v not in (a, b, c)]
        for a_prime in others:
            for b_prime in others:
                if b_prime == a_prime: continue
                for c_prime in others:
                    if c_prime == a_prime or c_prime == b_prime: continue
                    # Check 7 transversals
                    # If all 7 present, conflict
                    ok = True
                    t1 = tuple(sorted((a, b, c_prime)))
                    if not present[triple_to_idx[t1]]: ok = False
                    if ok:
                        t2 = tuple(sorted((a, b_prime, c)))
                        if not present[triple_to_idx[t2]]: ok = False
                    if ok:
                        t3 = tuple(sorted((a, b_prime, c_prime)))
                        if not present[triple_to_idx[t3]]: ok = False
                    if ok:
                        t4 = tuple(sorted((a_prime, b, c)))
                        if not present[triple_to_idx[t4]]: ok = False
                    if ok:
                        t5 = tuple(sorted((a_prime, b, c_prime)))
                        if not present[triple_to_idx[t5]]: ok = False
                    if ok:
                        t6 = tuple(sorted((a_prime, b_prime, c)))
                        if not present[triple_to_idx[t6]]: ok = False
                    if ok:
                        t7 = tuple(sorted((a_prime, b_prime, c_prime)))
                        if not present[triple_to_idx[t7]]: ok = False
                    if ok:
                        return False
        return True

    def try_add_all():
        missing = [i for i in range(total_triples) if not present[i]]
        random.shuffle(missing)
        for idx in missing:
            if time.time() > deadline: break
            if can_add(idx):
                present[idx] = True

    best_count = 0
    best_present = None

    for restart in range(50):
        if time.time() > deadline: break
        
        present = [True] * total_triples
        
        # Random removal to speed up cleanup
        for _ in range(100):
            if time.time() > deadline: break
            verts = random.sample(range(n), 6)
            part = random.choice(partitions)
            if is_k222_present(verts, part):
                p1, p2, p3 = part
                v1a, v1b = verts[p1[0]], verts[p1[1]]
                v2a, v2b = verts[p2[0]], verts[p2[1]]
                v3a, v3b = verts[p3[0]], verts[p3[1]]
                candidates = [
                    (v1a, v2a, v3a), (v1a, v2a, v3b),
                    (v1a, v2b, v3a), (v1a, v2b, v3b),
                    (v1b, v2a, v3a), (v1b, v2a, v3b),
                    (v1b, v2b, v3a), (v1b, v2b, v3b)
                ]
                t = tuple(sorted(random.choice(candidates)))
                present[triple_to_idx[t]] = False

        # Deterministic cleanup
        while deterministic_cleanup():
            pass

        # Addition
        try_add_all()

        count = sum(present)
        if count > best_count:
            best_count = count
            best_present = present[:]

    result = []
    for i in range(total_triples):
        if best_present[i]:
            result.append(list(triples[i]))
    return result
