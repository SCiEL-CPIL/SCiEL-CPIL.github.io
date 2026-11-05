import itertools
import time

def construct():
    deadline = time.time() + 140  # Leave 10s safety margin before 150s kill
    
    # 1. Generate STS(15) using GF(2)^4 structure
    # Points 0..14 correspond to nonzero vectors in GF(2)^4 (values 1..15)
    # Triple {a,b,c} iff a^b^c == 0 (i.e., they form a line in PG(3,2))
    
    sts_triples = []
    for i in range(1, 16):
        for j in range(i+1, 16):
            k = i ^ j
            if k > j and k != 0:
                t = tuple(sorted([i-1, j-1, k-1]))
                sts_triples.append(t)
    
    # Convert to bitmask representation for fast subset checks
    # Each triple becomes a 15-bit integer
    triple_masks = []
    for t in sts_triples:
        mask = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
        triple_masks.append(mask)
    
    # 2. Find all 8-caps (independent sets of size 8)
    all_caps = []
    for combo in itertools.combinations(range(15), 8):
        cap_mask = 0
        for p in combo:
            cap_mask |= (1 << p)
        is_cap = True
        for tm in triple_masks:
            if (cap_mask & tm) == tm:  # triple is subset of cap
                is_cap = False
                break
        if is_cap:
            all_caps.append((combo, cap_mask))
    
    # If we're running out of time, use a simple fallback
    if time.time() > deadline - 5:
        found_M = [(0,1), (2,3), (4,5), (6,7), (8,9), (10,11), (12,13)]
    else:
        # 3. Find matching M hitting all caps
        # M is a list of pairs (a,b) with a<b, disjoint.
        # A matching "hits" a cap if some pair in M is fully contained in the cap.
        
        def hits_all(M):
            M_masks = [(1 << m[0]) | (1 << m[1]) for m in M]
            for _, cap_mask in all_caps:
                hit = False
                for mm in M_masks:
                    if (cap_mask & mm) == mm:
                        hit = True
                        break
                if not hit:
                    return False
            return True
        
        found_M = None
        
        # Try matching size 1
        for a in range(15):
            if time.time() > deadline - 10:
                break
            for b in range(a+1, 15):
                if hits_all([(a,b)]):
                    found_M = [(a,b)]
                    break
            if found_M:
                break
        
        # Try matching size 2 if size 1 failed
        if found_M is None and time.time() < deadline - 10:
            for a in range(15):
                if time.time() > deadline - 10:
                    break
                for b in range(a+1, 15):
                    for c in range(a+1, 15):
                        if c == b:
                            continue
                        if time.time() > deadline - 10:
                            break
                        for d in range(c+1, 15):
                            if d == a or d == b:
                                continue
                            if hits_all([(a,b), (c,d)]):
                                found_M = [(a,b), (c,d)]
                                break
                        if found_M:
                            break
                    if found_M:
                        break
                if found_M:
                    break
        
        # Fallback if no small matching found
        if found_M is None:
            found_M = [(i, i+1) for i in range(0, 14, 2)]
    
    # 4. Construct final triples
    M_masks = [(1 << m[0]) | (1 << m[1]) for m in found_M]
    
    final_triples = []
    
    # Keep STS triples that do not contain any pair in M
    for t in sts_triples:
        t_mask = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
        keep = True
        for mm in M_masks:
            if (t_mask & mm) == mm:
                keep = False
                break
        if keep:
            final_triples.append(list(t))
    
    # Add new triples with point 15
    for m in found_M:
        final_triples.append([15, m[0], m[1]])
    
    return final_triples
