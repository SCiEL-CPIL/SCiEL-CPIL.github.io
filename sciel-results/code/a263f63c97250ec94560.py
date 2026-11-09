"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time
from itertools import combinations

# EVOLVE-BLOCK-START

def _independence_number_fast(triples, n=16):
    """Fast independence number computation using bitmask DP."""
    if not triples:
        return n
    
    # Precompute triple bitmasks
    triple_masks = []
    for t in triples:
        mask = 0
        for p in t:
            mask |= (1 << p)
        triple_masks.append(mask)
    
    # Check from n down to 1
    for size in range(n, 0, -1):
        if size == n:
            mask = (1 << n) - 1
            for tm in triple_masks:
                if (tm & mask) == tm:
                    break
            else:
                return size
            continue
        
        for subset in combinations(range(n), size):
            mask = 0
            for p in subset:
                mask |= (1 << p)
            independent = True
            for tm in triple_masks:
                if (tm & mask) == tm:
                    independent = False
                    break
            if independent:
                return size
    
    return 0

def _is_linear(triples):
    """Check if any two triples share at most one point."""
    pair_set = set()
    for t in triples:
        for i in range(3):
            for j in range(i+1, 3):
                pair = (min(t[i], t[j]), max(t[i], t[j]))
                if pair in pair_set:
                    return False
                pair_set.add(pair)
    return True

def _build_sts15_plus_point():
    """
    Build a linear triple system on 16 points by starting with STS(15) on points 1-15
    and replacing some triples to include point 0.
    """
    # Generate STS(15)
    sts_triples = []
    seen = set()
    for a in range(1, 16):
        for b in range(a+1, 16):
            c = a ^ b
            if c == 0 or c == a or c == b:
                continue
            t = tuple(sorted((a, b, c)))
            if t not in seen:
                seen.add(t)
                sts_triples.append(t)
    
    # Build pair-to-triple map
    pair_to_idx = {}
    for i, t in enumerate(sts_triples):
        for k in range(3):
            for l in range(k+1, 3):
                pair = (min(t[k], t[l]), max(t[k], t[l]))
                pair_to_idx[pair] = i
    
    # Start with all STS triples
    triples = [list(t) for t in sts_triples]
    used_pairs = set(pair_to_idx.keys())
    
    # Try to add triples with point 0 by replacing existing triples
    # We want to maximize the number of triples involving point 0
    # while maintaining linearity
    
    added_count = 0
    # Try all pairs (a, b) with 1 <= a < b <= 15
    for a in range(1, 16):
        for b in range(a+1, 16):
            if added_count >= 5:
                break
            
            pair_ab = (a, b)
            if pair_ab in used_pairs:
                # Find the triple containing this pair
                idx = pair_to_idx.get(pair_ab)
                if idx is not None and idx < len(triples):
                    old_triple = triples[idx]
                    # Check if old_triple is still present
                    if old_triple in triples:
                        # Check pairs (0,a) and (0,b) are free
                        pair_0a = (0, a)
                        pair_0b = (0, b)
                        if pair_0a not in used_pairs and pair_0b not in used_pairs:
                            # Replace old triple with {0, a, b}
                            triples[idx] = [0, a, b]
                            # Update used pairs
                            for k in range(3):
                                for l in range(k+1, 3):
                                    p = (min(old_triple[k], old_triple[l]), max(old_triple[k], old_triple[l]))
                                    if p != pair_ab:
                                        used_pairs.discard(p)
                            used_pairs.add(pair_0a)
                            used_pairs.add(pair_0b)
                            added_count += 1
                            # Update pair_to_idx for the new triple
                            for k in range(3):
                                for l in range(k+1, 3):
                                    p = (min(triples[idx][k], triples[idx][l]), max(triples[idx][k], triples[idx][l]))
                                    pair_to_idx[p] = idx
                            # Clear old pair mappings
                            for k in range(3):
                                for l in range(k+1, 3):
                                    p = (min(old_triple[k], old_triple[l]), max(old_triple[k], old_triple[l]))
                                    if p != pair_ab:
                                        if p in pair_to_idx and pair_to_idx[p] == idx:
                                            del pair_to_idx[p]
    
    return triples

def construct():
    start_time = time.time()
    deadline = start_time + 100  # Safety margin before 150s limit
    
    best_triples = None
    best_indep = 16
    
    # Try the algebraic construction
    alg = _build_sts15_plus_point()
    if alg and _is_linear(alg):
        indep = _independence_number_fast(alg, 16)
        best_indep = indep
        best_triples = alg
    
    # Try random constructions with local search
    for seed in range(50):
        if time.time() > deadline:
            break
        
        rng = random.Random(seed)
        triples = []
        pair_used = set()
        
        def try_add(a, b, c):
            nonlocal pair_used
            t = tuple(sorted((a, b, c)))
            if len(set(t)) < 3:
                return False
            pairs = [(min(t[0],t[1]), max(t[0],t[1])),
                     (min(t[0],t[2]), max(t[0],t[2])),
                     (min(t[1],t[2]), max(t[1],t[2]))]
            for p in pairs:
                if p in pair_used:
                    return False
            for p in pairs:
                pair_used.add(p)
            triples.append(list(t))
            return True
        
        # Generate random linear triples
        attempts = 0
        while len(triples) < 45 and attempts < 5000:
            attempts += 1
            a, b, c = rng.sample(range(16), 3)
            try_add(a, b, c)
        
        if triples and _is_linear(triples):
            indep = _independence_number_fast(triples, 16)
            if indep < best_indep:
                best_indep = indep
                best_triples = triples
    
    if best_triples is None:
        best_triples = [[3*i, 3*i+1, 3*i+2] for i in range(5)]
    
    return best_triples
# EVOLVE-BLOCK-END
