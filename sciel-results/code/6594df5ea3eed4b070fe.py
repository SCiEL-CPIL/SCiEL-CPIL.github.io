import random
import time
from itertools import combinations

# EVOLVE-BLOCK-START

def _independence_number(triples, n=16):
    """
    Calculate the independence number of a linear triple system.
    Returns the size of the largest subset of points containing no triple.
    """
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

def _build_pg32():
    """
    Build the Projective Geometry PG(3,2) on 15 non-zero points of F_2^4.
    Point 0 is isolated.
    """
    triples = []
    seen = set()
    # Points are 1..15 representing non-zero vectors
    for a in range(1, 16):
        for b in range(a+1, 16):
            c = a ^ b
            if c == 0 or c == a or c == b:
                continue
            t = tuple(sorted((a, b, c)))
            if t not in seen:
                seen.add(t)
                triples.append(list(t))
    return triples

def _local_search(triples, max_time=50):
    """
    Perform local search to minimize independence number.
    Uses swaps involving point 0 to integrate it.
    """
    start_time = time.time()
    deadline = start_time + max_time
    
    current_triples = [list(t) for t in triples]
    current_indep = _independence_number(current_triples, 16)
    best_triples = [list(t) for t in current_triples]
    best_indep = current_indep
    
    # Precompute pairs for fast lookup
    def get_pair_set(triples):
        pairs = set()
        for t in triples:
            for i in range(3):
                for j in range(i+1, 3):
                    pairs.add((min(t[i], t[j]), max(t[i], t[j])))
        return pairs
    
    current_pairs = get_pair_set(current_triples)
    
    # Identify triples NOT containing 0 (candidates for swap out)
    # Identify pairs (a,b) such that {0,a,b} can be added
    # This requires (0,a) and (0,b) to be free.
    
    # Simulated Annealing / Hill Climbing
    temperature = 1.0
    cooling = 0.99
    
    rng = random.Random(42)
    
    while time.time() < deadline:
        # Find all possible swaps
        # A swap replaces {a,b,c} (not containing 0) with {0,a,b}
        # Conditions:
        # 1. {a,b,c} is in current_triples
        # 2. 0 not in {a,b,c}
        # 3. (0,a) not in current_pairs
        # 4. (0,b) not in current_pairs
        
        candidates = []
        for i, t in enumerate(current_triples):
            if 0 in t:
                continue
            a, b, c = t
            # Try swapping with {0,a,b}, {0,a,c}, {0,b,c}
            for pair in [(a,b), (a,c), (b,c)]:
                p1, p2 = pair
                if (0, p1) not in current_pairs and (0, p2) not in current_pairs:
                    # Check if {0, p1, p2} is valid (distinct points, yes)
                    # And if we remove t, do we free up pairs?
                    # Yes, pairs in t are removed.
                    # Pairs in new triple are added.
                    # Since (0,p1) and (0,p2) are free, and (p1,p2) was in t (so it's removed then re-added? 
                    # No, (p1,p2) is in t. If we replace t with {0,p1,p2}, the pair (p1,p2) is still used.
                    # So the only new pairs are (0,p1) and (0,p2).
                    # The pairs (p1, other) and (p2, other) from t are freed.
                    candidates.append((i, p1, p2))
        
        if not candidates:
            # Try reverse swaps: Replace {0,a,b} with {a,b,a^b}
            for i, t in enumerate(current_triples):
                if 0 not in t:
                    continue
                # t is {0, a, b}
                a, b = t[1], t[2]
                c = a ^ b
                if c == 0 or c == a or c == b:
                    continue
                # Check if {a,b,c} can be added
                # Pairs (a,b) is in t (removed).
                # Pairs (a,c) and (b,c) must be free.
                if (a,c) not in current_pairs and (b,c) not in current_pairs:
                    candidates.append((i, a, b, 'reverse'))
        
        if not candidates:
            break
            
        # Pick a random candidate
        cand = rng.choice(candidates)
        
        if len(cand) == 3:
            idx, p1, p2 = cand
            old_triple = current_triples[idx]
            new_triple = [0, p1, p2]
            
            # Apply swap
            current_triples[idx] = new_triple
            
            # Update pairs
            # Remove pairs of old_triple
            for i in range(3):
                for j in range(i+1, 3):
                    current_pairs.discard((min(old_triple[i], old_triple[j]), max(old_triple[i], old_triple[j])))
            # Add pairs of new_triple
            for i in range(3):
                for j in range(i+1, 3):
                    current_pairs.add((min(new_triple[i], new_triple[j]), max(new_triple[i], new_triple[j])))
            
            new_indep = _independence_number(current_triples, 16)
            
            delta = new_indep - current_indep
            
            if delta < 0 or rng.random() < pow(2.718, -delta / temperature):
                current_indep = new_indep
                if new_indep < best_indep:
                    best_indep = new_indep
                    best_triples = [list(t) for t in current_triples]
            else:
                # Revert
                current_triples[idx] = old_triple
                for i in range(3):
                    for j in range(i+1, 3):
                        current_pairs.discard((min(new_triple[i], new_triple[j]), max(new_triple[i], new_triple[j])))
                for i in range(3):
                    for j in range(i+1, 3):
                        current_pairs.add((min(old_triple[i], old_triple[j]), max(old_triple[i], old_triple[j])))
        else:
            idx, p1, p2, _ = cand
            old_triple = current_triples[idx] # {0, p1, p2}
            c = p1 ^ p2
            new_triple = [p1, p2, c]
            
            current_triples[idx] = new_triple
            
            for i in range(3):
                for j in range(i+1, 3):
                    current_pairs.discard((min(old_triple[i], old_triple[j]), max(old_triple[i], old_triple[j])))
            for i in range(3):
                for j in range(i+1, 3):
                    current_pairs.add((min(new_triple[i], new_triple[j]), max(new_triple[i], new_triple[j])))
            
            new_indep = _independence_number(current_triples, 16)
            
            delta = new_indep - current_indep
            
            if delta < 0 or rng.random() < pow(2.718, -delta / temperature):
                current_indep = new_indep
                if new_indep < best_indep:
                    best_indep = new_indep
                    best_triples = [list(t) for t in current_triples]
            else:
                current_triples[idx] = old_triple
                for i in range(3):
                    for j in range(i+1, 3):
                        current_pairs.discard((min(new_triple[i], new_triple[j]), max(new_triple[i], new_triple[j])))
                for i in range(3):
                    for j in range(i+1, 3):
                        current_pairs.add((min(old_triple[i], old_triple[j]), max(old_triple[i], old_triple[j])))
        
        temperature *= cooling
        
    return best_triples, best_indep

def construct():
    start_time = time.time()
    deadline = start_time + 100
    
    best_triples = None
    best_indep = 16
    
    # 1. Algebraic Construction (PG(3,2))
    pg32 = _build_pg32()
    if pg32 and _is_linear(pg32):
        indep = _independence_number(pg32, 16)
        if indep < best_indep:
            best_indep = indep
            best_triples = pg32
            
        # Refine with local search
        if time.time() < deadline:
            refined, ref_indep = _local_search(pg32, max_time=30)
            if ref_indep < best_indep:
                best_indep = ref_indep
                best_triples = refined
    
    # 2. Random Greedy with Local Search
    for seed in range(20):
        if time.time() > deadline:
            break
            
        rng = random.Random(seed)
        triples = []
        pair_used = set()
        
        def try_add(a, b, c):
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
        
        # Generate random maximal packing
        attempts = 0
        while len(triples) < 40 and attempts < 5000:
            attempts += 1
            a, b, c = rng.sample(range(16), 3)
            try_add(a, b, c)
        
        if triples and _is_linear(triples):
            # Simple local search: try to swap triples to reduce alpha
            # For brevity, just evaluate and keep best
            indep = _independence_number(triples, 16)
            if indep < best_indep:
                best_indep = indep
                best_triples = triples
                
            # Try a few local improvements
            if time.time() < deadline:
                # Pick a random triple and try to swap it with another valid triple
                # This is a simplified version of the SA above
                for _ in range(100):
                    if time.time() > deadline:
                        break
                    i = rng.randrange(len(triples))
                    old_t = triples[i]
                    # Generate a random new triple
                    a, b, c = rng.sample(range(16), 3)
                    new_t = tuple(sorted((a,b,c)))
                    # Check if new_t is compatible with rest of system
                    # Remove old_t pairs
                    old_pairs = [(min(old_t[0],old_t[1]), max(old_t[0],old_t[1])),
                                 (min(old_t[0],old_t[2]), max(old_t[0],old_t[2])),
                                 (min(old_t[1],old_t[2]), max(old_t[1],old_t[2]))]
                    for p in old_pairs:
                        pair_used.discard(p)
                    
                    new_pairs = [(min(new_t[0],new_t[1]), max(new_t[0],new_t[1])),
                                 (min(new_t[0],new_t[2]), max(new_t[0],new_t[2])),
                                 (min(new_t[1],new_t[2]), max(new_t[1],new_t[2]))]
                    valid = True
                    for p in new_pairs:
                        if p in pair_used:
                            valid = False
                            break
                    if valid:
                        for p in new_pairs:
                            pair_used.add(p)
                        triples[i] = list(new_t)
                        new_indep = _independence_number(triples, 16)
                        if new_indep < indep:
                            indep = new_indep
                            if indep < best_indep:
                                best_indep = indep
                                best_triples = [list(t) for t in triples]
                        else:
                            # Revert
                            for p in new_pairs:
                                pair_used.discard(p)
                            for p in old_pairs:
                                pair_used.add(p)
                            triples[i] = list(old_t)
                    else:
                        for p in old_pairs:
                            pair_used.add(p)

    if best_triples is None:
        best_triples = [[3*i, 3*i+1, 3*i+2] for i in range(5)]
        
    return best_triples

# EVOLVE-BLOCK-END
