import random
import itertools

# Precompute point masks for speed
_POINT_MASKS = [1 << i for i in range(16)]

def _independence_number(triples, n=16):
    """Compute the independence number: largest subset with no triple."""
    if not triples:
        return n
    
    triple_masks = []
    for t in triples:
        mask = 0
        for p in t:
            mask |= _POINT_MASKS[p]
        triple_masks.append(mask)
    
    # Check for independent sets of size k, from n down to 1
    for k in range(n, 0, -1):
        for comb in itertools.combinations(range(n), k):
            mask = 0
            for p in comb:
                mask |= _POINT_MASKS[p]
            
            # Check if any triple is fully contained in mask
            is_independent = True
            for tm in triple_masks:
                if (mask & tm) == tm:
                    is_independent = False
                    break
            if is_independent:
                return k
    return 0

def _generate_packing(n=16, seed=0):
    """Generate a linear triple system using randomized greedy."""
    rng = random.Random(seed)
    used_pairs = set()
    triples = []
    
    all_triples = list(itertools.combinations(range(n), 3))
    rng.shuffle(all_triples)
    
    for t in all_triples:
        pairs = [(t[0], t[1]), (t[0], t[2]), (t[1], t[2])]
        if any(tuple(sorted(p)) in used_pairs for p in pairs):
            continue
        for p in pairs:
            used_pairs.add(tuple(sorted(p)))
        triples.append(list(t))
    
    return triples

def _local_search(triples, n=16, iterations=1000, seed=0):
    """Improve the packing using local search to minimize independence number."""
    rng = random.Random(seed)
    current_triples = [list(t) for t in triples]
    current_indep = _independence_number(current_triples, n)
    
    all_triples = list(itertools.combinations(range(n), 3))
    
    for i in range(iterations):
        # Remove a random triple
        if not current_triples:
            break
        idx = rng.randrange(len(current_triples))
        removed_triple = current_triples.pop(idx)
        
        # Rebuild used_pairs
        used_pairs = set()
        for t in current_triples:
            a, b, c = t
            used_pairs.add((a, b) if a < b else (b, a))
            used_pairs.add((a, c) if a < c else (c, a))
            used_pairs.add((b, c) if b < c else (c, b))
        
        # Try to find a valid triple to add
        added = False
        for _ in range(10):
            candidate = rng.choice(all_triples)
            a, b, c = candidate
            p1 = (a, b) if a < b else (b, a)
            p2 = (a, c) if a < c else (c, a)
            p3 = (b, c) if b < c else (c, b)
            if p1 not in used_pairs and p2 not in used_pairs and p3 not in used_pairs:
                current_triples.append(list(candidate))
                used_pairs.add(p1)
                used_pairs.add(p2)
                used_pairs.add(p3)
                added = True
                break
        
        if not added:
            current_triples.append(removed_triple)
            continue
            
        # Calculate new independence number only every 10 steps to save time
        if i % 10 == 0:
            new_indep = _independence_number(current_triples, n)
            if new_indep <= current_indep:
                current_indep = new_indep
            else:
                # Revert
                current_triples.pop()
                current_triples.append(removed_triple)
        else:
            # Assume it's okay, or do a fast check?
            # For now, accept to allow exploration
            pass

    # Final check
    final_indep = _independence_number(current_triples, n)
    return current_triples, final_indep

def construct():
    best_triples = None
    best_indep = float('inf')
    
    for seed in range(20):
        triples = _generate_packing(16, seed)
        improved_triples, indep = _local_search(triples, 16, iterations=500, seed=seed)
        
        if indep < best_indep:
            best_indep = indep
            best_triples = improved_triples
            
        if best_indep <= 4:
            break

    return best_triples
