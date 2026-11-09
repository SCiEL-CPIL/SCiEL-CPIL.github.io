import random
import itertools

def _independence_number(triples, n=16):
    """Compute the independence number: largest subset with no triple."""
    triple_set = set(tuple(sorted(t)) for t in triples)
    best = 0
    for mask in range(1 << n):
        # Count bits
        bits = bin(mask).count('1')
        if bits <= best:
            continue
        # Check if any triple is fully contained
        has_triple = False
        for t in triple_set:
            a, b, c = t
            if (mask >> a) & 1 and (mask >> b) & 1 and (mask >> c) & 1:
                has_triple = True
                break
        if not has_triple:
            best = bits
    return best

def _generate_packing(n=16, seed=0):
    """Generate a linear triple system using randomized greedy."""
    rng = random.Random(seed)
    used_pairs = set()
    triples = []
    
    # Generate all possible triples
    all_triples = list(itertools.combinations(range(n), 3))
    rng.shuffle(all_triples)
    
    for t in all_triples:
        pairs = [(t[0], t[1]), (t[0], t[2]), (t[1], t[2])]
        # Check if any pair is already used
        if any(tuple(sorted(p)) in used_pairs for p in pairs):
            continue
        # Add this triple
        for p in pairs:
            used_pairs.add(tuple(sorted(p)))
        triples.append(list(t))
    
    return triples

def construct():
    best_triples = None
    best_indep = float('inf')
    
    # Try multiple seeds
    for seed in range(200):
        triples = _generate_packing(16, seed)
        indep = _independence_number(triples, 16)
        if indep < best_indep:
            best_indep = indep
            best_triples = triples
        if best_indep <= 5:  # Good enough
            break
    
    return best_triples
