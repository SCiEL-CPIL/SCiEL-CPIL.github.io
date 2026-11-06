import random
import time
from itertools import combinations

def _is_linear(triples):
    """Check if any two triples share more than one point."""
    for i in range(len(triples)):
        for j in range(i + 1, len(triples)):
            shared = len(set(triples[i]) & set(triples[j]))
            if shared > 1:
                return False
    return True

def _calc_alpha(triples, n=16):
    """Compute independence number: largest subset with no triple."""
    triple_sets = [frozenset(t) for t in triples]
    best = 0
    # Check all subsets
    for mask in range(1 << n):
        # Quick size check
        size = bin(mask).count('1')
        if size <= best:
            continue
        # Check if this subset contains any triple
        has_triple = False
        for ts in triple_sets:
            # Check if ts is subset of mask
            ok = True
            for v in ts:
                if not (mask & (1 << v)):
                    ok = False
                    break
            if ok:
                has_triple = True
                break
        if not has_triple:
            best = size
    return best

def _greedy_build(n=16, seed=0):
    """Build a linear 3-uniform hypergraph greedily."""
    rng = random.Random(seed)
    triples = []
    # Precompute all possible triples
    all_triples = list(combinations(range(n), 3))
    rng.shuffle(all_triples)
    
    # Track pairs used
    pair_used = set()
    # Track vertex degrees for balance
    vertex_deg = [0] * n
    
    for t in all_triples:
        ts = frozenset(t)
        # Check linearity: no pair already used
        pairs = list(combinations(sorted(t), 2))
        if any(pair in pair_used for pair in pairs):
            continue
        # Add triple
        triples.append(list(t))
        for p in pairs:
            pair_used.add(p)
        for v in t:
            vertex_deg[v] += 1
    
    return triples

def _local_search(triples, n=16, max_iters=5000, seed=0):
    """Local search to minimize independence number."""
    rng = random.Random(seed)
    triples = [list(t) for t in triples]
    best_alpha = _calc_alpha(triples, n)
    
    for it in range(max_iters):
        # Try random modifications
        move_type = rng.choice(['add', 'remove', 'swap'])
        
        if move_type == 'add':
            # Try adding a random valid triple
            candidates = []
            for _ in range(50):
                t = list(rng.sample(range(n), 3))
                # Check linearity
                valid = True
                for existing in triples:
                    if len(set(t) & set(existing)) > 1:
                        valid = False
                        break
                if valid:
                    candidates.append(t)
            if candidates:
                new_triple = rng.choice(candidates)
                triples.append(new_triple)
                new_alpha = _calc_alpha(triples, n)
                if new_alpha < best_alpha:
                    best_alpha = new_alpha
                else:
                    triples.pop()
        
        elif move_type == 'remove' and triples:
            idx = rng.randrange(len(triples))
            removed = triples.pop(idx)
            new_alpha = _calc_alpha(triples, n)
            if new_alpha < best_alpha:
                best_alpha = new_alpha
            else:
                triples.insert(idx, removed)
        
        elif move_type == 'swap' and triples:
            # Remove one, try to add another
            idx = rng.randrange(len(triples))
            removed = triples.pop(idx)
            # Try to find a replacement
            added = False
            for _ in range(20):
                t = list(rng.sample(range(n), 3))
                valid = True
                for existing in triples:
                    if len(set(t) & set(existing)) > 1:
                        valid = False
                        break
                if valid:
                    triples.append(t)
                    new_alpha = _calc_alpha(triples, n)
                    if new_alpha < best_alpha:
                        best_alpha = new_alpha
                        added = True
                    else:
                        triples.pop()
                    break
            if not added:
                triples.insert(idx, removed)
    
    return triples, best_alpha

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    best_triples = None
    best_alpha = 16  # Worst case
    
    # Try multiple seeds with greedy + local search
    for seed in range(50):
        if time.time() > deadline - 10:
            break
        
        triples = _greedy_build(16, seed=seed)
        if not triples:
            continue
        
        alpha = _calc_alpha(triples, 16)
        
        # Local search
        triples, alpha = _local_search(triples, 16, max_iters=2000, seed=seed)
        
        if alpha < best_alpha:
            best_alpha = alpha
            best_triples = [list(t) for t in triples]
            # If we get a good result, we can stop early
            if best_alpha <= 5:
                break
    
    # Final check
    if best_triples is None:
        # Fallback: just return the seed
        return [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(5)]
    
    return best_triples
