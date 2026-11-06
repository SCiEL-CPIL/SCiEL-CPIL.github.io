import time
import random
import itertools
import math

def _compute_alpha_fast(triples):
    """
    Fast independence number computation using bitmask and early termination.
    Uses the fact that for small alpha, we can check smaller subsets first
    using a smarter approach: for each point, track which triples it's in,
    then use branch-and-bound or direct enumeration with pruning.
    """
    if not triples:
        return 16
    
    # Precompute: for each point, list of triples containing it
    point_to_triples = [[] for _ in range(16)]
    triple_masks = []
    for idx, t in enumerate(triples):
        m = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
        triple_masks.append(m)
        for p in t:
            point_to_triples[p].append(idx)
    
    # Build a list of all triples as bitmasks for fast containment check
    # To check if a set S is independent: no triple is fully contained in S
    # i.e., for all triples T, T is not a subset of S
    
    # Strategy: Try to find an independent set of size k, starting from a reasonable upper bound
    # Upper bound: 16 - min_degree (rough), but let's just go from 8 down since alpha is likely small
    
    # For efficiency, we'll use a recursive backtracking search for independent sets
    # with pruning: if adding a point creates a complete triple, prune
    
    # Alternative: enumerate all subsets of size k from largest to smallest,
    # but use bitmask operations for speed
    
    # Let's try sizes from 10 down to 1 (alpha is unlikely to be > 8 for good systems)
    for k in range(10, 0, -1):
        # Generate combinations of size k
        # To speed up, we can use a heuristic: start with points that have low "triple density"
        found = False
        for comb in itertools.combinations(range(16), k):
            mask = 0
            for p in comb:
                mask |= (1 << p)
            
            # Check if any triple is fully contained
            independent = True
            for tm in triple_masks:
                if (mask & tm) == tm:
                    independent = False
                    break
            
            if independent:
                return k
    
    # If we didn't find one of size >= 1, check size 1
    return 1

def _is_linear(triples):
    """Check that any two triples share at most one point."""
    pair_counts = {}
    for t in triples:
        for i in range(3):
            for j in range(i + 1, 3):
                a, b = t[i], t[j]
                key = (min(a, b), max(a, b))
                pair_counts[key] = pair_counts.get(key, 0) + 1
                if pair_counts[key] > 1:
                    return False
    return True

def _z2_4_construction():
    """
    Deterministic construction based on the vector space Z_2^4.
    Points are 4-bit vectors (0-15).
    We use lines (affine 1-dim subspaces) as triples.
    In AG(4,2), a line has 2 points, but we can use planes or other structures.
    
    Alternative: Use the Steiner triple system approach or a known design.
    For 16 points, we can use a partial Steiner triple system.
    
    Let's use a simple algebraic construction:
    - Take all triples {a, b, a^b} where a, b, a^b are distinct.
    This gives triples from XOR. But we need linearity (at most one shared point).
    
    Actually, let's use a known good construction: the affine geometry AG(4,2) has
    16 points. The lines have 2 points. To get triples, we can use "ovals" or
    other structures.
    
    Simpler: Use a random greedy but with a fixed seed that's known to work well,
    then improve with local search using the fast alpha computation.
    """
    # Let's use a deterministic greedy with a specific ordering
    points = list(range(16))
    all_triples = list(itertools.combinations(points, 3))
    
    # Sort triples by some heuristic: prefer triples that "cover" more pairs
    # or use a fixed random seed
    rng = random.Random(42)
    rng.shuffle(all_triples)
    
    chosen = []
    used_pairs = set()
    
    for t in all_triples:
        valid = True
        for i in range(3):
            for j in range(i + 1, 3):
                a, b = t[i], t[j]
                key = (min(a, b), max(a, b))
                if key in used_pairs:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            for i in range(3):
                for j in range(i + 1, 3):
                    a, b = t[i], t[j]
                    key = (min(a, b), max(a, b))
                    used_pairs.add(key)
            chosen.append(list(t))
            
    return chosen

def _local_search_fast(triples, start_temp=1.0, max_iters=500, seed=0, deadline=None):
    """
    Simulated Annealing with fast alpha computation.
    """
    rng = random.Random(seed)
    points = list(range(16))
    all_triples = list(itertools.combinations(points, 3))
    
    current = [list(t) for t in triples]
    current_alpha = _compute_alpha_fast(current)
    best_alpha = current_alpha
    best_triples = [list(t) for t in current]
    
    def get_used_pairs(triples):
        used = set()
        for t in triples:
            for i in range(3):
                for j in range(i + 1, 3):
                    a, b = t[i], t[j]
                    used.add((min(a, b), max(a, b)))
        return used
    
    used_pairs = get_used_pairs(current)
    
    temp = start_temp
    cooling = 0.99
    
    for i in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
            
        idx = rng.randrange(len(current))
        removed = current[idx]
        
        for k in range(3):
            for l in range(k + 1, 3):
                a, b = removed[k], removed[l]
                used_pairs.discard((min(a, b), max(a, b)))
        
        new_triple = None
        for _ in range(30):
            t = rng.choice(all_triples)
            valid = True
            for k in range(3):
                for l in range(k + 1, 3):
                    a, b = t[k], t[l]
                    if (min(a, b), max(a, b)) in used_pairs:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                new_triple = list(t)
                break
        
        if new_triple is None:
            for k in range(3):
                for l in range(k + 1, 3):
                    a, b = removed[k], removed[l]
                    used_pairs.add((min(a, b), max(a, b)))
            current[idx] = removed
            continue
            
        current[idx] = new_triple
        for k in range(3):
            for l in range(k + 1, 3):
                a, b = new_triple[k], new_triple[l]
                used_pairs.add((min(a, b), max(a, b)))
                
        new_alpha = _compute_alpha_fast(current)
        
        if new_alpha < current_alpha:
            current_alpha = new_alpha
            if new_alpha < best_alpha:
                best_alpha = new_alpha
                best_triples = [list(t) for t in current]
        else:
            delta = new_alpha - current_alpha
            if delta > 0:
                prob = math.exp(-delta / max(temp, 0.01))
                if rng.random() > prob:
                    for k in range(3):
                        for l in range(k + 1, 3):
                            a, b = new_triple[k], new_triple[l]
                            used_pairs.discard((min(a, b), max(a, b)))
                    for k in range(3):
                        for l in range(k + 1, 3):
                            a, b = removed[k], removed[l]
                            used_pairs.add((min(a, b), max(a, b)))
                    current[idx] = removed
                else:
                    current_alpha = new_alpha
            else:
                current_alpha = new_alpha
        
        temp *= cooling
        
    return best_triples, best_alpha

def construct():
    start_time = time.time()
    deadline = start_time + 100  # Safety margin
    
    best_triples = None
    best_alpha = 16
    
    # Try several deterministic greedy starts with different seeds
    for seed in range(30):
        if time.time() > deadline:
            break
        
        points = list(range(16))
        all_triples = list(itertools.combinations(points, 3))
        rng = random.Random(seed)
        rng.shuffle(all_triples)
        
        chosen = []
        used_pairs = set()
        
        for t in all_triples:
            valid = True
            for i in range(3):
                for j in range(i + 1, 3):
                    a, b = t[i], t[j]
                    key = (min(a, b), max(a, b))
                    if key in used_pairs:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                for i in range(3):
                    for j in range(i + 1, 3):
                        a, b = t[i], t[j]
                        key = (min(a, b), max(a, b))
                        used_pairs.add(key)
                chosen.append(list(t))
        
        if not _is_linear(chosen):
            continue
            
        alpha = _compute_alpha_fast(chosen)
        if alpha < best_alpha:
            best_alpha = alpha
            best_triples = [list(t) for t in chosen]
    
    # Local search on the best solution
    if best_triples is not None and time.time() < deadline - 5:
        improved, alpha = _local_search_fast(best_triples, start_temp=0.5, max_iters=300, seed=1000, deadline=deadline - 2)
        if alpha < best_alpha:
            best_alpha = alpha
            best_triples = improved
    
    # Try a few more starts with local search if time permits
    for seed in range(30, 50):
        if time.time() > deadline:
            break
        
        points = list(range(16))
        all_triples = list(itertools.combinations(points, 3))
        rng = random.Random(seed)
        rng.shuffle(all_triples)
        
        chosen = []
        used_pairs = set()
        
        for t in all_triples:
            valid = True
            for i in range(3):
                for j in range(i + 1, 3):
                    a, b = t[i], t[j]
                    key = (min(a, b), max(a, b))
                    if key in used_pairs:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                for i in range(3):
                    for j in range(i + 1, 3):
                        a, b = t[i], t[j]
                        key = (min(a, b), max(a, b))
                        used_pairs.add(key)
                chosen.append(list(t))
        
        if not _is_linear(chosen):
            continue
            
        alpha = _compute_alpha_fast(chosen)
        
        if alpha <= best_alpha and time.time() < deadline - 3:
            improved, a2 = _local_search_fast(chosen, start_temp=0.5, max_iters=150, seed=seed + 1000, deadline=deadline - 1)
            if a2 < best_alpha:
                best_alpha = a2
                best_triples = improved
    
    if best_triples is None:
        return [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(5)]
    
    return best_triples
