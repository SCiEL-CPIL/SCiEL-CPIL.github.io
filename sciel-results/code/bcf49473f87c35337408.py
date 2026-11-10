"""Linear triple systems on 16 points with small independence number."""
import random
import time
from itertools import combinations

# EVOLVE-BLOCK-START

def _independence_number(triples, n=16):
    """Compute the independence number: largest set with no triple fully contained."""
    if not triples:
        return n
    
    # Represent each triple as a bitmask
    triple_masks = []
    for t in triples:
        mask = 0
        for x in t:
            mask |= (1 << x)
        triple_masks.append(mask)
    
    # Check subsets in decreasing size
    for size in range(n, 0, -1):
        for subset in combinations(range(n), size):
            subset_mask = 0
            for x in subset:
                subset_mask |= (1 << x)
            # Check if any triple is contained
            found = False
            for tm in triple_masks:
                if (tm & subset_mask) == tm:
                    found = True
                    break
            if not found:
                return size
    return 1

def _independence_number_fast(triples, n=16):
    """Faster independence number computation using bitmasks and early pruning."""
    if not triples:
        return n
    
    # Represent each triple as a bitmask
    triple_masks = []
    for t in triples:
        mask = 0
        for x in t:
            mask |= (1 << x)
        triple_masks.append(mask)
    
    # Check subsets in decreasing size
    for size in range(n, 0, -1):
        for subset in combinations(range(n), size):
            subset_mask = 0
            for x in subset:
                subset_mask |= (1 << x)
            # Check if any triple is contained
            found = False
            for tm in triple_masks:
                if (tm & subset_mask) == tm:
                    found = True
                    break
            if not found:
                return size
    return 1

def _can_add(triples, new_triple):
    """Check if new_triple can be added without violating linearity."""
    new_set = set(new_triple)
    for t in triples:
        overlap = len(new_set & set(t))
        if overlap > 1:
            return False
    return True

def _local_search(triples, rng, deadline, max_iters=500):
    """Local search: try to swap triples to reduce independence number."""
    n = 16
    all_triples = list(combinations(range(n), 3))
    
    current = [list(t) for t in triples]
    current_alpha = _independence_number_fast(current)
    best = [list(t) for t in current]
    best_alpha = current_alpha
    
    for _ in range(max_iters):
        if time.time() > deadline:
            break
        
        # Try random swap: remove one triple, add another
        if not current:
            break
        
        idx = rng.randrange(len(current))
        old_triple = current[idx]
        current.remove(old_triple)
        
        # Find a valid triple to add
        valid_adds = []
        for t in all_triples:
            if t not in current:
                if _can_add(current, t):
                    valid_adds.append(t)
        
        if not valid_adds:
            current.append(old_triple)
            continue
        
        new_triple = rng.choice(valid_adds)
        current.append(new_triple)
        
        new_alpha = _independence_number_fast(current)
        if new_alpha < best_alpha:
            best_alpha = new_alpha
            best = [list(t) for t in current]
        else:
            # Revert
            current.remove(new_triple)
            current.append(old_triple)
    
    return best, best_alpha

def _greedy_construct(seed, deadline):
    """Randomized greedy construction with deadline guard."""
    rng = random.Random(seed)
    triples = []
    n = 16
    
    # Precompute all possible triples
    all_triples = list(combinations(range(n), 3))
    
    while time.time() < deadline:
        # Find all valid triples that can be added
        valid_triples = [t for t in all_triples if _can_add(triples, t)]
        if not valid_triples:
            break
        
        # Heuristic: prefer triples that include points not well-covered
        # Count how many times each point appears in existing triples
        point_count = [0] * n
        for t in triples:
            for x in t:
                point_count[x] += 1
        
        # Score each valid triple: prefer those with lower total point count
        # (i.e., points that are less covered)
        def score_triple(t):
            return sum(point_count[x] for x in t)
        
        # Sort by score, take top candidates, then pick randomly
        valid_triples.sort(key=score_triple)
        top_k = min(10, len(valid_triples))
        candidate = rng.choice(valid_triples[:top_k])
        triples.append(list(candidate))
        
        # Early termination: if we've added many triples, check progress
        if len(triples) > 40:
            break
    
    return triples

def _simulated_annealing(triples, rng, deadline):
    """Simulated annealing to improve the triple system."""
    n = 16
    all_triples = list(combinations(range(n), 3))
    
    current = [list(t) for t in triples]
    current_alpha = _independence_number_fast(current)
    best = [list(t) for t in current]
    best_alpha = current_alpha
    
    T = 1.0
    cooling = 0.99
    
    while time.time() < deadline and T > 0.01:
        if not current:
            break
        
        # Try random mutation: remove one triple, add another
        idx = rng.randrange(len(current))
        old_triple = current[idx]
        current.remove(old_triple)
        
        # Find a valid triple to add
        valid_adds = []
        for t in all_triples:
            if t not in current:
                if _can_add(current, t):
                    valid_adds.append(t)
        
        if not valid_adds:
            current.append(old_triple)
            T *= cooling
            continue
        
        new_triple = rng.choice(valid_adds)
        current.append(new_triple)
        
        new_alpha = _independence_number_fast(current)
        delta = new_alpha - current_alpha
        
        if delta <= 0 or rng.random() < pow(2.718281828, -delta / T):
            current_alpha = new_alpha
            if new_alpha < best_alpha:
                best_alpha = new_alpha
                best = [list(t) for t in current]
        else:
            # Revert
            current.remove(new_triple)
            current.append(old_triple)
        
        T *= cooling
    
    return best, best_alpha

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    best_triples = None
    best_indep = 17  # Start high
    
    # Try multiple random seeds with different strategies
    for seed in range(200):
        if time.time() > deadline - 10.0:
            break
        
        # Phase 1: Greedy construction
        triples = _greedy_construct(seed, deadline - 10.0)
        if not triples:
            continue
        
        indep = _independence_number_fast(triples)
        if indep < best_indep:
            best_indep = indep
            best_triples = triples
        
        # Phase 2: Local search or simulated annealing
        if time.time() > deadline - 5.0:
            break
        
        if seed % 2 == 0:
            improved, improved_alpha = _local_search(triples, random.Random(seed + 1000), deadline - 5.0, max_iters=200)
        else:
            improved, improved_alpha = _simulated_annealing(triples, random.Random(seed + 2000), deadline - 5.0)
        
        if improved_alpha < best_indep:
            best_indep = improved_alpha
            best_triples = improved
    
    if best_triples is None:
        # Fallback: return a simple valid system
        return [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(5)]
    
    return best_triples

# EVOLVE-BLOCK-END
