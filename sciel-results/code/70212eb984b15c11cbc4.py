import random
import time
import itertools
import copy

N = 16
ALL_PAIRS = set(itertools.combinations(range(N), 2))

def _triple_mask(triple):
    m = 0
    for x in triple:
        m |= (1 << x)
    return m

def _compute_alpha(triples):
    """Compute independence number: largest subset with no triple contained in it."""
    if not triples:
        return N
    triple_masks = [_triple_mask(t) for t in triples]
    n = len(triple_masks)
    for k in range(N, 0, -1):
        for subset in itertools.combinations(range(N), k):
            s_mask = 0
            for x in subset:
                s_mask |= (1 << x)
            ok = True
            for tm in triple_masks:
                if (s_mask & tm) == tm:
                    ok = False
                    break
            if ok:
                return k
    return 1

def _is_linear(triples):
    """Check linearity: no pair appears in more than one triple."""
    seen = set()
    for t in triples:
        for p in itertools.combinations(t, 2):
            if p in seen:
                return False
            seen.add(p)
    return True

def _pairs_of(triple):
    return set(itertools.combinations(triple, 2))

def _used_pairs(triples):
    s = set()
    for t in triples:
        s |= _pairs_of(t)
    return s

def _can_add(triples, new_triple, used_pairs_set):
    """Check if new_triple can be added maintaining linearity."""
    for p in _pairs_of(new_triple):
        if p in used_pairs_set:
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    random.seed(12345)
    
    best_triples = None
    best_alpha = N + 1
    
    def try_update(triples):
        nonlocal best_triples, best_alpha
        if not _is_linear(triples):
            return
        alpha = _compute_alpha(triples)
        if alpha < best_alpha:
            best_alpha = alpha
            best_triples = [list(t) for t in triples]
    
    # Precompute all triples
    all_triples = list(itertools.combinations(range(N), 3))
    
    # Phase 1: Build a good initial solution using greedy + local improvement
    # Start with a structured initial solution
    
    # Strategy: Build greedily, then do local search
    def greedy_build():
        triples = []
        used = set()
        # Randomize order for diversity
        indices = list(range(len(all_triples)))
        random.shuffle(indices)
        for idx in indices:
            t = all_triples[idx]
            if _can_add(triples, t, used):
                triples.append(t)
                used |= _pairs_of(t)
        return triples, used
    
    def local_search(triples, used, max_iters=5000):
        """Local search with add, drop, swap moves."""
        triples = [list(t) for t in triples]
        used = set(used)
        alpha = _compute_alpha(triples)
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            move_type = random.choice(['add', 'drop', 'swap'])
            
            if move_type == 'add':
                # Try to add a random triple
                for _ in range(20):
                    idx = random.randint(0, len(all_triples) - 1)
                    t = all_triples[idx]
                    if _can_add(triples, t, used):
                        new_triples = triples + [t]
                        new_used = used | _pairs_of(t)
                        new_alpha = _compute_alpha(new_triples)
                        if new_alpha <= alpha:
                            triples = new_triples
                            used = new_used
                            alpha = new_alpha
                        break
            
            elif move_type == 'drop' and len(triples) > 5:
                # Try to drop a random triple
                idx = random.randint(0, len(triples) - 1)
                t = triples[idx]
                new_triples = triples[:idx] + triples[idx+1:]
                new_used = set()
                for tt in new_triples:
                    new_used |= _pairs_of(tt)
                new_alpha = _compute_alpha(new_triples)
                if new_alpha <= alpha:
                    triples = new_triples
                    used = new_used
                    alpha = new_alpha
            
            elif move_type == 'swap' and triples:
                # Swap: remove one triple, add another
                idx = random.randint(0, len(triples) - 1)
                old_t = triples[idx]
                # Find a replacement
                for _ in range(20):
                    new_idx = random.randint(0, len(all_triples) - 1)
                    new_t = all_triples[new_idx]
                    if new_t == old_t:
                        continue
                    # Check if new_t is linear with remaining triples
                    remaining = triples[:idx] + triples[idx+1:]
                    rem_used = set()
                    for tt in remaining:
                        rem_used |= _pairs_of(tt)
                    if _can_add(remaining, new_t, rem_used):
                        new_triples = remaining + [new_t]
                        new_used = rem_used | _pairs_of(new_t)
                        new_alpha = _compute_alpha(new_triples)
                        if new_alpha <= alpha:
                            triples = new_triples
                            used = new_used
                            alpha = new_alpha
                        break
        
        return triples, used, alpha
    
    # Simulated annealing wrapper
    def simulated_annealing(triples, used, iters=3000):
        """SA with local search as neighborhood."""
        triples = [list(t) for t in triples]
        used = set(used)
        alpha = _compute_alpha(triples)
        T = 2.0
        cooling = 0.995
        
        for it in range(iters):
            if time.time() > deadline:
                break
            
            move_type = random.choice(['add', 'drop', 'swap'])
            new_triples = None
            new_used = None
            
            if move_type == 'add':
                for _ in range(10):
                    idx = random.randint(0, len(all_triples) - 1)
                    t = all_triples[idx]
                    if _can_add(triples, t, used):
                        new_triples = triples + [t]
                        new_used = used | _pairs_of(t)
                        break
            
            elif move_type == 'drop' and len(triples) > 3:
                idx = random.randint(0, len(triples) - 1)
                t = triples[idx]
                new_triples = triples[:idx] + triples[idx+1:]
                new_used = set()
                for tt in new_triples:
                    new_used |= _pairs_of(tt)
            
            elif move_type == 'swap' and triples:
                idx = random.randint(0, len(triples) - 1)
                old_t = triples[idx]
                for _ in range(10):
                    new_idx = random.randint(0, len(all_triples) - 1)
                    new_t = all_triples[new_idx]
                    if new_t == old_t:
                        continue
                    remaining = triples[:idx] + triples[idx+1:]
                    rem_used = set()
                    for tt in remaining:
                        rem_used |= _pairs_of(tt)
                    if _can_add(remaining, new_t, rem_used):
                        new_triples = remaining + [new_t]
                        new_used = rem_used | _pairs_of(new_t)
                        break
            
            if new_triples is None:
                T *= cooling
                continue
            
            new_alpha = _compute_alpha(new_triples)
            delta = new_alpha - alpha
            
            if delta <= 0 or random.random() < __import__('math').exp(-delta / max(T, 0.01)):
                triples = new_triples
                used = new_used
                alpha = new_alpha
            
            T *= cooling
        
        return triples, used, alpha
    
    # Run multiple restarts with different strategies
    for restart in range(50):
        if time.time() > deadline:
            break
        
        # Different initial constructions
        if restart % 3 == 0:
            # Pure random greedy
            triples, used = greedy_build()
        elif restart % 3 == 1:
            # Structured: try to cover points evenly
            triples = []
            used = set()
            point_counts = [0] * N
            indices = list(range(len(all_triples)))
            random.shuffle(indices)
            for idx in indices:
                t = all_triples[idx]
                if _can_add(triples, t, used):
                    # Prefer triples with least-covered points
                    triples.append(t)
                    used |= _pairs_of(t)
                    for p in t:
                        point_counts[p] += 1
        else:
            # Random seed then greedy
            triples = []
            used = set()
            for _ in range(random.randint(2, 5)):
                idx = random.randint(0, len(all_triples) - 1)
                t = all_triples[idx]
                if _can_add(triples, t, used):
                    triples.append(t)
                    used |= _pairs_of(t)
            # Then greedy fill
            indices = list(range(len(all_triples)))
            random.shuffle(indices)
            for idx in indices:
                t = all_triples[idx]
                if _can_add(triples, t, used):
                    triples.append(t)
                    used |= _pairs_of(t)
        
        if not triples:
            continue
        
        # Local search
        triples, used, alpha = local_search(triples, used, max_iters=2000)
        try_update(triples)
        
        if best_alpha <= 4:
            break
        
        # Simulated annealing
        if time.time() < deadline - 5:
            triples, used, alpha = simulated_annealing(triples, used, iters=2000)
            try_update(triples)
    
    # Final local search on best solution
    if best_triples is not None and time.time() < deadline - 5:
        used = _used_pairs(best_triples)
        best_triples, used, alpha = local_search(best_triples, used, max_iters=3000)
        try_update(best_triples)
    
    if best_triples is None:
        # Fallback
        best_triples, _ = greedy_build()
    
    return best_triples
