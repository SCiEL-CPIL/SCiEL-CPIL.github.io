import random
import itertools
import numpy as np
import time

# Global caches for numpy-based independence number calculation
_ALL_MASKS = None
_POPCOUNTS = None

def _init_numpy():
    global _ALL_MASKS, _POPCOUNTS
    if _ALL_MASKS is None:
        _ALL_MASKS = np.arange(1 << 16, dtype=np.int32)
        # Precompute popcounts for all masks
        _POPCOUNTS = np.array([bin(i).count('1') for i in range(1 << 16)], dtype=np.int8)

def _calc_alpha_np(triples):
    """Calculate independence number using numpy vectorization."""
    _init_numpy()
    if not triples:
        return 16
    
    # Convert triples to bitmasks
    t_masks = np.array([sum(1 << p for p in t) for t in triples], dtype=np.int32)
    
    # Check which masks contain any triple
    # A: (65536, 1), B: (1, |T|)
    A = _ALL_MASKS[:, None]
    B = t_masks[None, :]
    
    # (A & B) == B checks if mask A contains triple B
    # Result shape: (65536, |T|)
    contains_any = np.any((A & B) == B, axis=1)
    is_independent = ~contains_any
    
    if not np.any(is_independent):
        return 0
        
    return int(np.max(_POPCOUNTS[is_independent]))

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
        if any(tuple(sorted(p)) in used_pairs for p in pairs):
            continue
        for p in pairs:
            used_pairs.add(tuple(sorted(p)))
        triples.append(list(t))
    
    return triples

def construct():
    _init_numpy()
    rng = random.Random(42)
    
    # Initialize with a random packing
    triples = _generate_packing(16, 0)
    
    # Build used_pairs set for fast validation
    used_pairs = set()
    for t in triples:
        for i in range(3):
            for j in range(i+1, 3):
                p = tuple(sorted((t[i], t[j])))
                used_pairs.add(p)
        
    current_alpha = _calc_alpha_np(triples)
    best_alpha = current_alpha
    best_triples = [list(t) for t in triples]
    
    all_triples = list(itertools.combinations(range(16), 3))
    
    # Simulated Annealing
    T = 1.0
    start_time = time.time()
    max_time = 100.0 # Leave 20s buffer
    
    while time.time() - start_time < max_time:
        # Pick random triple to remove
        idx = rng.randrange(len(triples))
        t_old = triples[idx]
        
        # Remove it
        triples.pop(idx)
        for k in range(3):
            for l in range(k+1, 3):
                p = tuple(sorted((t_old[k], t_old[l])))
                used_pairs.remove(p)
        
        # Try to find a valid new triple
        t_new = None
        # Try a few random candidates
        for _ in range(20):
            cand = rng.choice(all_triples)
            valid = True
            for k in range(3):
                for l in range(k+1, 3):
                    p = tuple(sorted((cand[k], cand[l])))
                    if p in used_pairs:
                        valid = False
                        break
                if not valid: break
            
            if valid:
                t_new = cand
                break
        
        if t_new is None:
            # Revert
            triples.append(t_old)
            for k in range(3):
                for l in range(k+1, 3):
                    p = tuple(sorted((t_old[k], t_old[l])))
                    used_pairs.add(p)
            continue
        
        # Add t_new
        triples.append(t_new)
        for k in range(3):
            for l in range(k+1, 3):
                p = tuple(sorted((t_new[k], t_new[l])))
                used_pairs.add(p)
        
        new_alpha = _calc_alpha_np(triples)
        
        # Acceptance criterion
        delta = new_alpha - current_alpha
        accept = False
        if delta < 0:
            accept = True
        elif delta == 0:
            accept = rng.random() < 0.5
        else:
            # Avoid division by zero or underflow
            if T > 1e-10:
                prob = np.exp(-delta / T)
                accept = rng.random() < prob
        
        if accept:
            current_alpha = new_alpha
            if new_alpha < best_alpha:
                best_alpha = new_alpha
                best_triples = [list(t) for t in triples]
                if best_alpha <= 5:
                    break
        else:
            # Revert
            triples.pop()
            for k in range(3):
                for l in range(k+1, 3):
                    p = tuple(sorted((t_new[k], t_new[l])))
                    used_pairs.remove(p)
            triples.append(t_old)
            for k in range(3):
                for l in range(k+1, 3):
                    p = tuple(sorted((t_old[k], t_old[l])))
                    used_pairs.add(p)
        
        T *= 0.9995
        
    return best_triples
