import random
import time
import numpy as np

def compute_alpha_np(triples):
    """
    Computes the independence number of a linear triple system on 16 points.
    Uses NumPy for vectorized subset checking.
    """
    if not triples:
        return 16
        
    # Precompute masks for triples
    triple_masks = []
    for t in triples:
        m = 0
        for p in t:
            m |= (1 << p)
        triple_masks.append(m)
    
    if len(triple_masks) == 0:
        return 16

    triple_masks_np = np.array(triple_masks, dtype=np.int32)
    
    # Generate all subset masks 0..65535
    # We can cache this
    if not hasattr(compute_alpha_np, "all_masks"):
        compute_alpha_np.all_masks = np.arange(1 << 16, dtype=np.int32)
        # Precompute popcount
        pc = np.zeros(1 << 16, dtype=np.int8)
        for i in range(1, 1 << 16):
            pc[i] = pc[i >> 1] + (i & 1)
        compute_alpha_np.popcount = pc
        
    all_masks = compute_alpha_np.all_masks
    popcount = compute_alpha_np.popcount
    
    # Check if any triple is contained in each mask
    intersects = (all_masks[:, None] & triple_masks_np[None, :]) == triple_masks_np[None, :]
    has_triple = intersects.any(axis=1)
    independent = ~has_triple
    
    if not independent.any():
        return 0
        
    max_alpha = popcount[independent].max()
    return int(max_alpha)

def generate_all_triples():
    """Generate all possible triples on 16 points."""
    points = range(16)
    triples = []
    for a in points:
        for b in points:
            if b <= a: continue
            for c in points:
                if c <= b: continue
                triples.append((a, b, c))
    return triples

def greedy_construct(all_triples, seed=None):
    """
    Randomized greedy construction of a linear triple system.
    """
    if seed is not None:
        random.seed(seed)
        
    candidates = list(all_triples)
    random.shuffle(candidates)
    
    used_pairs = set()
    triples = []
    
    for t in candidates:
        a, b, c = t
        pairs = [(a, b), (a, c), (b, c)]
        
        # Check if any pair is already used
        conflict = False
        for p in pairs:
            if p in used_pairs:
                conflict = True
                break
                
        if not conflict:
            triples.append(t)
            for p in pairs:
                used_pairs.add(p)
                
    return triples

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    all_triples = generate_all_triples()
    
    # Initialize with a few greedy runs to find a good starting point
    best_alpha = 16
    best_triples = []
    
    for i in range(5):
        if time.time() > deadline:
            break
        triples = greedy_construct(all_triples, seed=i)
        alpha = compute_alpha_np(triples)
        if alpha < best_alpha:
            best_alpha = alpha
            best_triples = triples

    if not best_triples:
        return []

    # Local Search / Simulated Annealing
    current_triples = best_triples[:]
    current_alpha = best_alpha
    
    # Precompute pair sets for current triples to speed up move generation
    # We'll maintain a set of used pairs
    
    def get_used_pairs(triples):
        used = set()
        for t in triples:
            a, b, c = t
            used.add((a, b))
            used.add((a, c))
            used.add((b, c))
        return used

    used_pairs = get_used_pairs(current_triples)
    
    # Pre-filter candidates? No, just pick random from all_triples.
    # To make it faster, we can maintain a list of "free" triples?
    # That's complex. Random pick and check is O(1) for check.
    
    temperature = 1.0
    cooling = 0.999
    min_temp = 0.01
    
    rng = random.Random(42)
    
    while time.time() < deadline and temperature > min_temp:
        # Generate a move
        move_type = rng.random()
        
        new_triples = current_triples[:]
        new_used_pairs = used_pairs.copy()
        
        if move_type < 0.3:
            # Add a random triple
            t = rng.choice(all_triples)
            a, b, c = t
            pairs = [(a, b), (a, c), (b, c)]
            
            if all(p in new_used_pairs for p in pairs):
                continue
            
            # Check if any pair is used
            conflict = False
            for p in pairs:
                if p in new_used_pairs:
                    conflict = True
                    break
            
            if not conflict:
                new_triples.append(t)
                for p in pairs:
                    new_used_pairs.add(p)
            else:
                continue
                
        elif move_type < 0.6:
            # Remove a random triple
            if not current_triples:
                continue
            idx = rng.randint(0, len(current_triples) - 1)
            t = current_triples[idx]
            a, b, c = t
            pairs = [(a, b), (a, c), (b, c)]
            
            new_triples.pop(idx)
            for p in pairs:
                new_used_pairs.discard(p)
                
        else:
            # Swap: Remove one, try to add another
            if not current_triples:
                continue
            idx = rng.randint(0, len(current_triples) - 1)
            t_remove = current_triples[idx]
            a, b, c = t_remove
            pairs_remove = [(a, b), (a, c), (b, c)]
            
            # Temporarily remove
            temp_triples = current_triples[:]
            temp_triples.pop(idx)
            temp_used = used_pairs.copy()
            for p in pairs_remove:
                temp_used.discard(p)
                
            # Try to add a random new triple
            t_add = rng.choice(all_triples)
            if t_add == t_remove:
                continue
            x, y, z = t_add
            pairs_add = [(x, y), (x, z), (y, z)]
            
            conflict = False
            for p in pairs_add:
                if p in temp_used:
                    conflict = True
                    break
                    
            if conflict:
                continue
                
            new_triples = temp_triples
            new_triples.append(t_add)
            new_used_pairs = temp_used
            for p in pairs_add:
                new_used_pairs.add(p)

        # Evaluate
        new_alpha = compute_alpha_np(new_triples)
        
        delta = new_alpha - current_alpha
        
        if delta < 0:
            # Accept
            current_triples = new_triples
            current_alpha = new_alpha
            used_pairs = new_used_pairs
            
            if current_alpha < best_alpha:
                best_alpha = current_alpha
                best_triples = current_triples[:]
        else:
            # Metropolis criterion
            if temperature > 0:
                prob = np.exp(-delta / temperature)
                if rng.random() < prob:
                    current_triples = new_triples
                    current_alpha = new_alpha
                    used_pairs = new_used_pairs
        
        temperature *= cooling
        
    return [list(t) for t in best_triples]
