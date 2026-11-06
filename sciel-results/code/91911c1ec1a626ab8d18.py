import time
import random
import itertools

def min_hitting_set(triples):
    """
    Find the minimum size of a set of vertices that intersects every triple.
    This is equivalent to 16 - alpha.
    """
    n = 16
    # Convert triples to bitmasks for fast intersection checks
    triple_masks = []
    for t in triples:
        mask = 0
        for v in t:
            mask |= (1 << v)
        triple_masks.append(mask)
    
    # Sort triples by length (all 3) or maybe by frequency?
    # For B&B, order matters.
    
    best_size = 16  # Worst case: all vertices
    
    # Precompute for each vertex, which triples it hits?
    # Or just use the list of remaining triples.
    
    def search(remaining_triples, current_size):
        nonlocal best_size
        
        if current_size >= best_size:
            return
        
        if not remaining_triples:
            best_size = current_size
            return
        
        # Lower bound: 
        # If remaining_triples are disjoint, we need at least len(remaining_triples) more vertices.
        # Simple LB: 0. Better: Greedy packing of remaining triples.
        # For small N, simple LB is often enough with good branching.
        
        # Choose a triple to branch on
        # Heuristic: Choose the triple that appears in the most other triples? 
        # Or just the first one.
        # Let's pick the first one for simplicity, or one with max overlap.
        # To speed up, let's just pick index 0.
        t_mask = remaining_triples[0]
        
        # Find vertices in this triple
        # We must pick at least one vertex from this triple to cover it.
        
        # Get vertices
        verts = []
        m = t_mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            verts.append(v)
            m ^= lsb
            
        for v in verts:
            # Add v to hitting set
            new_remaining = []
            for t in remaining_triples[1:]: # Skip the one we are covering
                if not (t & (1 << v)):
                    new_remaining.append(t)
            
            search(new_remaining, current_size + 1)
            
            # Pruning: If we found a solution, we might be able to prune other branches
            # if current_size + 1 >= best_size (handled at start)
            
    search(triple_masks, 0)
    return best_size

def compute_alpha(triples):
    if not triples:
        return 16
    hit = min_hitting_set(triples)
    return 16 - hit

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    best_alpha = 16
    best_triples = []
    
    # Strategy: Randomized Greedy Packing
    # 1. Generate all possible triples
    # 2. Shuffle
    # 3. Greedily add if linear
    # 4. Evaluate alpha
    # 5. Local search?
    
    all_triples = list(itertools.combinations(range(16), 3))
    
    # To ensure diversity, we can use different random seeds or shuffles
    
    # Optimization: 
    # A maximal linear triple system on 16 points has at most 40 triples.
    # We want to find one with small alpha.
    
    # Let's try to build a system close to 40 triples.
    # Greedy might stop early.
    # We can use a "repair" strategy or just many random restarts.
    
    # Fast check for linearity:
    # Maintain a set of used pairs.
    
    def is_valid_add(triple, used_pairs):
        a, b, c = triple
        pairs = [(a,b), (a,c), (b,c)]
        for p in pairs:
            if p in used_pairs:
                return False
        return True

    def add_triple(triple, used_pairs):
        a, b, c = triple
        used_pairs.add((a,b))
        used_pairs.add((a,c))
        used_pairs.add((b,c))

    # Run many iterations
    for i in range(5000):
        if time.time() > deadline:
            break
            
        random.shuffle(all_triples)
        current_triples = []
        used_pairs = set()
        
        for t in all_triples:
            if is_valid_add(t, used_pairs):
                current_triples.append(t)
                add_triple(t, used_pairs)
                
        # Evaluate
        alpha = compute_alpha(current_triples)
        
        if alpha < best_alpha:
            best_alpha = alpha
            best_triples = current_triples[:]
            
            # If we found alpha=5, it's likely optimal (f(16) is known to be <=6, maybe 5).
            # If we find 5, we can stop or continue to see if 4 is possible (unlikely).
            if best_alpha <= 5:
                # Try to find 4? No, f(16) is likely 5 or 6.
                # If 5 is found, we can try to verify if 4 is possible with more time, 
                # but 5 is a strong result.
                pass

    # If we have time, try local search on the best solution
    if best_triples and time.time() < deadline - 10:
        # Local search: try to swap triples to reduce alpha
        # This is complex. Stick to random restarts which are robust.
        pass

    return [list(t) for t in best_triples]
