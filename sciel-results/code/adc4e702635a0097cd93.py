import time
import random
import itertools
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    # Build a dense linear triple system on 16 points.
    # Strategy: start from PG(3,2) lines on 15 points (a Steiner triple system),
    # then use simulated annealing to add triples involving point 15
    # and swap triples to maximize density while maintaining linearity.

    # Step 1: Construct STS(15) from PG(3,2)
    vectors = [
        0b0001, 0b0010, 0b0011, 0b0100, 0b0101,
        0b0110, 0b0111, 0b1000, 0b1001, 0b1010,
        0b1011, 0b1100, 0b1101, 0b1110, 0b1111
    ]
    vec_to_idx = {v: i for i, v in enumerate(vectors)}

    sts_triples = []
    n15 = 15
    for i in range(n15):
        for j in range(i + 1, n15):
            v_k = vectors[i] ^ vectors[j]
            k = vec_to_idx[v_k]
            t = tuple(sorted([i, j, k]))
            sts_triples.append(t)

    # Step 2: Fast alpha computation using bitmask
    def compute_alpha(triples):
        n_points = 16
        # Build a set of triple masks for quick lookup
        triple_masks = set()
        for t in triples:
            m = 0
            for p in t:
                m |= (1 << p)
            triple_masks.add(m)

        # Check if a given subset (as bitmask) is independent
        def is_independent(mask):
            # Check all triples: if any triple is subset of mask, not independent
            for tm in triple_masks:
                if (mask & tm) == tm:
                    return False
            return True

        # Find largest independent set
        # Try from largest to smallest
        for k in range(n_points, 0, -1):
            for combo in itertools.combinations(range(n_points), k):
                mask = 0
                for p in combo:
                    mask |= (1 << p)
                if is_independent(mask):
                    return k
        return 1

    # Step 3: Faster alpha using numpy
    def compute_alpha_fast(triples):
        n_points = 16
        n_triples = len(triples)
        if n_triples == 0:
            return n_points

        # Create triple masks
        triple_masks = np.zeros(n_triples, dtype=np.int64)
        for idx, t in enumerate(triples):
            m = 0
            for p in t:
                m |= (1 << p)
            triple_masks[idx] = m

        def has_independent_set(k):
            if k == 0:
                return True
            combos = list(itertools.combinations(range(n_points), k))
            if not combos:
                return False

            combo_masks = np.zeros(len(combos), dtype=np.int64)
            for i, s in enumerate(combos):
                m = 0
                for p in s:
                    m |= (1 << p)
                combo_masks[i] = m

            # Vectorized: for each combo, check if any triple is contained
            # (combo_mask & triple_mask) == triple_mask
            contains = (combo_masks[:, None] & triple_masks[None, :]) == triple_masks[None, :]
            any_contains = np.any(contains, axis=1)
            # If no combo has any triple contained, it's independent
            return not np.all(any_contains)

        for k in range(n_points, 0, -1):
            if has_independent_set(k):
                return k
        return 1

    # Step 4: Build initial system with triples involving point 15
    # We need to add triples containing point 15 while maintaining linearity
    # Each triple with 15 uses two other points. No two triples can share 2 points.
    # So the pairs of non-15 points used with 15 must form a matching (no shared point).
    # Actually, they can share at most 1 point total, so pairs can share at most 1 point.
    
    # Start with the STS(15) and try to add triples with point 15
    current_triples = [list(t) for t in sts_triples]
    
    # Precompute which pairs are already in a triple together (in STS)
    pair_in_triple = set()
    for t in sts_triples:
        pair_in_triple.add((t[0], t[1]))
        pair_in_triple.add((t[0], t[2]))
        pair_in_triple.add((t[1], t[2]))

    # Add triples with point 15: we need pairs (a,b) from 0..14 that are not in pair_in_triple
    # and also the new triple doesn't conflict with existing triples
    # A triple (a,b,15) conflicts with existing triple t if |{a,b,15} ∩ t| > 1
    # Since 15 is not in any existing triple, we need |{a,b} ∩ t| <= 1 for all t
    
    # This means: no existing triple contains both a and b (i.e., (a,b) not in pair_in_triple)
    # So we need a matching on the 15 points where no edge is in pair_in_triple
    # But pair_in_triple contains all pairs from STS(15), which is a 2-(15,3,1) design
    # So every pair IS in exactly one triple! That means we can't add any triple with 15
    # without removing an existing triple.
    
    # So we need to do swaps: remove some STS triples and add triples with 15
    # to increase density and reduce independence number.

    # Simulated Annealing approach:
    # State: set of triples
    # Moves: 
    #   1. Remove one triple, add one triple with point 15 (if linear)
    #   2. Remove one triple, add a random triple (if linear)
    #   3. Swap: remove two triples, add two triples (if linear)
    
    # Let's use a different approach: build from scratch using SA
    # to maximize the number of triples in a linear triple system on 16 points.
    
    def is_linear(triples):
        """Check if the triple system is linear (any two triples share at most 1 point)."""
        n = len(triples)
        for i in range(n):
            si = set(triples[i])
            for j in range(i + 1, n):
                if len(si & set(triples[j])) > 1:
                    return False
        return True

    def is_linear_fast(triple, others):
        """Check if a single triple is linear with all triples in others."""
        st = set(triple)
        for t in others:
            if len(st & set(t)) > 1:
                return False
        return True

    # SA to build a dense linear triple system
    random.seed(42)
    
    # Start with empty system and greedily add triples
    # Then use SA to improve
    
    # Actually, let's start with the STS(15) and do SA swaps
    current = [list(t) for t in sts_triples]
    current_alpha = compute_alpha_fast(current)
    best = current[:]
    best_alpha = current_alpha
    
    # Precompute all possible triples
    all_triples = list(itertools.combinations(range(16), 3))
    
    # SA parameters
    T_start = 10.0
    T_end = 0.01
    n_sa_iters = 50000
    
    def evaluate(triples):
        """Score: negative alpha (we want to minimize alpha) plus a bonus for more triples."""
        alpha = compute_alpha_fast(triples)
        # We want to minimize alpha, and among same alpha, maximize triples
        return alpha * 1000 - len(triples)
    
    current_score = evaluate(current)
    best_score = current_score
    
    T = T_start
    cooling = (T_end / T_start) ** (1.0 / max(1, n_sa_iters - 1))
    
    for it in range(n_sa_iters):
        if time.time() > deadline - 10:
            break
        
        T *= cooling
        
        # Generate a random move
        move_type = random.random()
        
        if move_type < 0.5 and len(current) > 0:
            # Remove one triple, add another
            idx = random.randint(0, len(current) - 1)
            removed = current[idx]
            remaining = current[:idx] + current[idx+1:]
            
            # Try to add a random triple
            new_triple = list(random.choice(all_triples))
            if is_linear_fast(new_triple, remaining) and tuple(sorted(new_triple)) not in set(tuple(sorted(t)) for t in remaining):
                new_state = remaining + [new_triple]
                new_score = evaluate(new_state)
                delta = new_score - current_score
                if delta < 0 or random.random() < np.exp(-delta / max(T, 1e-10)):
                    current = new_state
                    current_score = new_score
                    if current_score < best_score:
                        best_score = current_score
                        best = current[:]
        elif move_type < 0.8 and len(current) > 0:
            # Remove one triple, add one with point 15
            idx = random.randint(0, len(current) - 1)
            removed = current[idx]
            remaining = current[:idx] + current[idx+1:]
            
            # Generate a triple with point 15
            a, b = random.sample(range(15), 2)
            new_triple = sorted([a, b, 15])
            if is_linear_fast(new_triple, remaining) and tuple(new_triple) not in set(tuple(sorted(t)) for t in remaining):
                new_state = remaining + [new_triple]
                new_score = evaluate(new_state)
                delta = new_score - current_score
                if delta < 0 or random.random() < np.exp(-delta / max(T, 1e-10)):
                    current = new_state
                    current_score = new_score
                    if current_score < best_score:
                        best_score = current_score
                        best = current[:]
        else:
            # Swap: remove two, add two
            if len(current) >= 2:
                idx1, idx2 = random.sample(range(len(current)), 2)
                remaining = [t for i, t in enumerate(current) if i != idx1 and i != idx2]
                
                # Add two random triples
                t1 = list(random.choice(all_triples))
                t2 = list(random.choice(all_triples))
                if (is_linear_fast(t1, remaining) and is_linear_fast(t2, remaining + [t1])
                        and tuple(sorted(t1)) not in set(tuple(sorted(t)) for t in remaining)
                        and tuple(sorted(t2)) not in set(tuple(sorted(t)) for t in remaining + [t1])):
                    new_state = remaining + [t1, t2]
                    new_score = evaluate(new_state)
                    delta = new_score - current_score
                    if delta < 0 or random.random() < np.exp(-delta / max(T, 1e-10)):
                        current = new_state
                        current_score = new_score
                        if current_score < best_score:
                            best_score = current_score
                            best = current[:]
    
    # Final polish: try to add more triples greedily
    # Sort triples by how "useful" they are (cover more isolated points)
    if time.time() < deadline - 5:
        current = best[:]
        current_alpha = compute_alpha_fast(current)
        
        # Try adding triples one by one
        existing_set = set(tuple(sorted(t)) for t in current)
        for t in all_triples:
            if time.time() > deadline - 2:
                break
            if t in existing_set:
                continue
            if is_linear_fast(t, current):
                new_state = current + [list(t)]
                new_alpha = compute_alpha_fast(new_state)
                if new_alpha <= current_alpha:
                    current = new_state
                    current_alpha = new_alpha
                    existing_set.add(t)
        
        best = current[:]
        best_alpha = current_alpha
    
    # Verify final answer
    final_alpha = compute_alpha_fast(best)
    return best

# Quick test
if __name__ == "__main__":
    t0 = time.time()
    result = construct()
    t1 = time.time()
    print(f"Time: {t1-t0:.1f}s, Triples: {len(result)}")
    # Verify linearity
    for i in range(len(result)):
        si = set(result[i])
        for j in range(i+1, len(result)):
            assert len(si & set(result[j])) <= 1, f"Conflict: {result[i]} and {result[j]}"
    print("Linear: OK")
