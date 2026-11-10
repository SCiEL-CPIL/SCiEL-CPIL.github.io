# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def furstenberg_sarkozy_205() -> np.ndarray:
    """
    Finds a large subset of Z/205Z with no two elements differing by a nonzero square.
    Uses Bitmask representation and Randomized Local Search.
    """
    m = 205
    k = 2
    
    # 1. Compute Forbidden Differences
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, k, m))
    forbidden.discard(0)
    
    # 2. Precompute Conflict Masks
    # conflict_mask[i] is an integer bitmask where bit j is 1 if i and j conflict.
    # Since m=205, we need 205 bits. Python ints handle this easily.
    conflict_masks = [0] * m
    for i in range(m):
        mask = 0
        for j in range(m):
            if i == j:
                continue
            diff = (i - j) % m
            if diff in forbidden:
                mask |= (1 << j)
        conflict_masks[i] = mask
        
    # Helper: Greedy Fill
    # Tries to add vertices to S (and update mask) that are compatible.
    # Uses a provided list of candidates (shuffled for randomness).
    def greedy_fill(S, mask, candidates):
        added = False
        # Iterate through candidates
        # Note: We should check if candidate is already in S? 
        # The caller should ensure candidates are not in S.
        for v in candidates:
            if v in S:
                continue
            if (conflict_masks[v] & mask) == 0:
                S.add(v)
                mask |= (1 << v)
                added = True
        return S, mask, added

    # Initial Solution: Deterministic Greedy
    # To get a strong start, we use a greedy algorithm that picks vertices 
    # with minimum degree in the remaining graph? 
    # For simplicity and speed, we'll use a simple greedy first, 
    # then rely on local search.
    
    # Simple Greedy:
    init_S = set()
    init_mask = 0
    # Order by index? Or random? 
    # Let's try a few initializations and pick the best.
    
    best_S = set()
    best_mask = 0
    best_size = 0
    
    # Run a few deterministic greedy passes with different orders?
    # Or just one standard greedy.
    for v in range(m):
        if (conflict_masks[v] & init_mask) == 0:
            init_S.add(v)
            init_mask |= (1 << v)
            
    if len(init_S) > best_size:
        best_S = set(init_S)
        best_mask = init_mask
        best_size = len(init_S)
        
    # Local Search
    start_time = time.time()
    timeout = 600
    random.seed(42) # Fixed seed for reproducibility of the search process itself
    
    current_S = set(best_S)
    current_mask = best_mask
    
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() - start_time < timeout:
        # 1. Attempt to fill current_S
        # Generate candidates: all vertices not in current_S
        # Shuffling ensures we don't always pick the same "first" available vertex
        candidates = [v for v in range(m) if v not in current_S]
        random.shuffle(candidates)
        
        # We can try to add multiple vertices in one pass
        # The greedy_fill function adds as many as it can in the order provided.
        # However, adding one vertex might block another.
        # A better strategy: 
        # Loop:
        #   Find ALL compatible vertices.
        #   If none, break.
        #   Pick one (randomly from compatible ones).
        #   Add it.
        #   Repeat.
        
        # Let's implement this "Iterative Single Add" for better local search dynamics.
        improved_in_pass = False
        while True:
            # Find compatible vertices
            compatible = []
            for v in candidates: # candidates is static list of non-S
                if v not in current_S: # might have been added in previous iteration of this while? 
                                       # No, we add to current_S, so we should check membership.
                                       # But iterating full list every time is O(N^2).
                                       # N=205, so 205*205 = 42k ops. Fast.
                    if (conflict_masks[v] & current_mask) == 0:
                        compatible.append(v)
                        
            if not compatible:
                break
                
            # Pick a random compatible vertex
            v_to_add = random.choice(compatible)
            current_S.add(v_to_add)
            current_mask |= (1 << v_to_add)
            improved_in_pass = True
            
        if improved_in_pass:
            if len(current_S) > best_size:
                best_size = len(current_S)
                best_S = set(current_S)
                best_mask = current_mask
                no_improve_count = 0
            else:
                # This case (improved_in_pass True but size not > best) implies 
                # we added vertices but the total size is still <= best.
                # This can happen if we started from a smaller set.
                # But we always start from best_S or a perturbed best_S.
                # If we start from best_S, size is best_size. Adding increases it.
                # So if improved_in_pass, size MUST be > best_size (if started from best).
                # If started from perturbed (smaller), it might catch up.
                pass
        else:
            # Stuck. Maximal Independent Set.
            if len(current_S) > best_size:
                best_size = len(current_S)
                best_S = set(current_S)
                best_mask = current_mask
                no_improve_count = 0
            else:
                no_improve_count += 1
                
        # 2. Perturbation (Kick)
        # If we are stuck, remove some vertices to allow new configurations.
        # We do this every iteration to keep the search dynamic, 
        # but the "size" of the kick can vary.
        
        # If we haven't improved in a while, kick harder.
        if no_improve_count > max_no_improve:
            # Hard restart: Random small seed or Empty
            # Let's try empty restart with high probability if very stuck
            if random.random() < 0.1:
                current_S = set()
                current_mask = 0
            else:
                # Remove 3-5 random vertices
                kick_size = random.randint(3, 5)
                if len(current_S) > kick_size:
                    to_remove = random.sample(list(current_S), kick_size)
                    for v in to_remove:
                        current_S.remove(v)
                        current_mask &= ~(1 << v)
            no_improve_count = 0
        else:
            # Soft kick: Remove 1-2 random vertices
            # Only if we are not currently improving? 
            # Actually, if we just filled and couldn't add more, we are at a local max.
            # We should kick to move to a different local max.
            if len(current_S) > 0:
                kick_size = random.randint(1, 2)
                if len(current_S) >= kick_size:
                    to_remove = random.sample(list(current_S), kick_size)
                    for v in to_remove:
                        current_S.remove(v)
                        current_mask &= ~(1 << v)

    return np.array(list(best_S), dtype=np.int64)

# EVOLVE-BLOCK-END
