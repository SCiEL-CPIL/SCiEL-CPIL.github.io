import time
import numpy as np

# EVOLVE-BLOCK-START

def _precompute_ap_structure(n):
    """
    Precompute all arithmetic progressions and their membership.
    Returns:
    - aps: list of lists of global indices
    - ap_membership: list of lists of AP indices for each global index
    - pos_in_ap: 2D numpy array (n_aps, n) where pos_in_ap[ap_idx][i] is position of i in ap_idx, or -1
    """
    aps = []
    ap_membership = [[] for _ in range(n)]
    
    # d is step
    for d in range(1, n + 1):
        # r is start residue
        for r in range(d):
            if r >= n:
                break
            indices = list(range(r, n, d))
            if len(indices) < 1:
                continue
            
            ap_idx = len(aps)
            aps.append(indices)
            
            for pos, idx in enumerate(indices):
                ap_membership[idx].append(ap_idx)
                
    n_aps = len(aps)
    # Initialize pos_in_ap
    # Using int16 is enough for n=256
    pos_in_ap = np.full((n_aps, n), -1, dtype=np.int16)
    for ap_idx, indices in enumerate(aps):
        for pos, idx in enumerate(indices):
            pos_in_ap[ap_idx, idx] = pos
            
    return aps, ap_membership, pos_in_ap

def _compute_initial_state(aps, signs):
    """
    Compute prefix sums and max discrepancies for all APs.
    """
    n_aps = len(aps)
    ap_prefix_sums = []
    ap_max_disc = np.zeros(n_aps, dtype=np.int16)
    
    for ap_idx, indices in enumerate(aps):
        seq = np.asarray(signs, dtype=np.int16)[indices]
        cum = np.cumsum(seq)
        ap_prefix_sums.append(cum)
        ap_max_disc[ap_idx] = np.max(np.abs(cum))
        
    return ap_prefix_sums, ap_max_disc

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    n = 256
    
    # 1. Precompute structure
    aps, ap_membership, pos_in_ap = _precompute_ap_structure(n)
    n_aps = len(aps)
    
    # 2. Initialize signs (Thue-Morse)
    # Thue-Morse: (-1)^{number of 1s in binary representation}
    signs = np.array([1 if bin(i).count('1') % 2 == 0 else -1 for i in range(n)], dtype=np.int16)
    
    # 3. Initial state
    ap_prefix_sums, ap_max_disc = _compute_initial_state(aps, signs)
    current_global_max = np.max(ap_max_disc)
    best_signs = signs.copy()
    best_global_max = current_global_max
    
    # Helper to update state after flipping index i
    def flip_sign(i):
        nonlocal ap_prefix_sums, ap_max_disc, current_global_max
        
        old_val = signs[i]
        delta = -2 * old_val
        signs[i] = -old_val
        
        # Update all APs containing i
        affected_aps = ap_membership[i]
        
        for ap_idx in affected_aps:
            pos = pos_in_ap[ap_idx, i]
            if pos == -1:
                continue
            
            # Update prefix sums from pos onwards
            # Note: prefix sums are 1-indexed in concept, but 0-indexed in array
            # P[k] = sum(x[0..k])
            # If x[pos] changes by delta, P[pos], P[pos+1]... change by delta
            ap_prefix_sums[ap_idx][pos:] += delta
            
            # Recompute max for this AP
            new_max = np.max(np.abs(ap_prefix_sums[ap_idx]))
            ap_max_disc[ap_idx] = new_max
        
        # Update global max
        # Optimization: If the flipped AP was the unique max, we might need to re-scan all.
        # If not, we can just check if the new max of affected APs is greater than current,
        # or if the old max AP was affected and reduced, we might need to find the next max.
        # Simple approach: Recalculate global max from ap_max_disc array.
        # O(n_aps) ~ 32k ops. This is fast.
        current_global_max = np.max(ap_max_disc)
        
    # Helper to revert flip
    def revert_flip(i):
        # Just flip again
        flip_sign(i)
        
    # Search Loop
    rng = np.random.default_rng(42)
    
    # Phase 1: Iterated Local Search (ILS)
    # We run local search, then perturb, then local search again.
    
    max_outer_loops = 50
    perturb_strength = 10
    
    for outer_loop in range(max_outer_loops):
        if time.time() > deadline - 5.0:
            break
            
        # Inner Local Search
        no_improve_count = 0
        max_no_improve = 1000
        
        while time.time() < deadline - 5.0 and no_improve_count < max_no_improve:
            
            # Find worst AP
            worst_ap_idx = np.argmax(ap_max_disc)
            worst_disc = ap_max_disc[worst_ap_idx]
            
            if worst_disc == 1:
                break # Optimal (min possible is 1)
                
            # Find best flip within the worst AP (or a few worst APs)
            # To save time, only look at top 5 worst APs
            top_k = 5
            # Get indices of top k APs
            # np.argpartition is faster than full sort
            if n_aps > top_k:
                top_indices = np.argpartition(ap_max_disc, -top_k)[-top_k:]
            else:
                top_indices = np.arange(n_aps)
                
            best_flip_idx = -1
            best_new_global_max = current_global_max
            best_ap_improvement = 0
            
            for ap_idx in top_indices:
                indices = aps[ap_idx]
                ap_disc = ap_max_disc[ap_idx]
                
                # Try flipping each element in this AP
                # Optimization: Only try elements that are likely to help
                # i.e., elements where sign * sum > 0
                # But sum varies by prefix.
                # Just try all? Length of AP is small.
                for pos in range(len(indices)):
                    i = indices[pos]
                    
                    # Estimate improvement
                    # Flipping i changes prefix sums from pos onwards by delta = -2*signs[i]
                    # New max for this AP:
                    # We can simulate quickly
                    old_prefix = ap_prefix_sums[ap_idx]
                    delta = -2 * signs[i]
                    new_prefix = old_prefix.copy()
                    new_prefix[pos:] += delta
                    new_ap_disc = np.max(np.abs(new_prefix))
                    
                    # Improvement in this AP
                    ap_improvement = ap_disc - new_ap_disc
                    
                    # We don't know the global effect without updating, 
                    # but if this AP was the global max and we reduce it, 
                    # global max might drop.
                    
                    # Heuristic: Prefer flips that reduce the worst AP significantly
                    # and don't increase others too much? Hard to know without full update.
                    # Let's just pick the one that reduces the worst AP the most.
                    # If multiple, pick random.
                    
                    if ap_improvement > best_ap_improvement:
                        best_ap_improvement = ap_improvement
                        best_flip_idx = i
                        best_new_global_max = current_global_max # Placeholder, will verify after flip
                
            if best_flip_idx == -1:
                # No improvement found in top APs?
                # Try random flip
                break
                
            # Apply the best flip
            flip_sign(best_flip_idx)
            
            if current_global_max < best_global_max:
                best_global_max = current_global_max
                best_signs = signs.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
                # Revert if it didn't improve global max?
                # In strict local search, yes.
                # But maybe it moves us to a better local minimum?
                # ILS usually allows non-improving moves in the perturbation phase,
                # but in the local search phase, we usually only accept improvements.
                # Let's stick to strict improvement for the local search phase.
                revert_flip(best_flip_idx)
                
        # Perturbation for next ILS iteration
        if time.time() > deadline - 5.0:
            break
            
        # Perturb current state
        for _ in range(perturb_strength):
            i = rng.integers(0, n)
            flip_sign(i)
            
        # After perturbation, update best if improved?
        # Usually we just start the next local search from the perturbed state.
        
    # Final Polish
    if time.time() < deadline - 1.0:
        # Run a few more strict local search steps
        for _ in range(1000):
            if time.time() > deadline - 1.0:
                break
            worst_ap_idx = np.argmax(ap_max_disc)
            if ap_max_disc[worst_ap_idx] == 1:
                break
            indices = aps[worst_ap_idx]
            # Random flip in worst AP
            pos = rng.integers(0, len(indices))
            i = indices[pos]
            flip_sign(i)
            if current_global_max < best_global_max:
                best_global_max = current_global_max
                best_signs = signs.copy()
            else:
                revert_flip(i)
                
    return best_signs.tolist()

# EVOLVE-BLOCK-END
