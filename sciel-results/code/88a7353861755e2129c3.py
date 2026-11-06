import time
import random
import numpy as np
import itertools

DEADLINE_SECONDS = 110.0 # Safety margin
START_TIME = None

def _build_masks(adj):
    n = adj.shape[0]
    masks = [0] * n
    for u in range(n):
        mask = 0
        for v in range(n):
            if adj[u, v]:
                mask |= (1 << v)
        masks[u] = mask
    return masks

def _count_violations_size_k(masks, n, k):
    """Count regular induced subgraphs of exactly size k."""
    count = 0
    for combo in itertools.combinations(range(n), k):
        subset_mask = 0
        for v in combo:
            subset_mask |= (1 << v)
        
        degs = []
        valid = True
        for u in combo:
            d = (masks[u] & subset_mask).bit_count()
            if not degs:
                degs.append(d)
            else:
                if d != degs[0]:
                    valid = False
                    break
        if valid:
            count += 1
    return count

def _count_total_violations(masks, n, check_larger=True):
    """Count total violations. Optimized to stop early if only validity is needed? 
    No, we return the count for fitness."""
    total = 0
    # Size 6 is the bottleneck
    total += _count_violations_size_k(masks, n, 6)
    if check_larger:
        for k in range(7, n + 1):
            total += _count_violations_size_k(masks, n, k)
    return total

def construct():
    global START_TIME
    if START_TIME is None:
        START_TIME = time.time()
    
    deadline = START_TIME + DEADLINE_SECONDS
    rng = np.random.default_rng(42)
    
    best_n = 0
    best_edges = []
    
    # Try n from 21 down to 20 (20 is known to be possible)
    for target_n in [21, 20]:
        if time.time() > deadline - 5.0:
            break
            
        found = False
        # Multiple restarts
        max_restarts = 10 if target_n == 21 else 5
        
        for r in range(max_restarts):
            if time.time() > deadline - 2.0:
                break
                
            # Init random graph
            p = 0.5
            adj = (rng.random((target_n, target_n)) < p).astype(np.int8)
            adj = np.triu(adj, 1)
            adj = adj + adj.T
            np.fill_diagonal(adj, 0)
            
            masks = _build_masks(adj)
            
            # Local Search / Simulated Annealing
            # We focus on minimizing size-6 violations first.
            # If size-6 is 0, we check larger.
            
            current_viol_6 = _count_violations_size_k(masks, target_n, 6)
            # If we start with 0 size-6 violations, check larger
            if current_viol_6 == 0:
                total_viol = _count_total_violations(masks, target_n, check_larger=True)
                if total_viol == 0:
                    found = True
                    best_n = target_n
                    # Convert to edges
                    edges = []
                    for u in range(target_n):
                        for v in range(u + 1, target_n):
                            if adj[u, v]:
                                edges.append([u, v])
                    best_edges = edges
                    break
            
            # Search Loop
            max_steps = 500
            for step in range(max_steps):
                if time.time() > deadline - 1.0:
                    break
                
                # Find a violating subset of size 6
                # We can cache the first violation found
                viol_combo = None
                for combo in itertools.combinations(range(target_n), 6):
                    subset_mask = 0
                    for v in combo:
                        subset_mask |= (1 << v)
                    degs = []
                    valid = True
                    for u in combo:
                        d = (masks[u] & subset_mask).bit_count()
                        if not degs:
                            degs.append(d)
                        else:
                            if d != degs[0]:
                                valid = False
                                break
                    if valid:
                        viol_combo = combo
                        break
                
                if viol_combo is None:
                    # No size 6 violations. Check larger.
                    total_viol = _count_total_violations(masks, target_n, check_larger=True)
                    if total_viol == 0:
                        found = True
                        best_n = target_n
                        edges = []
                        for u in range(target_n):
                            for v in range(u + 1, target_n):
                                if adj[u, v]:
                                    edges.append([u, v])
                        best_edges = edges
                        break
                    else:
                        # There is a violation of size > 6.
                        # Find it (first one)
                        viol_combo = None
                        for k in range(7, target_n + 1):
                            for combo in itertools.combinations(range(target_n), k):
                                subset_mask = 0
                                for v in combo:
                                    subset_mask |= (1 << v)
                                degs = []
                                valid = True
                                for u in combo:
                                    d = (masks[u] & subset_mask).bit_count()
                                    if not degs:
                                        degs.append(d)
                                    else:
                                        if d != degs[0]:
                                            valid = False
                                            break
                                if valid:
                                    viol_combo = combo
                                    break
                            if viol_combo:
                                break
                        
                        if viol_combo is None:
                            # Should not happen if total_viol > 0
                            continue
                        
                        # Flip an edge in the violating subset
                        # Random pick from viol_combo
                        u, v = rng.choice(viol_combo, size=2, replace=False)
                else:
                    # Flip an edge in the violating subset of size 6
                    u, v = rng.choice(viol_combo, size=2, replace=False)
                
                # Perform flip
                old_val = adj[u, v]
                new_val = 1 - old_val
                adj[u, v] = new_val
                adj[v, u] = new_val
                
                # Update masks
                if new_val == 1:
                    masks[u] |= (1 << v)
                    masks[v] |= (1 << u)
                else:
                    masks[u] &= ~(1 << v)
                    masks[v] &= ~(1 << u)
                
                # Evaluate new state
                # Only check size 6 for speed in the inner loop
                new_viol_6 = _count_violations_size_k(masks, target_n, 6)
                
                # Simple Hill Climbing: Accept if better or equal (with random tie break to escape local minima)
                if new_viol_6 < current_viol_6:
                    current_viol_6 = new_viol_6
                else:
                    # Revert
                    adj[u, v] = old_val
                    adj[v, u] = old_val
                    if old_val == 1:
                        masks[u] |= (1 << v)
                        masks[v] |= (1 << u)
                    else:
                        masks[u] &= ~(1 << v)
                        masks[v] &= ~(1 << u)
                    
                    # If stuck, try a random flip (Annealing)
                    if rng.random() < 0.1:
                        u2, v2 = rng.choice(target_n, size=2, replace=False)
                        if u2 != v2:
                            old_val2 = adj[u2, v2]
                            new_val2 = 1 - old_val2
                            adj[u2, v2] = new_val2
                            adj[v2, u2] = new_val2
                            if new_val2 == 1:
                                masks[u2] |= (1 << v2)
                                masks[v2] |= (1 << u2)
                            else:
                                masks[u2] &= ~(1 << v2)
                                masks[v2] &= ~(1 << u2)
                            # Don't update current_viol_6 immediately, let it be evaluated next step
                            # Actually, better to revert if it doesn't help?
                            # For simplicity, accept random moves occasionally.
                            # Recalculate viol_6 for this random move? 
                            # That's expensive. Let's just accept and see.
                            # But we need to know if we improved.
                            # Let's just keep the random move and let the next iteration check it.
                            # But we need to update current_viol_6 to reflect the state.
                            # This is getting complex. 
                            # Simpler: Only accept if strictly better. If stuck, restart.
                            # Revert the random move too to keep state consistent with current_viol_6 logic?
                            # No, let's just restart if we don't improve for X steps.
                            
                    # Track stagnation
                    # If we don't improve, we might be stuck.
                    # For now, just continue. If we don't reach 0, we'll restart.
            
            if found:
                break
        
        if found:
            break
    
    if best_n == 0:
        # Fallback: Return a small valid graph (e.g., empty graph on 5 vertices)
        # The task says "Maximize n". A small n is valid but low score.
        # Better to return a known valid small graph than nothing.
        return {'n': 5, 'edges': []}
        
    return {'n': best_n, 'edges': best_edges}
