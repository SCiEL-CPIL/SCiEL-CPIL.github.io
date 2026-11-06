import time
import random
import math

def construct():
    t0 = time.time()
    DEADLINE = t0 + 55.0
    
    N = 256
    
    # 1. Build Conflict Masks
    # conflict_mask[i] has bit j set if (i+1, j+1) conflict
    conflict_mask = [0] * N
    for i in range(N):
        a = i + 1
        mask = 0
        for j in range(N):
            if i == j:
                continue
            b = j + 1
            if (a * b) % (a + b) == 0:
                mask |= (1 << j)
        conflict_mask[i] = mask

    # 2. Reduction: Remove dominated vertices
    # A vertex u is dominated by v if N[u] subseteq N[v]
    # N[u] = {u} U N(u)
    # We work with a list of active vertices
    active = [1] * N
    
    def is_dominated(u, active):
        # Check if there exists v != u such that N[u] subseteq N[v]
        # N[u] bits: u and its neighbors
        nu = (1 << u) | conflict_mask[u]
        for v in range(N):
            if not active[v] or v == u:
                continue
            nv = (1 << v) | conflict_mask[v]
            # Check if nu is subset of nv
            # This is (nu & ~nv) == 0
            if (nu & ~nv) == 0:
                return True
        return False

    # Iterative reduction
    changed = True
    while changed:
        changed = False
        for u in range(N):
            if not active[u]:
                continue
            if is_dominated(u, active):
                active[u] = 0
                changed = True
            if time.time() > DEADLINE - 10:
                break
        if time.time() > DEADLINE - 10:
            break

    # Get active indices
    active_indices = [i for i in range(N) if active[i]]
    
    # If all active, no reduction happened
    # Map active indices to 0..M-1 for easier B&B?
    # Or just use the original indices with a mask.
    
    # 3. Solve MIS on the active subgraph
    # Use Branch and Bound with simple bounds
    
    best_size = 0
    best_mask = 0
    
    # Initial greedy solution on active vertices
    # Sort by degree
    degrees = [bin(conflict_mask[i] & ((1 << N) - 1)).count('1') for i in active_indices]
    # Actually, degree within active subgraph
    active_mask = 0
    for i in active_indices:
        active_mask |= (1 << i)
        
    # Greedy
    current_mask = 0
    remaining = active_mask
    while remaining:
        # Pick vertex with min degree in remaining? Or max?
        # For MIS, picking min degree often works well? 
        # Or just pick first one.
        # Let's pick one with min degree to remaining
        min_deg = N
        min_idx = -1
        temp = remaining
        while temp:
            lsb = temp & -temp
            idx = lsb.bit_length() - 1
            temp ^= lsb
            # Count neighbors in remaining
            deg = bin(conflict_mask[idx] & remaining).count('1')
            if deg < min_deg:
                min_deg = deg
                min_idx = idx
        if min_idx != -1:
            current_mask |= (1 << min_idx)
            # Remove min_idx and its neighbors from remaining
            remaining &= ~( (1 << min_idx) | conflict_mask[min_idx] )
            
    best_size = bin(current_mask).count('1')
    best_mask = current_mask

    # B&B
    # State: candidates_mask, current_size, current_mask
    # Bound: current_size + popcount(candidates_mask) <= best_size -> prune
    
    def solve(candidates, curr_size, curr_mask):
        nonlocal best_size, best_mask
        
        if time.time() > DEADLINE:
            return
            
        if candidates == 0:
            if curr_size > best_size:
                best_size = curr_size
                best_mask = curr_mask
            return
            
        # Bound
        if curr_size + bin(candidates).count('1') <= best_size:
            return
            
        # Pick a vertex
        # Heuristic: Pick vertex with max degree in candidates
        temp = candidates
        max_deg = -1
        pivot = -1
        while temp:
            lsb = temp & -temp
            idx = lsb.bit_length() - 1
            temp ^= lsb
            deg = bin(conflict_mask[idx] & candidates).count('1')
            if deg > max_deg:
                max_deg = deg
                pivot = idx
                
        if pivot == -1:
            return

        # Branch 1: Include pivot
        # New candidates: candidates without pivot and without its neighbors
        new_candidates = candidates & ~( (1 << pivot) | conflict_mask[pivot] )
        solve(new_candidates, curr_size + 1, curr_mask | (1 << pivot))
        
        # Branch 2: Exclude pivot
        new_candidates = candidates & ~(1 << pivot)
        solve(new_candidates, curr_size, curr_mask)

    # Run B&B
    solve(active_mask, 0, 0)

    # 4. Local Search Polish on best_mask
    # Try to add any vertex that fits
    for i in active_indices:
        if not ((best_mask >> i) & 1):
            if (conflict_mask[i] & best_mask) == 0:
                best_mask |= (1 << i)
                best_size += 1

    # Simple 1-swap local search
    improved = True
    while improved and time.time() < DEADLINE:
        improved = False
        # Try to swap one out, one in
        for i in active_indices:
            if (best_mask >> i) & 1:
                # Try to remove i
                trial_mask = best_mask & ~(1 << i)
                # Try to add any j not in best_mask
                for j in active_indices:
                    if j == i:
                        continue
                    if not ((trial_mask >> j) & 1):
                        if (conflict_mask[j] & trial_mask) == 0:
                            # Swap i for j
                            # Size is same, so we only do this if it helps future moves?
                            # Or if j has fewer conflicts?
                            # For now, skip same-size swaps unless we have a specific heuristic.
                            pass
                            # Actually, if we just want max size, same size swaps don't increase size.
                            # They might help escape local optima.
                            # But with B&B, we might already be optimal or close.
                            # Let's stick to adding if possible.

    # Final check
    if bin(best_mask).count('1') == 0:
        # Fallback
        best_mask = 1 << active_indices[0]
        best_size = 1

    result = sorted([i + 1 for i in range(N) if (best_mask >> i) & 1])
    return result
