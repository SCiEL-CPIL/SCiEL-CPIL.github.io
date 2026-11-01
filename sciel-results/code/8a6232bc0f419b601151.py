"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill
    
    # Strategy:
    # Use vectors with norm squared = 4 (Norm = 2).
    # The constraint is min pairwise dist >= max norm.
    # If all vectors have norm 2, we need dist^2 >= 4.
    # dist^2 = ||u||^2 + ||v||^2 - 2 u.v = 4 + 4 - 2 u.v = 8 - 2 u.v.
    # 8 - 2 u.v >= 4 => u.v <= 2.
    
    # Type A: Weight 1 vectors in {-2, 0, 2}^11.
    # Norm^2 = 4.
    # Count: 11 positions * 2 signs = 22.
    # Mutual compatibility:
    # Same pos, diff sign: u.v = -4 <= 2.
    # Diff pos: u.v = 0 <= 2.
    # All 22 are compatible.
    
    # Type B: Weight 2 vectors in {-1, 0, 1}^11.
    # Norm^2 = 2.
    # Wait, if we mix Norm 2 and Norm sqrt(2), max norm is 2.
    # Constraint: dist^2 >= 4.
    # For u (Norm 2) and v (Norm sqrt(2)):
    # dist^2 = 4 + 2 - 2 u.v = 6 - 2 u.v.
    # 6 - 2 u.v >= 4 => 2 u.v <= 2 => u.v <= 1.
    
    # Let's check Type B vs Type B (both Norm sqrt(2)):
    # dist^2 = 2 + 2 - 2 u.v = 4 - 2 u.v.
    # 4 - 2 u.v >= 4 => -2 u.v >= 0 => u.v <= 0.
    
    # So Type B vectors must have dot product <= 0 with each other.
    # And Type B vs Type A must have dot product <= 1.
    
    # Let's re-evaluate Type B mutual compatibility.
    # u = (1, 1, 0...), v = (1, -1, 0...). u.v = 0. OK.
    # u = (1, 1, 0...), v = (1, 0, 1...). u.v = 1. FAIL (1 > 0).
    # So Type B vectors are NOT mutually compatible if they share a coordinate with same sign.
    
    # Therefore, we cannot just take ALL Type B vectors.
    # We need an independent set for Type B with constraint u.v <= 0.
    # And we need to check Type B vs Type A.
    
    # Let's simplify.
    # Just take Type A (22 vectors).
    # And try to find a large set of Type B vectors compatible with each other and Type A.
    
    # Actually, let's look at Type C: Weight 4 in {-1, 0, 1}^11.
    # We established that W4 vs W4 requires u.v <= 2.
    # W4 vs W1 (Type A) requires u.v <= 2. (Always true for W4, max dot 2).
    # W4 vs W2 (Type B) requires u.v <= 1.
    
    # Given the complexity of mixing, let's stick to the safest high-yield set.
    # The inspiration 2 got ~193 W4 vectors.
    # If we take those 193 W4 vectors, we can add ALL 22 W1 vectors.
    # Total ~215.
    
    # Can we do better?
    # What if we take NO W4, but a large set of W2?
    # W2 vs W2 requires u.v <= 0.
    # This is equivalent to finding a set of edges in K11 such that no two edges share a vertex with the same "sign orientation"?
    # Actually, u.v <= 0 for W2 vectors means they cannot share a coordinate with the same sign.
    # This is like a matching in a signed graph.
    # Max number of W2 vectors with u.v <= 0?
    # We can take all vectors with support {i, j} and signs (+, -) or (-, +)?
    # u=(1, -1, 0..), v=(1, 0, -1..). u.v = 1. Fail.
    
    # Let's just use the W4 + W1 strategy, which is proven to work and yields ~215.
    # We will implement the fast MIS for W4.
    
    # Generate W4 vectors
    v4_candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_candidates.append(v)
            
    # Generate W1 vectors
    v1_candidates = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_candidates.append(v)
            
    vectors_v4 = np.array(v4_candidates, dtype=np.int64)
    vectors_v1 = np.array(v1_candidates, dtype=np.int64)
    
    N4 = len(vectors_v4)
    
    if N4 == 0:
        return vectors_v1
        
    # Compute conflict matrix for W4
    # Conflict if u.v > 2
    dots = vectors_v4 @ vectors_v4.T
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)
    
    # Compute degrees
    degrees = conflicts.sum(axis=1).astype(np.int32)
    
    # Fast MIS using bitmask representation and aggressive local search
    rng = np.random.default_rng(42)
    
    def greedy_mis_fast(conflicts, degrees, deadline, rng):
        """
        Greedy Maximum Independent Set with degree-based ordering.
        Returns list of selected indices.
        """
        N = len(conflicts)
        alive = np.ones(N, dtype=bool)
        deg = degrees.copy()
        
        selected_indices = []
        
        while np.any(alive):
            if time.time() > deadline:
                break
                
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break
                
            # Find min degree among alive
            min_deg = np.min(deg[alive_indices])
            candidates = alive_indices[deg[alive_indices] == min_deg]
            
            # Random tie-breaking
            chosen = candidates[rng.integers(len(candidates))]
            selected_indices.append(chosen)
            
            # Find neighbors to remove
            neighbors = np.where(conflicts[chosen])[0]
            to_remove = np.unique(np.concatenate(([chosen], neighbors)))
            
            # Update alive
            alive[to_remove] = False
            
            # Update degrees for remaining alive nodes
            remaining_alive = np.where(alive)[0]
            if len(remaining_alive) > 0:
                removed_conflicts = conflicts[remaining_alive][:, to_remove].sum(axis=1)
                deg[remaining_alive] -= removed_conflicts

        return selected_indices
    
    def local_search_improve(conflicts, degrees, initial_set, deadline, rng):
        """
        Local search to improve an initial independent set.
        Uses add/remove/swap moves with simulated annealing.
        """
        N = len(conflicts)
        current_selected = set(initial_set)
        current_size = len(current_selected)
        best_selected = set(initial_set)
        best_size = current_size
        
        # Precompute conflict sets as arrays for faster lookup
        conflict_sets = [np.where(conflicts[i])[0] for i in range(N)]
        
        # Temperature schedule
        T_start = 2.0
        T_end = 0.005
        cooling_rate = 0.998
        
        T = T_start
        step = 0
        no_improve_count = 0
        
        while time.time() < deadline and T > T_end:
            step += 1
            
            # Choose a random move: add, remove, or swap
            move_type = rng.integers(3)
            
            if move_type == 0:  # Add a vertex
                # Find vertices that can be added (no conflicts with current set)
                if current_selected:
                    # A vertex can be added if it has no conflicts with any selected vertex
                    can_add_mask = np.ones(N, dtype=bool)
                    for v in current_selected:
                        can_add_mask[conflict_sets[v]] = False
                        can_add_mask[v] = False
                    candidates = np.where(can_add_mask)[0]
                else:
                    candidates = np.arange(N)
                
                if len(candidates) > 0:
                    v_add = candidates[rng.integers(len(candidates))]
                    new_size = current_size + 1
                    
                    # Acceptance: always accept if improves
                    delta = current_size - new_size  # negative if improves
                    if delta < 0 or rng.random() < np.exp(-delta / T):
                        current_selected.add(v_add)
                        current_size = new_size
                        if current_size > best_size:
                            best_selected = set(current_selected)
                            best_size = current_size
                            no_improve_count = 0
                        else:
                            no_improve_count += 1
                            
            elif move_type == 1:  # Remove a vertex
                if current_selected:
                    v_remove = list(current_selected)[rng.integers(len(current_selected))]
                    current_selected.remove(v_remove)
                    current_size -= 1
                    
                    delta = 1  # positive if worsens
                    if delta < 0 or rng.random() < np.exp(-delta / T):
                        # Already removed, accept
                        if current_size > best_size:
                            best_selected = set(current_selected)
                            best_size = current_size
                            no_improve_count = 0
                        else:
                            no_improve_count += 1
                    else:
                        # Reject: put back
                        current_selected.add(v_remove)
                        current_size += 1
                            
            else:  # Swap: remove one, add another
                if current_selected:
                    v_remove = list(current_selected)[rng.integers(len(current_selected))]
                    current_selected.remove(v_remove)
                    
                    # Find vertices that can be added
                    can_add_mask = np.ones(N, dtype=bool)
                    for v in current_selected:
                        can_add_mask[conflict_sets[v]] = False
                        can_add_mask[v] = False
                    candidates = np.where(can_add_mask)[0]
                    
                    if len(candidates) > 0:
                        v_add = candidates[rng.integers(len(candidates))]
                        current_selected.add(v_add)
                        
                        delta = 0  # size unchanged
                        # For swap, always accept if valid (size same)
                        if current_size > best_size:
                            best_selected = set(current_selected)
                            best_size = current_size
                            no_improve_count = 0
                        else:
                            no_improve_count += 1
                    else:
                        # Put back
                        current_selected.add(v_remove)
            
            # Cool down
            T *= cooling_rate
            
            # If no improvement for a while, try a restart from best
            if no_improve_count > 500 and step % 100 == 0:
                # Reset to best and try again with higher temperature
                current_selected = set(best_selected)
                current_size = best_size
                T = max(T, 1.0)
                no_improve_count = 0
        
        return list(best_selected)
    
    def greedy_completion(conflicts, degrees, current_set, deadline, rng):
        """
        Try to add more vertices to current set greedily.
        """
        N = len(conflicts)
        selected = set(current_set)
        size = len(selected)
        
        # Build conflict mask for selected vertices
        conflict_mask = np.zeros(N, dtype=bool)
        for v in selected:
            conflict_mask[v] = True
            conflict_mask[conflicts[v]] = True
        
        # Try to add vertices that don't conflict
        alive = ~conflict_mask
        while np.any(alive):
            if time.time() > deadline:
                break
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break
            
            # Find min degree among alive
            min_deg = np.min(degrees[alive_indices])
            candidates = alive_indices[degrees[alive_indices] == min_deg]
            chosen = candidates[rng.integers(len(candidates))]
            
            # Add it
            selected.add(chosen)
            size += 1
            conflict_mask[chosen] = True
            conflict_mask[conflicts[chosen]] = True
            alive[chosen] = False
            alive[conflicts[chosen]] = False
        
        return list(selected)
    
    # Run multiple strategies and keep the best
    best_result = []
    
    # Strategy 1: Multiple greedy restarts with different orderings
    greedy_deadline = deadline - 60
    for restart in range(5):
        if time.time() > greedy_deadline:
            break
        # Shuffle degrees slightly to get different orderings
        perturbed_degrees = degrees.astype(np.float64) + rng.random(len(degrees)) * 0.1
        result = greedy_mis_fast(conflicts, perturbed_degrees.astype(np.int32), min(greedy_deadline, time.time() + 10), rng)
        if len(result) > len(best_result):
            best_result = result
    
    # Strategy 2: Local search from best greedy result
    if best_result and time.time() < deadline - 20:
        ls_deadline = deadline - 15
        ls_result = local_search_improve(conflicts, degrees, best_result, ls_deadline, rng)
        if len(ls_result) > len(best_result):
            best_result = ls_result
    
    # Strategy 3: Greedy completion from various starting points
    if time.time() < deadline - 10:
        gc_deadline = deadline - 5
        # Try completion from best result
        gc_result = greedy_completion(conflicts, degrees, best_result, gc_deadline, rng)
        if len(gc_result) > len(best_result):
            best_result = gc_result
    
    # Final: try one more local search if time permits
    if best_result and time.time() < deadline - 5:
        final_ls_deadline = deadline - 2
        final_ls_result = local_search_improve(conflicts, degrees, best_result, final_ls_deadline, rng)
        if len(final_ls_result) > len(best_result):
            best_result = final_ls_result
    
    if not best_result:
        return vectors_v1
    
    result_v4 = vectors_v4[best_result]
    
    # Combine with W1
    final_vectors = np.vstack([result_v4, vectors_v1])
    
    return final_vectors

# EVOLVE-BLOCK-END
