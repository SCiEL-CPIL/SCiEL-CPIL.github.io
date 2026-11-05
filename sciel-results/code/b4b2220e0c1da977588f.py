"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # We will try to find a valid graph for n from 20 down to 10
    # If we find one for n, we return it.
    # We prioritize larger n.
    
    best_n = 5
    best_edges = []
    
    # Helper to extract edges from adj_bits
    def get_edges(n, adj_bits):
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj_bits[u] & (1 << v):
                    edges.append([u, v])
        return edges

    # Precomputation for subsets
    # We will do this inside the loop for each n, or precompute for max n?
    # Precomputing for max n=22 is fine.
    
    max_n = 22
    all_subsets_6 = []
    all_subsets_7 = []
    
    # Generate subsets for max_n
    # For n < max_n, we only use subsets where all vertices < n
    
    # To save memory and time, we generate subsets for the specific n in the search loop
    # But generating combinations is O(C(n,6)).
    
    def search_for_n(n, time_limit):
        """
        Attempt to find a valid graph on n vertices within time_limit.
        Returns (adj_bits, is_valid)
        """
        # Precompute subsets for this n
        subsets_6 = []
        for combo in combinations(range(n), 6):
            mask = 0
            for v in combo:
                mask |= (1 << v)
            subsets_6.append((combo, mask))
            
        subsets_7 = []
        for combo in combinations(range(n), 7):
            mask = 0
            for v in combo:
                mask |= (1 << v)
            subsets_7.append((combo, mask))
            
        num_subsets_6 = len(subsets_6)
        num_subsets_7 = len(subsets_7)
        
        # Map vertex to list of subset indices containing it
        # This is crucial for fast incremental updates
        vertex_to_subsets_6 = [[] for _ in range(n)]
        for i, (combo, mask) in enumerate(subsets_6):
            for v in combo:
                vertex_to_subsets_6[v].append(i)
                
        vertex_to_subsets_7 = [[] for _ in range(n)]
        for i, (combo, mask) in enumerate(subsets_7):
            for v in combo:
                vertex_to_subsets_7[v].append(i)

        def init_state():
            adj_bits = [0] * n
            for u in range(n):
                for v in range(u + 1, n):
                    if random.random() < 0.5:
                        adj_bits[u] |= (1 << v)
                        adj_bits[v] |= (1 << u)
            
            # Compute degrees for all subsets
            degs_6 = [[0]*6 for _ in range(num_subsets_6)]
            bad_6 = set()
            for i, (combo, mask) in enumerate(subsets_6):
                for j, v in enumerate(combo):
                    degs_6[i][j] = (adj_bits[v] & mask).bit_count()
                if len(set(degs_6[i])) == 1:
                    bad_6.add(i)
                    
            degs_7 = [[0]*7 for _ in range(num_subsets_7)]
            bad_7 = set()
            for i, (combo, mask) in enumerate(subsets_7):
                for j, v in enumerate(combo):
                    degs_7[i][j] = (adj_bits[v] & mask).bit_count()
                if len(set(degs_7[i])) == 1:
                    bad_7.add(i)
                    
            return adj_bits, degs_6, bad_6, degs_7, bad_7

        def flip_edge(adj_bits, u, v):
            adj_bits[u] ^= (1 << v)
            adj_bits[v] ^= (1 << u)

        def update_state(adj_bits, degs_6, bad_6, degs_7, bad_7, u, v):
            """
            Update degrees and bad sets after flipping edge (u, v).
            """
            # Identify affected subsets
            # For 6-subsets
            affected_6 = set(vertex_to_subsets_6[u] + vertex_to_subsets_6[v])
            for i in affected_6:
                combo, mask = subsets_6[i]
                # Determine which vertices in this subset are u or v
                # We need to update degrees for u and v in this subset
                # Note: If u or v is not in the subset, this index shouldn't be here,
                # but vertex_to_subsets ensures they are.
                
                # Check if u is in combo
                if u in combo:
                    # Find index of u in combo
                    idx_u = combo.index(u)
                    # Degree of u changes by +1 if v was neighbor, -1 if not?
                    # We just flipped, so we don't know old state easily without storing it.
                    # Easier: Recalculate degree for u in this subset?
                    # No, that's O(6).
                    # We can track the change.
                    # Old state: was v in adj_bits[u]?
                    # We flipped it.
                    # If we are in the middle of update, adj_bits is already flipped.
                    # So we need to know the OLD value.
                    # Let's pass old_val? Or just recalculate?
                    # Recalculating 2 degrees per affected subset is cheap.
                    pass
                
                # Let's just recalculate the degrees for u and v in this subset
                # This is O(1) per vertex if we have the mask.
                # deg = (adj_bits[vertex] & mask).bit_count()
                
                # Update degs_6[i]
                for j, vertex in enumerate(combo):
                    if vertex == u or vertex == v:
                        degs_6[i][j] = (adj_bits[vertex] & mask).bit_count()
                        
                # Check regularity
                if len(set(degs_6[i])) == 1:
                    bad_6.add(i)
                else:
                    bad_6.discard(i)
                    
            # For 7-subsets
            affected_7 = set(vertex_to_subsets_7[u] + vertex_to_subsets_7[v])
            for i in affected_7:
                combo, mask = subsets_7[i]
                for j, vertex in enumerate(combo):
                    if vertex == u or vertex == v:
                        degs_7[i][j] = (adj_bits[vertex] & mask).bit_count()
                        
                if len(set(degs_7[i])) == 1:
                    bad_7.add(i)
                else:
                    bad_7.discard(i)

        def sample_check_8plus(adj_bits, n, num_samples=100):
            """
            Check if any 8+ subset is regular.
            """
            for sz in range(8, n + 1):
                if sz > n:
                    break
                samples = min(num_samples, 50) # Reduce samples for speed
                for _ in range(samples):
                    idx = random.sample(range(n), sz)
                    mask = 0
                    for v in idx:
                        mask |= (1 << v)
                    degs = [(adj_bits[v] & mask).bit_count() for v in idx]
                    if len(set(degs)) == 1:
                        return False
            return True

        # Main Search Loop
        best_adj = None
        best_bad_6 = float('inf')
        best_bad_7 = float('inf')
        
        # Try multiple restarts
        max_restarts = 20
        for r in range(max_restarts):
            if time.time() > time_limit:
                break
                
            adj_bits, degs_6, bad_6, degs_7, bad_7 = init_state()
            
            current_bad_6 = len(bad_6)
            current_bad_7 = len(bad_7)
            
            if current_bad_6 < best_bad_6 or (current_bad_6 == best_bad_6 and current_bad_7 < best_bad_7):
                best_adj = adj_bits[:]
                best_bad_6 = current_bad_6
                best_bad_7 = current_bad_7
                if current_bad_6 == 0 and current_bad_7 == 0:
                    if sample_check_8plus(adj_bits, n, 50):
                        return adj_bits, True

            stuck = 0
            max_stuck = 100
            
            while time.time() < time_limit:
                if current_bad_6 == 0 and current_bad_7 == 0:
                    if sample_check_8plus(adj_bits, n, 100):
                        return adj_bits, True
                    # If 8+ fails, we are stuck. Break to restart?
                    # Or try to fix 8+? Hard.
                    break
                
                # Pick a move
                if current_bad_6 > 0:
                    # Pick a random bad 6-subset
                    bad_idx = random.choice(list(bad_6))
                    combo, mask = subsets_6[bad_idx]
                    # Pick a random edge in this subset to flip
                    u, v = random.sample(combo, 2)
                else:
                    # current_bad_6 == 0, so current_bad_7 > 0
                    bad_idx = random.choice(list(bad_7))
                    combo, mask = subsets_7[bad_idx]
                    u, v = random.sample(combo, 2)
                
                # Flip edge
                flip_edge(adj_bits, u, v)
                
                # Update state
                update_state(adj_bits, degs_6, bad_6, degs_7, bad_7, u, v)
                
                new_bad_6 = len(bad_6)
                new_bad_7 = len(bad_7)
                
                # Acceptance Criterion: Always accept if it improves the primary metric (bad_6), 
                # or if bad_6 is 0 and it improves bad_7.
                # If it worsens, we might accept with small prob (SA), but let's stick to Greedy for simplicity first.
                # If Greedy gets stuck, we restart.
                
                if new_bad_6 < current_bad_6 or (new_bad_6 == current_bad_6 and new_bad_7 < current_bad_7):
                    current_bad_6 = new_bad_6
                    current_bad_7 = new_bad_7
                    stuck = 0
                else:
                    # Revert
                    flip_edge(adj_bits, u, v)
                    update_state(adj_bits, degs_6, bad_6, degs_7, bad_7, u, v)
                    stuck += 1
                    if stuck > max_stuck:
                        break
                
                # Update best
                if current_bad_6 < best_bad_6 or (current_bad_6 == best_bad_6 and current_bad_7 < best_bad_7):
                    best_adj = adj_bits[:]
                    best_bad_6 = current_bad_6
                    best_bad_7 = current_bad_7
                    if current_bad_6 == 0 and current_bad_7 == 0:
                        if sample_check_8plus(adj_bits, n, 50):
                            return adj_bits, True

        # Return best found, even if not fully valid (caller will verify)
        return best_adj, (best_bad_6 == 0 and best_bad_7 == 0)

    # Try n from 20 down to 10
    for n in range(20, 9, -1):
        time_remaining = deadline - time.time()
        if time_remaining < 5:
            break
            
        # Allocate time for this n
        # If n is large, give it more time
        alloc_time = min(time_remaining - 2, 20) # Max 20s per n
        
        adj_bits, is_valid = search_for_n(n, time.time() + alloc_time)
        
        if is_valid:
            # Final full verification
            # Re-verify 6 and 7 exactly
            subsets_6 = list(combinations(range(n), 6))
            valid_6 = True
            for combo in subsets_6:
                mask = 0
                for v in combo: mask |= (1 << v)
                degs = [(adj_bits[v] & mask).bit_count() for v in combo]
                if len(set(degs)) == 1:
                    valid_6 = False
                    break
            
            if valid_6:
                subsets_7 = list(combinations(range(n), 7))
                valid_7 = True
                for combo in subsets_7:
                    mask = 0
                    for v in combo: mask |= (1 << v)
                    degs = [(adj_bits[v] & mask).bit_count() for v in combo]
                    if len(set(degs)) == 1:
                        valid_7 = False
                        break
                
                if valid_7:
                    # Sample 8+
                    if sample_check_8plus_local(adj_bits, n, 200):
                        edges = get_edges(n, adj_bits)
                        return {'n': n, 'edges': edges}
        
        # If not valid, continue to next n

    # Fallback
    return {'n': 5, 'edges': []}

def sample_check_8plus_local(adj_bits, n, num_samples=100):
    """
    Local helper for sampling check.
    """
    for sz in range(8, n + 1):
        if sz > n:
            break
        samples = min(num_samples, 50)
        for _ in range(samples):
            idx = random.sample(range(n), sz)
            mask = 0
            for v in idx:
                mask |= (1 << v)
            degs = [(adj_bits[v] & mask).bit_count() for v in idx]
            if len(set(degs)) == 1:
                return False
    return True

# EVOLVE-BLOCK-END
