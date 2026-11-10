import numpy as np
import time
import random
import math

def furstenberg_sarkozy_205() -> np.ndarray:
    """
    Finds a large subset of Z/205Z with no two elements differing by a nonzero square.
    Uses Multi-Start Simulated Annealing with Bitset-based state representation.
    """
    m = 205
    
    # 1. Precompute forbidden differences and conflict masks
    forbidden = set(pow(x, 2, m) for x in range(1, m))
    forbidden.discard(0)
    
    # conflict_mask[i] has bit j set if i and j are adjacent (conflict)
    conflict_mask = [0] * m
    for i in range(m):
        mask = 0
        for j in range(m):
            if i != j and (i - j) % m in forbidden:
                mask |= (1 << j)
        conflict_mask[i] = mask
        
    # Precompute all vertices list for random selection
    all_vertices = list(range(m))
    
    best_mask = 0
    best_size = 0
    
    start_time = time.time()
    deadline = start_time + 590.0  # Safety margin
    
    def get_size(mask):
        return mask.bit_count()
        
    def random_bit(mask):
        """Returns the index of a random set bit in mask, or -1 if mask is 0."""
        if mask == 0:
            return -1
        # Get list of set bits? Too slow if done every time.
        # Better: pick a random position and check?
        # Or: convert to list of bits?
        # For m=205, converting to list is fast enough?
        # Let's try a faster way:
        # We can precompute a list of indices for each possible mask? No, 2^205 is huge.
        # We can just iterate?
        # Actually, picking a random index and checking if bit is set is O(1) expected if density is high, O(n) if low.
        # Density is ~ 100/205 = 0.5. So it's fast.
        while True:
            idx = random.randint(0, m - 1)
            if mask & (1 << idx):
                return idx
            # If mask is very sparse, this might loop.
            # But we usually operate on dense-ish sets or we check size first.
            
    def random_unset_bit(mask):
        """Returns the index of a random unset bit in mask, or -1 if mask is full."""
        # Complement mask within m bits
        full_mask = (1 << m) - 1
        unset_mask = full_mask ^ mask
        if unset_mask == 0:
            return -1
        while True:
            idx = random.randint(0, m - 1)
            if unset_mask & (1 << idx):
                return idx

    def greedy_fill():
        """
        Creates a random initial independent set using randomized greedy.
        """
        curr_mask = 0
        # Randomize order of vertices
        order = all_vertices[:]
        random.shuffle(order)
        for v in order:
            if (curr_mask & conflict_mask[v]) == 0:
                curr_mask |= (1 << v)
        return curr_mask

    def simulate_annealing(initial_mask, t_start, t_end, cooling_rate, max_iter):
        """
        Runs SA on the given initial mask.
        Returns the best mask found during the run.
        """
        curr_mask = initial_mask
        curr_size = get_size(curr_mask)
        best_local_mask = curr_mask
        best_local_size = curr_size
        
        T = t_start
        
        for _ in range(max_iter):
            if time.time() > deadline:
                break
                
            # Generate a neighbor
            move_type = random.random()
            
            if move_type < 0.5:
                # Try to add a vertex
                v = random_unset_bit(curr_mask)
                if v == -1:
                    continue # Full
                if (curr_mask & conflict_mask[v]) == 0:
                    new_mask = curr_mask | (1 << v)
                    new_size = curr_size + 1
                    # Always accept improvement
                    curr_mask = new_mask
                    curr_size = new_size
                    if curr_size > best_local_size:
                        best_local_size = curr_size
                        best_local_mask = curr_mask
                else:
                    # If can't add, maybe swap?
                    # Try 1-swap: remove u, add v
                    u = random_bit(curr_mask)
                    if u == -1:
                        continue
                    temp_mask = curr_mask & ~(1 << u)
                    if (temp_mask & conflict_mask[v]) == 0:
                        new_mask = temp_mask | (1 << v)
                        new_size = curr_size # Size unchanged
                        # Accept with probability 1 (neutral move) or slight bias?
                        # For exploration, accept neutral moves.
                        curr_mask = new_mask
                        # Size same, so no update to best_local unless we track diversity?
                        # We just keep the state.
            elif move_type < 0.8:
                # Try 1-swap (remove u, add v) where v is random
                u = random_bit(curr_mask)
                if u == -1:
                    continue
                v = random_unset_bit(curr_mask & ~(1 << u)) # v not in mask, and v != u (u is in mask)
                # Actually random_unset_bit(curr_mask) gives v not in curr_mask.
                # We need v such that (curr_mask \ {u}) is compatible with v.
                temp_mask = curr_mask & ~(1 << u)
                # Find a random v not in temp_mask that is compatible?
                # Or just pick random v not in curr_mask and check.
                v = random_unset_bit(curr_mask)
                if v == -1:
                    continue
                if (temp_mask & conflict_mask[v]) == 0:
                    new_mask = temp_mask | (1 << v)
                    new_size = curr_size
                    curr_mask = new_mask
            else:
                # Remove a vertex (worsening move, for escape)
                u = random_bit(curr_mask)
                if u == -1:
                    continue
                new_mask = curr_mask & ~(1 << u)
                new_size = curr_size - 1
                # Accept with probability exp((new_size - curr_size) / T) = exp(-1/T)
                if random.random() < math.exp(-1.0 / T):
                    curr_mask = new_mask
                    curr_size = new_size
                    
            # Cool down
            T *= cooling_rate
            
        return best_local_mask, best_local_size

    # Main Loop: Multi-Start SA
    # We run multiple SA instances.
    # Each instance gets a time slice.
    # Or we run one long SA with restarts?
    # Let's do short runs with different random seeds/initializations.
    
    while time.time() < deadline:
        # Randomized Greedy Init
        init_mask = greedy_fill()
        
        # SA Parameters
        # We want to finish this instance quickly to try more.
        # Let's aim for ~1000-5000 iterations per SA run depending on speed.
        # Estimate time per iter: ~50-100us?
        # 5000 iters ~ 0.25-0.5s.
        # So we can do ~1000-2000 runs in 600s.
        
        t_start = 1.0
        t_end = 0.01
        cooling_rate = 0.995 # Slow cooling
        max_iter = 5000
        
        best_local_mask, best_local_size = simulate_annealing(init_mask, t_start, t_end, cooling_rate, max_iter)
        
        if best_local_size > best_size:
            best_size = best_local_size
            best_mask = best_local_mask
            
        # Optional: Also try a deterministic local search on the result?
        # The SA already does local search via acceptance of improvements.
        
    # Convert best_mask to array
    result = []
    for i in range(m):
        if best_mask & (1 << i):
            result.append(i)
            
    return np.array(result, dtype=np.int64)

# EVOLVE-BLOCK-END
