import numpy as np
import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    # Safety margin: Evaluator kills at 150s. We use 145s.
    deadline = start_time + 145.0
    
    N = 256
    
    # 1. Precompute all 4-term arithmetic progressions (0-indexed)
    # A 4-AP is (a, a+d, a+2d, a+3d) with 0 <= a, a+3d < N
    # d >= 1
    ap_list = []
    for d in range(1, N // 3 + 1):
        max_a = N - 3 * d
        for a in range(0, max_a):
            ap_list.append((a, a+d, a+2*d, a+3*d))
            
    if not ap_list:
        return [0] * N
        
    aps = np.array(ap_list, dtype=np.int32)
    num_aps = len(aps)
    
    # 2. Precompute adjacency for fast vectorized delta calculation
    # For each index i, we need to know the indices of the other 3 elements in each AP it belongs to.
    # Let's build a structure that allows us to compute the delta for ALL i at once?
    # Or at least for a batch?
    
    # Actually, let's stick to the per-index vectorized delta but use it in a deterministic greedy loop.
    # The key difference from SA is that we don't flip randomly; we evaluate all positions 
    # and flip those that improve the score.
    
    # Precompute neighbor data for each index i:
    # neighbors[i] = list of (o1, o2, o3) for each AP containing i.
    index_to_aps = [[] for _ in range(N)]
    
    for ap_idx in range(num_aps):
        a, b, c, d = aps[ap_idx]
        index_to_aps[a].append((b, c, d))
        index_to_aps[b].append((a, c, d))
        index_to_aps[c].append((a, b, d))
        index_to_aps[d].append((a, b, c))
        
    # Convert to numpy arrays for fast access
    neighbor_data = []
    for i in range(N):
        if not index_to_aps[i]:
            neighbor_data.append(None)
        else:
            o1s = []
            o2s = []
            o3s = []
            for o1, o2, o3 in index_to_aps[i]:
                o1s.append(o1)
                o2s.append(o2)
                o3s.append(o3)
            neighbor_data.append((np.array(o1s, dtype=np.int32), 
                                  np.array(o2s, dtype=np.int32), 
                                  np.array(o3s, dtype=np.int32)))

    def compute_score(c):
        ca = c[aps[:, 0]]
        cb = c[aps[:, 1]]
        cc = c[aps[:, 2]]
        cd = c[aps[:, 3]]
        is_mono = (ca == cb) & (cb == cc) & (cc == cd)
        return np.sum(is_mono)

    def greedy_local_search(c, max_sweeps=100):
        """
        Perform deterministic greedy local search.
        In each sweep, we iterate through indices 0..N-1.
        For each index, we compute the delta.
        If delta < 0, we flip.
        We repeat until no flips are made in a full sweep or max_sweeps is reached.
        """
        current_score = compute_score(c)
        no_improvement_count = 0
        
        for sweep in range(max_sweeps):
            if time.time() > deadline - 2:
                break
                
            flips_made = 0
            # We can iterate in a fixed order. Randomizing the order per sweep might help escape cycles, 
            # but for deterministic greedy on a binary quadratic-ish problem, fixed order is often fine.
            # Let's use a fixed order but maybe randomize the initial order per restart?
            # For now, fixed order 0..N-1.
            
            for i in range(N):
                data = neighbor_data[i]
                if data is None:
                    continue
                
                o1s, o2s, o3s = data
                vals_o1 = c[o1s]
                vals_o2 = c[o2s]
                vals_o3 = c[o3s]
                
                others_all_equal = (vals_o1 == vals_o2) & (vals_o2 == vals_o3)
                
                val_i = c[i]
                new_val_i = 1 - val_i
                
                # Mono before: others_all_equal AND val_i == vals_o1
                is_mono_before = others_all_equal & (val_i == vals_o1)
                
                # Mono after: others_all_equal AND new_val_i == vals_o1
                is_mono_after = others_all_equal & (new_val_i == vals_o1)
                
                delta = np.sum(is_mono_after.astype(np.int32) - is_mono_before.astype(np.int32))
                
                if delta < 0:
                    c[i] = new_val_i
                    current_score += delta
                    flips_made += 1
            
            if flips_made == 0:
                break
                
        return current_score, c

    best_score = np.inf
    best_coloring = None
    
    rng = np.random.default_rng()
    
    # Multi-start: Try many random initializations and run greedy local search on each.
    # Since greedy is fast, we can afford many restarts.
    
    num_restarts = 1000 # We will break early if time runs out
    
    for r in range(num_restarts):
        if time.time() > deadline - 5:
            break
            
        # Initialize coloring
        # Random start
        c = rng.integers(0, 2, size=N, dtype=np.int8)
        
        # Calculate initial score
        current_score = compute_score(c)
        
        if current_score < best_score:
            best_score = current_score
            best_coloring = c.copy()
            
        # Run greedy local search
        # We can pass the current score to avoid recomputation if we modify the function, 
        # but compute_score is cheap enough.
        final_score, c_final = greedy_local_search(c)
        
        if final_score < best_score:
            best_score = final_score
            best_coloring = c_final.copy()
            
        # Optional: If we hit a very low score, we might reduce restarts?
        # No, keep going to find better.
        
    # Final check
    if best_coloring is None:
        # Fallback
        return [0] * N
        
    return best_coloring.tolist()
# EVOLVE-BLOCK-END
