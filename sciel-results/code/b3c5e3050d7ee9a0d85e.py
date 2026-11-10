import time
import random
import numpy as np

P = {'n': 12, 'm': 40}

def construct():
    start_time = time.time()
    deadline = start_time + 88  # 88s safety margin
    
    n = P['n']
    m = P['m']
    total = 1 << n
    
    # Build comparability matrix
    # C[i, j] = 1 if i subset j or j subset i, i != j
    masks = np.arange(total, dtype=np.uint32)
    # To save memory and time, we can build it in blocks or use broadcasting
    # 4096x4096 is 16M bytes for uint8, which is fine.
    
    # Efficient construction:
    # i subset j iff (i & j) == i
    # j subset i iff (i & j) == j
    # We can compute this for all pairs.
    
    # Broadcasting:
    # i_col = masks[:, None]
    # j_row = masks[None, :]
    # comp = ((i_col & j_row) == i_col) | ((i_col & j_row) == j_row)
    # This creates two 4096x4096 boolean arrays.
    # Memory: 2 * 16MB = 32MB. OK.
    
    i_col = masks[:, None]
    j_row = masks[None, :]
    
    # Compute subset relations
    # Note: i == j is a subset, but we want STRICT subset for comparability?
    # "one member is a strict subset of the other"
    # So C[i, i] should be 0.
    # (i & i) == i is True.
    # So we need to mask out the diagonal.
    
    # Let's compute A = (i & j) == i and B = (i & j) == j
    # C = A | B
    # C[diag] = 0
    
    # To avoid huge intermediate arrays if possible, but 32MB is fine.
    # However, creating boolean arrays might be slow.
    # Let's try to do it in chunks if needed, but 4096 is small enough.
    
    # Optimization:
    # We only need C for the update.
    # We can store C as uint8.
    
    # Constructing C
    # Using numpy operations
    # i_col & j_row
    inter = i_col & j_row
    # (inter == i_col) | (inter == j_row)
    # This creates two boolean arrays of size 16M.
    # Then OR them.
    # Then set diagonal to 0.
    # Then cast to uint8.
    
    # Let's do it.
    is_sub1 = (inter == i_col)
    is_sub2 = (inter == j_row)
    comp_bool = is_sub1 | is_sub2
    
    # Zero out diagonal
    idx = np.arange(total)
    comp_bool[idx, idx] = False
    
    C = comp_bool.astype(np.uint8)
    
    # Precompute neighbors? 
    # For vectorized update, we just use C[:, k].
    
    def evaluate(x):
        """x is a boolean array of length total."""
        # F = 0.5 * sum(x * (C @ x))
        # C @ x is vector of degrees
        # x * (C @ x) is sum of degrees for selected nodes
        # Since C is symmetric and C_ii=0, sum x_i d_i = 2 * F
        deg = C.dot(x)
        return int(np.dot(x, deg) / 2)
    
    def greedy_init():
        x = np.zeros(total, dtype=np.uint8)
        d = np.zeros(total, dtype=np.int32)
        # Greedy: pick node with max degree
        # Initially d is 0.
        # We need to update d as we add nodes.
        # This is O(m * total) if we recompute, or O(m * deg) if we update.
        # m=40, total=4096. 40*4096 = 160k. Very fast.
        
        for _ in range(m):
            # Pick argmax of d among unselected
            # Mask selected
            d_masked = d.copy()
            d_masked[x == 1] = -1
            best = np.argmax(d_masked)
            x[best] = 1
            # Update d
            d += C[:, best]
        return x
    
    def random_init():
        x = np.zeros(total, dtype=np.uint8)
        chosen = np.random.choice(total, size=m, replace=False)
        x[chosen] = 1
        return x
        
    def simulated_annealing(x0, t_max=100.0, t_min=0.01, alpha=0.9999):
        x = x0.copy()
        d = C.dot(x).astype(np.int32)
        current_score = evaluate(x)
        best_x = x.copy()
        best_score = current_score
        
        t = t_max
        no_improve = 0
        
        # Precompute indices for random selection?
        # We can just pick random indices and check if they are in/out.
        # Or maintain lists of in/out indices.
        # Maintaining lists is faster for selection.
        
        in_idx = np.where(x == 1)[0]
        out_idx = np.where(x == 0)[0]
        
        while t > t_min:
            if time.time() > deadline - 2:
                break
            
            # Pick random in and out
            if len(in_idx) == 0 or len(out_idx) == 0:
                break
                
            k_out = in_idx[np.random.randint(len(in_idx))]
            k_in = out_idx[np.random.randint(len(out_idx))]
            
            if k_in == k_out:
                continue
                
            delta = int(d[k_in]) - int(d[k_out])
            
            if delta >= 0 or random.random() < np.exp(delta / t):
                # Accept
                x[k_out] = 0
                x[k_in] = 1
                
                # Update d
                d -= C[:, k_out]
                d += C[:, k_in]
                
                # Update in/out indices
                # This is O(N) if we use lists, but N=4096.
                # We can just rebuild them occasionally or use boolean masks.
                # Rebuilding is 4096 ops.
                # If we do this every step, it's slow.
                # Let's just use boolean mask x for selection in the next step?
                # No, we need to pick from in/out.
                # We can maintain the lists by swapping.
                
                # Find k_out in in_idx and replace with k_in?
                # No, k_in was in out_idx.
                
                # Simple approach: rebuild in_idx and out_idx every K steps or if time permits.
                # Or just use x to filter?
                # in_idx = np.where(x==1)[0] is 4096 ops.
                # If we do 100k steps, 400M ops. Might be too slow in Python loop.
                
                # Alternative: Keep x as the source of truth.
                # Pick random k_in, k_out from 0..total-1.
                # If x[k_in]==1 or x[k_out]==0, reject.
                # Probability of rejection is high if m is small?
                # m=40, total=4096.
                # P(k_in not in S) = (4096-40)/4096 ~ 0.99.
                # P(k_out in S) = 40/4096 ~ 0.01.
                # P(valid pair) = 0.99 * 0.01 = 0.0099.
                # So we need ~100 tries per valid swap.
                # This is inefficient.
                
                # Better: Maintain lists.
                # To update lists efficiently:
                # in_idx is a list of ints.
                # Find index of k_out in in_idx.
                # Replace it with k_in.
                # Find index of k_in in out_idx.
                # Replace it with k_out.
                
                # Using numpy arrays for in_idx/out_idx makes replacement O(N).
                # Using Python lists makes replacement O(N) for search, but we can use a set/dict for index lookup?
                # Or just rebuild every 100 steps?
                
                # Let's try rebuilding every step first, if it's fast enough.
                # 4096 ops per step.
                # 100k steps -> 400M ops.
                # NumPy where is C-speed. 4096 is small.
                # Let's assume it's okay.
                
                in_idx = np.where(x == 1)[0]
                out_idx = np.where(x == 0)[0]
                
                current_score += delta
                if current_score > best_score:
                    best_score = current_score
                    best_x = x.copy()
                    no_improve = 0
                else:
                    no_improve += 1
                    
                if no_improve > 5000:
                    break
                    
            t *= alpha
            
        return best_x, best_score

    # Initializations
    inits = [
        greedy_init(),
        random_init(),
        random_init()
    ]
    
    best_x = None
    best_score = -1
    
    for x0 in inits:
        if time.time() > deadline - 20:
            break
        s = evaluate(x0)
        if s > best_score:
            best_score = s
            best_x = x0.copy()
            
    # SA from best
    if best_x is not None:
        x_best, s = simulated_annealing(best_x)
        if s > best_score:
            best_score = s
            best_x = x_best
            
    # More restarts
    while time.time() < deadline - 5:
        x0 = random_init()
        x_best, s = simulated_annealing(x0, t_max=50.0)
        if s > best_score:
            best_score = s
            best_x = x_best
            
    if best_x is None:
        best_x = greedy_init()
        
    selected = np.where(best_x == 1)[0]
    result = []
    for mask in selected:
        s = [b for b in range(n) if (mask >> b) & 1]
        result.append(s)
        
    return result

