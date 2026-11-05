import numpy as np
import time

def _precompute_progression_membership(n):
    """For each position, store list of (progression_id, index_in_progression).
    Also store all progressions as (step, start) pairs and their lengths."""
    # Enumerate all arithmetic progressions [1..n] with any start, positive step, length >= 1
    # A progression is identified by (start, step, length)
    # For efficiency, we group by (step, start) and store the full sequence
    
    # prog_list[k] = list of positions in progression k (0-indexed)
    # pos_to_progs[i] = list of (prog_id, idx_in_prog) for position i
    
    prog_list = []
    pos_to_progs = [[] for _ in range(n)]
    
    prog_id = 0
    for step in range(1, n + 1):
        for start in range(step):
            # positions: start, start+step, start+2*step, ... < n
            positions = list(range(start, n, step))
            if len(positions) == 0:
                continue
            prog_list.append(positions)
            for idx, pos in enumerate(positions):
                pos_to_progs[pos].append((prog_id, idx))
            prog_id += 1
    
    return prog_list, pos_to_progs

def _compute_max_abs_ap_sum(signs):
    """Compute max absolute prefix sum over all arithmetic progressions."""
    n = len(signs)
    max_abs = 0
    
    for step in range(1, n):
        max_len = (n - 1) // step + 1
        if max_len <= max_abs:
            break
        
        for r in range(step):
            subseq = signs[r::step].astype(np.int32)
            if len(subseq) == 0:
                continue
            prefix_sums = np.cumsum(subseq)
            current_max = np.max(np.abs(prefix_sums))
            if current_max > max_abs:
                max_abs = int(current_max)
                
    return max_abs

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 50  # 50s deadline, 10s margin
    
    rng = np.random.RandomState(42)
    
    # Precompute progression membership
    prog_list, pos_to_progs = _precompute_progression_membership(n)
    num_progs = len(prog_list)
    
    # For each progression, store:
    # - current prefix sums (array)
    # - current max abs prefix sum
    
    best_signs = None
    best_score = float('inf')
    
    num_restarts = 5
    
    for restart in range(num_restarts):
        if time.time() > deadline - 5:
            break
            
        if restart == 0:
            signs = np.array([1 if i % 2 == 0 else -1 for i in range(n)], dtype=np.int8)
        elif restart == 1:
            signs = np.array([1 if bin(i).count('1') % 2 == 0 else -1 for i in range(n)], dtype=np.int8)
        else:
            signs = rng.choice([-1, 1], size=n).astype(np.int8)
        
        # Compute initial prefix sums and max abs for each progression
        prog_prefix_sums = []
        prog_max_abs = []
        for positions in prog_list:
            subseq = signs[positions].astype(np.int32)
            ps = np.cumsum(subseq)
            prog_prefix_sums.append(ps)
            prog_max_abs.append(int(np.max(np.abs(ps))))
        
        cur_score = max(prog_max_abs)
        
        if cur_score < best_score:
            best_score = cur_score
            best_signs = signs.copy()
        
        improved = True
        passes = 0
        max_passes = 50  # Allow more passes since each flip is cheaper
        
        while improved and passes < max_passes:
            improved = False
            passes += 1
            idx_order = rng.permutation(n)
            
            for i in idx_order:
                if time.time() > deadline - 2:
                    break
                
                old_val = signs[i]
                new_val = -old_val
                delta = new_val - old_val  # +/- 2
                
                # Find all progressions containing position i
                affected = pos_to_progs[i]
                
                if not affected:
                    continue
                
                # For each affected progression, update prefix sums from the index onward
                # and track if the max abs changes
                old_maxes = {}
                new_maxes = {}
                
                for prog_id, idx_in_prog in affected:
                    old_ps = prog_prefix_sums[prog_id]
                    # Update prefix sums from idx_in_prog to end
                    old_ps[idx_in_prog:] += delta
                    new_ps = old_ps
                    new_max = int(np.max(np.abs(new_ps)))
                    old_maxes[prog_id] = prog_max_abs[prog_id]
                    new_maxes[prog_id] = new_max
                    prog_max_abs[prog_id] = new_max
                
                # Compute new overall max
                # We need to check if any progression's max increased beyond cur_score
                # or if the previous worst progression's max decreased
                new_overall_max = 0
                for pid in new_maxes:
                    if new_maxes[pid] > new_overall_max:
                        new_overall_max = new_maxes[pid]
                
                # Also need to consider progressions not in new_maxes that might have high max
                # This is expensive to check fully, so we use an approximation:
                # If the old worst progression is not affected, its max is unchanged
                # We track the global max separately
                
                # Simpler approach: just compute the max over all affected + keep a global counter
                # Actually, let's just recompute the global max from all prog_max_abs
                # But that's O(num_progs) which is too slow per flip.
                
                # Better: maintain a count of how many progressions have max == cur_score
                # For now, use a heuristic: if new_overall_max < cur_score, accept
                # Otherwise, do a full check occasionally
                
                # For correctness, we need the true max. Let's use a bucket approach.
                # Store prog_max_abs in a way that allows fast max query.
                # Since max is at most ~sqrt(n) or so, we can use a frequency array.
                
                # Actually, let's just do: if any affected prog has max > cur_score, reject
                # If the worst prog is affected and its max dropped, we need to find new worst
                # This is getting complex. Let's use a simpler but correct approach:
                # Maintain a list of (max_val, prog_id) sorted, or just track the max.
                
                # Given the complexity, let's just recompute global max when needed.
                # num_progs is about sum_{step=1}^{n} step = n(n+1)/2 ~ 32896 for n=256
                # That's too many to check per flip.
                
                # Alternative: only accept if new_overall_max <= cur_score - 1 (strict improvement in affected)
                # And periodically do a full check.
                
                # Let's use a different strategy: keep track of the top few worst progressions.
                # If none of the top 3 are affected, the global max is unchanged.
                
                # For simplicity and correctness, let's limit the affected check:
                # If max(new_maxes.values()) < cur_score: accept
                # Else if any old_maxes[pid] == cur_score and new_maxes[pid] < cur_score: need to find new max
                # Else: reject (since some affected prog still has max >= cur_score)
                
                accept = False
                if max(new_maxes.values()) < cur_score:
                    accept = True
                else:
                    # Check if the previous worst is improved
                    prev_worst_improved = False
                    for pid in new_maxes:
                        if old_maxes[pid] == cur_score and new_maxes[pid] < cur_score:
                            prev_worst_improved = True
                            break
                    
                    if prev_worst_improved:
                        # Need to find new global max - expensive
                        # Do a full scan but only if this happens rarely
                        full_max = max(prog_max_abs)
                        if full_max < cur_score:
                            accept = True
                            cur_score = full_max
                        else:
                            # Roll back
                            for prog_id, idx_in_prog in affected:
                                ps = prog_prefix_sums[prog_id]
                                ps[idx_in_prog:] -= delta
                                prog_max_abs[prog_id] = old_maxes[prog_id]
                    else:
                        # Roll back
                        for prog_id, idx_in_prog in affected:
                            ps = prog_prefix_sums[prog_id]
                            ps[idx_in_prog:] -= delta
                            prog_max_abs[prog_id] = old_maxes[prog_id]
                
                if accept:
                    signs[i] = new_val
                    if cur_score < best_score:
                        best_score = cur_score
                        best_signs = signs.copy()
                    improved = True
                else:
                    # Already rolled back in the else branch above
                    pass
            
            if time.time() > deadline - 2:
                break
                
    if best_signs is None:
        best_signs = np.ones(n, dtype=np.int8)
        
    return [int(x) for x in best_signs]
