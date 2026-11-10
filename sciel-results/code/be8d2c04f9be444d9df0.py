import numpy as np
import time
import random

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 50  # Safety margin before 90s limit
    
    # Precompute all APs and their index lists
    # AP: (start, step, length) where start in [0, n), step in [1, n), 
    # and start + (length-1)*step < n
    # We store each AP as a list of indices
    ap_lists = []
    for step in range(1, n):
        for start in range(step):
            seq = list(range(start, n, step))
            if len(seq) >= 2:
                ap_lists.append(seq)
    
    m = len(ap_lists)
    
    # Build incidence: for each index, which APs contain it
    index_to_aps = [[] for _ in range(n)]
    for ap_id, seq in enumerate(ap_lists):
        for idx in seq:
            index_to_aps[idx].append(ap_id)
    
    # Convert to numpy arrays for speed
    ap_sums = np.zeros(m, dtype=np.int32)
    
    # Initialize with Legendre symbol mod 257
    p = 257
    signs = np.zeros(n, dtype=np.int8)
    for i in range(n):
        val = (i + 1) % p
        if val == 0:
            signs[i] = 1
        else:
            ls = pow(val, (p - 1) // 2, p)
            signs[i] = 1 if ls == 1 else -1
    
    # Compute initial AP sums
    for ap_id, seq in enumerate(ap_lists):
        s = 0
        for idx in seq:
            s += signs[idx]
        ap_sums[ap_id] = s
    
    current_max = np.max(np.abs(ap_sums))
    best_signs = signs.copy()
    best_max = current_max
    
    # Precompute for each index the list of AP ids it belongs to as numpy array
    index_to_aps_np = [np.array(aps, dtype=np.int32) for aps in index_to_aps]
    
    # Also precompute the APs sorted by their absolute sum for targeted search
    # We'll recompute worst APs periodically
    
    def get_worst_aps(k=20):
        """Return indices of the k worst APs (by absolute sum)."""
        abs_sums = np.abs(ap_sums)
        # Get top k
        if k >= m:
            return np.argsort(-abs_sums)
        top_k = np.argpartition(-abs_sums, k)[:k]
        return top_k
    
    def flip_index(i):
        """Flip sign at index i and update ap_sums. Returns old sign."""
        old_sign = signs[i]
        new_sign = -old_sign
        signs[i] = new_sign
        delta = 2 * new_sign
        for ap_id in index_to_aps_np[i]:
            ap_sums[ap_id] += delta
        return old_sign
    
    # Targeted local search: focus on worst APs
    no_improve = 0
    max_no_improve = 15
    temperature = 3.0
    cooling = 0.93
    
    while time.time() < deadline - 3:
        # Get worst APs
        worst = get_worst_aps(30)
        
        # Collect indices that appear in worst APs
        candidate_indices = set()
        for ap_id in worst:
            for idx in ap_lists[ap_id]:
                candidate_indices.add(idx)
        
        candidates = list(candidate_indices)
        if not candidates:
            candidates = list(range(n))
        random.shuffle(candidates)
        
        improved = False
        
        for i in candidates:
            if time.time() > deadline - 3:
                break
            
            old_sign = signs[i]
            new_sign = -old_sign
            delta = 2 * new_sign
            
            # Tentatively flip
            for ap_id in index_to_aps_np[i]:
                ap_sums[ap_id] += delta
            signs[i] = new_sign
            
            new_max = np.max(np.abs(ap_sums))
            
            if new_max < current_max:
                current_max = new_max
                improved = True
            elif new_max == current_max:
                # Neutral move, accept with some probability to explore
                if random.random() < 0.3:
                    improved = True
                else:
                    # Revert
                    for ap_id in index_to_aps_np[i]:
                        ap_sums[ap_id] -= delta
                    signs[i] = old_sign
            else:
                # Worse move
                delta_max = new_max - current_max
                if random.random() < np.exp(-delta_max / temperature):
                    current_max = new_max
                    improved = True
                else:
                    # Revert
                    for ap_id in index_to_aps_np[i]:
                        ap_sums[ap_id] -= delta
                    signs[i] = old_sign
        
        if current_max < best_max:
            best_max = current_max
            best_signs = signs.copy()
            no_improve = 0
        elif improved:
            no_improve = 0
        else:
            no_improve += 1
            temperature *= cooling
            
            if no_improve > max_no_improve:
                if best_max > 3:
                    # Restart from best with perturbation
                    signs = best_signs.copy()
                    num_perturb = random.randint(5, 12)
                    for _ in range(num_perturb):
                        idx = random.randint(0, n - 1)
                        signs[idx] = -signs[idx]
                    # Recompute ap_sums
                    ap_sums[:] = 0
                    for ap_id, seq in enumerate(ap_lists):
                        s = 0
                        for idx in seq:
                            s += signs[idx]
                        ap_sums[ap_id] = s
                    current_max = np.max(np.abs(ap_sums))
                    temperature = 2.5
                    no_improve = 0
                else:
                    break
    
    return best_signs.tolist()
