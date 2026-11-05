import numpy as np
import time
import random

def construct():
    N = 51975
    # Find divisors of 51975
    divisors = []
    for i in range(2, int(np.sqrt(N)) + 1):
        if N % i == 0:
            divisors.append(i)
            if i != N // i:
                divisors.append(N // i)
    divisors.append(N)
    divisors = sorted(list(set(divisors)))
    divisors = [d for d in divisors if d > 1]
    num_mods = len(divisors)
    div_to_idx = {d: i for i, d in enumerate(divisors)}

    # Precompute index arrays for each divisor
    # For divisor m, indices for residue r are r, r+m, r+2m, ...
    # We'll precompute the base index array for r=0 for each m
    base_indices = {}
    for m in divisors:
        base_indices[m] = np.arange(0, N, m)
    
    # Also precompute the number of elements for each m
    len_map = {m: len(base_indices[m]) for m in divisors}

    def compute_uncovered(counts):
        return int(np.sum(counts == 0))

    def evaluate_move(residues, counts, i, new_r):
        """Compute the delta in uncovered count if we change modulus i to residue new_r."""
        m = divisors[i]
        old_r = residues[i]
        if new_r == old_r:
            return 0
        
        # Get indices for old and new residues
        idx_old = base_indices[m] + old_r
        idx_new = base_indices[m] + new_r
        
        # Mask to ensure indices < N (though arange + r should be < N since r < m and arange goes up to N-m)
        # Actually, base_indices[m] goes from 0 to N-m with step m.
        # So base_indices[m] + r goes from r to N-m+r. Since r < m, N-m+r < N. So no mask needed.
        
        c_old = counts[idx_old]
        c_new = counts[idx_new]
        
        # Elements in idx_old that were covered only by this class (count == 1) will become uncovered
        inc = np.sum(c_old == 1)
        # Elements in idx_new that were uncovered (count == 0) will become covered
        dec = np.sum(c_new == 0)
        
        return int(inc - dec)

    def apply_move(residues, counts, i, new_r):
        m = divisors[i]
        old_r = residues[i]
        if new_r == old_r:
            return 0
        
        idx_old = base_indices[m] + old_r
        idx_new = base_indices[m] + new_r
        
        counts[idx_old] -= 1
        counts[idx_new] += 1
        residues[i] = new_r
        
        c_old_before = None # We already applied, so we need to calculate delta before applying or store it
        # It's better to calculate delta first, then apply.
        # Let's refactor: this function just applies.
        # The caller should compute delta first.
        # But for now, this works if we assume caller knows delta.
        # Actually, let's just return the delta computed inside or pass it in.
        # Better: compute delta in caller, pass it in, or just re-compute? No, too slow.
        # Let's change design: 
        # 1. compute_delta
        # 2. if good, apply
        pass

    def local_refine(residues, counts, time_limit):
        start_time = time.time()
        current_score = compute_uncovered(counts)
        best_score = current_score
        best_residues = residues.copy()
        
        # Precompute candidate lists for each modulus
        # For small m, check all residues. For large m, sample.
        candidates = {}
        for m in divisors:
            if m <= 20:
                candidates[m] = list(range(m))
            else:
                # Sample 20 random residues, plus 0, 1, m//2
                sample_size = min(20, m)
                sample = random.sample(range(m), sample_size)
                # Add some deterministic ones
                extra = [0, 1, m // 2, (m // 2) + 1]
                for e in extra:
                    if e not in sample and e < m:
                        sample.append(e)
                candidates[m] = sample
        
        improved = True
        while improved and time.time() - start_time < time_limit:
            improved = False
            best_delta = 0
            best_idx = -1
            best_new_r = -1
            
            # Randomize order of checking moduli to avoid cycles
            indices = list(range(num_mods))
            random.shuffle(indices)
            
            for i in indices:
                m = divisors[i]
                old_r = residues[i]
                for new_r in candidates[m]:
                    if new_r == old_r:
                        continue
                    delta = evaluate_move(residues, counts, i, new_r)
                    if delta < best_delta:
                        best_delta = delta
                        best_idx = i
                        best_new_r = new_r
            
            if best_idx != -1 and best_delta < 0:
                # Apply the move
                m = divisors[best_idx]
                old_r = residues[best_idx]
                new_r = best_new_r
                
                idx_old = base_indices[m] + old_r
                idx_new = base_indices[m] + new_r
                
                counts[idx_old] -= 1
                counts[idx_new] += 1
                residues[best_idx] = new_r
                
                current_score += best_delta
                if current_score < best_score:
                    best_score = current_score
                    best_residues = residues.copy()
                
                improved = True
        
        return best_residues, best_score

    def greedy_construct(seed):
        random.seed(seed)
        counts = np.zeros(N, dtype=np.int32)
        residues = [0] * num_mods
        used_moduli = set()
        
        available_moduli = list(divisors)
        random.shuffle(available_moduli)
        
        for m in available_moduli:
            best_gain = -1
            best_r = 0
            
            # Determine candidate residues
            if m <= 20:
                r_candidates = range(m)
            else:
                # For larger m, sampling is key.
                # The gain is the number of uncovered spots in the class.
                # We want to maximize this.
                sample_size = min(50, m)
                r_candidates = random.sample(range(m), sample_size)
                # Always include 0
                if 0 not in r_candidates:
                    r_candidates.append(0)
            
            for r in r_candidates:
                idxs = base_indices[m] + r
                gain = np.sum(counts[idxs] == 0)
                if gain > best_gain:
                    best_gain = gain
                    best_r = r
            
            residues[div_to_idx[m]] = best_r
            idxs = base_indices[m] + best_r
            counts[idxs] += 1
            used_moduli.add(m)
        
        return residues, counts

    # Main loop
    start_time = time.time()
    timeout = 115.0
    best_global_score = N
    best_global_residues = None
    
    restart_count = 0
    while time.time() - start_time < timeout:
        restart_count += 1
        time_left = timeout - (time.time() - start_time)
        if time_left < 5.0:
            break
            
        # Allocate time for refinement
        # Greedy is fast, so most time for refinement
        refine_time = min(10.0, time_left - 1.0)
        
        seed = restart_count
        residues, counts = greedy_construct(seed)
        
        refined_residues, score = local_refine(residues, counts, refine_time)
        
        if score < best_global_score:
            best_global_score = score
            best_global_residues = refined_residues.copy()
            
        if best_global_score == 0:
            break

    if best_global_residues is None:
        # Fallback
        best_global_residues = np.array([random.randint(0, d - 1) for d in divisors])

    classes = [[divisors[i], int(best_global_residues[i])] for i in range(num_mods)]
    return classes
