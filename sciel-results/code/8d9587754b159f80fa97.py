import time
import numpy as np
import random

def _score_set(U):
    """Compute the lower bound 1 + log(|U-U|/|U+U|)/log(2 max U + 1)."""
    if not U or len(U) < 2:
        return 0.0
    
    # Ensure unique, sorted, non-negative integers including 0
    U_arr = np.array(U, dtype=np.int64)
    U_arr = U_arr[U_arr >= 0]
    if 0 not in U_arr:
        U_arr = np.concatenate(([0], U_arr))
    U_arr = np.unique(U_arr)
    
    n = len(U_arr)
    if n < 2:
        return 0.0
        
    mx = int(U_arr.max())
    if mx == 0:
        return 0.0
    
    N = 2 * mx + 1
    
    # Efficient counting via FFT
    sig = np.zeros(N, dtype=np.float64)
    sig[U_arr] = 1.0
    
    fft_size = 1
    while fft_size < 2 * N:
        fft_size *= 2
        
    F = np.fft.rfft(sig, n=fft_size)
    
    # Sumset U+U
    S_sum = np.fft.irfft(F * F, n=fft_size)
    sumset_size = np.sum(S_sum[:N] > 1e-6)
    
    # Diffset U-U
    # rev_sig[i] = sig[N-1-i]
    flip_sig = np.zeros(N, dtype=np.float64)
    flip_sig[:] = sig[::-1]
    
    F_flip = np.fft.rfft(flip_sig, n=fft_size)
    S_diff = np.fft.irfft(F * F_flip, n=fft_size)
    
    diffset_size = np.sum(S_diff > 1e-6)
    
    if sumset_size == 0:
        return 0.0
        
    ratio = diffset_size / sumset_size
    if ratio <= 0:
        return 0.0
        
    denom = np.log(2 * mx + 1)
    if denom == 0:
        return 0.0
        
    return 1.0 + np.log(ratio) / denom

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Strategy:
    # 1. Generate multiple candidate sets using Greedy Mian-Chowla construction 
    #    with varying "density" or stopping criteria.
    # 2. Use Local Search (Swap) to refine the best candidate.
    #    Swap: Replace element x with y (y not in U) if it improves score.
    #    Since full swap search is expensive, we will try:
    #    a) Adding a new large element.
    #    b) Removing an element that causes many sum collisions.
    #    c) Swapping an element with a nearby candidate.
    
    best_score = 0.0
    best_set = [0, 1]
    
    def greedy_mian_chowla(max_elements, max_value=10**5):
        """
        Generates a Mian-Chowla sequence.
        U[0] = 0.
        U[i] is the smallest integer > U[i-1] such that all pairwise sums 
        U[j] + U[i] (for j < i) are distinct from all previous pairwise sums.
        """
        U = [0]
        used_sums = {0}
        
        while len(U) < max_elements:
            candidate = U[-1] + 1
            while True:
                valid = True
                for u in U:
                    s = candidate + u
                    if s in used_sums:
                        valid = False
                        break
                    if s > max_value:
                        valid = False
                        break
                if valid:
                    # Accept candidate
                    U.append(candidate)
                    for u in U[:-1]: # Add new sums involving candidate
                        used_sums.add(candidate + u)
                    break
                candidate += 1
                if candidate > max_value:
                    return U
        return U

    def local_search_improve(U, time_limit=5.0):
        """
        Attempts to improve the set U via local moves.
        Moves:
        1. Add a new element x (x > max(U)) if it improves score.
        2. Remove an element x (x > 0) if it improves score.
        3. Swap: Remove x, add y. (Approximated by trying adds and removes).
        
        To keep it fast, we evaluate score changes heuristically or use full score
        only on promising moves. However, FFT score is fast enough for small sets.
        For larger sets, we need to be careful.
        
        Let's restrict local search to sets with max(U) < 5000 to keep FFT fast.
        """
        if max(U) > 5000:
            return U, _score_set(U)
            
        U = sorted(list(set(U)))
        current_score = _score_set(U)
        
        # Precompute sumset and diffset indicators for quick checking?
        # No, just recompute score. It's O(N log N).
        
        steps = 0
        max_steps = 500
        start_ls = time.time()
        
        while steps < max_steps and time.time() - start_ls < time_limit:
            improved = False
            steps += 1
            
            # Move 1: Add a new element
            # Try candidates just above max(U) or random larger ones
            candidates_add = []
            mx = max(U)
            # Try a few nearby values
            for delta in range(1, 20):
                candidates_add.append(mx + delta)
            # Try some random larger values
            for _ in range(5):
                candidates_add.append(mx + random.randint(1, 100))
                
            for cand in candidates_add:
                if cand in U:
                    continue
                new_U = U + [cand]
                s = _score_set(new_U)
                if s > current_score + 1e-9:
                    U = new_U
                    current_score = s
                    improved = True
                    break
            
            # Move 2: Remove an element
            # Try removing elements that are "expensive" (large) or cause collisions
            # We'll just try removing the largest element, or a random non-zero element
            candidates_remove = []
            if len(U) > 2:
                # Try largest
                candidates_remove.append(max(U))
                # Try random
                for _ in range(3):
                    candidates_remove.append(random.choice(U[1:]))
            
            for cand in candidates_remove:
                if cand == 0:
                    continue
                new_U = [x for x in U if x != cand]
                if len(new_U) < 2:
                    continue
                s = _score_set(new_U)
                if s > current_score + 1e-9:
                    U = new_U
                    current_score = s
                    improved = True
                    break
                    
            if not improved:
                break
                
        return U, current_score

    # Phase 1: Generate diverse greedy seeds
    # Mian-Chowla is known to have good additive properties.
    # We will generate sequences of different lengths.
    
    seed_lengths = [10, 15, 20, 25, 30, 40, 50, 70, 100, 150, 200]
    
    candidates = []
    
    for L in seed_lengths:
        if time.time() > deadline:
            break
        try:
            seq = greedy_mian_chowla(L, max_value=10**6)
            if len(seq) > 1:
                candidates.append(seq)
        except Exception:
            pass

    # Also try a "Prime Step" greedy variant mentioned in inspirations
    # U = {0, p1, p1+p2, ...} where pi are primes?
    # Or just standard greedy but with a bias towards primes?
    # Let's stick to standard Mian-Chowla first as it's robust.
    
    # Phase 2: Local Search on the best candidates
    # We should prioritize candidates with higher initial scores or potential.
    
    scored_candidates = []
    for seq in candidates:
        s = _score_set(seq)
        scored_candidates.append((s, seq))
        
    # Sort by score descending
    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    
    # Run local search on the top few candidates
    # Budget: ~10s per local search, run on top 5-10 candidates if time permits
    
    for s_init, seq in scored_candidates[:10]:
        if time.time() > deadline - 20.0: # Leave 20s buffer
            break
            
        # Only run local search if the set is small enough for FFT to be fast
        # If max(U) is huge, FFT is slow. 
        # Mian-Chowla grows fast. 
        # For L=100, max(U) can be ~10^4-10^5. FFT size ~ 2*10^5 is okay.
        # For L=200, max(U) might be ~10^6. FFT size 2*10^6 is ~1-2s.
        
        # Adjust time limit for local search based on size
        mx = max(seq)
        if mx > 200000:
            ls_time = 1.0
        elif mx > 50000:
            ls_time = 2.0
        else:
            ls_time = 5.0
            
        try:
            refined_seq, s_refined = local_search_improve(seq, time_limit=ls_time)
            if s_refined > best_score:
                best_score = s_refined
                best_set = refined_seq
        except Exception:
            pass
            
        # Update best if initial was better (in case local search failed or didn't improve)
        if s_init > best_score:
            best_score = s_init
            best_set = seq

    # Phase 3: If time remains, try a random restart with different greedy parameters
    # e.g., Greedy with a constraint on max difference?
    # Or just more random local searches on the best set found.
    
    if time.time() < deadline - 5.0:
        # Try to "mutate" the best set found
        # Randomly swap elements
        current_best = list(best_set)
        current_best_score = best_score
        
        # Simple random perturbation loop
        for _ in range(50):
            if time.time() > deadline - 5.0:
                break
            if len(current_best) < 2:
                break
            
            # Pick a random element to remove (not 0)
            if len(current_best) > 1:
                idx = random.randint(1, len(current_best)-1)
                removed = current_best.pop(idx)
                
                # Pick a random new element to add
                # Range: around max(current_best)
                mx = max(current_best) if current_best else 0
                new_val = mx + random.randint(-10, 20)
                if new_val < 0:
                    new_val = 1
                if new_val not in current_best:
                    current_best.append(new_val)
                    current_best.sort()
                    
                s = _score_set(current_best)
                if s > current_best_score:
                    current_best_score = s
                    best_score = s
                    best_set = list(current_best)
                else:
                    # Revert
                    current_best.remove(new_val)
                    current_best.insert(idx, removed)
                    current_best.sort()

    return best_set

if __name__ == "__main__":
    s = construct()
    print(f"Result: {s[:10]}... Size: {len(s)}, Score: {_score_set(s)}")
