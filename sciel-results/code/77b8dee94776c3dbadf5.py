import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    MIN_SUM = 128
    MAX_SUM = 256
    NUM_SUMS = MAX_SUM - MIN_SUM + 1  # 129
    N = 129
    
    # Precompute FFT size and window
    # We need sums up to 256.
    # Convolution of length 129 signal with itself gives length 257.
    # We want to extract indices [128, 256] from the convolution result.
    
    def calc_mult_fft(A_set):
        """
        A_set: a set of integers in [0, 128].
        Returns an array of multiplicities for sums in [128, 256].
        Uses FFT for O(N log N) complexity.
        """
        # Create signal
        sig = np.zeros(N, dtype=np.float64)
        for a in A_set:
            sig[a] = 1.0
        
        # FFT size needs to be at least 2*N - 1 to avoid circular convolution artifacts
        # 2*129 - 1 = 257. Next power of 2 is 512.
        fft_size = 512 
        
        # Compute FFT
        sig_fft = np.fft.fft(sig, n=fft_size)
        
        # Convolution in frequency domain
        conv_fft = sig_fft * sig_fft
        
        # Inverse FFT
        conv = np.fft.ifft(conv_fft).real
        
        # Extract the relevant range [128, 256]
        # The index in conv corresponds to the sum.
        # We want sums s where 128 <= s <= 256.
        # Note: conv[s] gives the number of pairs (i, j) such that i+j=s.
        # Since A is a set, sig[i] is 1 if i in A.
        # The convolution counts ordered pairs (i, j).
        # The problem asks for unordered pairs {a, b} with a <= b.
        # For s = a + b:
        # If a != b, the ordered pairs are (a, b) and (b, a). So count is 2 * (unordered count).
        # If a == b, the ordered pair is (a, a). So count is 1 * (unordered count).
        # Wait, the problem says "Pair representations are unordered and permit equal summands."
        # "Minimize the largest representation multiplicity".
        # Usually, multiplicity means the number of distinct pairs {a,b} in A x A such that a+b=s.
        # Let's verify the definition.
        # "Every integer in [128,256] must be a sum of two elements of A."
        # "Minimize the largest representation multiplicity".
        # If A={1,2}, sum=3 is 1+2. One representation.
        # If A={1,2,3}, sum=4 is 1+3, 2+2. Two representations.
        # The FFT convolution gives ordered pairs.
        # Let M_ordered[s] be the count from FFT.
        # Let M_unordered[s] be the desired count.
        # M_ordered[s] = 2 * M_unordered[s] - (1 if s is even and s/2 in A else 0)
        # Because if a=b=s/2, the pair (a,a) is counted once in ordered, but contributes 1 to unordered.
        # Pairs with a!=b are counted twice in ordered, once in unordered.
        # So: M_unordered[s] = (M_ordered[s] + I(s/2 in A and s%2==0)) / 2
        
        # Let's extract M_ordered for s in [128, 256]
        M_ordered = conv[MIN_SUM:MAX_SUM+1].astype(np.int32)
        
        # Calculate M_unordered
        M_unordered = np.zeros(NUM_SUMS, dtype=np.int32)
        A_arr = np.zeros(N, dtype=np.bool_)
        for a in A_set:
            A_arr[a] = True
            
        for i in range(NUM_SUMS):
            s = MIN_SUM + i
            ordered_count = M_ordered[i]
            if s % 2 == 0:
                half = s // 2
                if 0 <= half < N and A_arr[half]:
                    # a=b case exists
                    M_unordered[i] = (ordered_count + 1) // 2
                else:
                    M_unordered[i] = ordered_count // 2
            else:
                M_unordered[i] = ordered_count // 2
                
        return M_unordered

    def calc_score(A_set):
        """
        Returns (gaps, peak_mult).
        Lower is better. (0, 1) is optimal.
        """
        M = calc_mult_fft(A_set)
        gaps = int(np.sum(M == 0))
        if gaps > 0:
            return (gaps, float('inf'))
        peak = int(np.max(M))
        return (0, peak)

    def get_contributors(A_set, target_sum):
        """
        Returns a list of elements a in A_set such that target_sum - a is also in A_set and a <= target_sum - a.
        Actually, we just need to know which elements are involved in high multiplicity sums.
        For a sum S, the pairs are (a, S-a).
        If we remove a, we reduce the multiplicity of S by 1 (if S-a is in A and S-a != a) or 1 (if S-a == a).
        Wait, if we remove a, the pair (a, S-a) is gone.
        If we swap a with b, the pair (a, S-a) is gone, and (b, S-b) is added (if S-b in A).
        """
        contributors = []
        for a in A_set:
            b = target_sum - a
            if b in A_set and a <= b:
                contributors.append(a)
        return contributors

    def local_search(A_set, deadline):
        """
        Local search using FFT-based evaluation.
        """
        current_score = calc_score(A_set)
        if current_score == (0, 1):
            return current_score, A_set
            
        improved = True
        max_iters = 10000 # High limit, time is the real constraint
        
        while improved and time.time() < deadline and max_iters > 0:
            improved = False
            max_iters -= 1
            
            # Identify bottleneck sums
            M = calc_mult_fft(A_set)
            max_mult = int(np.max(M))
            if max_mult <= 1:
                break
                
            # Get sums with max multiplicity
            high_mult_indices = np.where(M == max_mult)[0]
            
            # Pick a random bottleneck sum to focus on
            target_idx = random.choice(high_mult_indices)
            target_sum = MIN_SUM + target_idx
            
            # Get contributors to this sum
            contributors = get_contributors(A_set, target_sum)
            
            if not contributors:
                continue
                
            # Try to modify the set to reduce multiplicity of target_sum
            # Strategy 1: Remove a contributor
            # Strategy 2: Swap a contributor with a random element
            
            A_list = list(A_set)
            available = [x for x in range(129) if x not in A_set]
            
            # Try removing one of the contributors
            random.shuffle(contributors)
            for a in contributors:
                if time.time() > deadline:
                    break
                new_A_set = set(A_set)
                new_A_set.remove(a)
                if not new_A_set:
                    continue
                new_score = calc_score(new_A_set)
                if new_score < current_score:
                    current_score = new_score
                    A_set = new_A_set
                    improved = True
                    break
                    
            if improved:
                continue
                
            # Try swapping a contributor with an available element
            if available:
                random.shuffle(contributors)
                for a in contributors:
                    if time.time() > deadline:
                        break
                    # Pick a few random candidates
                    candidates = random.sample(available, min(5, len(available)))
                    for b in candidates:
                        new_A_set = set(A_set)
                        new_A_set.remove(a)
                        new_A_set.add(b)
                        new_score = calc_score(new_A_set)
                        if new_score < current_score:
                            current_score = new_score
                            A_set = new_A_set
                            improved = True
                            break
                    if improved:
                        break
                        
        return current_score, A_set

    best_score = (float('inf'), float('inf'))
    best_A_set = None
    
    # Phase 1: Multi-restart local search
    # Start with random sets of various sizes
    # Smaller sets are harder to cover but have lower peak if they do cover.
    # Larger sets are easier to cover but have higher peak.
    # We want to find the sweet spot.
    
    # Let's try sizes from 20 to 60
    for k in range(20, 61, 2):
        if time.time() > deadline - 5: # Stop if close to deadline
            break
            
        n_restarts = 5
        for restart in range(n_restarts):
            if time.time() > deadline:
                break
                
            # Random initialization
            A_set = set(random.sample(range(129), k))
            
            # Local search
            current_score, A_set = local_search(A_set, deadline)
            
            if current_score < best_score:
                best_score = current_score
                best_A_set = A_set
                
            if best_score == (0, 1):
                return sorted(best_A_set)
                
    # Phase 2: If we found a valid solution, try to shrink it
    if best_A_set is not None and best_score[0] == 0:
        # Try to remove elements while maintaining validity
        # This is a "greedy removal" phase
        A_set = set(best_A_set)
        current_score = best_score
        improved = True
        while improved and time.time() < deadline:
            improved = False
            A_list = list(A_set)
            random.shuffle(A_list)
            for a in A_list:
                if time.time() > deadline:
                    break
                if len(A_set) <= 1:
                    break
                new_A_set = set(A_set)
                new_A_set.remove(a)
                new_score = calc_score(new_A_set)
                if new_score <= current_score: # Keep if score doesn't increase (prefer smaller set? No, score is peak multiplicity)
                    # Actually, we want to minimize peak multiplicity.
                    # Removing an element might increase peak multiplicity if it was the only one covering some sum? No, it removes a pair, so it decreases multiplicity of sums involving a.
                    # Wait, if we remove a, we lose pairs (a, b). This decreases multiplicity of a+b.
                    # It might create gaps if a+b was the only representation.
                    # calc_score checks for gaps.
                    # If new_score is valid (gaps=0) and peak is lower or equal, we take it.
                    # If peak is lower, great. If peak is equal, we have a smaller set, which might be better for future removals?
                    # The objective is just peak multiplicity.
                    # However, smaller sets are generally "tighter" and might be easier to optimize?
                    # Let's only accept if peak strictly decreases, or if peak is same but set is smaller (heuristic).
                    if new_score[0] == 0:
                        if new_score[1] < current_score[1]:
                            current_score = new_score
                            A_set = new_A_set
                            improved = True
                            break
                        elif new_score[1] == current_score[1] and len(new_A_set) < len(A_set):
                            # Accept to potentially enable further reductions
                            current_score = new_score
                            A_set = new_A_set
                            improved = True
                            break
                            
        if current_score < best_score:
            best_score = current_score
            best_A_set = A_set

    if best_A_set is None:
        # Fallback: A set that is guaranteed to cover, though peak might be high
        # [0, 128] covers everything. Peak is high.
        # Let's return a random valid set if we found one, otherwise a default.
        # A simple valid set: [64, 128]. Sums: 128..256.
        # 64+64=128, 64+65=129... 64+128=192, 65+128=193... 128+128=256.
        # Does it cover all?
        # Sums of [64, 128] are [128, 256].
        # Yes.
        # Peak multiplicity?
        # For s in [128, 192], pairs are (64, s-64) ... (s-64, 64) is same.
        # Number of pairs for s:
        # If s=128: (64,64) -> 1
        # If s=129: (64,65) -> 1
        # ...
        # If s=192: (64,128) -> 1
        # If s=193: (65,128) -> 1
        # ...
        # If s=256: (128,128) -> 1
        # Wait, is the multiplicity always 1?
        # Let's check s=160. Pairs: (64,96), (65,95)... (96,64) is same.
        # Since A is a contiguous interval [64, 128], for any sum s, the number of pairs is determined by the overlap of [64, 128] and [s-128, s-64].
        # The length of the intersection is min(128, s-64) - max(64, s-128) + 1.
        # For s=128: min(128, 64) - max(64, 0) + 1 = 64 - 64 + 1 = 1.
        # For s=192: min(128, 128) - max(64, 64) + 1 = 128 - 64 + 1 = 65.
        # So peak is 65.
        # This is a valid fallback.
        return list(range(64, 129))
        
    return sorted(best_A_set)

# Run construct to test
if __name__ == "__main__":
    start = time.time()
    A = construct()
    end = time.time()
    print(f"Time: {end-start:.2f}s")
    print(f"Set size: {len(A)}")
    # Verify
    A_set = set(A)
    M = calc_mult_fft(A_set)
    print(f"Min mult: {np.min(M)}, Max mult: {np.max(M)}")
    print(f"Gaps: {np.sum(M==0)}")
    print(f"Score: {calc_score(A_set)}")
