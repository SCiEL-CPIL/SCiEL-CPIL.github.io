import time
import random
import math
import numpy as np

# EVOLVE-BLOCK-START
_START = time.time()
_DEADLINE = _START + 115.0  # Safety margin vs 150 s kill

def _score_from_set(U):
    """
    Compute the score for a set U.
    Score = 1 + log(|U-U|/|U+U|) / log(2*max(U) + 1)
    """
    if len(U) < 2:
        return 0.0
    max_val = max(U)
    if max_val < 1:
        return 0.0
    
    # Use FFT for efficient computation
    # We need a vector long enough to hold the convolution result without wrap-around.
    # Convolution of U with U has length 2*max(U) + 1.
    # To be safe and efficient, we pad to the next power of 2 >= 2*max(U) + 1.
    
    L = max_val + 1
    fft_len = 1
    while fft_len < 2 * L - 1:
        fft_len *= 2
        
    vec = np.zeros(fft_len, dtype=np.float64)
    for u in U:
        vec[u] = 1.0
        
    f = np.fft.fft(vec)
    
    # Sumset |U+U|: Non-zero coefficients in convolution IFFT(f^2)
    conv = np.fft.ifft(f * f).real
    # Threshold to handle numerical noise
    sums_count = np.sum(np.abs(conv) > 1e-9)
    
    # Differenceset |U-U|: Non-zero coefficients in cross-correlation IFFT(conj(f)*f)
    # Note: The correlation result is centered. The indices in the IFFT output correspond to 
    # shifts. For a set of non-negative integers, the differences range from -max to max.
    # The IFFT of conj(f)*f gives the circular cross-correlation.
    # Since we padded sufficiently, there is no wrap-around.
    # The output length is fft_len. The "zero" difference is at index 0.
    # Positive differences are at indices 1..max_val.
    # Negative differences are at indices fft_len - 1 .. fft_len - max_val.
    # All non-zero coefficients in the entire array correspond to valid differences 
    # because the support of the true correlation is within [-max, max] and we padded 
    # to 2*max+1 (or more), so the circular correlation equals the linear correlation.
    corr = np.fft.ifft(np.conj(f) * f).real
    diffs_count = np.sum(np.abs(corr) > 1e-9)
    
    if sums_count == 0:
        return 0.0
        
    ratio = diffs_count / float(sums_count)
    if ratio <= 0:
        return 0.0
        
    return 1.0 + math.log(ratio) / math.log(2 * max_val + 1)

def _is_sidon(U):
    """
    Check if U is a Sidon set (B2 sequence).
    A set is Sidon if all sums a+b (with a<=b) are distinct.
    Equivalently, all differences a-b (with a!=b) are distinct in absolute value? 
    No, Sidon means sums are distinct. 
    For a set of integers, it is Sidon if and only if the only solution to a+b=c+d 
    with a,b,c,d in U is {a,b}={c,d}.
    This is equivalent to: all pairwise differences are distinct in absolute value?
    Actually, if a-b = c-d, then a+d = c+b. 
    If all differences a-b (a>b) are distinct, then it's a Golomb Ruler.
    A Golomb Ruler is a stronger condition than a Sidon set?
    Let's check:
    If U is a Golomb Ruler, all differences are unique.
    If a+b = c+d, then a-d = c-b. If a!=d and c!=b, this implies two differences are equal.
    So a Golomb Ruler is a Sidon set.
    We will construct a Golomb Ruler (distinct differences) as it maximizes |U-U|.
    """
    diffs = set()
    n = len(U)
    for i in range(n):
        for j in range(i):
            d = U[i] - U[j]
            if d in diffs:
                return False
            diffs.add(d)
    return True

def _greedy_golomb(num_elements):
    """
    Greedy construction of a Golomb Ruler.
    Starts with 0, 1.
    Adds the smallest integer that keeps the set a Golomb Ruler.
    """
    U = [0, 1]
    diffs = {1}
    current_max = 1
    
    for _ in range(num_elements - 2):
        candidate = current_max + 1
        while True:
            # Check if adding candidate maintains Golomb property
            valid = True
            for u in U:
                d = candidate - u
                if d in diffs:
                    valid = False
                    break
            if valid:
                # Add it
                U.append(candidate)
                for u in U[:-1]:
                    diffs.add(candidate - u)
                current_max = candidate
                break
            else:
                candidate += 1
                
    return U

def _optimize_golomb(U, time_limit):
    """
    Local search to minimize max(U) while maintaining Golomb property.
    This reduces the denominator log(2*max(U)+1).
    """
    best_score = _score_from_set(U)
    best_U = U[:]
    current_U = U[:]
    
    # Calculate current diffs
    def get_diffs(S):
        d = set()
        for i in range(len(S)):
            for j in range(i):
                d.add(S[i] - S[j])
        return d
        
    current_diffs = get_diffs(current_U)
    
    # Simple local search: try to reduce the largest element
    # Or swap elements to pack them tighter.
    # Since it's a Golomb ruler, packing is hard. 
    # We can try to remove the largest element and see if the score improves? 
    # No, we want to keep the size. 
    # Let's try to shift elements down if possible.
    
    # A more effective move: try to replace the largest element with a smaller one
    # that doesn't conflict, or shift the whole set? No, 0 is fixed.
    
    # Let's do a random perturbation: pick a random element (not 0), try to move it to a 
    # smaller random value, check if Golomb property holds.
    
    random.seed(42)
    start_time = time.time()
    
    while time.time() < start_time + time_limit:
        if len(current_U) < 2:
            break
            
        # Pick a random index > 0
        idx = random.randint(1, len(current_U) - 1)
        old_val = current_U[idx]
        
        # Try to move to a smaller value
        # The new value must be > 0 and not in U
        # And must not create duplicate differences.
        
        # Strategy: Try values from old_val - 1 down to 1, or random small values.
        # To be efficient, we can check a few random candidates below old_val.
        found_better = False
        
        # Limit search space for the new value
        # The new value must be distinct from all other elements.
        # And all differences with other elements must be new.
        
        # Let's try a few random candidates in [1, old_val)
        for _ in range(10):
            new_val = random.randint(1, max(1, old_val - 1))
            if new_val in current_U:
                continue
                
            # Check if we can replace old_val with new_val
            # Remove old_val from U
            test_U = current_U[:]
            test_U.remove(old_val)
            test_U.append(new_val)
            test_U.sort()
            
            # Check Golomb property
            # This is O(n^2) which is fine for small n.
            # But we can do it faster by checking differences involving new_val.
            
            # Differences involving new_val:
            # new_val - u for u in test_U where u < new_val
            # u - new_val for u in test_U where u > new_val
            
            # We need to ensure these are not in the set of differences of test_U \ {new_val}
            # And that they are distinct from each other (which they are if new_val is distinct).
            
            # Actually, it's easier to just check the full set for small n.
            # But let's optimize:
            # Precompute diffs of test_U without new_val?
            # That's expensive.
            
            # Let's just do the O(n^2) check. n is likely < 100.
            is_golomb = True
            diffs_set = set()
            for i in range(len(test_U)):
                for j in range(i):
                    d = test_U[i] - test_U[j]
                    if d in diffs_set:
                        is_golomb = False
                        break
                    diffs_set.add(d)
                if not is_golomb:
                    break
            
            if is_golomb:
                # Check if score improves
                new_score = _score_from_set(test_U)
                if new_score > best_score:
                    best_score = new_score
                    best_U = test_U[:]
                    current_U = test_U[:]
                    found_better = True
                    break
                elif new_score > _score_from_set(current_U) - 0.01:
                    # Accept slightly worse to escape local minima? 
                    # Let's stick to strict improvement for simplicity and safety, 
                    # or use simulated annealing logic.
                    # For now, strict improvement on max_val reduction is the goal.
                    # If score doesn't improve, we might still want to accept if max_val decreased?
                    # The score depends on max_val. If max_val decreases, denominator decreases, 
                    # score likely increases even if ratio changes slightly.
                    if max(test_U) < max(current_U):
                        current_U = test_U[:]
                        found_better = True
                        # Update best if score is actually better
                        cur_s = _score_from_set(current_U)
                        if cur_s > best_score:
                            best_score = cur_s
                            best_U = current_U[:]
                        break
                        
        if not found_better:
            # If we didn't find a move, try a random swap between two non-zero elements?
            # Or just break if we are stuck for a while?
            # Let's try a few random swaps.
            if len(current_U) > 3:
                i = random.randint(1, len(current_U) - 1)
                j = random.randint(1, len(current_U) - 1)
                if i != j:
                    test_U = current_U[:]
                    test_U[i], test_U[j] = test_U[j], test_U[i]
                    test_U.sort()
                    # Check Golomb
                    is_golomb = True
                    diffs_set = set()
                    for k in range(len(test_U)):
                        for l in range(k):
                            d = test_U[k] - test_U[l]
                            if d in diffs_set:
                                is_golomb = False
                                break
                            diffs_set.add(d)
                        if not is_golomb:
                            break
                    if is_golomb:
                        new_score = _score_from_set(test_U)
                        if new_score > best_score:
                            best_score = new_score
                            best_U = test_U[:]
                            current_U = test_U[:]
                            
    return best_U

def construct():
    global _DEADLINE
    
    best_score = 0.0
    best_set = [0, 1]
    
    # Try different sizes of Golomb Rulers
    # The score is 1 + log(|U-U|/|U+U|) / log(2M+1)
    # For a Golomb Ruler of size n, |U-U| = n(n-1) + 1 (including 0).
    # |U+U| is at most n(n+1)/2 (if sums are distinct, which is true for Sidon/Golomb).
    # So ratio is approx n(n-1) / (n^2/2) = 2.
    # log(2) is small.
    # We want to maximize log(2) / log(2M+1).
    # This suggests we want M to be as small as possible for a given n.
    # But M grows with n.
    # For small n, M is small, denominator is small.
    # Let's check n=2: U={0,1}. |U-U|=3, |U+U|=3. Ratio=1. log(1)=0. Score=1.
    # n=3: U={0,1,3}. |U-U|=7, |U+U|=6. Ratio=7/6. log(7/6)/log(7) > 0.
    # n=4: U={0,1,4,6}. |U-U|=13, |U+U|=10. Ratio=1.3. log(1.3)/log(13).
    # As n increases, M grows roughly as n^2 (for optimal Golomb rulers, M ~ n^2).
    # log(2M+1) ~ 2 log n.
    # Ratio stays ~ 2.
    # So score ~ 1 + log(2) / (2 log n).
    # This decreases as n increases!
    # So we want SMALL n?
    # Wait, the ratio |U-U|/|U+U| can be larger than 2 if sums collide?
    # No, for Sidon sets, sums are distinct.
    # For non-Sidon sets, sums collide, so |U+U| is smaller, ratio is larger.
    # But |U-U| might also be smaller if differences collide.
    # The "sweet spot" might be sets that are NOT Sidon, but have "more" collisions in sums than in differences.
    # However, the greedy Golomb construction is a strong baseline.
    # Let's test a few small n with the greedy Golomb, and then optimize.
    
    sizes_to_try = [3, 4, 5, 6, 7, 8, 10, 12, 15, 20]
    
    for n in sizes_to_try:
        if time.time() > _DEADLINE - 30.0:
            break
            
        # Generate greedy Golomb ruler
        U = _greedy_golomb(n)
        
        # Optimize it
        # Allocate time based on remaining budget
        remaining = _DEADLINE - time.time()
        time_per_config = max(1.0, remaining / len(sizes_to_try))
        
        U_opt = _optimize_golomb(U, time_per_config)
        
        score = _score_from_set(U_opt)
        if score > best_score:
            best_score = score
            best_set = U_opt[:]
            
    # If we have time left, try a random search on small sets to find non-Sidon structures
    if time.time() < _DEADLINE - 10.0:
        # Random search for small sets
        for n in [3, 4, 5, 6]:
            for _ in range(1000):
                if time.time() > _DEADLINE - 10.0:
                    break
                # Generate random set
                U = [0]
                for _ in range(n - 1):
                    U.append(random.randint(1, 50))
                U = sorted(set(U))
                if len(U) < n:
                    continue
                score = _score_from_set(U)
                if score > best_score:
                    best_score = score
                    best_set = U[:]

    # Final sanity check
    best_set = sorted(set(int(x) for x in best_set))
    if 0 not in best_set:
        best_set.append(0)
    if len(best_set) < 2:
        best_set = [0, 1]
        
    return best_set
# EVOLVE-BLOCK-END
