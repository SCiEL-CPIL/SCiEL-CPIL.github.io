"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import random
import numpy as np

# EVOLVE-BLOCK-START

def _score_from_set(U):
    """Compute the score for a set U (list of nonnegative ints containing 0)."""
    if not U:
        return 0.0
    U_sorted = sorted(set(U))
    if 0 not in U_sorted:
        U_sorted.append(0)
        U_sorted = sorted(U_sorted)
    n = len(U_sorted)
    if n <= 1:
        return 0.0
    max_val = U_sorted[-1]
    if max_val == 0:
        return 0.0
    
    # Compute |U+U| and |U-U|
    # For efficiency with larger sets, use FFT-based convolution
    # But for small sets, direct computation is fine
    
    if n <= 200:
        # Direct computation
        sum_set = set()
        diff_set = set()
        for i in range(n):
            for j in range(n):
                sum_set.add(U_sorted[i] + U_sorted[j])
                diff_set.add(U_sorted[i] - U_sorted[j])
        num_sum = len(sum_set)
        num_diff = len(diff_set)
    else:
        # FFT-based approach
        # U+U: convolution of indicator with itself
        # U-U: convolution of indicator with reversed indicator
        max_val_plus = 2 * max_val
        M = 1
        while M < max_val_plus + 1:
            M *= 2
        indicator = np.zeros(M, dtype=np.float64)
        for v in U_sorted:
            indicator[v] = 1.0
        
        # Sum set: |U+U|
        conv_sum = np.fft.fft(indicator)
        conv_sum = conv_sum * conv_sum
        conv_sum = np.fft.ifft(conv_sum).real
        # Round to handle floating point
        conv_sum = np.round(conv_sum)
        num_sum = int(np.sum(conv_sum > 0.5))
        
        # Diff set: |U-U|
        # U-U = {a-b : a,b in U}
        # This is the convolution of indicator with its "reverse"
        # Let's use: diff indicator at position (a-b + max_val) for the range
        # Actually, |U-U| can be computed as: for each d in range [-max_val, max_val],
        # count if there exist a,b in U with a-b = d
        # Using FFT: create array where we put 1 at positions, and 1 at positions shifted by max_val
        # Then convolve... Actually simpler: 
        # |U-U| = number of distinct differences
        # We can compute this via: create array A of length M with A[x]=1 if x in U
        # Create array B of length M with B[max_val - x]=1 if x in U (for x in U)
        # Then convolution gives counts of a-b+max_val... 
        # Let me think more carefully.
        # a-b ranges from -max_val to max_val
        # Let's create C where C[x] = 1 if x in U (for x in [0, max_val])
        # And D where D[x] = 1 if (max_val - x) in U, i.e., D[max_val - u] = 1 for u in U
        # Then (C * D)[k] = sum_x C[x] D[k-x] = sum_x C[x] D[k-x]
        # D[k-x] = 1 iff max_val - (k-x) in U iff (max_val - k + x) in U
        # So (C*D)[k] = #{x in U : max_val - k + x in U} = #{x in U : x + (max_val-k) in U}
        # Let d = max_val - k, so k = max_val - d
        # (C*D)[max_val - d] = #{x in U : x + d in U}
        # This counts pairs (x, x+d) both in U, i.e., the autocorrelation at lag d
        # We want to know if there EXISTS a pair with difference d, not the count
        # So |U-U| = #{d in [-max_val, max_val] : autocorrelation at lag d > 0}
        
        indicator_diff = np.zeros(M, dtype=np.float64)
        for v in U_sorted:
            indicator_diff[max_val - v] = 1.0
        
        conv_diff = np.fft.fft(indicator)
        conv_diff2 = np.fft.fft(indicator_diff)
        conv_diff = conv_diff * conv_diff2
        conv_diff = np.fft.ifft(conv_diff).real
        conv_diff = np.round(conv_diff)
        # The difference d corresponds to index max_val - d in the convolution
        # d ranges from -max_val to max_val, so indices range from 0 to 2*max_val
        num_diff = int(np.sum(conv_diff[:2*max_val+1] > 0.5))
    
    if num_sum == 0 or num_diff == 0:
        return 0.0
    
    ratio = num_diff / num_sum
    if ratio <= 0:
        return 0.0
    
    log_ratio = math.log(ratio)
    log_denom = math.log(2 * max_val + 1)
    
    if log_denom <= 0:
        return 0.0
    
    score = 1.0 + log_ratio / log_denom
    return score


def _golomb_ruler(n):
    """Generate a Golomb ruler of length n (n marks including 0).
    Uses a simple greedy/random approach for small n.
    """
    if n <= 1:
        return [0]
    if n == 2:
        return [0, 1]
    
    # For small n, use known optimal Golomb rulers
    known = {
        3: [0, 1, 4],
        4: [0, 1, 4, 9],
        5: [0, 1, 4, 9, 15],
        6: [0, 1, 4, 9, 15, 22],
        7: [0, 1, 4, 9, 15, 22, 30],
        8: [0, 1, 4, 9, 15, 22, 30, 39],
        9: [0, 1, 4, 9, 15, 22, 30, 39, 49],
        10: [0, 1, 4, 9, 15, 22, 30, 39, 49, 60],
        11: [0, 1, 4, 9, 15, 22, 30, 39, 49, 60, 72],
        12: [0, 1, 4, 9, 15, 22, 30, 39, 49, 60, 72, 85],
        13: [0, 1, 4, 9, 15, 22, 30, 39, 49, 60, 72, 85, 99],
        14: [0, 1, 4, 9, 15, 22, 30, 39, 49, 60, 72, 85, 99, 114],
        15: [0, 1, 4, 9, 15, 22, 30, 39, 49, 60, 72, 85, 99, 114, 130],
    }
    if n in known:
        return known[n]
    
    # Greedy construction
    ruler = [0]
    for i in range(1, n):
        best_pos = None
        best_min_diff = -1
        # Try positions from current max + 1 upward
        start = ruler[-1] + 1
        # Limit search to avoid infinite loop
        for pos in range(start, start + 200):
            # Check all differences are distinct
            diffs = set()
            valid = True
            for j in range(len(ruler)):
                d = pos - ruler[j]
                if d in diffs:
                    valid = False
                    break
                diffs.add(d)
            # Also check differences among existing marks
            for j in range(len(ruler)):
                for k in range(j+1, len(ruler)):
                    d = ruler[k] - ruler[j]
                    if d in diffs:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                # This is a valid position; take the smallest
                best_pos = pos
                break
        
        if best_pos is None:
            # Fallback: just extend
            best_pos = ruler[-1] + max(1, int(1.5 ** i))
        
        ruler.append(best_pos)
    
    return ruler


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_score = 0.0
    best_set = [0, 1, 4]  # Golomb ruler of length 3
    
    # Try various set sizes and constructions
    # Strategy: Golomb rulers give good |U-U|/|U+U| ratios
    # Larger sets with Golomb property have |U-U| = n(n-1) (all differences distinct)
    # and |U+U| is roughly n(n+1)/2 (for "nice" sets)
    # So ratio ~ 2(n-1)/(n+1) which approaches 2 as n grows
    # But max U grows too, so log(2*max+1) also grows
    
    # Let's try Golomb rulers of various sizes
    for n in range(3, 50):
        if time.time() > deadline:
            break
        
        ruler = _golomb_ruler(n)
        score = _score_from_set(ruler)
        if score > best_score:
            best_score = score
            best_set = list(ruler)
    
    # Now do local search on the best set
    current_set = list(best_set)
    current_score = best_score
    
    # Local search: try moving elements
    random.seed(42)
    iterations = 0
    while time.time() < deadline:
        iterations += 1
        n = len(current_set)
        
        # Try modifying a random element
        idx = random.randint(0, n - 1)
        old_val = current_set[idx]
        
        # Try several perturbations
        found_improvement = False
        for _ in range(20):
            # Perturb the value
            if idx == 0:
                # Keep 0 fixed
                new_val = 0
            else:
                # Try random changes
                scale = max(1, int(old_val * 0.3))
                new_val = old_val + random.randint(-scale, scale)
                if new_val < 0:
                    new_val = 0
                if new_val == old_val:
                    continue
            
            candidate = current_set[:]
            candidate[idx] = new_val
            candidate = sorted(set(candidate))
            
            if 0 not in candidate:
                continue
            
            new_score = _score_from_set(candidate)
            if new_score > current_score + 1e-12:
                current_set = candidate
                current_score = new_score
                found_improvement = True
                break
        
        if not found_improvement:
            # Try adding a new element
            if random.random() < 0.3 and len(current_set) < 100:
                max_val = current_set[-1] if current_set else 0
                new_val = random.randint(1, max_val + max(1, int(max_val * 0.5)))
                candidate = sorted(set(current_set + [new_val]))
                if 0 in candidate:
                    new_score = _score_from_set(candidate)
                    if new_score > current_score + 1e-12:
                        current_set = candidate
                        current_score = new_score
        
        if current_score > best_score:
            best_score = current_score
            best_set = list(current_set)
    
    # Also try a few different initial constructions
    # Fibonacci-like sets
    for base in [2, 3, 4, 5]:
        if time.time() > deadline:
            break
        fib_set = [0]
        val = 1
        for i in range(20):
            fib_set.append(val)
            val *= base
            if val > 100000:
                break
        score = _score_from_set(fib_set)
        if score > best_score:
            best_score = score
            best_set = list(fib_set)
    
    # Power of 2 sets
    for max_exp in range(5, 20):
        if time.time() > deadline:
            break
        pow2_set = [2**i for i in range(max_exp)]
        if 0 not in pow2_set:
            pow2_set.append(0)
        pow2_set = sorted(pow2_set)
        score = _score_from_set(pow2_set)
        if score > best_score:
            best_score = score
            best_set = list(pow2_set)
    
    # Final: try to improve best_set with more local search
    current_set = list(best_set)
    current_score = best_score
    no_improve_count = 0
    while time.time() < deadline and no_improve_count < 50:
        n = len(current_set)
        idx = random.randint(0, n - 1)
        old_val = current_set[idx]
        
        improved = False
        for _ in range(30):
            if idx == 0:
                new_val = 0
            else:
                scale = max(1, int(old_val * 0.2))
                new_val = old_val + random.randint(-scale, scale)
                if new_val < 0:
                    new_val = 0
                if new_val == old_val:
                    continue
            
            candidate = current_set[:]
            candidate[idx] = new_val
            candidate = sorted(set(candidate))
            if 0 not in candidate:
                continue
            
            new_score = _score_from_set(candidate)
            if new_score > current_score + 1e-12:
                current_set = candidate
                current_score = new_score
                improved = True
                break
        
        if improved:
            no_improve_count = 0
            if current_score > best_score:
                best_score = current_score
                best_set = list(current_set)
        else:
            no_improve_count += 1
    
    return sorted(set(best_set))

# EVOLVE-BLOCK-END
