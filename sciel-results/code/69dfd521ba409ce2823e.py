"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
import numpy as np

def compute_score_fast(U, max_val_hint=None):
    """
    Computes the lower bound exponent using FFT for sumset and bitsets for diffset.
    U: list of nonnegative integers
    Returns: score
    """
    if len(U) < 2:
        return 0.0
    U = sorted(set(U))
    maxU = U[-1]
    if maxU == 0:
        return 0.0
    
    # 1. Compute Sumset |U+U| using FFT
    # We need to find if u1 + u2 = k.
    # Create an array A where A[u] = 1 if u in U.
    # Convolution C = A * A. C[k] > 0 implies k is in U+U.
    
    # Determine size for FFT.
    # Max sum is 2 * maxU.
    # Next power of 2 >= 2 * maxU + 1.
    max_sum = 2 * maxU
    n_fft = 1
    while n_fft < max_sum + 1:
        n_fft *= 2
    
    A = np.zeros(n_fft, dtype=np.float64)
    for u in U:
        A[u] = 1.0
        
    # FFT convolution
    A_fft = np.fft.rfft(A)
    C_fft = A_fft * A_fft
    C = np.fft.irfft(C_fft, n=n_fft)
    
    # Count non-zero entries in C[0..2*maxU]
    # Due to floating point precision, values might be very close to 0 or 1.
    # Threshold at 0.5 is safe for integer convolutions of 0/1 arrays.
    sum_set_size = np.sum(C[:max_sum + 1] > 0.5)
    
    # 2. Compute Difference Set |U-U| using Bitsets
    # Diff set is {u1 - u2 | u1, u2 in U}.
    # Range is [-maxU, maxU].
    # We can compute this by checking for each d in [-maxU, maxU] if (U + d) intersects U.
    # Alternatively, use bit operations.
    # Let's use a simple loop for the difference set since N is likely small enough 
    # that the O(N^2) set construction is fast, BUT the FFT part is the bottleneck 
    # for large ranges. 
    # Actually, for the difference set, we can also use bitsets efficiently.
    # Let's stick to the set method for diffs first, as it's O(N^2) and N is small 
    # compared to maxU. If N is 100, N^2 is 10k, which is trivial.
    
    diff_set = set()
    for u1 in U:
        for u2 in U:
            diff_set.add(u1 - u2)
            
    size_diff = len(diff_set)
    size_sum = sum_set_size
    
    if size_sum == 0:
        return 0.0
        
    log_ratio = math.log(size_diff / size_sum)
    log_range = math.log(2 * maxU + 1)
    
    return 1.0 + log_ratio / log_range

def construct():
    """
    Entry point. Returns a list of nonnegative integers containing 0.
    Uses a greedy constructive approach with FFT-accelerated scoring.
    """
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Initial candidates
    # Geometric progressions tend to have large difference sets relative to sum sets
    # because sums are unique (Sidon-like) while differences can overlap less?
    # Actually, for geometric, sums are unique, so |U+U| ~ N^2/2.
    # Differences: u_i - u_j. Also mostly unique.
    # The ratio is close to 1.
    # We want |U-U| > |U+U|.
    # Since U is non-negative, U-U contains negative numbers, 0, and positive numbers.
    # U+U contains only non-negative numbers.
    # If U is "symmetric" in some sense or has structure that makes differences more spread out...
    # Actually, |U-U| is always >= |U+U| for sets containing 0?
    # Let U = {0, a, b}.
    # Sums: 0, a, b, 2a, a+b, 2b. (6 elements max)
    # Diffs: 0, a, b, -a, -b, a-b, b-a, 2a, -2a, 2b, -2b.
    # If a, b are such that no collisions, |U-U| = 1 + 2*(3-1) = 5? No.
    # Pairs (u1, u2): 9 pairs.
    # Diffs: 0-0=0, 0-a=-a, 0-b=-b, a-0=a, a-a=0, a-b=a-b, b-0=b, b-a=b-a, b-b=0.
    # Unique: 0, -a, -b, a, a-b, b, b-a. (7 elements)
    # Sums: 0, a, b, 2a, a+b, 2b. (6 elements)
    # Ratio 7/6 > 1.
    
    # Strategy: Greedy addition.
    # Start with small set, try adding elements that maximize score.
    
    best_U = [0, 1, 3, 9]
    best_score = compute_score_fast(best_U)
    
    current_U = list(best_U)
    current_score = best_score
    
    # Greedy Search
    # We want to add elements. 
    # Candidate pool: integers up to some limit.
    # Since max(U) grows, the log_range term grows.
    # We want to keep max(U) small while increasing |U-U|/|U+U|.
    
    max_allowed_val = 200 # Start small
    
    while time.time() < deadline:
        improved = False
        current_max = max(current_U) if current_U else 0
        
        # Generate candidates to add
        # 1. Small integers not in U
        # 2. Multiples of existing elements
        candidates = set()
        
        # Range to search for new elements
        # If max is small, search up to 100. If max is large, search around max.
        search_upper = max(50, int(current_max * 1.5))
        
        for x in range(1, search_upper + 1):
            if x not in current_U:
                candidates.add(x)
                
        # Also try some larger jumps
        if current_max > 0:
            for mult in [2, 3, 5]:
                candidates.add(current_max * mult)
                candidates.add(current_max * mult + 1)
                candidates.add(current_max * mult - 1)

        # Filter candidates that are too large (to keep log_range small)
        # But also we might need to go large if we are stuck?
        # The score is 1 + log(ratio)/log(range).
        # If ratio is constant, larger range is bad.
        # If ratio increases with size, it might be worth it.
        
        best_move_score = current_score
        best_move_U = None
        
        # Evaluate candidates
        # To speed up, we can sort candidates by some heuristic?
        # Or just evaluate all. FFT is fast.
        
        # Limit number of candidates to evaluate to prevent timeout if search space is huge
        candidate_list = list(candidates)
        # Shuffle to avoid bias? Or sort by value?
        # Let's evaluate all, but break if time is tight.
        
        for x in candidate_list:
            if time.time() > deadline - 10:
                break
                
            new_U = sorted(list(set(current_U + [x])))
            # Pruning: if new max is much larger, score might drop.
            # But let's compute.
            s = compute_score_fast(new_U)
            
            if s > best_move_score:
                best_move_score = s
                best_move_U = new_U
        
        if best_move_U:
            current_U = best_move_U
            current_score = best_move_score
            if current_score > best_score:
                best_score = current_score
                best_U = list(current_U)
            improved = True
        else:
            # No improvement found in this neighborhood.
            # Try a random restart or perturb?
            # Or try removing an element?
            # Let's try removing the largest element to see if it helps (reduces range).
            if len(current_U) > 2:
                # Try removing the largest element
                largest = max(current_U)
                new_U = [x for x in current_U if x != largest]
                s = compute_score_fast(new_U)
                if s > best_move_score:
                    current_U = new_U
                    current_score = s
                    if current_score > best_score:
                        best_score = current_score
                        best_U = list(current_U)
                    improved = True
            
            if not improved:
                # Try a random restart with a different geometric base
                # or a random set
                for _ in range(3):
                    if time.time() > deadline - 5:
                        break
                    base = random.randint(2, 5)
                    size = random.randint(4, 8)
                    random_U = [0]
                    val = 1
                    for i in range(size - 1):
                        random_U.append(val)
                        val *= base
                        # Add some noise
                        if random.random() > 0.5:
                            val += random.randint(-1, 1)
                            if val <= 0:
                                val = 1
                    random_U = sorted(set([x for x in random_U if x >= 0]))
                    if 0 not in random_U:
                        random_U.append(0)
                    s = compute_score_fast(random_U)
                    if s > best_score:
                        best_score = s
                        best_U = list(random_U)
                        current_U = list(random_U)
                        current_score = s
                        break
                
                # If still no improvement, we might be stuck.
                # Let's try expanding the search range slightly?
                # Or just break to avoid wasting time.
                # Given the strict time limit, breaking is safer.
                # But we can also try to "shake" the current set by replacing a random element.
                if len(current_U) > 2:
                    idx = random.randint(1, len(current_U)-1)
                    old_val = current_U[idx]
                    new_val = old_val + random.randint(-5, 5)
                    if new_val > 0 and new_val not in current_U:
                        new_U = sorted([new_val] + [x for x in current_U if x != old_val])
                        s = compute_score_fast(new_U)
                        if s > current_score:
                            current_U = new_U
                            current_score = s
                            if current_score > best_score:
                                best_score = current_score
                                best_U = list(current_U)
                            improved = True
                
                if not improved:
                    # Stuck. Break.
                    break

    return best_U

# EVOLVE-BLOCK-END
