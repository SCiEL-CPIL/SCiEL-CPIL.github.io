import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    MIN_SUM = 128
    MAX_SUM = 256
    NUM_SUMS = MAX_SUM - MIN_SUM + 1  # 129
    N = 129  # elements 0..128
    
    # Precompute FFT parameters
    fft_size = 512  # next power of 2 >= 2*129-1 = 257
    
    def calc_mult_fft(A_arr):
        """
        A_arr: boolean array of length 129, where A_arr[i] is True if i in A.
        Returns an array of unordered pair multiplicities for sums in [128, 256].
        """
        sig = A_arr.astype(np.float64)
        sig_fft = np.fft.fft(sig, n=fft_size)
        conv_fft = sig_fft * sig_fft
        conv = np.fft.ifft(conv_fft).real
        
        # Extract ordered counts for sums [128, 256]
        M_ordered = conv[MIN_SUM:MAX_SUM+1].astype(np.int32)
        
        # Convert to unordered: M_unordered[s] = (M_ordered[s] + I(s/2 in A and s even)) / 2
        M_unordered = np.zeros(NUM_SUMS, dtype=np.int32)
        for i in range(NUM_SUMS):
            s = MIN_SUM + i
            ordered_count = M_ordered[i]
            if s % 2 == 0:
                half = s // 2
                if 0 <= half < N and A_arr[half]:
                    M_unordered[i] = (ordered_count + 1) // 2
                else:
                    M_unordered[i] = ordered_count // 2
            else:
                M_unordered[i] = ordered_count // 2
                
        return M_unordered
    
    def calc_score(M):
        """
        M: array of multiplicities for sums [128, 256].
        Returns (gaps, peak_mult). Lower is better.
        """
        gaps = int(np.sum(M == 0))
        if gaps > 0:
            return (gaps, float('inf'))
        peak = int(np.max(M))
        return (0, peak)
    
    def get_contributors(A_set, target_sum):
        """Returns elements a in A such that a <= target_sum-a and target_sum-a in A."""
        contributors = []
        for a in A_set:
            b = target_sum - a
            if b in A_set and a <= b:
                contributors.append(a)
        return contributors
    
    def structured_init(k, pattern=0):
        """
        Generate structured initializations.
        pattern 0: even numbers
        pattern 1: odd numbers  
        pattern 2: middle range
        pattern 3: ends
        pattern 4: mixed
        """
        if pattern == 0:  # Even numbers
            evens = list(range(0, 129, 2))
            if len(evens) >= k:
                return set(random.sample(evens, k))
            return set(random.sample(range(129), k))
        elif pattern == 1:  # Odd numbers
            odds = list(range(1, 129, 2))
            if len(odds) >= k:
                return set(random.sample(odds, k))
            return set(random.sample(range(129), k))
        elif pattern == 2:  # Middle range [32, 96]
            middle = list(range(32, 97))
            if len(middle) >= k:
                return set(random.sample(middle, k))
            return set(random.sample(range(129), k))
        elif pattern == 3:  # Ends: [0, 16] and [112, 128]
            ends = list(range(0, 17)) + list(range(112, 129))
            if len(ends) >= k:
                return set(random.sample(ends, k))
            return set(random.sample(range(129), k))
        else:  # pattern 4: mixed structured
            # Take some from each quarter
            q1 = list(range(0, 32))
            q2 = list(range(32, 64))
            q3 = list(range(64, 96))
            q4 = list(range(96, 129))
            result = set()
            per_quarter = k // 4
            for q in [q1, q2, q3, q4]:
                take = min(per_quarter, len(q))
                result.update(random.sample(q, take))
            # Fill remaining randomly
            if len(result) < k:
                remaining = [x for x in range(129) if x not in result]
                if remaining:
                    result.update(random.sample(remaining, k - len(result)))
            return result
    
    def fix_gaps(A_set, max_iters=100):
        """
        Try to fix gaps by adding elements. Returns (A_set, success).
        """
        A_arr = np.zeros(N, dtype=np.bool_)
        for a in A_set:
            A_arr[a] = True
        M = calc_mult_fft(A_arr)
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            gaps = np.where(M == 0)[0]
            if len(gaps) == 0:
                break
            
            # Pick a random gap
            gap_idx = random.choice(gaps)
            target_sum = MIN_SUM + gap_idx
            
            # Find which elements could help cover this gap
            # We need a, b in A such that a+b = target_sum
            # Try adding elements that pair with existing elements
            candidates = []
            for a in A_set:
                b = target_sum - a
                if 0 <= b <= 128 and b not in A_set:
                    candidates.append(b)
            
            if not candidates:
                # Try adding pairs that sum to target_sum
                for a in range(0, target_sum // 2 + 1):
                    b = target_sum - a
                    if 0 <= b <= 128:
                        if a not in A_set and b not in A_set:
                            candidates.append(a)
                            candidates.append(b)
                        elif a not in A_set:
                            candidates.append(a)
                        elif b not in A_set:
                            candidates.append(b)
            
            if not candidates:
                continue
            
            # Try adding a random candidate
            cand = random.choice(candidates)
            if cand not in A_set:
                A_set.add(cand)
                A_arr[cand] = True
                M = calc_mult_fft(A_arr)
        
        A_arr = np.zeros(N, dtype=np.bool_)
        for a in A_set:
            A_arr[a] = True
        M = calc_mult_fft(A_arr)
        score = calc_score(M)
        success = score[0] == 0
        return A_set, success
    
    def simulated_annealing(A_set, initial_temp=10.0, cooling_rate=0.995, min_temp=0.1, max_steps=5000):
        """
        Simulated annealing to minimize peak multiplicity while maintaining validity.
        """
        A_arr = np.zeros(N, dtype=np.bool_)
        for a in A_set:
            A_arr[a] = True
        M = calc_mult_fft(A_arr)
        current_score = calc_score(M)
        
        best_score = current_score
        best_A_set = set(A_set)
        
        temp = initial_temp
        step = 0
        
        while temp > min_temp and step < max_steps and time.time() < deadline:
            step += 1
            
            # Generate a random move
            move_type = random.random()
            
            if move_type < 0.3 and len(A_set) > 1:
                # Remove a random element
                a = random.choice(list(A_set))
                A_set.remove(a)
                A_arr[a] = False
            elif move_type < 0.6:
                # Add a random element
                available = [x for x in range(129) if x not in A_set]
                if available:
                    b = random.choice(available)
                    A_set.add(b)
                    A_arr[b] = True
            else:
                # Swap: remove one, add one
                if len(A_set) > 0:
                    a = random.choice(list(A_set))
                    A_set.remove(a)
                    A_arr[a] = False
                    available = [x for x in range(129) if x not in A_set]
                    if available:
                        b = random.choice(available)
                        A_set.add(b)
                        A_arr[b] = True
            
            # Evaluate new state
            M = calc_mult_fft(A_arr)
            new_score = calc_score(M)
            
            # Acceptance criterion
            if new_score < current_score:
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_A_set = set(A_set)
            else:
                # Compute delta in score (use peak if valid, else use gaps as penalty)
                if current_score[0] == 0 and new_score[0] == 0:
                    delta = new_score[1] - current_score[1]
                elif current_score[0] > 0 and new_score[0] > 0:
                    delta = new_score[0] - current_score[0]
                elif current_score[0] == 0:
                    delta = 100 + new_score[0]  # Penalty for becoming invalid
                else:
                    delta = -(100 + current_score[0]) + new_score[1] if new_score[0] == 0 else new_score[0] - current_score[0]
                
                if delta < 0:
                    accept = True
                else:
                    accept_prob = np.exp(-delta / temp)
                    accept = random.random() < accept_prob
                
                if accept:
                    current_score = new_score
                else:
                    # Revert
                    if move_type < 0.3:
                        A_set.add(a)
                        A_arr[a] = True
                    elif move_type < 0.6:
                        A_set.discard(b)
                        A_arr[b] = False
                    else:
                        A_set.discard(b)
                        A_arr[b] = False
                        A_set.add(a)
                        A_arr[a] = True
            
            temp *= cooling_rate
        
        return best_A_set, best_score
    
    def targeted_improvement(A_set, max_iters=200):
        """
        Targeted local search focusing on high-multiplicity sums.
        """
        A_arr = np.zeros(N, dtype=np.bool_)
        for a in A_set:
            A_arr[a] = True
        M = calc_mult_fft(A_arr)
        current_score = calc_score(M)
        
        if current_score[0] > 0:
            # Fix gaps first
            A_set, success = fix_gaps(A_set)
            if not success:
                return A_set, current_score
            A_arr = np.zeros(N, dtype=np.bool_)
            for a in A_set:
                A_arr[a] = True
            M = calc_mult_fft(A_arr)
            current_score = calc_score(M)
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            
            max_mult = int(np.max(M))
            if max_mult <= 1:
                break
            
            # Find sums with max multiplicity
            high_idx = np.where(M == max_mult)[0]
            target_idx = random.choice(high_idx)
            target_sum = MIN_SUM + target_idx
            
            contributors = get_contributors(A_set, target_sum)
            if not contributors:
                continue
            
            # Try removing a contributor
            random.shuffle(contributors)
            improved = False
            for a in contributors:
                if time.time() > deadline:
                    break
                A_set.remove(a)
                A_arr[a] = False
                M_new = calc_mult_fft(A_arr)
                new_score = calc_score(M_new)
                
                if new_score < current_score:
                    current_score = new_score
                    M = M_new
                    improved = True
                    break
                else:
                    A_set.add(a)
                    A_arr[a] = True
            
            if improved:
                continue
            
            # Try swapping a contributor with a random element
            random.shuffle(contributors)
            for a in contributors:
                if time.time() > deadline:
                    break
                available = [x for x in range(129) if x not in A_set]
                if not available:
                    break
                candidates = random.sample(available, min(3, len(available)))
                for b in candidates:
                    A_set.remove(a)
                    A_arr[a] = False
                    A_set.add(b)
                    A_arr[b] = True
                    M_new = calc_mult_fft(A_arr)
                    new_score = calc_score(M_new)
                    
                    if new_score < current_score:
                        current_score = new_score
                        M = M_new
                        improved = True
                        break
                    else:
                        A_set.discard(b)
                        A_arr[b] = False
                        A_set.add(a)
                        A_arr[a] = True
                if improved:
                    break
            
            if not improved:
                # Try adding an element to break up high multiplicity
                available = [x for x in range(129) if x not in A_set]
                if available:
                    b = random.choice(available)
                    A_set.add(b)
                    A_arr[b] = True
                    M_new = calc_mult_fft(A_arr)
                    new_score = calc_score(M_new)
                    if new_score < current_score:
                        current_score = new_score
                        M = M_new
                        improved = True
                    else:
                        A_set.discard(b)
                        A_arr[b] = False
        
        return A_set, current_score
    
    best_score = (float('inf'), float('inf'))
    best_A_set = None
    
    # Phase 1: Structured initialization + gap fixing + targeted improvement
    sizes = [25, 30, 35, 40, 45, 50]
    patterns = [0, 1, 2, 3, 4]
    
    for k in sizes:
        if time.time() > deadline - 10:
            break
        for pattern in patterns:
            if time.time() > deadline - 5:
                break
            
            A_set = structured_init(k, pattern)
            
            # Fix gaps
            A_set, success = fix_gaps(A_set, max_iters=50)
            
            if not success:
                continue
            
            # Targeted improvement
            A_set, score = targeted_improvement(A_set, max_iters=100)
            
            if score < best_score:
                best_score = score
                best_A_set = set(A_set)
            
            if best_score == (0, 1):
                return sorted(best_A_set)
    
    # Phase 2: Simulated annealing from best solution
    if best_A_set is not None:
        A_set, score = simulated_annealing(best_A_set, initial_temp=5.0, cooling_rate=0.998, min_temp=0.05, max_steps=3000)
        if score < best_score:
            best_score = score
            best_A_set = A_set
    
    # Phase 3: Final targeted improvement
    if best_A_set is not None:
        A_set, score = targeted_improvement(best_A_set, max_iters=200)
        if score < best_score:
            best_score = score
            best_A_set = A_set
    
    # Phase 4: Greedy removal to minimize set size while maintaining peak
    if best_A_set is not None and best_score[0] == 0:
        A_set = set(best_A_set)
        current_score = best_score
        A_arr = np.zeros(N, dtype=np.bool_)
        for a in A_set:
            A_arr[a] = True
        M = calc_mult_fft(A_arr)
        
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            A_list = list(A_set)
            random.shuffle(A_list)
            for a in A_list:
                if time.time() > deadline - 2:
                    break
                if len(A_set) <= 1:
                    break
                A_set.remove(a)
                A_arr[a] = False
                M_new = calc_mult_fft(A_arr)
                new_score = calc_score(M_new)
                
                if new_score[0] == 0 and new_score[1] <= current_score[1]:
                    if new_score[1] < current_score[1] or len(A_set) < len(best_A_set):
                        current_score = new_score
                        best_A_set = set(A_set)
                        M = M_new
                        improved = True
                        break
                else:
                    A_set.add(a)
                    A_arr[a] = True
        
        if current_score < best_score:
            best_score = current_score
    
    if best_A_set is None:
        # Fallback: contiguous interval [64, 128]
        return list(range(64, 129))
    
    return sorted(best_A_set)

if __name__ == "__main__":
    start = time.time()
    A = construct()
    end = time.time()
    print(f"Time: {end-start:.2f}s")
    print(f"Set size: {len(A)}")
    A_set = set(A)
    A_arr = np.zeros(129, dtype=np.bool_)
    for a in A_set:
        A_arr[a] = True
    M = calc_mult_fft(A_arr)
    print(f"Min mult: {np.min(M)}, Max mult: {np.max(M)}")
    print(f"Gaps: {np.sum(M==0)}")
    print(f"Score: {calc_score(M)}")
