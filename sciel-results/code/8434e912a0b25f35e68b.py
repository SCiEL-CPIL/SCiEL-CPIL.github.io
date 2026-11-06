import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin for safety
    
    n = 128
    target_low = 128
    target_high = 256
    target_range = range(target_low, target_high + 1)
    
    def compute_reps(A_set):
        a = np.zeros(129, dtype=np.float64)
        for x in A_set:
            a[x] = 1.0
        
        size = 512
        fa = np.fft.fft(a, size)
        conv = np.fft.ifft(fa * fa).real
        
        reps = np.zeros(257, dtype=int)
        for s in target_range:
            c = int(round(conv[s]))
            d = 1 if (s % 2 == 0 and s // 2 in A_set) else 0
            reps[s] = (c + d) // 2
        return reps

    def is_valid(reps):
        return np.all(reps[target_low:target_high+1] > 0)

    def get_score(reps):
        if not is_valid(reps):
            return float('inf')
        return int(np.max(reps[target_low:target_high+1]))

    def get_stats(reps):
        # Return peak and sum of squares (for smoothing)
        if not is_valid(reps):
            return float('inf'), 0
        peak = int(np.max(reps[target_low:target_high+1]))
        ss = int(np.sum(reps[target_low:target_high+1]**2))
        return peak, ss

    best_A = None
    best_score = float('inf')
    best_ss = 0

    for restart in range(50): # Multiple restarts
        if time.time() > deadline:
            break
            
        # 1. Initialization
        # Try different initial sizes
        init_size = 25 + (restart % 10) # 25 to 34
        A = set(random.sample(range(0, 129), init_size))
        
        # Ensure validity by adding elements
        reps = compute_reps(A)
        while not is_valid(reps):
            # Find sums with 0 reps
            zero_sums = [s for s in target_range if reps[s] == 0]
            if not zero_sums:
                break
            
            # Find candidate a to add
            best_a = None
            best_gain = -1
            # Sample candidates if A is large, otherwise check all
            candidates = [x for x in range(129) if x not in A]
            if len(candidates) > 50:
                candidates = random.sample(candidates, 50)
                
            for a in candidates:
                new_A = A | {a}
                # Quick check: does a help cover zero_sums?
                # a + b = s => b = s - a
                # b must be in A
                gain = 0
                for s in zero_sums:
                    b = s - a
                    if 0 <= b <= 128 and b in A:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_a = a
                    
            if best_a is None:
                # Fallback: add random
                best_a = random.choice(candidates)
            
            A.add(best_a)
            reps = compute_reps(A)
            
        if not is_valid(reps):
            continue

        current_score, current_ss = get_stats(reps)
        
        # 2. Pruning
        improved = True
        while improved:
            improved = False
            for a in list(A):
                if time.time() > deadline:
                    break
                new_A = A - {a}
                if not new_A:
                    continue
                new_reps = compute_reps(new_A)
                if is_valid(new_reps):
                    new_score, new_ss = get_stats(new_reps)
                    # Accept if score is better or same (and we remove an element, reducing future complexity)
                    if new_score < current_score or (new_score == current_score and len(new_A) < len(A)):
                        A = new_A
                        reps = new_reps
                        current_score = new_score
                        current_ss = new_ss
                        improved = True
                        break # Restart scan

        if current_score < best_score or (current_score == best_score and current_ss < best_ss):
            best_score = current_score
            best_ss = current_ss
            best_A = set(A)

        # 3. Local Search / Swaps
        # Try to reduce peak or smooth distribution
        stuck_count = 0
        while time.time() < deadline - 5.0 and stuck_count < 100:
            # Identify elements contributing to high multiplicity sums
            # For simplicity, try random swaps
            a = random.choice(list(A))
            # Find a new position
            # Prefer positions that don't create new high peaks
            candidates = [x for x in range(129) if x not in A]
            if not candidates:
                break
            
            # Sample candidates
            if len(candidates) > 20:
                candidates = random.sample(candidates, 20)
            
            found_improvement = False
            for b in candidates:
                new_A = (A - {a}) | {b}
                new_reps = compute_reps(new_A)
                if not is_valid(new_reps):
                    continue
                
                new_score, new_ss = get_stats(new_reps)
                
                # Accept if peak is lower
                if new_score < current_score:
                    A = new_A
                    reps = new_reps
                    current_score = new_score
                    current_ss = new_ss
                    found_improvement = True
                    break
                # Accept if peak is same, but variance is lower (smoother)
                elif new_score == current_score and new_ss < current_ss:
                    A = new_A
                    reps = new_reps
                    current_score = new_score
                    current_ss = new_ss
                    found_improvement = True
                    break
                    
            if found_improvement:
                stuck_count = 0
                if current_score < best_score or (current_score == best_score and current_ss < best_ss):
                    best_score = current_score
                    best_ss = current_ss
                    best_A = set(A)
            else:
                stuck_count += 1
                
                # If stuck for a while, try a more aggressive move?
                # Or just restart the loop with a new random a?
                # The loop already picks a random a each iteration.

    if best_A is None:
        # Fallback: return the seed solution if all else fails
        return list(range(64, 129))
        
    return list(best_A)
