import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    n = 128
    # Target: every t in [128, 256] must be expressible as a+b with a<=b, a,b in A
    # Minimize max multiplicity over t in [128, 256]
    
    # Precompute FFT size for fast convolution
    # v has length 129, conv has length 257, need FFT size >= 257
    fft_size = 512  # power of 2 >= 257
    
    def fast_convolve(v):
        """Compute convolution v*v using FFT."""
        V = np.fft.rfft(v, n=fft_size)
        conv = np.fft.irfft(V * V, n=fft_size)
        return np.round(conv).astype(np.int32)
    
    def evaluate(A_set):
        """Return (max_mult, covered) for the given set A."""
        v = np.zeros(129, dtype=np.float64)
        for x in A_set:
            v[x] = 1.0
        conv = fast_convolve(v)
        
        max_mult = 0
        covered = True
        for t in range(128, 257):
            ordered = int(conv[t])
            diag = 1 if (t % 2 == 0 and (t // 2) in A_set) else 0
            unordered = (ordered + diag) // 2
            if unordered == 0:
                covered = False
            if unordered > max_mult:
                max_mult = unordered
        return max_mult, covered
    
    # Generate a good initial solution
    # Strategy: use a structured set that covers [128, 256]
    # If A contains all multiples of k in some range, we can cover sums
    # Let's try: A = {0, k, 2k, ..., mk} union some extras
    # For k=4: A = {0,4,8,...,128} = 33 elements
    # Sums range from 0 to 256, step 4. Need to cover all [128,256]
    # So we need A to be dense enough
    
    # Better: use a greedy construction
    # Start with elements that help cover the target range
    # t in [128, 256]: need a+b=t with a,b in A, a<=b
    # So for each t, need at least one a in A with t-a in A and t-a >= a
    
    # Let's use a two-phase approach:
    # Phase 1: Build a covering set
    # Phase 2: Optimize via simulated annealing
    
    def build_initial_set():
        """Build a valid covering set."""
        A = set()
        # Add 0 and 128 to help cover extremes
        A.add(0)
        A.add(128)
        
        # For each t in [128, 256], ensure coverage
        # Greedy: for uncovered t, add a useful element
        for t in range(128, 257):
            # Check if t is covered
            covered_t = False
            for a in A:
                b = t - a
                if b in A and b >= a:
                    covered_t = True
                    break
            if not covered_t:
                # Add an element that helps cover t
                # Try to add a = t // 2 (if integer) or nearby
                candidates = []
                for a in range(max(0, t - 128), min(128, t // 2) + 1):
                    if a not in A:
                        candidates.append(a)
                if candidates:
                    # Pick the one that covers the most uncovered targets
                    best_a = candidates[0]
                    best_count = -1
                    for a in candidates:
                        # Count how many t' in [128,256] would be covered
                        count = 0
                        for t2 in range(128, 257):
                            b2 = t2 - a
                            if b2 in A and b2 >= a:
                                count += 1
                        if count > best_count:
                            best_count = count
                            best_a = a
                    A.add(best_a)
        
        return A
    
    A0 = build_initial_set()
    score0, covered0 = evaluate(A0)
    
    if not covered0:
        # Fallback to a dense set
        A0 = set(range(0, 129, 2))  # even numbers
        score0, covered0 = evaluate(A0)
        if not covered0:
            A0 = set(range(0, 129))
            score0, covered0 = evaluate(A0)
    
    best_A = set(A0)
    best_score = score0
    best_covered = covered0
    
    # Multi-restart simulated annealing
    num_restarts = 5
    time_per_restart = (deadline - start_time) / num_restarts
    
    for restart in range(num_restarts):
        if time.time() > deadline - 2:
            break
        
        # Different starting points for each restart
        if restart == 0:
            current_A = set(A0)
        elif restart == 1:
            # Start with even numbers
            current_A = set(range(0, 129, 2))
            s, c = evaluate(current_A)
            if not c:
                current_A = set(A0)
        elif restart == 2:
            # Start with a different structured set
            current_A = set(range(0, 65)) | set(range(64, 129, 2))
            s, c = evaluate(current_A)
            if not c:
                current_A = set(A0)
        elif restart == 3:
            # Random perturbation of best
            current_A = set(best_A)
            for _ in range(20):
                if random.random() < 0.5 and len(current_A) > 1:
                    x = random.choice(sorted(current_A))
                    current_A.discard(x)
                else:
                    x = random.randint(0, 128)
                    current_A.add(x)
            s, c = evaluate(current_A)
            if not c:
                current_A = set(best_A)
        else:
            # Another random perturbation
            current_A = set(best_A)
            for _ in range(40):
                if random.random() < 0.5 and len(current_A) > 1:
                    x = random.choice(sorted(current_A))
                    current_A.discard(x)
                else:
                    x = random.randint(0, 128)
                    current_A.add(x)
            s, c = evaluate(current_A)
            if not c:
                current_A = set(best_A)
        
        current_score, current_covered = evaluate(current_A)
        
        # Simulated annealing parameters
        T = 30.0
        T_min = 0.01
        cooling = 0.9995
        
        while time.time() < deadline - 2 and T > T_min:
            # Generate a move
            r = random.random()
            if r < 0.3 and len(current_A) > 1:
                # Remove
                x = random.choice(sorted(current_A))
                new_A = current_A - {x}
            elif r < 0.6:
                # Add
                x = random.randint(0, 128)
                if x in current_A:
                    continue
                new_A = current_A | {x}
            elif r < 0.8 and len(current_A) > 1:
                # Swap: remove one, add another
                x_remove = random.choice(sorted(current_A))
                x_add = random.randint(0, 128)
                if x_add in current_A:
                    continue
                new_A = current_A - {x_remove} | {x_add}
            else:
                # Double swap: remove two, add two
                if len(current_A) < 2:
                    continue
                x1, x2 = random.sample(sorted(current_A), 2)
                x3 = random.randint(0, 128)
                x4 = random.randint(0, 128)
                if x3 in current_A or x4 in current_A or x3 == x4:
                    continue
                new_A = current_A - {x1, x2} | {x3, x4}
            
            new_score, new_covered = evaluate(new_A)
            
            # Cost function with heavy penalty for uncovered
            if new_covered:
                new_cost = new_score
            else:
                new_cost = new_score + 10000
            
            if current_covered:
                current_cost = current_score
            else:
                current_cost = current_score + 10000
            
            delta = new_cost - current_cost
            
            if delta < 0 or random.random() < np.exp(-delta / T):
                current_A = new_A
                current_score = new_score
                current_covered = new_covered
                
                if current_covered and (not best_covered or current_score < best_score):
                    best_A = set(current_A)
                    best_score = current_score
                    best_covered = True
            
            T *= cooling
    
    # Final polish: try to remove elements one by one
    if best_covered and time.time() < deadline - 1:
        # Try removing each element if possible
        for x in sorted(best_A):
            if time.time() > deadline - 1:
                break
            new_A = best_A - {x}
            s, c = evaluate(new_A)
            if c and s <= best_score:
                best_A = new_A
                best_score = s
    
    # Ensure validity
    if not best_covered:
        best_A = set(range(0, 129))
    
    return sorted(best_A)
