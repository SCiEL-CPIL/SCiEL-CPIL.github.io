import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 5s margin
    
    N_LOW = 128
    N_HIGH = 256
    MAX_VAL = 128
    UNIVERSE_SIZE = 129 # 0..128
    
    # Helper to calculate reps using FFT/Convolution
    # A is a set of ints in [0, 128]
    # We need to count unordered pairs (a, b) with a <= b such that a+b = s.
    # Convolution of A with A gives counts of ordered pairs (a, b) for all a, b in A.
    # Let C[s] = sum_{a in A} A[a] * A[s-a]. This counts (a, b) and (b, a) separately if a != b.
    # For a == b, it counts 1.
    # We want unordered: 
    # If a != b, (a, b) and (b, a) are the same. So count is C[s] / 2? No.
    # Let's look at the definition: "Pair representations are unordered and permit equal summands."
    # Usually, multiplicity means the number of such pairs.
    # If A = {1, 2}, sums: 1+1=2, 1+2=3, 2+2=4.
    # Convolution of [0,1,1,0] with itself:
    # idx: 0 1 2 3 4
    # vals: 0 1 2 1 0 (wait, 1*1=1 at 2, 1*1+1*1=2 at 3? No)
    # A = [0, 1, 1, 0] (indices 0,1,2,3)
    # Conv:
    # 0: 0
    # 1: 1 (1*1)
    # 2: 2 (1*1 from 1+1, 1*1 from 2+0? No, A[0]=0)
    # Let's use np.convolve correctly.
    # If A is indicator vector x, x*x gives number of ordered pairs.
    # For s, ordered count N_ord(s).
    # Unordered count N_unord(s) = (N_ord(s) + N_diag(s)) / 2, where N_diag(s) is 1 if s/2 in A and s is even, else 0.
    # Why? 
    # Pairs are {a,b}.
    # If a != b, they appear twice in convolution (a,b) and (b,a). So contribution to N_ord is 2. Contribution to N_unord is 1.
    # If a == b, they appear once in convolution (a,a). So contribution to N_ord is 1. Contribution to N_unord is 1.
    # So N_unord(s) = (N_ord(s) - N_diag(s) + 2*N_diag(s)) / 2 = (N_ord(s) + N_diag(s)) / 2.
    
    def calc_reps(A_set):
        """
        Returns an array of length 257 (indices 0..256) representing unordered pair counts.
        """
        if not A_set:
            return np.zeros(257, dtype=np.int64)
        
        # Create indicator vector
        # Size 129 is enough for A, but convolution will be larger.
        # Max sum is 256.
        x = np.zeros(129, dtype=np.int64)
        for a in A_set:
            x[a] = 1
            
        # Convolve
        # Result size will be 129 + 129 - 1 = 257
        conv = np.convolve(x, x, mode='full')
        
        # Adjust for unordered
        # conv[s] is ordered count.
        # diag is 1 if s is even and s/2 is in A.
        # We only care about s in [128, 256], but let's compute all.
        reps = np.zeros(257, dtype=np.int64)
        
        for s in range(257):
            ord_count = conv[s]
            # Check if s/2 is in A
            if s % 2 == 0:
                half = s // 2
                if 0 <= half < 129 and x[half] == 1:
                    diag_count = 1
                else:
                    diag_count = 0
            else:
                diag_count = 0
            
            reps[s] = (ord_count + diag_count) // 2
            
        return reps

    def get_objective(reps, A_set):
        """
        Objective: Minimize peak multiplicity in [128, 256], subject to coverage.
        We return a tuple (uncovered_count, peak) to compare lexicographically.
        """
        target_reps = reps[N_LOW:N_HIGH+1]
        uncovered = np.sum(target_reps == 0)
        if uncovered > 0:
            # Penalize uncovered heavily
            return (uncovered, 1000) 
        peak = np.max(target_reps)
        return (0, peak)

    def is_valid(reps):
        return np.all(reps[N_LOW:N_HIGH+1] > 0)

    # Initialization Strategy:
    # We want a set A such that A+A covers [128, 256].
    # A good starting point is a "balanced" set.
    # Let's try a random set of size ~40-50, but structured.
    # Or start with a known good structure: 
    # If A contains all even numbers in [0, 128], sums cover [0, 256] evens.
    # If A contains all odds, sums cover evens.
    # To cover all, we need a mix.
    
    # Let's use a "Bit Reversal" or "Van der Corput" like distribution to spread out sums.
    # Actually, simple random with repair is often best if evaluation is fast.
    
    # Initial A
    # Size estimation: 
    # To cover 129 sums with peak P, we need ~129*P pairs.
    # Pairs ~ K^2/2.
    # If P=10, pairs ~ 1300. K^2 ~ 2600 -> K ~ 50.
    # If P=5, pairs ~ 650. K^2 ~ 1300 -> K ~ 36.
    
    random.seed(42)
    
    # Generate initial candidate using a heuristic:
    # Pick K elements.
    # We will run a few restarts with different K and initial patterns.
    
    best_A = None
    best_obj = (10000, 1000) # (uncovered, peak)
    
    # Pre-compute neighbors for fast swap evaluation? 
    # With FFT, evaluating a new set takes ~O(1) numpy calls (fast).
    # We can afford many iterations.
    
    num_restarts = 5
    
    for restart in range(num_restarts):
        if time.time() > deadline - 15:
            break
            
        # Vary size
        K = 35 + restart * 5
        if K > 60: K = 60
        
        # Initialize A
        # Strategy: Random sample, but ensure 0 and 128 are present to help extremes?
        # 128 is needed for sum 256 (128+128). If 128 not in A, max sum < 256 unless 127+129 (impossible).
        # So 128 MUST be in A to cover 256? 
        # Wait, max element is 128. Max sum is 128+128=256.
        # So 128 must be in A to get 256.
        # Min sum for target is 128.
        # If 0 in A, 128 can be 0+128.
        # If 0 not in A, min sum is 1+1=2 (if 1 in A). 
        # We need to cover 128.
        
        A = set(random.sample(range(0, 128), K-1))
        A.add(128)
        
        reps = calc_reps(A)
        curr_obj = get_objective(reps, A)
        
        if curr_obj < best_obj:
            best_obj = curr_obj
            best_A = set(A)
            
        # Local Search Phase
        # We use a "Greedy Repair + Peak Shaving" loop.
        
        # 1. Ensure Coverage
        # If not covered, find the sum s with reps[s]==0.
        # Find a pair (a,b) such that a+b=s. 
        # We can add an element x to A such that x + y = s for some y in A.
        # Or swap.
        
        # 2. Reduce Peak
        # Find s with max reps.
        # Find a pair (a,b) in A with a+b=s.
        # Try to replace a with x such that x+b is not s (or has lower rep) and coverage is maintained.
        
        max_iters = 2000
        for _ in range(max_iters):
            if time.time() > deadline - 5:
                break
                
            reps = calc_reps(A)
            if not is_valid(reps):
                # Repair Coverage
                target_reps = reps[N_LOW:N_HIGH+1]
                holes = np.where(target_reps == 0)[0]
                if len(holes) == 0:
                    continue
                s = holes[random.randint(0, len(holes)-1)] + N_LOW
                
                # Try to add an element to cover s
                # We need x in [0,128] such that exists y in A with x+y=s.
                # i.e., x = s - y.
                # We want to pick x that also helps other holes or doesn't hurt peak too much.
                
                candidates = []
                for y in A:
                    x = s - y
                    if 0 <= x <= 128:
                        # Check if adding x creates a new peak
                        # Rough estimate: just pick one
                        candidates.append(x)
                
                if candidates:
                    # Pick random candidate
                    x = random.choice(candidates)
                    if x not in A:
                        A.add(x)
                        # Check if this improved objective
                        new_reps = calc_reps(A)
                        new_obj = get_objective(new_reps, A)
                        if new_obj <= curr_obj:
                            curr_obj = new_obj
                            if curr_obj < best_obj:
                                best_obj = curr_obj
                                best_A = set(A)
                        else:
                            # Revert if worse (though adding usually helps coverage)
                            # Actually, adding increases size, might increase peak elsewhere.
                            # If coverage was broken, obj was high. 
                            # If we fixed coverage, obj drops significantly.
                            pass
                else:
                    # No single addition fixes it? 
                    # Might need a swap.
                    # Try swapping u in A with v not in A.
                    # We want v + y = s for some y in A (after swap).
                    # This is complex. Let's try random swap if stuck.
                    u = random.choice(list(A))
                    v = random.randint(0, 128)
                    if v != u and v not in A:
                        A.remove(u)
                        A.add(v)
                        new_reps = calc_reps(A)
                        new_obj = get_objective(new_reps, A)
                        if new_obj < curr_obj:
                            curr_obj = new_obj
                            if curr_obj < best_obj:
                                best_obj = curr_obj
                                best_A = set(A)
                        else:
                            # Revert
                            A.remove(v)
                            A.add(u)
                continue # Re-loop to recalc reps
            
            # 2. Reduce Peak (Coverage is valid)
            target_reps = reps[N_LOW:N_HIGH+1]
            max_rep = np.max(target_reps)
            if max_rep == 1:
                # Optimal? Can't get lower than 1 if covered.
                break
            
            # Find sums with max_rep
            peak_sums = np.where(target_reps == max_rep)[0] + N_LOW
            s = peak_sums[random.randint(0, len(peak_sums)-1)]
            
            # Find a pair (a,b) in A such that a+b=s
            # Iterate through A
            pair_found = None
            for a in A:
                b = s - a
                if b in A and a <= b: # Unordered, check a<=b to avoid duplicates? 
                    # Actually, we just need one representation to break.
                    pair_found = (a, b)
                    break
            
            if pair_found is None:
                continue # Should not happen if reps[s] > 0
                
            a, b = pair_found
            
            # Try to replace a with x
            # We want x such that:
            # 1. x not in A
            # 2. x in [0, 128]
            # 3. New reps for sums involving x are low.
            # 4. Coverage is not broken.
            
            # Candidate x: 
            # Ideally, x + b != s (so we don't just shift the peak).
            # x + b = s => x = a. (So we must pick x != a).
            
            # Let's try a few random x
            improved = False
            for _ in range(20): # Try 20 random swaps for this peak
                x = random.randint(0, 128)
                if x in A or x == a:
                    continue
                
                # Simulate swap: remove a, add x
                A.remove(a)
                A.add(x)
                
                new_reps = calc_reps(A)
                
                if not is_valid(new_reps):
                    # Revert
                    A.remove(x)
                    A.add(a)
                    continue
                
                new_peak = np.max(new_reps[N_LOW:N_HIGH+1])
                new_obj = (0, new_peak)
                
                if new_obj < curr_obj:
                    curr_obj = new_obj
                    if curr_obj < best_obj:
                        best_obj = curr_obj
                        best_A = set(A)
                    improved = True
                    break
                else:
                    # Revert
                    A.remove(x)
                    A.add(a)
            
            if not improved:
                # Try swapping b instead?
                # Or try a "2-swap" (remove a, add x; remove b, add y)? Too complex.
                # Let's just try another random peak sum.
                pass

    if best_A is None:
        # Fallback
        return list(range(0, 129, 2))
        
    return list(best_A)
