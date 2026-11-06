import random
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Leave 10s margin for 60s limit
    
    n = 128
    target_start = 128
    target_end = 256
    
    # A is a set of integers in [0, 128]
    # We represent it as a boolean list for fast lookup and sum calculation
    # mult[s] = number of pairs {a,b} in A such that a+b=s
    
    # Helper to compute multiplicities
    def compute_mults(A_set):
        mults = [0] * (2 * n + 1)
        A_list = list(A_set)
        L = len(A_list)
        for i in range(L):
            a = A_list[i]
            # Pairs (a, b) with b >= a to handle unordered pairs with equal summands
            # If a+b == s, and we count each unordered pair once.
            # For a given a, b can be a..n.
            # But we only care about sums in [128, 256].
            # Actually, let's just compute all sums and filter later or check bounds.
            for j in range(i, L):
                b = A_list[j]
                s = a + b
                if target_start <= s <= target_end:
                    mults[s] += 1
        return mults
    
    # Initialize A
    # A good starting point is a random subset of appropriate size.
    # Size estimate: To cover 129 sums, we need enough pairs.
    # If |A|=k, max pairs is k(k+1)/2.
    # We want mults to be small.
    # Let's try k=50.
    random.seed(42)
    
    best_A = set()
    best_peak = float('inf')
    best_var = float('inf')
    
    # Greedy initialization
    # Start with a structured set to ensure coverage
    # e.g. every 3rd number?
    # Or random.
    
    A = set(random.sample(range(n+1), 50))
    
    # Check coverage and peak
    mults = compute_mults(A)
    # Check if all sums in [128, 256] are covered
    covered = all(mults[s] > 0 for s in range(target_start, target_end+1))
    
    if covered:
        peak = max(mults[target_start:target_end+1])
        # Calculate variance for tie-breaking
        mean = sum(mults[target_start:target_end+1]) / 129.0
        var = sum((m - mean)**2 for m in mults[target_start:target_end+1])
        best_A = set(A)
        best_peak = peak
        best_var = var
    else:
        # If not covered, add elements until covered
        # Greedy: add element that covers most missing sums
        missing = [s for s in range(target_start, target_end+1) if mults[s] == 0]
        while missing and time.time() < start_time + 5:
            best_elem = None
            best_gain = 0
            for x in range(n+1):
                if x in A:
                    continue
                gain = 0
                for s in missing:
                    # Does adding x cover s?
                    # Need y in A such that x+y=s => y = s-x
                    # If y in A and y >= x (to avoid double counting? No, just existence)
                    # Actually, if y in A, then {x,y} is a pair.
                    # We just need existence.
                    if 0 <= s-x <= n and (s-x) in A:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_elem = x
            if best_elem is None:
                break
            A.add(best_elem)
            mults = compute_mults(A)
            missing = [s for s in range(target_start, target_end+1) if mults[s] == 0]
        
        if not missing:
            peak = max(mults[target_start:target_end+1])
            mean = sum(mults[target_start:target_end+1]) / 129.0
            var = sum((m - mean)**2 for m in mults[target_start:target_end+1])
            best_A = set(A)
            best_peak = peak
            best_var = var
    
    if not best_A:
        # Fallback to seed
        return list(range(64, 129))
        
    # Local Search / Simulated Annealing
    T = 10.0
    cooling = 0.999
    
    A = set(best_A)
    mults = compute_mults(A)
    peak = best_peak
    mean = sum(mults[target_start:target_end+1]) / 129.0
    var = best_var
    
    while time.time() < deadline:
        T *= cooling
        if T < 0.01:
            break
            
        # Generate a move
        move_type = random.random()
        new_A = set(A)
        
        if move_type < 0.3:
            # Add
            candidates = [x for x in range(n+1) if x not in A]
            if not candidates: continue
            x = random.choice(candidates)
            new_A.add(x)
        elif move_type < 0.6:
            # Remove
            if len(A) < 30: continue
            x = random.choice(list(A))
            new_A.remove(x)
        else:
            # Swap
            if len(A) < 30: continue
            x = random.choice(list(A))
            candidates = [y for y in range(n+1) if y not in A]
            if not candidates: continue
            y = random.choice(candidates)
            new_A.remove(x)
            new_A.add(y)
            
        new_mults = compute_mults(new_A)
        # Check coverage
        if any(new_mults[s] == 0 for s in range(target_start, target_end+1)):
            # Invalid, reject
            continue
            
        new_peak = max(new_mults[target_start:target_end+1])
        new_mean = sum(new_mults[target_start:target_end+1]) / 129.0
        new_var = sum((m - new_mean)**2 for m in new_mults[target_start:target_end+1])
        
        # Objective: Minimize peak, then variance
        # Score function for comparison
        # Let's use a composite score: peak * 1000 + var
        score = peak * 1000.0 + var
        new_score = new_peak * 1000.0 + new_var
        
        if new_score < score:
            A = new_A
            mults = new_mults
            peak = new_peak
            var = new_var
            score = new_score
        else:
            # Accept with probability
            delta = new_score - score
            if delta < 0:
                prob = 1.0
            else:
                prob = math.exp(-delta / T)
                
            if random.random() < prob:
                A = new_A
                mults = new_mults
                peak = new_peak
                var = new_var
                score = new_score
                
        # Update global best
        if peak < best_peak or (peak == best_peak and var < best_var):
            best_A = set(A)
            best_peak = peak
            best_var = var
            
    return list(best_A)
