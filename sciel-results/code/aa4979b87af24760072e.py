import time
import random

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    max_val = 1000000
    
    # Deadline guard
    start_time = time.time()
    deadline = 80.0  # Leave 10s margin
    
    best_x = None
    best_span = float('inf')
    
    def optimize(x):
        """
        Greedy local search to minimize span.
        Returns optimized x and span.
        """
        x = x[:]  # Copy
        improved = True
        while improved:
            improved = False
            current_span = max(x) - min(x)
            for i in range(n):
                others = [x[j] for j in range(n) if j != i]
                lo = min(others)
                hi = max(others)
                
                # Candidates:
                # 1. Multiples of d_i in [lo, hi] not in others
                # 2. Largest multiple < lo
                # 3. Smallest multiple > hi
                
                d_i = divisors[i]
                
                # Find multiples in range [lo, hi]
                # k_min = ceil(lo / d_i)
                # k_max = floor(hi / d_i)
                
                candidates = []
                
                # Check inside [lo, hi]
                k_start = (lo + d_i - 1) // d_i
                k_end = hi // d_i
                
                if k_start <= k_end:
                    for k in range(k_start, k_end + 1):
                        val = k * d_i
                        if val not in others:
                            candidates.append(val)
                
                # Check left (largest < lo)
                k_left = (lo - 1) // d_i
                if k_left > 0:
                    val_left = k_left * d_i
                    if val_left > 0:
                        candidates.append(val_left)
                        
                # Check right (smallest > hi)
                k_right = (hi // d_i) + 1
                val_right = k_right * d_i
                if val_right <= max_val:
                    candidates.append(val_right)
                
                if not candidates:
                    continue
                
                # Evaluate best candidate
                # If a candidate is in [lo, hi], span is hi - lo.
                # If candidate < lo, span is hi - c.
                # If candidate > hi, span is c - lo.
                
                best_cand = None
                best_new_span = float('inf')
                
                for c in candidates:
                    if lo <= c <= hi:
                        new_span = hi - lo
                    elif c < lo:
                        new_span = hi - c
                    else:
                        new_span = c - lo
                    
                    if new_span < best_new_span:
                        best_new_span = new_span
                        best_cand = c
                        # If we found a span equal to current inside span, we can break?
                        # No, we want to minimize global span.
                        # If new_span == current_span, it doesn't improve, but might help?
                        # Greedy: only accept if strictly better?
                        # If equal, it might allow a better move for another index?
                        # Let's accept if <=, but prioritize smaller.
                        # Actually, if new_span == current_span, the set of values changes,
                        # which might open up moves for others.
                        # However, to avoid cycles, let's only accept if strictly smaller,
                        # OR if equal, we break ties by some heuristic (e.g., closer to mean).
                        # For now, strictly smaller ensures termination.
                        # If we allow equal, we need a cycle detection or limit.
                        # Let's stick to strictly smaller for the main loop,
                        # and use randomization to escape local minima.
                        
                if best_cand is not None and best_new_span < current_span:
                    x[i] = best_cand
                    current_span = best_new_span
                    improved = True
                    # Restart the loop to re-evaluate all indices with new x
                    break 
                
        return x, max(x) - min(x)
        
    # Phase 1: Deterministic Sweep
    # Try bases from 1 to 2000
    for B in range(1, 2001):
        if time.time() > deadline:
            break
            
        x_init = []
        valid = True
        for d in divisors:
            k = (B + d - 1) // d
            val = k * d
            if val > max_val:
                valid = False
                break
            x_init.append(val)
            
        if not valid:
            continue
            
        x_opt, span = optimize(x_init)
        if span < best_span:
            best_span = span
            best_x = x_opt[:]
            
    # Phase 2: Random Restarts
    # Use remaining time for random initializations
    while time.time() < deadline:
        # Random start
        x_init = []
        for d in divisors:
            # Pick a random multiple
            # Range roughly 1000 to 100000
            min_k = 1000 // d
            max_k = 100000 // d
            if min_k > max_k:
                continue
            k = random.randint(min_k, max_k)
            x_init.append(k * d)
            
        x_opt, span = optimize(x_init)
        if span < best_span:
            best_span = span
            best_x = x_opt[:]
            
    # Fallback if best_x is None (should not happen)
    if best_x is None:
        best_x = [d for d in divisors]
        
    return best_x
