import math
import time

def construct():
    start_time = time.time()
    timeout = 140  # Safety margin for 150s limit
    
    N = 512
    # Weights: w[a] = 1 / (N - a)
    # We want to maximize sum of weights.
    
    # 1. Greedy Initialization
    # Select largest a such that gcd(a, b) == 1 for all b in selected
    selected = []
    selected_set = set()
    
    # Precompute gcds or just use math.gcd
    # Iterate from N-1 down to 1
    for a in range(N - 1, 0, -1):
        if time.time() - start_time > timeout:
            break
        # Check if a is coprime to all currently selected
        is_coprime = True
        for b in selected:
            if math.gcd(a, b) != 1:
                is_coprime = False
                break
        if is_coprime:
            selected.append(a)
            selected_set.add(a)
            
    # 2. Local Search: Swaps
    # Try to replace any a in selected with a b not in selected
    # such that b is coprime to (selected - {a}) and w(b) > w(a)
    
    # Sort selected by value (weight) to prioritize swapping low-value items?
    # Or just iterate through all.
    
    improved = True
    while improved:
        if time.time() - start_time > timeout:
            break
        improved = False
        
        # Create a list of indices to iterate, as we might modify the set
        # We iterate through the current selected set
        for i, a in enumerate(list(selected)):
            if time.time() - start_time > timeout:
                break
            
            # Try removing a
            remaining = [x for x in selected if x != a]
            remaining_set = set(remaining)
            
            # Find best candidate b not in selected
            # b must be coprime to all in remaining
            # We want max w(b), so we search from high to low
            # Optimization: Only search b > a? No, maybe a is large but blocks many, 
            # but we are swapping a with b. If b < a, w(b) < w(a), so no improvement.
            # So we only need to check b > a.
            
            best_b = -1
            best_gain = 0
            
            # Search for b from N-1 down to a+1
            # If we find one, it's the best because weights decrease with a.
            for b in range(N - 1, a, -1):
                if b in selected_set:
                    continue
                
                # Check coprime with remaining
                ok = True
                for x in remaining:
                    if math.gcd(b, x) != 1:
                        ok = False
                        break
                
                if ok:
                    # Since we iterate b from high to low, the first valid b is the best
                    # w(b) > w(a) is guaranteed if b > a
                    best_b = b
                    break
            
            if best_b != -1:
                # Perform swap
                selected.remove(a)
                selected_set.remove(a)
                selected.append(best_b)
                selected_set.add(best_b)
                improved = True
                # Restart the loop or continue? 
                # Restarting is safer to find new opportunities.
                # But for efficiency, we can break and restart the while loop.
                break
                
    return selected
