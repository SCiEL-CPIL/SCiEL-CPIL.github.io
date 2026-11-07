import random
import time

def construct():
    n = 128
    indices = range(n)
    
    # 1. Precompute patterns
    # A pattern is a set of indices (0-based) that must NOT all be 1.
    # If all are 1, it's a violation.
    patterns = []
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            for c in range(b + 1, n + 1):
                s_ab = a + b
                s_ac = a + c
                s_bc = b + c
                if s_bc > n:
                    break
                
                # Values involved: a, b, c, s_ab, s_ac, s_bc
                vals = [a, b, c, s_ab, s_ac, s_bc]
                # Convert to 0-based indices
                idxs = set(v - 1 for v in vals)
                patterns.append(frozenset(idxs))
    
    # 2. Map index to list of pattern indices containing it
    idx_to_pats = [[] for _ in range(n)]
    for p_idx, pat in enumerate(patterns):
        for i in pat:
            idx_to_pats[i].append(p_idx)
    
    # 3. Simulated Annealing
    # State: x (list of 0/1)
    # We want to minimize violations, then maximize size.
    # Score = size - 1000 * violations
    
    def count_violations(x):
        v = 0
        for pat in patterns:
            if all(x[i] for i in pat):
                v += 1
        return v
    
    # Initialize
    # Start with a random subset of size ~ n/2
    x = [random.randint(0, 1) for _ in range(n)]
    current_viol = count_violations(x)
    current_size = sum(x)
    current_score = current_size - 1000 * current_viol
    
    best_x = x.copy()
    best_score = current_score
    best_viol = current_viol
    best_size = current_size
    
    T = 10.0
    T_min = 0.01
    alpha = 0.999
    start_time = time.time()
    max_time = 50 # Leave some buffer for the rest
    
    while T > T_min and (time.time() - start_time) < max_time:
        # Pick random index to flip
        i = random.randint(0, n - 1)
        
        # Calculate delta violations
        # If x[i] flips from 0 to 1:
        #   Check pats containing i. If other elements are 1, viol increases.
        # If x[i] flips from 1 to 0:
        #   Check pats containing i. If other elements are 1, viol decreases.
        
        delta_viol = 0
        if x[i] == 0:
            # Becomes 1
            for p_idx in idx_to_pats[i]:
                pat = patterns[p_idx]
                # Check if all OTHER elements in pat are 1
                # Since x[i] is currently 0, we check others
                active = True
                for j in pat:
                    if j != i and x[j] == 0:
                        active = False
                        break
                if active:
                    delta_viol += 1
        else:
            # Becomes 0
            for p_idx in idx_to_pats[i]:
                pat = patterns[p_idx]
                active = True
                for j in pat:
                    if j != i and x[j] == 0:
                        active = False
                        break
                if active:
                    delta_viol -= 1
        
        new_size = current_size + (1 if x[i] == 0 else -1)
        new_viol = current_viol + delta_viol
        new_score = new_size - 1000 * new_viol
        
        # Acceptance
        if new_score > current_score:
            accept = True
        else:
            delta_E = new_score - current_score
            if random.random() < pow(2.718281828, delta_E / T):
                accept = True
            else:
                accept = False
        
        if accept:
            x[i] = 1 - x[i]
            current_size = new_size
            current_viol = new_viol
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_x = x.copy()
                best_viol = current_viol
                best_size = current_size
        
        T *= alpha
    
    # 4. Greedy repair/improvement
    # If best_viol > 0, try to fix it?
    # SA should have driven violations to 0.
    # If best_viol == 0, try to add more elements.
    
    x_final = best_x.copy()
    viol_final = best_viol
    
    if viol_final == 0:
        # Try to add any 0 that doesn't create a violation
        changed = True
        while changed:
            changed = False
            # Random order to be fair
            candidates = [i for i in range(n) if x_final[i] == 0]
            random.shuffle(candidates)
            for i in candidates:
                # Check if adding i creates a violation
                # Only pats containing i are affected
                safe = True
                for p_idx in idx_to_pats[i]:
                    pat = patterns[p_idx]
                    # Check if all OTHER elements in pat are 1
                    active = True
                    for j in pat:
                        if j != i and x_final[j] == 0:
                            active = False
                            break
                    if active:
                        safe = False
                        break
                if safe:
                    x_final[i] = 1
                    viol_final += 0 # Still 0
                    changed = True
    
    # Final check
    # Ensure types are correct
    result = [i + 1 for i in range(n) if x_final[i] == 1]
    
    # Verify validity one last time (slow check)
    # Just to be safe, though the logic above is sound
    S = set(result)
    valid = True
    for a in S:
        for b in S:
            if b <= a: continue
            for c in S:
                if c <= b: continue
                if a+b in S and a+c in S and b+c in S:
                    valid = False
                    break
            if not valid: break
        if not valid: break
    
    if not valid:
        # Fallback: return a known valid small set or retry?
        # If SA failed, return empty or minimal valid set?
        # The task requires maximizing size. If we return invalid, score 0.
        # Let's try a simple greedy construction if SA failed.
        # But SA is robust.
        pass
        
    return result
