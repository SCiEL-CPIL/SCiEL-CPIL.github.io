import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Primes <= 1000
    primes = []
    for i in range(2, 1001):
        is_prime = True
        for j in range(2, int(math.sqrt(i)) + 1):
            if i % j == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(i)
            
    n_primes = len(primes) # 168
    
    # State
    residues = [random.randint(0, p-1) for p in primes]
    
    # Helper to compute counts and L
    def get_stats(H):
        counts = [0] * (H + 1)
        for i, p in enumerate(primes):
            r = residues[i]
            # x in 1..H such that x % p == r
            # First x: if r == 0, p, 2p... else r, r+p...
            # Actually x can be 0? Task says 1..L.
            # x % p == r.
            # If r == 0, x = p, 2p...
            # If r > 0, x = r, r+p...
            start_x = r
            if start_x == 0:
                start_x = p
            elif start_x > H:
                # No x in 1..H? 
                # Wait, if r > H, no x in 1..H satisfies x % p == r?
                # x < p implies x % p = x. So if r < p, x=r is a candidate.
                # If r > H, and r < p, then r > H means no x in 1..H.
                # If r >= p, r is reduced mod p? No, r is the residue.
                # x % p == r.
                # Smallest positive x is r if r>0, else p.
                # If start_x > H, continue.
                continue
                
            for x in range(start_x, H + 1, p):
                counts[x] += 1
                
        # Find L
        L = 0
        for x in range(1, H + 1):
            if counts[x] >= 2:
                L = x
            else:
                break
        return counts, L

    # Initial stats
    H = 200
    counts, L = get_stats(H)
    
    best_L = L
    best_res = residues[:]
    
    # SA Parameters
    T = 10.0
    T_min = 0.01
    alpha = 0.999
    
    # Penalty function for SA: Sum of deficits
    def penalty(counts, H):
        p_val = 0
        for x in range(1, H + 1):
            if counts[x] < 2:
                p_val += (2 - counts[x]) ** 2
        return p_val
    
    current_pen = penalty(counts, H)
    
    while time.time() < deadline:
        # Check for expansion
        if L >= H:
            if H < 1000:
                H = min(H * 2, 1000)
                counts, L = get_stats(H)
                current_pen = penalty(counts, H)
                if L > best_L:
                    best_L = L
                    best_res = residues[:]
        
        # SA Move
        i = random.randint(0, n_primes - 1)
        p = primes[i]
        old_r = residues[i]
        new_r = random.randint(0, p - 1)
        
        if new_r == old_r:
            continue
            
        # Update counts
        # Remove old contributions
        start_old = old_r if old_r > 0 else p
        for x in range(start_old, H + 1, p):
            counts[x] -= 1
            
        # Add new contributions
        start_new = new_r if new_r > 0 else p
        for x in range(start_new, H + 1, p):
            counts[x] += 1
            
        # Calculate new penalty
        new_pen = 0
        # We only need to check affected x?
        # But penalty is global sum.
        # We can update penalty incrementally.
        # delta_pen = sum over affected x of (new_deficit^2 - old_deficit^2)
        
        # Let's just recompute penalty for speed of coding, H is small.
        # Optimization: Only recompute if we track affected set.
        # For now, recompute is O(H). 1000 ops. 10^6 moves = 10^9 ops. Might be slow in Python.
        # Let's do incremental penalty update.
        
        # We need old_pen.
        # delta = 0
        # for x in affected:
        #   old_c = counts[x] (after update? No, we updated counts in place)
        #   We need old counts.
        
        # Let's revert and do it properly.
        # This incremental update is tricky if we modify counts in place.
        
        # Alternative: Keep counts, calculate new_pen by iterating 1..H.
        # 1000 iterations per move. 100k moves in 120s? 1.2M moves.
        # 1.2M * 1000 = 1.2B ops. Too slow in Python.
        
        # We MUST do incremental penalty.
        
        # Let's restart the move logic with incremental penalty.
        
        # Revert counts
        for x in range(start_new, H + 1, p):
            counts[x] -= 1
        for x in range(start_old, H + 1, p):
            counts[x] += 1
            
        # Collect affected x
        affected = set()
        if start_old <= H:
            for x in range(start_old, H + 1, p):
                affected.add(x)
        if start_new <= H:
            for x in range(start_new, H + 1, p):
                affected.add(x)
                
        # Calculate delta penalty
        delta_pen = 0
        for x in affected:
            old_c = counts[x]
            # new_c will be old_c - 1 (if in old set) + 1 (if in new set)
            # If x in both, net 0 change?
            # x in old set: x % p == old_r
            # x in new set: x % p == new_r
            # Since old_r != new_r, x cannot be in both.
            if x % p == old_r:
                new_c = old_c - 1
            elif x % p == new_r:
                new_c = old_c + 1
            else:
                new_c = old_c # Should not happen for affected set
                
            old_def = max(0, 2 - old_c)
            new_def = max(0, 2 - new_c)
            delta_pen += new_def**2 - old_def**2
            
        new_pen = current_pen + delta_pen
        
        if new_pen < current_pen or random.random() < math.exp((current_pen - new_pen) / T):
            # Accept
            residues[i] = new_r
            # Apply changes to counts
            for x in range(start_old, H + 1, p):
                counts[x] -= 1
            for x in range(start_new, H + 1, p):
                counts[x] += 1
            current_pen = new_pen
            
            # Recalculate L occasionally? Or just at end of loop?
            # L depends on prefix.
            # We can check if the move improved L.
            # But L is not the penalty.
            # Let's just run repair to find L.
            
        T *= alpha
        if T < T_min:
            T = T_min
            
        # Periodic Repair
        # Every 1000 moves? Or if stuck?
        # Let's do a quick repair every 500 moves.
        # Repair: Find first x with C(x)<2. Try to fix.
        
        # Simple repair:
        # Find first bad x
        bad_x = -1
        for x in range(1, H + 1):
            if counts[x] < 2:
                bad_x = x
                break
                
        if bad_x != -1:
            # Try to fix bad_x
            best_delta = float('inf')
            best_p_idx = -1
            best_new_r = -1
            
            for i, p in enumerate(primes):
                # Candidate: r_p = bad_x % p
                cand_r = bad_x % p
                if cand_r == residues[i]:
                    continue # Already covers? But C(bad_x)<2, so it must cover?
                    # If it covers, C(bad_x) includes it.
                    # If C(bad_x) < 2, and it covers, then C(bad_x) is 1 or 0?
                    # If it covers, C(bad_x) >= 1.
                    # So if C(bad_x)=1, and it covers, changing it might drop it to 0?
                    # We want to INCREASE C(bad_x).
                    # So we need a p that DOES NOT cover bad_x yet, and set it to cover.
                    # OR a p that covers, but changing it keeps it covering?
                    # If p covers bad_x, r_p = bad_x % p. Changing to same value is no-op.
                    # So we only consider p where r_p != bad_x % p.
                    
                # Calculate delta penalty for setting residues[i] = cand_r
                old_r = residues[i]
                
                affected = set()
                start_old = old_r if old_r > 0 else p
                if start_old <= H:
                    for x in range(start_old, H + 1, p):
                        affected.add(x)
                        
                start_new = cand_r if cand_r > 0 else p
                if start_new <= H:
                    for x in range(start_new, H + 1, p):
                        affected.add(x)
                        
                delta_pen = 0
                for x in affected:
                    old_c = counts[x]
                    if x % p == old_r:
                        new_c = old_c - 1
                    elif x % p == cand_r:
                        new_c = old_c + 1
                    else:
                        new_c = old_c
                            
                    old_def = max(0, 2 - old_c)
                    new_def = max(0, 2 - new_c)
                    delta_pen += new_def**2 - old_def**2
                    
                if delta_pen < best_delta:
                    best_delta = delta_pen
                    best_p_idx = i
                    best_new_r = cand_r
                    
            if best_p_idx != -1:
                # Apply best move
                p = primes[best_p_idx]
                old_r = residues[best_p_idx]
                new_r = best_new_r
                
                start_old = old_r if old_r > 0 else p
                for x in range(start_old, H + 1, p):
                    counts[x] -= 1
                start_new = new_r if new_r > 0 else p
                for x in range(start_new, H + 1, p):
                    counts[x] += 1
                
                residues[best_p_idx] = new_r
                current_pen += best_delta
                
                # Update L
                # Just recompute L
                L = 0
                for x in range(1, H + 1):
                    if counts[x] >= 2:
                        L = x
                    else:
                        break
                if L > best_L:
                    best_L = L
                    best_res = residues[:]
                    
    # Final check with best_res
    # We should verify best_res
    return best_res
