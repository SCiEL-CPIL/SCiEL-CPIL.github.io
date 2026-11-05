"""Optimized residue assignment using structured greedy + aggressive local search."""
# EVOLVE-BLOCK-START
import math
import random
import time

def construct():
    # Generate primes <= 1000
    primes = []
    for p in range(2, 1001):
        is_prime = True
        for q in range(2, int(p ** 0.5) + 1):
            if p % q == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(p)
    
    n_primes = len(primes)
    L_max = 100000  # Cap for scoring
    
    def find_L(coverage, L_max):
        for x in range(1, min(L_max + 1, 100001)):
            if coverage[x] < 2:
                return x - 1
        return min(L_max, 100000)
    
    # Phase 1: Structured initialization
    # Key insight: for each prime p, residue r covers positions r, r+p, r+2p, ...
    # We want to maximize coverage density in [1, L_target]
    # Strategy: process primes from smallest to largest, for each pick the residue
    # that maximizes the number of newly-covered (currently < 2) positions
    
    L_target = 2000
    residues = [0] * n_primes
    coverage = [0] * (L_target + 1)
    
    # Process primes in increasing order (small primes first, they have more hits)
    for i, p in enumerate(primes):
        best_r = 0
        best_score = -1
        for r in range(p):
            score = 0
            x = r if r > 0 else p
            while x <= L_target:
                if coverage[x] < 2:
                    score += 1
                x += p
            if score > best_score:
                best_score = score
                best_r = r
        residues[i] = best_r
        x = best_r if best_r > 0 else p
        while x <= L_target:
            coverage[x] += 1
            x += p
    
    current_L = find_L(coverage, L_target)
    best_L = current_L
    best_residues = residues[:]
    
    # Phase 2: Aggressive local search with multi-move capability
    start_time = time.time()
    no_improve = 0
    max_no_improve = 20000
    
    while time.time() - start_time < 115:
        if no_improve > max_no_improve:
            break
        
        # Find first gap
        gap_pos = None
        search_end = min(current_L + 200, L_target)
        for x in range(current_L + 1, search_end + 1):
            if coverage[x] < 2:
                gap_pos = x
                break
        
        if gap_pos is None:
            current_L = search_end
            if current_L > best_L:
                best_L = current_L
                best_residues = residues[:]
            no_improve = 0
            continue
        
        # Try to fix gap_pos by reassigning one prime
        # Prefer large primes (less disruption to prefix)
        # But also try small primes if large ones fail
        
        found_fix = False
        
        # First try: large primes (p > gap_pos/2, they cover at most 2 spots in [1, current_L])
        for i in range(n_primes - 1, -1, -1):  # Descending prime order
            p = primes[i]
            r_old = residues[i]
            r_new = gap_pos % p
            if r_old == r_new:
                continue
            
            # Check safety: no position in [1, current_L] with coverage==2 loses this prime
            safe = True
            x = r_old if r_old > 0 else p
            while x <= current_L:
                if coverage[x] == 2:
                    safe = False
                    break
                x += p
            
            if safe:
                # Apply move
                x = r_old if r_old > 0 else p
                while x <= L_target:
                    coverage[x] -= 1
                    x += p
                x = r_new if r_new > 0 else p
                while x <= L_target:
                    coverage[x] += 1
                    x += p
                residues[i] = r_new
                found_fix = True
                no_improve = 0
                break
        
        if not found_fix:
            # Try all primes including small ones with a more permissive check
            # Allow temporary degradation but recover
            best_move = None
            best_penalty = float('inf')
            
            for i in range(n_primes):
                p = primes[i]
                r_old = residues[i]
                r_new = gap_pos % p
                if r_old == r_new:
                    continue
                
                # Count how many positions in [1, current_L] would drop to 1
                penalty = 0
                x = r_old if r_old > 0 else p
                while x <= current_L:
                    if coverage[x] == 2:
                        penalty += 1
                    x += p
                
                if penalty < best_penalty:
                    best_penalty = penalty
                    best_move = (i, r_old, r_new)
            
            if best_move is not None and best_penalty <= 3:
                i, r_old, r_new = best_move
                # Apply the move
                x = r_old if r_old > 0 else primes[i]
                while x <= L_target:
                    coverage[x] -= 1
                    x += primes[i]
                x = r_new if r_new > 0 else primes[i]
                while x <= L_target:
                    coverage[x] += 1
                    x += primes[i]
                residues[i] = r_new
                no_improve = 0
                found_fix = True
        
        if not found_fix:
            # Random kick to escape local minimum
            # Try 3 random single-prime changes, keep the best
            best_kick = None
            best_kick_L = -1
            
            for _ in range(3):
                i = random.randint(0, n_primes - 1)
                p = primes[i]
                r_old = residues[i]
                r_new = random.randint(0, p - 1)
                if r_old == r_new:
                    continue
                
                # Apply tentatively
                x = r_old if r_old > 0 else p
                while x <= L_target:
                    coverage[x] -= 1
                    x += p
                x = r_new if r_new > 0 else p
                while x <= L_target:
                    coverage[x] += 1
                    x += p
                residues[i] = r_new
                
                kick_L = find_L(coverage, L_target)
                
                if kick_L > best_kick_L:
                    best_kick_L = kick_L
                    best_kick = (i, r_new, r_old)
                
                # Revert
                x = r_new if r_new > 0 else p
                while x <= L_target:
                    coverage[x] -= 1
                    x += p
                x = r_old if r_old > 0 else p
                while x <= L_target:
                    coverage[x] += 1
                    x += p
                residues[i] = r_old
            
            if best_kick is not None and best_kick_L > current_L:
                i, r_new, r_old = best_kick
                x = r_old if r_old > 0 else primes[i]
                while x <= L_target:
                    coverage[x] -= 1
                    x += primes[i]
                x = r_new if r_new > 0 else primes[i]
                while x <= L_target:
                    coverage[x] += 1
                    x += primes[i]
                residues[i] = r_new
                no_improve = 0
            else:
                no_improve += 1
        
        # Update current_L
        current_L = find_L(coverage, L_target)
        if current_L > best_L:
            best_L = current_L
            best_residues = residues[:]
    
    return best_residues
# EVOLVE-BLOCK-END
