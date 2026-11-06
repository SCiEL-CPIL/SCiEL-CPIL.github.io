import time
import numpy as np
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n ** 0.5) + 1):
            if is_prime[i]:
                for j in range(i * i, n + 1, i):
                    is_prime[j] = False
        return [i for i, prime in enumerate(is_prime) if prime]
        
    primes = sieve(1000)
    n_primes = len(primes)
    MAX_L = 100000
    coverage = np.zeros(MAX_L + 1, dtype=np.int32)
    
    residues = [random.randint(0, p - 1) for p in primes]
    
    for i, p in enumerate(primes):
        r = residues[i]
        coverage[r::p] += 1
        
    def get_L():
        invalid = np.where(coverage[1:] < 2)[0]
        if len(invalid) == 0:
            return MAX_L
        else:
            return int(invalid[0])
            
    current_L = get_L()
    best_L = current_L
    best_residues = residues.copy()
    
    T = 10.0
    cooling = 0.999
    min_T = 0.1
    
    exp = math.exp
    rand = random.random
    randint = random.randint
    
    steps = 0
    
    while time.time() < deadline:
        # 1. Identify Gap
        if current_L >= MAX_L:
            break
        G = current_L + 1
        
        # 2. Try Safe Moves
        safe_move_found = False
        # Try a few random candidates that don't cover G
        # We want primes where residues[i] != G % p
        # To speed up, we can just pick random indices and check
        for _ in range(10): # Try up to 10 candidates
            i = randint(0, n_primes - 1)
            p = primes[i]
            r_old = residues[i]
            r_new = G % p
            
            if r_old == r_new:
                continue
            
            # Check safety: 
            # Decrease at x == r_old (mod p) for x in [1, G-1]
            # We need min coverage in that set to be >= 3
            # Find first x in [1, G-1] such that x % p == r_old
            # x = r_old + k*p. 
            # We want 1 <= x < G.
            
            # If r_old == 0, x can be p, 2p...
            # If r_old > 0, x can be r_old, r_old+p...
            
            # Calculate start index for the slice in coverage[1:G]
            # We are checking coverage[x] for x in [1, G-1]
            
            # The values in coverage array are indexed by x.
            # We need to check indices x such that 1 <= x < G and x % p == r_old.
            
            # If r_old == 0:
            #   x must be multiple of p. x = p, 2p...
            #   First x is p (if p < G).
            # If r_old > 0:
            #   x = r_old, r_old+p...
            #   First x is r_old (if r_old < G).
            
            first_x = r_old
            if first_x == 0:
                first_x = p
            elif first_x >= G:
                # No x in [1, G-1] is affected by the decrease?
                # If r_old >= G, then the first x is r_old which is >= G.
                # So no decrease in [1, G-1]. Move is safe!
                # Wait, if r_old < G, there are decreases.
                # If r_old >= G, the first decrease is at r_old >= G, so [1, G-1] is safe.
                # But we also need to ensure we don't break coverage at G? 
                # No, we are adding to G.
                # So if r_old >= G, the move is always safe with respect to prefix [1, G-1].
                # Let's verify.
                # Decrease at r_old, r_old+p...
                # If r_old >= G, then all decreases are at x >= G.
                # So coverage[1:G] is unchanged.
                # Thus min coverage in [1, G-1] is unchanged.
                # If it was >= 2, it is still >= 2.
                # So YES, if r_old >= G, the move is safe.
                
                # Apply move
                coverage[r_old::p] -= 1
                coverage[r_new::p] += 1
                new_L = get_L()
                if new_L > current_L:
                    current_L = new_L
                    residues[i] = r_new
                    if current_L > best_L:
                        best_L = current_L
                        best_residues = residues.copy()
                else:
                    # Revert if not improved? 
                    # Actually, if safe, it shouldn't hurt L. 
                    # But it might not improve L if G was already covered by others? 
                    # No, G was the first gap, so coverage[G] was < 2.
                    # We increased coverage[G] by 1.
                    # If coverage[G] was 1, it becomes 2. Gap moves.
                    # If coverage[G] was 0, it becomes 1. Gap might stay or move if G+1 was also gap?
                    # get_L finds the FIRST gap.
                    # If we fix G, L increases to at least G.
                    # So new_L should be >= G.
                    # If new_L == current_L, it means G was not the only issue? 
                    # No, G was the first gap. So L < G.
                    # If we fix G, L >= G.
                    # So new_L > current_L is expected.
                    # If new_L == current_L, something is wrong or G wasn't the first gap?
                    # get_L returns the first index < 2.
                    # So if we increase coverage at G, and G was the first < 2,
                    # then either G becomes >= 2 (L increases) or it was already >= 2 (impossible).
                    # So L must increase.
                    
                    # However, if we made a mistake in safety check, we might have broken a previous x.
                    # But we determined it was safe.
                    # So we accept it.
                    residues[i] = r_new
                safe_move_found = True
                break
            else:
                # r_old < G
                # Check min coverage in [first_x, G-1] with step p
                # Slice: coverage[first_x : G : p]
                # We need min >= 3
                if len(coverage[first_x : G : p]) > 0:
                    min_cov = np.min(coverage[first_x : G : p])
                    if min_cov >= 3:
                        # Safe
                        coverage[r_old::p] -= 1
                        coverage[r_new::p] += 1
                        new_L = get_L()
                        current_L = new_L
                        residues[i] = r_new
                        if current_L > best_L:
                            best_L = current_L
                            best_residues = residues.copy()
                        safe_move_found = True
                        break
                else:
                    # No points affected? Should not happen if r_old < G and r_old >= 1
                    # If r_old == 0, first_x = p. If p >= G, handled above.
                    # If p < G, slice is not empty.
                    pass
        
        if not safe_move_found:
            # Kick
            i = randint(0, n_primes - 1)
            p = primes[i]
            old_r = residues[i]
            new_r = randint(0, p - 1)
            
            if old_r == new_r:
                continue
                
            coverage[old_r::p] -= 1
            coverage[new_r::p] += 1
            
            new_L = get_L()
            
            delta = new_L - current_L
            if delta >= 0:
                current_L = new_L
                residues[i] = new_r
                if current_L > best_L:
                    best_L = current_L
                    best_residues = residues.copy()
            else:
                prob = exp(delta / T)
                if rand() < prob:
                    current_L = new_L
                    residues[i] = new_r
                else:
                    coverage[new_r::p] -= 1
                    coverage[old_r::p] += 1
        
        T *= cooling
        if T < min_T:
            T = min_T
            
        steps += 1
        if steps % 100 == 0 and time.time() >= deadline:
            break
            
    return best_residues
