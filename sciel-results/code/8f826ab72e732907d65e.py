import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin before 150s kill
    
    # Generate primes up to 1000
    primes = [p for p in range(2, 1001) if all(p % q for q in range(2, int(p ** 0.5) + 1))]
    n_primes = len(primes)
    
    # Initial assignment: spread residues to cover early integers
    residues = [(i + 1) % p for i, p in enumerate(primes)]
    
    def compute_L(residues, max_check=100000):
        """Compute the largest L such that every 1..L has >= 2 covers."""
        coverage = np.zeros(max_check + 1, dtype=np.int32)
        for i, p in enumerate(primes):
            r = residues[i]
            # Add 1 to positions r, r+p, r+2p, ... up to max_check
            # Only add if r > 0 (since we care about 1..max_check)
            start = r if r > 0 else p
            if start <= max_check:
                coverage[start::p] += 1
        # Find the first position with coverage < 2
        for L in range(1, max_check + 1):
            if coverage[L] < 2:
                return L - 1
        return max_check
    
    def try_improve(residues, max_iter=5000):
        """Local search: try changing individual residues to improve L."""
        current_L = compute_L(residues)
        best_residues = residues[:]
        best_L = current_L
        improved = True
        iteration = 0
        while improved and iteration < max_iter:
            improved = False
            iteration += 1
            if time.time() > deadline - 5:
                break
            for i, p in enumerate(primes):
                if time.time() > deadline - 5:
                    break
                # Try all residues for small primes, sample for large
                if p <= 100:
                    candidates = range(p)
                else:
                    # Sample residues around the current one and some random ones
                    candidates = set()
                    candidates.add(residues[i])
                    candidates.add((residues[i] + 1) % p)
                    candidates.add((residues[i] - 1) % p)
                    candidates.add(0)
                    candidates.add(1)
                    candidates.add((p - 1) % p)
                    # Add some random candidates
                    for _ in range(min(20, p)):
                        candidates.add(np.random.randint(0, p))
                    candidates = list(candidates)
                
                best_new_L = best_L
                best_new_r = best_residues[i]
                for r in candidates:
                    if r == best_residues[i]:
                        continue
                    test_residues = best_residues[:]
                    test_residues[i] = r
                    L_val = compute_L(test_residues)
                    if L_val > best_new_L:
                        best_new_L = L_val
                        best_new_r = r
                    if time.time() > deadline - 5:
                        break
                
                if best_new_L > best_L:
                    best_residues[i] = best_new_r
                    best_L = best_new_L
                    improved = True
                    if time.time() > deadline - 5:
                        break
        
        return best_residues, best_L
    
    # Run initial local search
    residues, current_L = try_improve(residues)
    
    # Try multiple restarts with different initializations
    best_overall = residues[:]
    best_L_overall = current_L
    
    num_restarts = 5
    for restart in range(num_restarts):
        if time.time() > deadline - 10:
            break
        
        # Different initialization strategies
        if restart == 0:
            # Original spread
            init_residues = [(i + 1) % p for i, p in enumerate(primes)]
        elif restart == 1:
            # Cluster early primes on early numbers
            init_residues = []
            for i, p in enumerate(primes):
                # Try to cover number i+1 if possible
                if i + 1 < p:
                    init_residues.append(i + 1)
                else:
                    init_residues.append((i + 1) % p)
        elif restart == 2:
            # Random
            init_residues = [np.random.randint(0, p) for p in primes]
        elif restart == 3:
            # All zeros
            init_residues = [0] * n_primes
        else:
            # Mix
            init_residues = [0 if i % 2 == 0 else (i + 1) % p for i, p in enumerate(primes)]
        
        improved_residues, improved_L = try_improve(init_residues, max_iter=3000)
        if improved_L > best_L_overall:
            best_L_overall = improved_L
            best_overall = improved_residues
    
    return best_overall
