"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin before 150s limit
    
    # Generate primes up to 1000
    primes = [p for p in range(2, 1001) if all(p % q for q in range(2, int(p ** 0.5) + 1))]
    n_primes = len(primes)
    
    # Initial solution: use the seed approach
    # The seed formula is ((i + 2) // 2) % p
    residues = [((i + 2) // 2) % p for i, p in enumerate(primes)]
    
    MAX_X = 10000  # Increase search horizon to allow for larger L
    
    def compute_coverage(res):
        """Compute coverage count for each x in 1..MAX_X."""
        coverage = [0] * (MAX_X + 1)
        for i, p in enumerate(primes):
            r = res[i]
            # x = r, r+p, r+2p...
            # Note: x must be >= 1. If r==0, first x is p.
            start = r if r >= 1 else p
            for x in range(start, MAX_X + 1, p):
                coverage[x] += 1
        return coverage

    def find_L(coverage, start_idx=1):
        """Find the largest L such that coverage[x] >= 2 for all x in 1..L.
        Starts scanning from start_idx. Assumes coverage[1..start_idx-1] is already valid (>=2).
        Returns the new L."""
        L = start_idx - 1
        for x in range(start_idx, MAX_X + 1):
            if coverage[x] < 2:
                return L
        return MAX_X

    # Initial coverage and score
    coverage = compute_coverage(residues)
    best_score = find_L(coverage)
    best_residues = residues[:]
    
    # Simulated Annealing
    current_residues = residues[:]
    current_score = best_score
    current_coverage = coverage
    
    # Temperature parameters
    T = 50.0  # Start high to allow moves
    cooling = 0.999
    min_T = 0.1
    
    # To speed up score calculation, we track the current L
    # When we make a move, we know which indices are affected.
    
    while T > min_T:
        if time.time() > deadline:
            break
        
        # Pick a random prime to change
        i = random.randint(0, n_primes - 1)
        p = primes[i]
        old_r = current_residues[i]
        
        # Generate a new residue
        # Bias towards residues that might help? 
        # For now, uniform random is robust.
        new_r = random.randint(0, p - 1)
        
        if new_r == old_r:
            continue
        
        # Determine affected indices
        # Old indices: old_r, old_r+p, ...
        # New indices: new_r, new_r+p, ...
        
        # We need to update coverage for these indices.
        # To find the new L efficiently:
        # 1. Update coverage array.
        # 2. Check if any x <= current_score has coverage < 2.
        #    If yes, new_score < current_score. Scan from 1.
        #    If no, new_score >= current_score. Scan from current_score + 1.
        
        # Collect affected indices to check validity quickly
        # Actually, just updating the array is O(MAX_X/p).
        
        start_old = old_r if old_r >= 1 else p
        start_new = new_r if new_r >= 1 else p
        
        # Update coverage
        for x in range(start_old, MAX_X + 1, p):
            current_coverage[x] -= 1
        for x in range(start_new, MAX_X + 1, p):
            current_coverage[x] += 1
        
        # Determine new score
        # Check if current_score is still valid
        valid = True
        # We only need to check indices that were modified and are <= current_score
        # But modifying the array already happened.
        # Let's just check the modified indices <= current_score.
        # If any of them dropped below 2, we have a problem.
        
        # Note: An index could be in both start_old and start_new sequences?
        # No, old_r != new_r mod p, so the sets of indices are disjoint.
        
        # Check old indices (which lost coverage)
        for x in range(start_old, current_score + 1, p):
            if current_coverage[x] < 2:
                valid = False
                break
        
        if not valid:
            # Score decreased. Find new L from 1.
            new_score = find_L(current_coverage, 1)
        else:
            # Score might have increased. Check from current_score + 1.
            # But wait, if we didn't break, current_score is still valid.
            # We need to see if we can extend.
            new_score = find_L(current_coverage, current_score + 1)
        
        # Accept or reject
        if new_score >= current_score:
            current_residues[i] = new_r
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_residues = current_residues[:]
        else:
            delta = current_score - new_score
            # Accept with probability exp(-delta/T)
            if random.random() < math.exp(-delta / T):
                current_residues[i] = new_r
                current_score = new_score
            else:
                # Revert
                for x in range(start_new, MAX_X + 1, p):
                    current_coverage[x] -= 1
                for x in range(start_old, MAX_X + 1, p):
                    current_coverage[x] += 1
        
        T *= cooling
        
        # Optional: Restart if stuck? 
        # If we haven't improved in a long time, we might be stuck.
        # But SA with cooling should handle this.
        
    return best_residues
# EVOLVE-BLOCK-END
