"""Improved program for odd covering systems with moduli dividing 51975."""
import time
import random

# EVOLVE-BLOCK-START
def construct():
    base = 51975
    deadline = time.time() + 115  # safety margin before 120s limit
    
    # Precompute all odd divisors of 51975 > 1, sorted in increasing order
    divisors = [d for d in range(3, base + 1, 2) if base % d == 0]
    n_div = len(divisors)
    
    # Greedy initialization using bytearray (fast)
    covered = bytearray(base)
    moduli = list(divisors)
    residues = [0] * n_div
    
    for i, m in enumerate(moduli):
        if time.time() > deadline:
            break
        best_r = 0
        best_count = -1
        for r in range(m):
            count = covered[r::m].count(0)
            if count > best_count:
                best_count = count
                best_r = r
        residues[i] = best_r
        covered[best_r::m] = b'\x01' * len(range(best_r, base, m))
    
    current_uncovered = covered.count(0)
    best_uncovered = current_uncovered
    best_residues = residues[:]
    
    # Fast local search: single-residue changes with efficient delta computation
    # Precompute for each modulus the positions for each residue class
    # But that's too much memory, so we compute on the fly
    
    # For efficient delta: when changing modulus i from old_r to new_r,
    # positions that change: old positions become uncovered (if not covered by others),
    # new positions become covered (if not already covered by others)
    # We need to track coverage counts per position to know if "others" cover it
    
    # Alternative: use a coverage count array
    cov_count = bytearray(base)  # how many moduli cover each position
    for i, m in enumerate(moduli):
        r = residues[i]
        for pos in range(r, base, m):
            cov_count[pos] += 1
    
    def apply_change(i, old_r, new_r):
        """Apply a residue change and return delta in uncovered count."""
        m = moduli[i]
        old_pos = list(range(old_r, base, m))
        new_pos = list(range(new_r, base, m))
        
        # Remove old coverage
        for pos in old_pos:
            cov_count[pos] -= 1
        # Add new coverage
        for pos in new_pos:
            cov_count[pos] += 1
        
        # Delta = (positions that went from 0 to 1) - (positions that went from 1 to 0)
        # Actually: delta = new_uncovered - old_uncovered
        # = (positions with cov_count==0 after) - (positions with cov_count==0 before)
        # We can compute this from the changes
        
        # Positions that were uncovered before but are covered now: -1 to uncovered
        # Positions that were covered before but are uncovered now: +1 to uncovered
        delta = 0
        for pos in old_pos:
            if cov_count[pos] == 0:
                delta += 1  # became uncovered
        for pos in new_pos:
            if cov_count[pos] == 1:
                delta -= 1  # was uncovered, now covered
        
        return delta
    
    def revert_change(i, old_r, new_r):
        """Revert a residue change."""
        m = moduli[i]
        old_pos = list(range(old_r, base, m))
        new_pos = list(range(new_r, base, m))
        
        for pos in new_pos:
            cov_count[pos] -= 1
        for pos in old_pos:
            cov_count[pos] += 1
    
    # Local search with single-residue moves
    temp = 50.0
    cooling = 0.995
    no_improve = 0
    max_no_improve = 2000
    
    while time.time() < deadline - 10:
        if temp < 1.0:
            break
        if no_improve > max_no_improve:
            break
        
        # Pick a random modulus and try a new residue
        i = random.randint(0, n_div - 1)
        m = moduli[i]
        old_r = residues[i]
        new_r = random.randint(0, m - 1)
        
        if new_r == old_r:
            continue
        
        delta = apply_change(i, old_r, new_r)
        
        if delta < 0 or random.random() < pow(2.718281828, -delta / temp):
            residues[i] = new_r
            current_uncovered += delta
            no_improve = 0
            
            if current_uncovered < best_uncovered:
                best_uncovered = current_uncovered
                best_residues = residues[:]
        else:
            revert_change(i, old_r, new_r)
            no_improve += 1
        
        temp *= cooling
    
    # Final polish: try all single-residue changes that improve
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        for i in range(n_div):
            m = moduli[i]
            old_r = residues[i]
            for new_r in range(m):
                if new_r == old_r:
                    continue
                delta = apply_change(i, old_r, new_r)
                if delta < 0:
                    residues[i] = new_r
                    current_uncovered += delta
                    improved = True
                    if current_uncovered < best_uncovered:
                        best_uncovered = current_uncovered
                        best_residues = residues[:]
                else:
                    revert_change(i, old_r, new_r)
    
    # Return the best solution found
    classes = [[moduli[i], best_residues[i]] for i in range(n_div)]
    return classes
# EVOLVE-BLOCK-END
