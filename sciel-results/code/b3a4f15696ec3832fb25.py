"""
Odd covering systems with moduli dividing 51975.
Uses bitset-based greedy maximum coverage with local search refinement.
"""
import time
import numpy as np

BASE = 51975  # 3^3 * 5^2 * 7 * 11

# Precompute all divisors > 1 of BASE
_DIVISORS = None

def _get_divisors():
    global _DIVISORS
    if _DIVISORS is None:
        divs = []
        # BASE = 51975
        # Prime factorization: 3^3 * 5^2 * 7 * 11
        # Generate all divisors
        def gen_divisors(n, primes, exponents):
            if not primes:
                return [1]
            result = []
            for d in gen_divisors(n // primes[0], primes[1:], exponents[1:]):
                for e in range(exponents[0] + 1):
                    result.append(d * primes[0] ** e)
            return result
        
        # 51975 = 3^3 * 5^2 * 7^1 * 11^1
        all_divs = gen_divisors(BASE, [3, 5, 7, 11], [3, 2, 1, 1])
        _DIVISORS = sorted([d for d in all_divs if d > 1])
    return _DIVISORS

def _build_mask(m, r):
    """Build a boolean numpy array representing residues covered by class (m, r)."""
    mask = np.zeros(BASE, dtype=np.uint8)
    mask[r::m] = 1
    return mask

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 150s limit
    
    divisors = _get_divisors()
    n_divs = len(divisors)
    
    # Precompute masks for all (m, r) pairs
    # For each divisor m, there are m possible residues
    # Total masks = sum of all divisors > 1
    all_masks = []  # list of (m, r, mask)
    mask_by_mod = {}  # m -> list of (r, mask)
    
    for m in divisors:
        mask_by_mod[m] = []
        for r in range(m):
            mask = _build_mask(m, r)
            all_masks.append((m, r, mask))
            mask_by_mod[m].append((r, mask))
    
    # Greedy maximum coverage
    # uncovered is a boolean array of length BASE
    uncovered = np.ones(BASE, dtype=np.uint8)
    covered_count = 0
    selected = []  # list of (m, r, mask)
    used_moduli = set()
    
    # For efficiency, precompute popcount of each mask
    # Actually, we need intersection with uncovered, so we compute dynamically
    
    # Greedy: repeatedly pick the mask that covers the most uncovered residues
    # To speed this up, we can use the fact that smaller moduli cover more residues
    
    # Sort divisors by size (ascending) - smaller moduli have more residues per class
    divisors_sorted = sorted(divisors)
    
    # First pass: greedy selection
    # For each modulus (in order of increasing size), pick the best residue
    # But we should consider all moduli at each step
    
    # Better approach: at each step, find the (m, r) pair that maximizes
    # popcount(uncovered & mask)
    
    # To make this efficient, we'll iterate through all masks
    # But that's sum(m for m in divisors) which could be large
    
    # Let's estimate: sum of divisors of 51975
    # sigma(51975) = sigma(3^3)*sigma(5^2)*sigma(7)*sigma(11)
    # = (1+3+9+27)*(1+5+25)*(1+7)*(1+11)
    # = 40 * 31 * 8 * 12 = 119040
    # So about 119k masks total. Each mask is 51975 bytes.
    # That's about 6GB which is too much to precompute all at once.
    
    # Instead, let's use a different strategy:
    # Use numpy arrays and compute intersections on the fly, but be smart about it.
    
    # Alternative: use the structure. For a given modulus m, the mask for residue r
    # is periodic with period m. We can compute the intersection with uncovered
    # more efficiently.
    
    # Let's use a hybrid approach:
    # 1. Greedy with a limited set of candidates at each step
    # 2. Local search to refine
    
    # Actually, let's just do the greedy properly with numpy
    # The key insight: for each m, we can compute all m residues' intersections
    # with uncovered in one vectorized operation
    
    # For modulus m, uncovered[r::m] gives the uncovered residues in that class
    # We want to find r that maximizes sum(uncovered[r::m])
    
    # This can be done by reshaping uncovered appropriately
    
    def best_residue_for_mod(m, uncovered_arr):
        """Find the residue r that covers the most uncovered residues for modulus m."""
        # uncovered_arr has length BASE
        # We want to find r in [0, m) that maximizes sum(uncovered_arr[r::m])
        
        # Reshape: create a 2D array where column j contains uncovered_arr[j],
        # uncovered_arr[j+m], uncovered_arr[j+2m], ...
        # Actually, we can use numpy's advanced indexing
        
        # For each r, the covered positions are r, r+m, r+2m, ...
        # The number of such positions is either BASE//m or BASE//m + 1
        
        # Let's compute this efficiently
        n_full = BASE // m
        remainder = BASE % m
        
        # For r < remainder, there are n_full + 1 elements
        # For r >= remainder, there are n_full elements
        
        # We need sum of uncovered at positions r, r+m, r+2m, ...
        
        # Create indices: for r in range(m), indices = np.arange(r, BASE, m)
        # But doing this for each r is O(m * BASE/m) = O(BASE) per m, total O(sum(m)) which is fine
        
        # More efficient: reshape uncovered into (n_full, m) + extra
        # Actually, let's just do it directly but vectorized
        
        # uncovered_arr[r::m] for all r at once:
        # We can create a 2D array of shape (m, max_len) where max_len = (BASE - r + m - 1) // m
        
        # Simpler approach:
        best_r = 0
        best_count = -1
        
        # Use numpy to compute all sums at once
        # Create an array of indices for each residue class
        # This might be memory intensive for large m, but m <= 51975
        
        if m <= 1000:
            # Direct computation
            for r in range(m):
                cnt = int(np.sum(uncovered_arr[r::m]))
                if cnt > best_count:
                    best_count = cnt
                    best_r = r
        else:
            # For large m, each residue class has few elements
            # BASE // m is small
            n_per_class = BASE // m
            if n_per_class == 1:
                # Each residue class has exactly 1 or 2 elements
                # For r < remainder: 2 elements, else 1
                # Just pick the r with most uncovered
                # Actually with n_per_class=1, most have 1 element, some have 2
                # Let's just iterate
                for r in range(m):
                    cnt = int(np.sum(uncovered_arr[r::m]))
                    if cnt > best_count:
                        best_count = cnt
                        best_r = r
            else:
                for r in range(m):
                    cnt = int(np.sum(uncovered_arr[r::m]))
                    if cnt > best_count:
                        best_count = cnt
                        best_r = r
        
        return best_r, best_count
    
    # Greedy construction
    # At each step, try all unused moduli and pick the one with best gain
    # To limit computation, we'll do multiple rounds
    
    uncovered_arr = np.ones(BASE, dtype=np.uint8)
    selected_classes = []
    used_moduli = set()
    
    # Number of greedy steps: use all divisors
    for step in range(n_divs):
        if time.time() > deadline:
            break
        
        best_m = None
        best_r = None
        best_gain = -1
        
        # Try each unused modulus
        for m in divisors_sorted:
            if m in used_moduli:
                continue
            
            r, gain = best_residue_for_mod(m, uncovered_arr)
            if gain > best_gain:
                best_gain = gain
                best_m = m
                best_r = r
        
        if best_m is None or best_gain <= 0:
            break
        
        # Add this class
        used_moduli.add(best_m)
        selected_classes.append((best_m, best_r))
        uncovered_arr[best_r::best_m] = 0
    
    current_score = int(np.sum(uncovered_arr))
    
    # Local search: try to improve by changing residues or swapping moduli
    # Phase 1: For each selected class, try all residues for its modulus
    improved = True
    max_local_iters = 20
    local_iter = 0
    
    while improved and local_iter < max_local_iters and time.time() < deadline:
        improved = False
        local_iter += 1
        
        for i in range(len(selected_classes)):
            if time.time() > deadline:
                break
            
            m, r_old = selected_classes[i]
            
            # Compute uncovered without this class
            # Rebuild uncovered from all other classes
            unc_without = np.ones(BASE, dtype=np.uint8)
            for j in range(len(selected_classes)):
                if j == i:
                    continue
                mj, rj = selected_classes[j]
                unc_without[rj::mj] = 0
            
            # Find best residue for m given unc_without
            best_r = r_old
            best_gain = -1
            for r in range(m):
                gain = int(np.sum(unc_without[r::m]))
                if gain > best_gain:
                    best_gain = gain
                    best_r = r
            
            if best_r != r_old:
                # Check if this actually improves the total score
                # New uncovered = unc_without & ~mask(best_m, best_r)
                new_unc = unc_without.copy()
                new_unc[best_r::m] = 0
                new_score = int(np.sum(new_unc))
                
                if new_score < current_score:
                    selected_classes[i] = (m, best_r)
                    current_score = new_score
                    uncovered_arr = new_unc
                    improved = True
    
    # Phase 2: Try adding unused moduli
    for m in divisors_sorted:
        if time.time() > deadline:
            break
        if m in used_moduli:
            continue
        
        # Find best residue for m
        best_r = 0
        best_gain = -1
        for r in range(m):
            gain = int(np.sum(uncovered_arr[r::m]))
            if gain > best_gain:
                best_gain = gain
                best_r = r
        
        if best_gain > 0:
            # Check if adding improves
            new_unc = uncovered_arr.copy()
            new_unc[best_r::m] = 0
            new_score = int(np.sum(new_unc))
            if new_score < current_score:
                used_moduli.add(m)
                selected_classes.append((m, best_r))
                current_score = new_score
                uncovered_arr = new_unc
    
    # Phase 3: More local search after additions
    improved = True
    local_iter = 0
    while improved and local_iter < max_local_iters and time.time() < deadline:
        improved = False
        local_iter += 1
        
        for i in range(len(selected_classes)):
            if time.time() > deadline:
                break
            
            m, r_old = selected_classes[i]
            
            unc_without = np.ones(BASE, dtype=np.uint8)
            for j in range(len(selected_classes)):
                if j == i:
                    continue
                mj, rj = selected_classes[j]
                unc_without[rj::mj] = 0
            
            best_r = r_old
            best_gain = -1
            for r in range(m):
                gain = int(np.sum(unc_without[r::m]))
                if gain > best_gain:
                    best_gain = gain
                    best_r = r
            
            if best_r != r_old:
                new_unc = unc_without.copy()
                new_unc[best_r::m] = 0
                new_score = int(np.sum(new_unc))
                
                if new_score < current_score:
                    selected_classes[i] = (m, best_r)
                    current_score = new_score
                    uncovered_arr = new_unc
                    improved = True
    
    # Convert to required format
    result = [[m, r] for m, r in selected_classes]
    return result
