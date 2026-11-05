import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    best_count = float('inf')
    best_coeffs = None
    
    def square_count(coeffs_list):
        """Compute the square of a sparse polynomial and return (count, full_list)."""
        if not coeffs_list:
            return 0, []
        max_deg = max(e for e, c in coeffs_list if c != 0)
        square = [0] * (2 * max_deg + 1)
        for i, (ei, ci) in enumerate(coeffs_list):
            if ci == 0:
                continue
            for j, (ej, cj) in enumerate(coeffs_list):
                if cj == 0:
                    continue
                square[ei + ej] += ci * cj
        count = sum(1 for x in square if x != 0)
        return count, square
    
    def coeffs_from_pairs(pairs):
        """Convert list of (exp, coeff) to full coefficient list."""
        if not pairs:
            return []
        max_deg = max(e for e, c in pairs if c != 0)
        full = [0] * (max_deg + 1)
        for e, c in pairs:
            full[e] = c
        return full
    
    # Phase 1: Brute force signs for 12 consecutive terms (0..11)
    # This is fast: 2^12 = 4096 patterns, each square takes ~144 multiplications
    for bits in range(1 << 12):
        if time.time() > deadline:
            break
        coeffs = [1 if (bits >> i) & 1 else -1 for i in range(12)]
        square = [0] * 23
        for i in range(12):
            ci = coeffs[i]
            for j in range(12):
                square[i + j] += ci * coeffs[j]
        count = sum(1 for x in square if x != 0)
        if count < best_count:
            best_count = count
            best_coeffs = coeffs
            if best_count == 12:
                return best_coeffs
    
    # Phase 2: Try small coefficient sets {-2,-1,1,2} with consecutive exponents
    # Use structured search: fix some signs, vary others
    random.seed(123)
    while time.time() < deadline:
        coeffs = [random.choice([-2, -1, 1, 2]) for _ in range(12)]
        square = [0] * 23
        for i in range(12):
            ci = coeffs[i]
            for j in range(12):
                square[i + j] += ci * coeffs[j]
        count = sum(1 for x in square if x != 0)
        if count < best_count:
            best_count = count
            best_coeffs = coeffs
            if best_count == 12:
                return best_coeffs
    
    # Phase 3: Random sparse exponents with small coefficients
    # Try 12 terms with exponents chosen from a larger range
    while time.time() < deadline:
        # Choose 12 distinct exponents from 0..30
        exponents = random.sample(range(31), 12)
        exponents.sort()
        coeffs = [random.choice([-2, -1, 1, 2]) for _ in range(12)]
        
        max_deg = exponents[-1]
        square = [0] * (2 * max_deg + 1)
        for i in range(12):
            ei = exponents[i]
            ci = coeffs[i]
            for j in range(12):
                ej = exponents[j]
                cj = coeffs[j]
                square[ei + ej] += ci * cj
        
        count = sum(1 for x in square if x != 0)
        if count < best_count:
            best_count = count
            full_coeffs = [0] * (max_deg + 1)
            for e, c in zip(exponents, coeffs):
                full_coeffs[e] = c
            best_coeffs = full_coeffs
            if best_count == 12:
                return best_coeffs
    
    # Phase 4: Targeted search - try to construct polynomials with specific structures
    # e.g., palindromic or anti-palindromic patterns
    while time.time() < deadline:
        # Generate random patterns with symmetry
        n = 12
        base = [random.choice([-1, 1]) for _ in range(n // 2)]
        if random.random() < 0.5:
            # Palindromic
            coeffs = base + base[::-1] if n % 2 == 0 else base + [1] + base[::-1]
            coeffs = coeffs[:n]
        else:
            # Anti-palindromic
            coeffs = base + [-b for b in base[::-1]] if n % 2 == 0 else base + [0] + [-b for b in base[::-1]]
            # Remove zeros to maintain 12 nonzero terms
            nonzero = [(i, c) for i, c in enumerate(coeffs) if c != 0]
            if len(nonzero) < 12:
                continue
            # Map to consecutive exponents
            coeffs = [c for _, c in nonzero[:12]]
        
        if len(coeffs) != 12 or any(c == 0 for c in coeffs):
            continue
            
        square = [0] * (2 * 11 + 1)
        for i in range(12):
            ci = coeffs[i]
            for j in range(12):
                square[i + j] += ci * coeffs[j]
        count = sum(1 for x in square if x != 0)
        if count < best_count:
            best_count = count
            best_coeffs = coeffs
            if best_count == 12:
                return best_coeffs
    
    # Phase 5: Local optimization - perturb best solution
    if best_coeffs is not None:
        max_deg = len(best_coeffs) - 1
        while time.time() < deadline:
            # Perturb one coefficient
            new_coeffs = best_coeffs[:]
            idx = random.randint(0, len(new_coeffs) - 1)
            old_val = new_coeffs[idx]
            if old_val == 0:
                new_coeffs[idx] = random.choice([-2, -1, 1, 2])
            else:
                new_coeffs[idx] = old_val + random.choice([-2, -1, 1, 2])
                if abs(new_coeffs[idx]) > 4 or new_coeffs[idx] == 0:
                    new_coeffs[idx] = random.choice([-2, -1, 1, 2])
            
            # Check exactly 12 nonzero terms
            nz = sum(1 for c in new_coeffs if c != 0)
            if nz != 12:
                continue
            
            max_deg = len(new_coeffs) - 1
            square = [0] * (2 * max_deg + 1)
            for i in range(len(new_coeffs)):
                ci = new_coeffs[i]
                if ci == 0:
                    continue
                for j in range(len(new_coeffs)):
                    cj = new_coeffs[j]
                    if cj == 0:
                        continue
                    square[i + j] += ci * cj
            count = sum(1 for x in square if x != 0)
            if count < best_count:
                best_count = count
                best_coeffs = new_coeffs
                if best_count == 12:
                    return best_coeffs
            elif count == best_count:
                # Keep as tie (could help further perturbation)
                if random.random() < 0.3:
                    best_coeffs = new_coeffs
    
    # Final validation
    if best_coeffs is None:
        best_coeffs = [1] * 12
    
    # Ensure exactly 12 nonzero terms and valid range
    nz_count = sum(1 for c in best_coeffs if c != 0)
    if nz_count != 12:
        # Fallback to all ones
        best_coeffs = [1] * 12
    
    # Ensure length <= 41
    if len(best_coeffs) > 41:
        best_coeffs = best_coeffs[:41]
        nz_count = sum(1 for c in best_coeffs if c != 0)
        if nz_count != 12:
            best_coeffs = [1] * 12
    
    # Ensure nonzero final coefficient
    if best_coeffs[-1] == 0:
        best_coeffs[-1] = 1
    
    # Ensure entries in [-64, 64]
    best_coeffs = [max(-64, min(64, c)) for c in best_coeffs]
    
    return best_coeffs

