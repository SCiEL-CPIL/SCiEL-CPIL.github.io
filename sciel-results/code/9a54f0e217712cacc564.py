import time
import itertools
from collections import defaultdict

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit

    # Fallback: Simple polynomial, score is likely 21
    # (1 + x + ... + x^11)^2 has 21 non-zero terms
    fallback = [1] * 12
    fallback_score = 21

    best_score = fallback_score
    best_candidate = fallback

    def compute_square_nonzeros(exps, coeffs):
        """
        Efficiently compute the number of non-zero terms in P(x)^2.
        exps: list of 12 exponents
        coeffs: list of 12 coefficients
        """
        square = defaultdict(int)
        n = len(exps)
        for i in range(n):
            ei = exps[i]
            ci = coeffs[i]
            # Square term
            square[2 * ei] += ci * ci
            # Cross terms
            for j in range(i + 1, n):
                ej = exps[j]
                cj = coeffs[j]
                square[ei + ej] += 2 * ci * cj
        
        # Count non-zeros
        count = 0
        for val in square.values():
            if val != 0:
                count += 1
        return count

    def try_candidate(exps, coeffs):
        """
        Check if candidate is valid and update best if better.
        """
        nonlocal best_score, best_candidate
        
        # Validation
        if len(exps) != 12 or len(coeffs) != 12:
            return
        if len(set(exps)) != 12:
            return
        if max(exps) > 40:
            return
        if any(abs(c) > 64 for c in coeffs):
            return
        if any(c == 0 for c in coeffs): # Must be exactly 12 non-zero terms in P
            return
        
        score = compute_square_nonzeros(exps, coeffs)
        
        if score < best_score:
            best_score = score
            # Build dense list
            max_deg = max(exps)
            dense = [0] * (max_deg + 1)
            for e, c in zip(exps, coeffs):
                dense[e] = c
            # Ensure final coeff is non-zero (guaranteed by max(exps))
            if dense[-1] != 0:
                best_candidate = dense

    def optimize_signs(exps, max_iter=200):
        """
        Local search for best sign assignment {-1, 1} for given exponents.
        """
        if time.time() > deadline:
            return
        
        # Initialize with random or all 1s
        # Try a few random starts
        import random
        random.seed(42) # Fixed seed for determinism
        
        best_local_score = float('inf')
        best_local_signs = [1] * 12

        for _ in range(10): # 10 random restarts
            if time.time() > deadline:
                break
            signs = [1 if random.random() > 0.5 else -1 for _ in range(12)]
            current_score = compute_square_nonzeros(exps, signs)
            
            for _ in range(max_iter):
                if time.time() > deadline:
                    break
                improved = False
                # Single flip
                for i in range(12):
                    old = signs[i]
                    signs[i] = -signs[i]
                    new_score = compute_square_nonzeros(exps, signs)
                    if new_score < current_score:
                        current_score = new_score
                        improved = True
                    else:
                        signs[i] = old
                
                if not improved:
                    break
            
            if current_score < best_local_score:
                best_local_score = current_score
                best_local_signs = signs[:]
        
        if best_local_score < best_score:
            try_candidate(exps, best_local_signs)

    def optimize_coeffs(exps, max_iter=100):
        """
        Local search for small integer coefficients.
        """
        if time.time() > deadline:
            return
        
        # Start with best signs from optimize_signs? Or random small ints.
        # Let's start with all 1s and -1s, then allow changes to +-2, +-3
        
        import random
        random.seed(99)
        
        best_local_score = float('inf')
        best_local_coeffs = [1] * 12
        
        for _ in range(5):
            if time.time() > deadline:
                break
            coeffs = [random.choice([-2, -1, 1, 2]) for _ in range(12)]
            current_score = compute_square_nonzeros(exps, coeffs)
            
            for _ in range(max_iter):
                if time.time() > deadline:
                    break
                improved = False
                i = random.randint(0, 11)
                old_val = coeffs[i]
                new_val = random.choice([-3, -2, -1, 1, 2, 3])
                if new_val == old_val:
                    continue
                coeffs[i] = new_val
                new_score = compute_square_nonzeros(exps, coeffs)
                
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                else:
                    coeffs[i] = old_val
                
                if not improved:
                    # If no improvement in a step, break to save time? 
                    # No, just continue random search.
                    pass
            
            if current_score < best_local_score:
                best_local_score = current_score
                best_local_coeffs = coeffs[:]
        
        if best_local_score < best_score:
            try_candidate(exps, best_local_coeffs)

    # --- Generate Candidate Exponent Sets ---
    
    candidate_exps = []
    
    # 1. Arithmetic Progressions
    for start in range(0, 10):
        for step in range(1, 5):
            exps = [start + i * step for i in range(12)]
            if max(exps) <= 40:
                candidate_exps.append(exps)
    
    # 2. Two Interleaved APs
    # e.g., evens and odds scaled
    for s1 in range(0, 5):
        for d1 in range(1, 3):
            for s2 in range(5, 15):
                for d2 in range(1, 3):
                    exps1 = [s1 + i * d1 for i in range(6)]
                    exps2 = [s2 + i * d2 for i in range(6)]
                    exps = sorted(exps1 + exps2)
                    if len(set(exps)) == 12 and max(exps) <= 40:
                        candidate_exps.append(exps)
    
    # 3. Clustered Sets
    # Try to pack exponents into small ranges to force collisions
    for base in range(0, 10):
        for spread in range(1, 8):
            # Generate 12 distinct exponents in [base, base+spread]
            # This is only possible if spread >= 11
            # So let's try [base, base+11] with some gaps removed? No, must be distinct.
            # If spread < 11, we can't fit 12 distinct ints.
            # So min spread is 11.
            pass

    # 4. Random Sets with High Collision Potential
    import random
    random.seed(123)
    for _ in range(200):
        if time.time() > deadline:
            break
        # Generate 12 random exponents in [0, 40]
        exps = random.sample(range(0, 41), 12)
        exps.sort()
        candidate_exps.append(exps)

    # Deduplicate
    seen = set()
    unique_exps = []
    for exps in candidate_exps:
        key = tuple(exps)
        if key not in seen:
            seen.add(key)
            unique_exps.append(exps)

    # Sort candidates? 
    # APs with smaller step might have more collisions?
    # Actually, smaller step = denser = more collisions.
    # Let's just iterate.
    
    # Run optimization on top candidates
    # Prioritize APs with step 1 and 2
    unique_exps.sort(key=lambda e: max(e) - min(e)) # Smaller span first?
    
    for exps in unique_exps:
        if time.time() > deadline:
            break
            
        # Try signs first
        optimize_signs(exps)
        
        # If we found a very good score (e.g., < 15), try coeff optimization
        # But coeff optimization is slower.
        if best_score <= 18 and time.time() < deadline - 10:
            optimize_coeffs(exps)
            
        # Early termination if we hit theoretical lower bound?
        # Theoretical lower bound is 12 (if all cross terms cancel).
        # Unlikely to reach 12, but 13-15 is possible.
        if best_score <= 13:
            break

    return best_candidate

# EVOLVE-BLOCK-END
