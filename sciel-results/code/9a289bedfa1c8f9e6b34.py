import time
import itertools
from collections import defaultdict
import random

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

    def optimize_coeffs_sa(exps, max_time=10.0):
        """
        Simulated Annealing / Local Search for coefficients.
        """
        if time.time() > deadline:
            return
        
        # Initialize
        best_local_score = float('inf')
        best_local_coeffs = [1] * 12
        
        # Multiple restarts
        num_restarts = 5
        restart_time = max_time / num_restarts
        
        for r in range(num_restarts):
            if time.time() > deadline:
                break
            
            # Random init
            coeffs = [random.choice([-1, 1, -2, 2]) for _ in range(12)]
            current_score = compute_square_nonzeros(exps, coeffs)
            
            # Local Search with Perturbation (Iterated Local Search)
            local_start = time.time()
            while time.time() - local_start < restart_time and time.time() < deadline:
                improved = False
                # Try all single coefficient changes
                for i in range(12):
                    old_val = coeffs[i]
                    # Try neighbors
                    for delta in [-2, -1, 1, 2]:
                        new_val = old_val + delta
                        if -64 <= new_val <= 64 and new_val != 0:
                            coeffs[i] = new_val
                            new_score = compute_square_nonzeros(exps, coeffs)
                            if new_score < current_score:
                                current_score = new_score
                                improved = True
                            else:
                                coeffs[i] = old_val
                
                if not improved:
                    # Perturb: random change to escape local minima
                    if random.random() < 0.5:
                        i = random.randint(0, 11)
                        coeffs[i] = random.choice([-3, -2, -1, 1, 2, 3])
                        current_score = compute_square_nonzeros(exps, coeffs)
                    else:
                        break # If no improvement, stop this restart? Or continue?
                        # Let's break to save time for next restart if stuck.
                    if current_score < best_local_score:
                        best_local_score = current_score
                        best_local_coeffs = coeffs[:]
                else:
                    if current_score < best_local_score:
                        best_local_score = current_score
                        best_local_coeffs = coeffs[:]
        
        if best_local_score < best_score:
            try_candidate(exps, best_local_coeffs)

    # --- Generate Candidate Exponent Sets ---
    
    candidate_exps = []
    
    # Structure 1: 3 blocks of 4 (as per directive)
    # Try various gaps between blocks and steps within blocks
    for gap1 in range(1, 10):
        for gap2 in range(1, 10):
            for step in [1, 2, 3]:
                # Block 1: 0, step, 2*step, 3*step
                # Block 2: gap1, gap1+step, ...
                # Block 3: gap1+gap2, ...
                b1 = [i * step for i in range(4)]
                b2 = [gap1 + i * step for i in range(4)]
                b3 = [gap1 + gap2 + i * step for i in range(4)]
                exps = sorted(b1 + b2 + b3)
                if len(set(exps)) == 12 and max(exps) <= 40:
                    candidate_exps.append(tuple(exps))

    # Structure 2: 2 blocks of 6
    for gap in range(1, 15):
        for step in [1, 2]:
            b1 = [i * step for i in range(6)]
            b2 = [gap + i * step for i in range(6)]
            exps = sorted(b1 + b2)
            if len(set(exps)) == 12 and max(exps) <= 40:
                candidate_exps.append(tuple(exps))

    # Structure 3: Random sets with high collision potential
    # Use a deterministic seed for reproducibility
    random.seed(42)
    for _ in range(100):
        # Bias towards smaller numbers to increase collision probability in square
        exps = random.sample(range(0, 25), 12)
        exps.sort()
        candidate_exps.append(tuple(exps))
        
    # Deduplicate
    unique_exps = list(set(candidate_exps))
    
    # Sort by "density" or span? Smaller span usually means more collisions.
    # Let's sort by max(exps) - min(exps)
    unique_exps.sort(key=lambda e: max(e) - min(e))
    
    # Run optimization on top candidates
    # We have ~85 seconds. 
    # Let's allocate time per candidate.
    num_candidates = len(unique_exps)
    if num_candidates > 0:
        time_per_candidate = (deadline - time.time()) / num_candidates
        # Cap time per candidate to avoid one taking too long
        time_per_candidate = min(time_per_candidate, 2.0)
        
        for exps in unique_exps:
            if time.time() > deadline - 5.0:
                break
            
            optimize_coeffs_sa(list(exps), max_time=time_per_candidate)
            
            # Early termination if we hit theoretical lower bound?
            # Theoretical lower bound is 12.
            # 13-15 is good.
            if best_score <= 13:
                break

    return best_candidate

# EVOLVE-BLOCK-END
