import random
import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin
    
    def count_nonzero(coeff_list):
        return sum(1 for c in coeff_list if c != 0)
    
    def poly_square_sparse(coeff_list):
        """Compute the square of a polynomial given as coefficient list (ascending degree).
        Returns the coefficient list of the square (ascending degree)."""
        n = len(coeff_list)
        # Square has degree 2*(n-1), so length 2*n - 1
        result = [0] * (2 * n - 1)
        for i in range(n):
            if coeff_list[i] == 0:
                continue
            ci = coeff_list[i]
            for j in range(n):
                if coeff_list[j] == 0:
                    continue
                result[i + j] += ci * coeff_list[j]
        return result
    
    def eval_candidate(coeff_list):
        """Return (num_nonzero_in_square, coeff_list). Lower num_nonzero is better."""
        sq = poly_square_sparse(coeff_list)
        return count_nonzero(sq), sq
    
    def validate(coeff_list):
        """Check if coeff_list satisfies all constraints."""
        if len(coeff_list) > 41:
            return False
        if len(coeff_list) == 0:
            return False
        if coeff_list[-1] == 0:
            return False
        for c in coeff_list:
            if not isinstance(c, int) or c < -64 or c > 64:
                return False
        if count_nonzero(coeff_list) != 12:
            return False
        return True
    
    # Start with a good initial guess
    # Try structured polynomials first
    best_coeff = None
    best_nnz = float('inf')
    
    # Initial candidates to try
    initial_candidates = []
    
    # Candidate 1: 12 terms with alternating signs at consecutive powers
    for start in range(0, 10):
        for d in range(1, 5):
            cand = [0] * (start + 11 * d + 1)
            for i in range(12):
                cand[start + i * d] = (-1) ** i
            if len(cand) <= 41:
                initial_candidates.append(cand)
    
    # Candidate 2: 12 terms with all +1
    for start in range(0, 10):
        for d in range(1, 5):
            cand = [0] * (start + 11 * d + 1)
            for i in range(12):
                cand[start + i * d] = 1
            if len(cand) <= 41:
                initial_candidates.append(cand)
    
    # Candidate 3: random small coefficients
    for _ in range(200):
        # Choose 12 distinct exponents
        max_deg = 40
        exponents = sorted(random.sample(range(0, max_deg + 1), 12))
        cand = [0] * (max_deg + 1)
        for e in exponents:
            cand[e] = random.randint(-64, 64)
            if cand[e] == 0:
                cand[e] = 1
        # Ensure last coefficient is nonzero
        if cand[-1] == 0:
            cand[-1] = 1
        # Ensure exactly 12 nonzero
        if count_nonzero(cand) != 12:
            # Fix: make sure only the chosen exponents are nonzero
            cand = [0] * (max_deg + 1)
            for e in exponents:
                cand[e] = random.choice([-1, 1, 2, -2, 3, -3])
            if cand[-1] == 0:
                cand[-1] = 1
        initial_candidates.append(cand)
    
    # Evaluate initial candidates
    for cand in initial_candidates:
        if not validate(cand):
            continue
        nnz, _ = eval_candidate(cand)
        if nnz < best_nnz:
            best_nnz = nnz
            best_coeff = cand[:]
    
    # Local search: try to improve by flipping signs or adjusting coefficients
    def local_search(coeff, max_iters=50000):
        nonlocal best_nnz, best_coeff
        current = coeff[:]
        current_nnz, _ = eval_candidate(current)
        
        if current_nnz < best_nnz:
            best_nnz = current_nnz
            best_coeff = current[:]
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Pick a random nonzero term to modify
            nonzero_idx = [i for i, c in enumerate(current) if c != 0]
            if not nonzero_idx:
                break
            idx = random.choice(nonzero_idx)
            old_val = current[idx]
            
            # Try changing the coefficient
            if random.random() < 0.5:
                # Flip sign
                new_val = -old_val
            else:
                # Random small change
                new_val = old_val + random.choice([-2, -1, 1, 2])
            
            if new_val == old_val:
                continue
            if new_val < -64 or new_val > 64:
                continue
            if new_val == 0:
                # Would reduce nonzero count below 12, skip
                continue
            
            current[idx] = new_val
            new_nnz, _ = eval_candidate(current)
            
            if new_nnz <= current_nnz:
                current_nnz = new_nnz
                if new_nnz < best_nnz:
                    best_nnz = new_nnz
                    best_coeff = current[:]
            else:
                # Revert
                current[idx] = old_val
        
        return current_nnz
    
    # Run local search from best initial
    if best_coeff is not None:
        local_search(best_coeff, max_iters=10000)
    
    # More random restarts with local search
    restarts = 0
    while time.time() < deadline - 5:
        restarts += 1
        # Generate a new random candidate
        max_deg = random.randint(12, 40)
        exponents = sorted(random.sample(range(0, max_deg + 1), 12))
        cand = [0] * (max_deg + 1)
        for e in exponents:
            cand[e] = random.choice([-1, 1, -2, 2, -3, 3])
        if cand[-1] == 0:
            cand[-1] = 1
        if count_nonzero(cand) != 12:
            continue
        
        if not validate(cand):
            continue
        
        nnz, _ = eval_candidate(cand)
        if nnz < best_nnz:
            best_nnz = nnz
            best_coeff = cand[:]
        
        # Short local search
        local_search(cand, max_iters=5000)
    
    # Final: try to trim trailing zeros
    if best_coeff is None:
        # Fallback: return a valid 12-term polynomial
        return [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    
    # Trim trailing zeros (but keep at least 1 element, and last must be nonzero)
    while len(best_coeff) > 1 and best_coeff[-1] == 0:
        best_coeff.pop()
    
    # Ensure validity
    if not validate(best_coeff):
        # Fallback
        return [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    
    return best_coeff
