import time
import itertools
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = 55.0  # 55s limit to be safe within 60s/90s constraints
    
    best_poly = None
    best_score = float('inf')
    
    # Helper to count nonzero terms in square
    def count_square_terms(coeffs):
        # coeffs is a dict {exp: coeff} or list
        if isinstance(coeffs, list):
            # Convert list to dict for efficient convolution if sparse, 
            # but for small degrees list is fine.
            # Let's use numpy for speed if available, but manual is safer for exact ints.
            # Actually, for degree <= 40, manual convolution is fast enough.
            n = len(coeffs)
            square = [0] * (2*n - 1)
            for i in range(n):
                if coeffs[i] != 0:
                    ci = coeffs[i]
                    for j in range(n):
                        if coeffs[j] != 0:
                            square[i+j] += ci * coeffs[j]
            return sum(1 for c in square if c != 0)
        return 0

    # Helper to validate and score a candidate
    def evaluate_candidate(coeffs_list):
        # Check constraints
        if len(coeffs_list) > 41:
            return None
        if coeffs_list[-1] == 0:
            return None
        nonzero_count = sum(1 for c in coeffs_list if c != 0)
        if nonzero_count != 12:
            return None
        for c in coeffs_list:
            if not (-64 <= c <= 64):
                return None
        
        score = count_square_terms(coeffs_list)
        return score

    # Strategy:
    # 1. Try structured exponent sets.
    #    The number of distinct sums $a_i + a_j$ determines the minimum possible terms 
    #    if we could cancel perfectly. 
    #    For 12 terms, min distinct sums is 12 (if all $a_i+a_j$ are distinct? No, 
    #    min is related to Sidon sets or additive bases).
    #    Actually, if we use symmetric exponents around a center, we can pair terms.
    #    
    #    Let's try specific exponent patterns:
    #    Pattern A: Consecutive 0..11. Distinct sums: 23. 
    #    Pattern B: Sparse exponents to reduce collisions? No, we want MORE collisions 
    #    to allow cancellation. We want a small sumset.
    #    Arithmetic progression 0, 1, ..., 11 has sumset size 23.
    #    Can we get smaller? 
    #    If we use exponents like {0, 1, 2, 4, 8, ...} (powers of 2), sums are unique (Sidon). 
    #    We want the OPPOSITE: we want many collisions to cancel.
    #    So we want exponents to be "additively dense". 
    #    The densest set of size 12 is an arithmetic progression.
    #    So, exponents 0..11 is likely optimal for minimizing distinct sums.
    
    #    Therefore, we fix exponents to 0..11 (or a shift/scale thereof) and focus 
    #    on finding coefficients that maximize cancellation in the 23 sum positions.
    
    #    The square of $P(x) = \sum c_i x^i$ has coefficients $S_k = \sum_{i+j=k} c_i c_j$.
    #    We want $S_k = 0$ for as many $k$ as possible.
    
    #    Let's use a local search / hill climbing on the coefficients for exponents 0..11.
    #    We can also try other small exponent sets if 0..11 fails to go low.
    
    exponent_sets = [
        list(range(12)), # 0..11
        list(range(0, 24, 2)), # 0,2,...22 (shifted/scaled)
        list(range(1, 13)), # 1..12
        [0, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144], # Fibonacci (unlikely, but check)
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], # Same as first
    ]
    
    # Remove duplicates
    seen = set()
    unique_exp_sets = []
    for e in exponent_sets:
        key = tuple(e)
        if key not in seen:
            seen.add(key)
            unique_exp_sets.append(e)

    for exps in unique_exp_sets:
        if time.time() > deadline:
            break
            
        # Initialize with random signs
        # We want to minimize nonzero terms in square.
        # Start with all 1s or random signs.
        
        # Try a few random initializations
        for init_iter in range(20):
            if time.time() > deadline:
                break
                
            # Random signs
            coeffs = [0] * 41
            for e in exps:
                coeffs[e] = random.choice([-1, 1, -2, 2, -3, 3])
            
            # Ensure 12 nonzero terms
            # (The loop above ensures this if exps are unique)
            
            curr_score = evaluate_candidate(coeffs)
            if curr_score is None:
                continue
            
            if curr_score < best_score:
                best_score = curr_score
                best_poly = coeffs[:]
                
            # Local Search: Flip signs or change values
            improved = True
            while improved and time.time() < deadline:
                improved = False
                # Try changing each coefficient
                for i, e in enumerate(exps):
                    old_val = coeffs[e]
                    # Try neighbors
                    for new_val in range(-64, 65):
                        if new_val == old_val:
                            continue
                        # Quick check: is new_val reasonable?
                        # Just try a subset of values to save time
                        # Actually, let's try flipping sign first, then small changes
                        pass
                    
                    # More efficient: try flipping sign
                    new_val = -old_val
                    coeffs[e] = new_val
                    new_score = evaluate_candidate(coeffs)
                    if new_score is not None:
                        if new_score < curr_score:
                            curr_score = new_score
                            improved = True
                            if curr_score < best_score:
                                best_score = curr_score
                                best_poly = coeffs[:]
                    else:
                        coeffs[e] = old_val
                        continue
                        
                    # If no improvement from flip, try small perturbations?
                    # For now, just flips are fast.
                    if not improved:
                        coeffs[e] = old_val # Revert if we didn't take it? 
                        # Wait, the logic above sets coeffs[e] = new_val, evaluates, 
                        # and if it's better, keeps it. If not, we should revert.
                        # Let's fix the logic.
                        pass

            # Redo local search with cleaner logic
            improved = True
            while improved and time.time() < deadline:
                improved = False
                for e in exps:
                    old_val = coeffs[e]
                    # Try flipping sign
                    new_val = -old_val
                    coeffs[e] = new_val
                    new_score = evaluate_candidate(coeffs)
                    if new_score is not None and new_score < curr_score:
                        curr_score = new_score
                        improved = True
                        if curr_score < best_score:
                            best_score = curr_score
                            best_poly = coeffs[:]
                    else:
                        coeffs[e] = old_val
                    
                    # Try adding/subtracting 1 or 2
                    for delta in [-2, -1, 1, 2]:
                        new_val = old_val + delta
                        if new_val < -64 or new_val > 64:
                            continue
                        if new_val == 0:
                            # Cannot be 0, must be 12 nonzero terms
                            # If we set to 0, we need to set another to non-zero?
                            # This complicates the "12 nonzero" constraint.
                            # So we stick to non-zero values.
                            continue
                        coeffs[e] = new_val
                        new_score = evaluate_candidate(coeffs)
                        if new_score is not None and new_score < curr_score:
                            curr_score = new_score
                            improved = True
                            if curr_score < best_score:
                                best_score = curr_score
                                best_poly = coeffs[:]
                                old_val = new_val # Update base for further deltas? 
                                # No, keep it simple: revert to old_val after trying all deltas
                                # But if we improved, we want to stay there.
                                # Let's just break out of delta loop if improved?
                                pass
                        else:
                            coeffs[e] = old_val

    # If we found a good one, return it.
    if best_poly is not None:
        # Final validation
        score = evaluate_candidate(best_poly)
        if score is not None:
            return best_poly
            
    # Fallback: The seed program
    return [1] * 12
