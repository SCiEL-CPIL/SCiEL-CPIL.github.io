import numpy as np
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin

    def evaluate(coeffs):
        sq = np.convolve(coeffs, coeffs)
        return int(np.count_nonzero(sq))

    def validate(coeffs):
        if len(coeffs) != 12:
            return False
        if coeffs[-1] == 0:
            return False
        if coeffs[0] == 0:
            return False
        if not all(isinstance(c, int) for c in coeffs):
            return False
        if not all(-64 <= c <= 64 for c in coeffs):
            return False
        if sum(1 for c in coeffs if c != 0) != 12:
            return False
        return True

    best_score = float('inf')
    best_poly = None

    # Candidate zero indices: 2 to 20
    # 0, 1, 21, 22 are always non-zero
    candidates = list(range(2, 21))
    
    # We will try to find the largest subset Z of candidates that can be zeroed out.
    # We iterate m from len(candidates) down to 0.
    
    def solve_for_Z(Z):
        """
        Finds a polynomial that has zeros at indices in Z.
        Returns (coeffs, score) or (None, None).
        """
        Z_set = set(Z)
        
        # c0, c1 are the root variables
        # We iterate over possible values for c0, c1
        for c0 in range(1, 65):
            for sign0 in [1, -1]:
                v0 = sign0 * c0
                for c1 in range(1, 65):
                    for sign1 in [1, -1]:
                        v1 = sign1 * c1
                        
                        coeffs = [0] * 12
                        coeffs[0] = v0
                        coeffs[1] = v1
                        
                        valid = True
                        # Determine c2..c11
                        for k in range(2, 12):
                            # S_k = 2*c0*ck + sum_{j=1}^{k-1} cj*c(k-j)
                            # If k is in Z, we must have S_k = 0 => ck is determined
                            # If k is not in Z, ck is free, but we want to minimize non-zeros.
                            # Strategy: If k is in Z, determine ck.
                            # If k is not in Z, we have a choice.
                            # To keep search space small, if k is not in Z, we can try to determine ck
                            # such that S_k=0 anyway? If that fails, we might need to branch.
                            # However, if we just want to find ANY solution for Z, we can pick
                            # arbitrary values for free ks? No, that might prevent future zeros.
                            # But if Z is large, most ks are fixed.
                            # Let's assume for now that if k not in Z, we can pick ck to try to zero S_k.
                            # If it doesn't work out to be integer, we pick a default?
                            # Actually, if k not in Z, we don't CARE if S_k is 0 or not,
                            # as long as we can complete the polynomial.
                            # But we want to MINIMIZE total non-zeros.
                            # So we SHOULD try to make S_k=0.
                            
                            # Calculate sum of known terms
                            sum_known = 0
                            for j in range(1, k):
                                sum_known += coeffs[j] * coeffs[k-j]
                            
                            if k in Z_set:
                                # 2*v0*ck + sum_known = 0
                                num = -sum_known
                                den = 2 * v0
                                if den == 0:
                                    valid = False
                                    break
                                if num % den != 0:
                                    valid = False
                                    break
                                ck = num // den
                                if ck == 0 or not (-64 <= ck <= 64):
                                    valid = False
                                    break
                                coeffs[k] = ck
                            else:
                                # k not in Z. We want to try to make S_k=0.
                                num = -sum_known
                                den = 2 * v0
                                if den != 0 and num % den == 0:
                                    ck = num // den
                                    if ck != 0 and -64 <= ck <= 64:
                                        coeffs[k] = ck
                                        continue
                                # If we can't make it zero, or it's invalid, we have to pick a value.
                                # To avoid branching explosion, let's just pick a small value?
                                # This is a heuristic. If Z is large, few free vars.
                                # If Z is small, this might fail to find optimal, but we start with large Z.
                                # Let's pick 1 or -1?
                                # Actually, if we are here, it means we couldn't zero S_k.
                                # We just need to assign SOME valid value.
                                # Let's try 1.
                                coeffs[k] = 1
                                # Note: This might make S_k non-zero, which is fine if k not in Z.
                                
                        if not valid:
                            continue
                        
                        # We have assigned c0..c11.
                        # Check if the polynomial is valid (all non-zero, in range)
                        if not all(c != 0 and -64 <= c <= 64 for c in coeffs):
                            continue
                        
                        # Calculate score
                        score = evaluate(coeffs)
                        
                        # We found a solution for Z.
                        # The score might be lower than 23 - len(Z) if we accidentally zeroed more.
                        return coeffs, score
                        
        return None, None

    # Iterate m from 19 down to 0
    for m in range(len(candidates), -1, -1):
        if time.time() > deadline - 10:
            break
            
        # Generate combinations of size m
        # To save time, we can break early if we find a solution with a very good score?
        # No, we want the BEST score.
        # But if we find a solution with score S, we can stop if we are sure we can't do better?
        # Min possible score is 4.
        # If we find score 4, we can stop.
        
        found_better = False
        for Z in combinations(candidates, m):
            if time.time() > deadline - 5:
                break
                
            coeffs, score = solve_for_Z(Z)
            
            if coeffs is not None:
                if score < best_score:
                    best_score = score
                    best_poly = coeffs
                    # If we found a very good score, we might want to break out of combination loop?
                    # If score == 4, it's optimal.
                    if score <= 4:
                        found_better = True
                        break
                # Even if not better, we found a valid poly for this Z.
                # But we want to check other Z of same size?
                # Maybe a different Z of same size yields lower score?
                # Yes. So we should continue.
        
        if found_better:
            break
            
    if best_poly is None:
        # Fallback
        best_poly = [1] * 12

    if not validate(best_poly):
        best_poly = [1] * 12

    return best_poly
