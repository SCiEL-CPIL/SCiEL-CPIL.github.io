import time
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 50s limit to be safe within 60s
    
    n = 30
    # We want to minimize |sum(c_i / i)| for c_i in {-1, 0, 1}, sum != 0
    
    # 1. Greedy Initialization
    # Process from i=1 to n=30.
    # Maintain current sum S.
    # For each i, choose c_i in {-1, 1} (initially avoiding 0 to maximize cancellation potential)
    # such that |S + c_i/i| is minimized.
    # Actually, we can consider {-1, 0, 1} in greedy too, but usually non-zero is better for fine tuning later.
    # Let's stick to {-1, 1} for greedy to ensure we are using the full dynamic range, then allow 0 in local search.
    
    coeffs = [0] * n
    S = Fraction(0, 1)
    
    for i in range(1, n + 1):
        term = Fraction(1, i)
        # Check c = 1
        s1 = S + term
        # Check c = -1
        s_minus1 = S - term
        
        # Choose the one with smaller absolute value
        if abs(s1) < abs(s_minus1):
            coeffs[i-1] = 1
            S = s1
        else:
            coeffs[i-1] = -1
            S = s_minus1
            
    # If S is 0, we need to break the zero. 
    # The greedy might land exactly on 0. 
    # If S == 0, we can flip the last coefficient or any coefficient to make it non-zero.
    # Flipping the last coefficient (1/n) changes S by 2/n.
    if S == 0:
        # Flip the last coefficient
        coeffs[n-1] *= -1
        S += Fraction(2, n) * (-1) # Wait, if we flip -1 to 1, S increases by 2/n.
        # Let's just recalculate S to be safe
        S = sum(Fraction(coeffs[i], i+1) for i in range(n))
        
    best_S = S
    best_coeffs = coeffs[:]
    
    # 2. Local Search
    # Try to improve by changing individual coefficients.
    # Moves: change c_i to -1, 0, 1.
    # Accept if new |S| < old |S| and new S != 0.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(n):
            if time.time() > deadline:
                break
            original_c = coeffs[i]
            original_S = S
            
            # Try all 3 values
            for new_c in [-1, 0, 1]:
                if new_c == original_c:
                    continue
                
                # Calculate delta
                delta = Fraction(new_c - original_c, i + 1)
                new_S = original_S + delta
                
                # Check validity: new_S != 0
                if new_S == 0:
                    continue
                
                # Check improvement: |new_S| < |original_S|
                if abs(new_S) < abs(original_S):
                    coeffs[i] = new_c
                    S = new_S
                    improved = True
                    # Do not break out of for new_c loop immediately, 
                    # we might find an even better one? 
                    # Actually, if we accept the first improvement, we update S.
                    # We can continue checking other new_c values for this i?
                    # No, because original_S is now stale.
                    # We should restart the loop for this i or just break to next i.
                    # Breaking to next i is standard hill climbing.
                    break 
                    
    # 3. Final Verification and Return
    # Ensure the result is valid
    final_S = sum(Fraction(coeffs[i], i+1) for i in range(n))
    if final_S == 0:
        # Fallback: use a known small sum
        # e.g. 1/2 - 1/3 - 1/6 = 0? No.
        # 1/1 - 1/2 - 1/3 + 1/6 = 0?
        # Let's just flip the last bit to ensure non-zero if we somehow ended up at 0.
        coeffs[n-1] *= -1
        final_S = sum(Fraction(coeffs[i], i+1) for i in range(n))
        
    return coeffs
# EVOLVE-BLOCK-END
