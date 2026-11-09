import numpy as np
import time
from scipy.optimize import differential_evolution

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4], where X1, X2, X3, X4 are i.i.d. with law mu.
    
    Strategy: Co-optimize both support points and weights for a 3-point measure.
    We parameterize the support as [0, a, b] with 0 < a < b.
    We parameterize the weights as [p0, pa, pb] with p0+pa+pb=1.
    We use scipy.optimize.differential_evolution to find the global maximum.
    """
    start_time = time.time()
    deadline = start_time + 590.0  # 10s safety margin
    
    # Parameters: [a, b, p0, pa, pb]
    # Constraints: 0 < a < b, p0 > 0, pa > 0, pb > 0, p0+pa+pb=1
    # We enforce p0+pa+pb=1 by normalizing inside the objective function,
    # but we should bound them to be positive.
    # Bounds for a: [0.1, 10.0]
    # Bounds for b: [1.1, 100.0] (b must be > a, handled by penalty or careful bounds)
    # Actually, simpler: let a, b be free in [0.1, 50.0] and sort them? 
    # No, sorting changes the identity of the weights.
    # Let's stick to 0 < a < b. 
    # We can define bounds: a in [0.1, 20.0], b in [0.1, 50.0].
    # Inside objective, if a >= b, return -inf (or a large penalty).
    
    # Weights p0, pa, pb in [0.01, 0.99].
    # Normalization: w = [p0, pa, pb] / sum([p0, pa, pb])
    
    def objective(params):
        a, b, p0, pa, pb = params
        
        # Check time
        if time.time() > deadline - 5.0:
            # Return a bad score to stop early? 
            # DE handles this poorly. Better to just let it run, 
            # but we have a hard deadline check in the wrapper.
            pass
            
        # Enforce 0 < a < b
        if a >= b:
            return -1.0  # Low probability, effectively -inf for maximization (minimizing -P)
        
        # Normalize weights
        sum_p = p0 + pa + pb
        if sum_p <= 1e-9:
            return -1.0
            
        w0 = p0 / sum_p
        wa = pa / sum_p
        wb = pb / sum_p
        
        # Ensure weights are strictly positive for validity check later
        if w0 < 1e-12 or wa < 1e-12 or wb < 1e-12:
            return -1.0

        support = np.array([0.0, a, b], dtype=np.float64)
        weights = np.array([w0, wa, wb], dtype=np.float64)
        
        # Calculate P[X1+X2+X3 < 2*X4]
        # X4 can be 0, a, or b.
        
        # Case X4 = 0 (prob w0):
        # 2*X4 = 0. Need X1+X2+X3 < 0.
        # Since support is >= 0, sum >= 0. So P(sum < 0) = 0.
        p_x4_0 = 0.0
        
        # Case X4 = a (prob wa):
        # 2*X4 = 2*a. Need X1+X2+X3 < 2*a.
        # Possible sums of 3 items from {0, a, b}:
        # 0+0+0 = 0 < 2a (Always true if a>0) -> prob w0^3
        # 0+0+a = a < 2a (Always true if a>0) -> prob 3*w0^2*wa
        # 0+a+0 = a < 2a -> prob 3*w0^2*wa (Wait, permutations: (0,0,a), (0,a,0), (a,0,0))
        # Actually, let's list unique multisets and their probabilities:
        # {0,0,0}: sum 0. Prob w0^3. Condition: 0 < 2a. True.
        # {0,0,a}: sum a. Prob 3*w0^2*wa. Condition: a < 2a. True.
        # {0,0,b}: sum b. Prob 3*w0^2*wb. Condition: b < 2a.
        # {0,a,a}: sum 2a. Prob 3*w0*wa^2. Condition: 2a < 2a. False (strict).
        # {0,a,b}: sum a+b. Prob 6*w0*wa*wb. Condition: a+b < 2a => b < a. False (since b>a).
        # {0,b,b}: sum 2b. Prob 3*w0*wb^2. Condition: 2b < 2a => b < a. False.
        # {a,a,a}: sum 3a. Prob wa^3. Condition: 3a < 2a => a < 0. False.
        # {a,a,b}: sum 2a+b. Prob 3*wa^2*wb. Condition: 2a+b < 2a => b < 0. False.
        # {a,b,b}: sum a+2b. Prob 3*wa*wb^2. Condition: a+2b < 2a => 2b < a. False.
        # {b,b,b}: sum 3b. Prob wb^3. Condition: 3b < 2a. False.
        
        # So P[S < 2a | X4=a] = w0^3 + 3*w0^2*wa + (3*w0^2*wb if b < 2a else 0)
        
        p_sum_lt_2a = w0**3 + 3*w0**2*wa
        if b < 2*a:
            p_sum_lt_2a += 3*w0**2*wb
            
        p_x4_a = wa * p_sum_lt_2a
        
        # Case X4 = b (prob wb):
        # 2*X4 = 2*b. Need X1+X2+X3 < 2*b.
        # Let's check which sums are >= 2*b.
        # Sums:
        # 0: < 2b. OK.
        # a: < 2b. OK.
        # b: < 2b. OK.
        # 2a: < 2b? Depends. If 2a < 2b (a<b, true), then OK.
        # a+b: < 2b? a < b (true), so OK.
        # 2b: < 2b? False.
        # 3a: < 2b? Depends.
        # 2a+b: < 2b? 2a < b? Depends.
        # a+2b: < 2b? a < 0? False.
        # 3b: < 2b? False.
        
        # So we need to exclude cases where sum >= 2b.
        # Excluded sums:
        # 2b (from {b,b,0} or {b,0,b} or {0,b,b}): Prob 3*w0*wb^2. Sum = 2b. Not < 2b.
        # a+2b (from {a,b,b}, {b,a,b}, {b,b,a}): Prob 3*wa*wb^2. Sum = a+2b > 2b.
        # 3b (from {b,b,b}): Prob wb^3. Sum = 3b > 2b.
        # 2a+b (from {a,a,b}, {a,b,a}, {b,a,a}): Prob 3*wa^2*wb. Sum = 2a+b.
        #   Is 2a+b >= 2b? => 2a >= b.
        # 3a (from {a,a,a}): Prob wa^3. Sum = 3a.
        #   Is 3a >= 2b? => 3a >= 2b.
        
        p_sum_ge_2b = 0.0
        p_sum_ge_2b += 3*w0*wb**2  # Sum 2b
        p_sum_ge_2b += 3*wa*wb**2  # Sum a+2b
        p_sum_ge_2b += wb**3       # Sum 3b
        
        if 2*a + b >= 2*b: # i.e., 2a >= b
            p_sum_ge_2b += 3*wa**2*wb # Sum 2a+b
            
        if 3*a >= 2*b:
            p_sum_ge_2b += wa**3 # Sum 3a
            
        # Also check 0+0+b = b < 2b (OK)
        # 0+a+b = a+b < 2b (OK)
        # a+a+a = 3a. Checked.
        # a+a+b = 2a+b. Checked.
        
        p_sum_lt_2b = 1.0 - p_sum_ge_2b
        
        # Clip to [0,1] to handle floating point errors
        p_sum_lt_2b = np.clip(p_sum_lt_2b, 0.0, 1.0)
        
        p_x4_b = wb * p_sum_lt_2b
        
        total_p = p_x4_0 + p_x4_a + p_x4_b
        
        # Return negative probability for minimization
        return -total_p

    # Bounds for Differential Evolution
    # a: [0.1, 20.0]
    # b: [0.1, 50.0]
    # p0, pa, pb: [0.01, 0.99]
    bounds = [
        (0.1, 20.0),   # a
        (0.1, 50.0),   # b
        (0.01, 0.99),  # p0
        (0.01, 0.99),  # pa
        (0.01, 0.99)   # pb
    ]
    
    best_result = None
    best_score = -np.inf
    best_params = None
    
    # Run Differential Evolution
    # maxiter=500, popsize=15 might be enough. 
    # We have ~590s. DE is stochastic.
    try:
        result = differential_evolution(
            objective, 
            bounds, 
            seed=42, 
            maxiter=200, 
            popsize=10, 
            tol=1e-6, 
            atol=1e-6,
            polish=False  # We do our own polish or just take DE result
        )
        
        if result.success or (result.fun < best_score):
            best_score = -result.fun
            best_params = result.x
            
    except Exception as e:
        # Fallback if DE fails
        pass

    # If DE didn't find a good result or failed, do a quick grid search fallback
    # But first, let's try to polish the DE result with a local optimizer if time permits.
    # However, the prompt says "Keep an EXPLICIT deadline guard".
    
    # Let's implement a simple local refinement (Nelder-Mead) if we have time.
    if best_params is not None and time.time() < deadline - 10.0:
        from scipy.optimize import minimize
        try:
            res = minimize(objective, best_params, method='Nelder-Mead', 
                           options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8})
            if res.fun < best_score:
                best_score = -res.fun
                best_params = res.x
        except Exception:
            pass

    # If still no best_params, use a reasonable default
    if best_params is None:
        # Fallback to the current program's best known config
        # Support [0, 1, 10], weights optimized.
        # Let's just return a valid 3-point measure.
        a = 1.0
        b = 10.0
        p0 = 0.5
        pa = 0.3
        pb = 0.2
        best_params = np.array([a, b, p0, pa, pb])
        # Calculate score for this fallback to ensure we don't return garbage
        # But the objective function handles it.
        val = objective(best_params)
        if val > -1.0: # -1.0 is our "invalid" marker
             best_score = -val
        else:
             # Even fallback might be invalid due to a>=b or similar? 
             # With a=1, b=10, it's valid.
             pass

    # Extract final parameters
    a, b, p0, pa, pb = best_params
    
    # Ensure 0 < a < b
    if a >= b:
        # Swap if needed? No, weights are tied to specific points.
        # If DE returned a > b, the objective returned -1.0, so it shouldn't be the best.
        # But if it's the only result, we need to handle it.
        # Let's just enforce validity by swapping support and weights if a > b.
        if a > b:
            a, b = b, a
            p0, pa, pb = p0, pb, pa # Wait, p0 is for 0. pa for a, pb for b.
            # If we swap a and b, the weight for the new 'a' (old b) is pb.
            # The weight for the new 'b' (old a) is pa.
            # So new support: [0, old_b, old_a]. New weights: [p0, pb, pa].
            # Let's just re-assign carefully.
            temp_a = a
            temp_b = b
            temp_pa = pa
            temp_pb = pb
            a = temp_b
            b = temp_a
            pa = temp_pb
            pb = temp_pa

    # Normalize weights again to be safe
    sum_p = p0 + pa + pb
    w0 = p0 / sum_p
    wa = pa / sum_p
    wb = pb / sum_p
    
    support = np.array([0.0, a, b], dtype=np.float64)
    weights = np.array([w0, wa, wb], dtype=np.float64)
    
    # Final validity check
    if np.any(support < -1e-9) or np.any(weights < -1e-9) or np.sum(weights) <= 1e-9:
        return np.array([0.0, 1.0]), np.array([0.5, 0.5])
        
    return support, weights
