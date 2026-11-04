import numpy as np
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    n, m, p = 4, 5, 6
    nm, mp, pn = n * m, m * p, p * n  # 20, 30, 24
    
    # Schoolbook fallback
    def schoolbook():
        R = n * m * p
        U = [[0.0] * R for _ in range(nm)]
        V = [[0.0] * R for _ in range(mp)]
        W = [[0.0] * R for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1.0
                    V[j * p + k][col] = 1.0
                    W[k * n + i][col] = 1.0
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best = schoolbook()
    
    def try_rank_exact(R, time_budget):
        """
        Try to find a rank R decomposition by fixing W=1 and solving for U, V.
        We choose V randomly, then solve for U.
        """
        deadline = time.time() + time_budget
        np.random.seed(42) # Reproducibility for debugging, but we want randomness. Use time seed?
        # Better to use random seed based on time for variety
        np.random.seed(int(time.time() * 1000) % (2**32))
        
        for attempt in range(50):
            if time.time() > deadline:
                break
                
            # Generate random V matrix (m*p x R)
            # We want V to be such that for each j, V_j (p x R) has full row rank p (if R>=p)
            # or specific structure if R<p.
            # Let's try dense random integers in {-1, 0, 1} or {-1, 1}
            V = np.random.randint(-1, 2, size=(mp, R))
            
            # Check if V is "good" enough
            # For each j, V_j is p x R. We need to solve V_j^T U = 1.
            # This requires rank(V_j) = rank(V_j | 1).
            # If R >= p, rank(V_j) is likely p.
            
            U = np.zeros((nm, R))
            valid = True
            
            for i in range(n):
                for j in range(m):
                    # Extract V_j: rows corresponding to index j
                    # V is indexed by (j, k). V[j*p + k, :]
                    Vj = V[j*p:(j+1)*p, :] # shape (p, R)
                    
                    # Solve Vj.T @ U_ij = ones(p)
                    # Vj.T is (R, p). U_ij is (R,).
                    # System: A x = b, where A = Vj.T, b = ones(p)
                    A = Vj.T
                    b = np.ones(p)
                    
                    # Use least squares / pseudo inverse
                    try:
                        # If R < p, check if solution exists
                        # If R >= p, solution always exists (in real numbers)
                        Uij, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
                        
                        # Check residual
                        if np.linalg.norm(A @ Uij - b) > 1e-9:
                            valid = False
                            break
                            
                        U[i*m + j, :] = Uij
                    except:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                # We have U, V, W=1.
                # Verify the decomposition exactly
                W = np.ones((pn, R))
                
                max_viol = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for r in range(R):
                                s += U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
                            max_viol = max(max_viol, abs(s - 1.0))
                
                if max_viol < 1e-9:
                    # Convert to lists
                    # The task allows integer/half-integer. 
                    # Our U, V are floats. We can try to round them if they are close to integers/half-integers.
                    # Or just return them as floats if the evaluator accepts "integer/half-integer" as a type hint for precision, 
                    # but the contract says "with integer/half-integer entries".
                    # This implies the entries MUST be exactly representable as k/2.
                    # Let's check if U and V are close to half-integers.
                    
                    U_rounded = np.round(U * 2) / 2
                    V_rounded = np.round(V * 2) / 2
                    
                    if np.allclose(U, U_rounded, atol=1e-9) and np.allclose(V, V_rounded, atol=1e-9):
                        U_list = [[float(U_rounded[i, r]) for r in range(R)] for i in range(nm)]
                        V_list = [[float(V_rounded[i, r]) for r in range(R)] for i in range(mp)]
                        W_list = [[float(W[i, r]) for r in range(R)] for i in range(pn)]
                        return {"U": U_list, "V": V_list, "W": W_list}
                    else:
                        # Try to find a scaling factor to make them integers?
                        # If U, V are rational, we can scale.
                        # But we don't know the rationals exactly from floats.
                        # For now, if they are not half-integers, we might fail the "integer/half-integer" constraint.
                        # However, we can try to construct V with integers and solve for U in rationals.
                        # If we use Fraction arithmetic, we can get exact rationals.
                        pass
            
        return None

    # Try ranks from low to high
    # Lower bound for 4x5x6 is not 6. 
    # Strassen rank for 4x4x4 is 23.
    # For 4x5x6, it's likely higher.
    # Let's try a range.
    ranks_to_try = [20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 110, 119]
    
    for R in ranks_to_try:
        if time.time() > 140:
            break
        time_left = 140 - time.time() - 5
        if time_left < 5:
            break
        
        # We need to be careful with time. 
        # The linear solve is fast. The bottleneck is the attempts.
        result = try_rank_exact(R, min(time_left, 10))
        if result is not None:
            best = result
            break
            
    return best
