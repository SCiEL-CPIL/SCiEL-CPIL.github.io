import numpy as np
import time
import sys
from fractions import Fraction

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 100.0  # Keep 20s margin
    
    # Try ranks from low to high
    # Known bounds suggest R(2,5,5) is around 40-45.
    # Start from a reasonable lower bound, e.g., 30, up to 50.
    for r in range(30, 51):
        if time.time() > deadline:
            break
            
        # Try multiple random initializations for U and V
        for attempt in range(100):
            if time.time() > deadline:
                break
                
            # Initialize U and V with small random integers
            # U shape: (n*m, r) = (10, r)
            # V shape: (m*p, r) = (25, r)
            # Using -1, 0, 1 to keep numbers small initially
            U = np.random.choice([-1, 0, 1], size=(n * m, r))
            V = np.random.choice([-1, 0, 1], size=(m * p, r))
            
            # Check if we can solve for W
            W = np.zeros((p * n, r))
            valid = True
            
            # For each (i, k), we have a system for W[k*n+i, :]
            for i in range(n):
                for k in range(p):
                    # Construct A (5 x r)
                    # A[j, c] = U[i*m+j, c] * V[j*p+k, c]
                    A = np.zeros((m, r))
                    for j in range(m):
                        for c in range(r):
                            A[j, c] = U[i * m + j, c] * V[j * p + k, c]
                    
                    # We need A * W_vec = 1
                    # Check rank of A
                    if np.linalg.matrix_rank(A) < m:
                        valid = False
                        break
                    
                    # Solve A x = 1
                    # Use least squares or pinv, but since rank is m, we can use lstsq
                    # Actually, we want an exact solution. Since A is integer, 
                    # the solution will be rational.
                    try:
                        # Using Fraction for exactness? 
                        # Numpy lstsq gives floats. We need exact rational solution.
                        # Let's use sympy? No, std lib + numpy/scipy.
                        # We can solve the linear system using Gaussian elimination with Fractions.
                        # But m=5, r<=50. 5x50 system.
                        # Let's implement a simple Fraction-based solver for Ax=b.
                        
                        # Convert A to Fractions
                        A_frac = [[Fraction(int(A[j, c])) for c in range(r)] for j in range(m)]
                        b_frac = [Fraction(1)] * m
                        
                        # Gaussian elimination
                        # Augmented matrix
                        M = [A_frac[j] + [b_frac[j]] for j in range(m)]
                        
                        # Eliminate
                        for col in range(m):
                            # Find pivot
                            pivot = -1
                            for row in range(col, m):
                                if M[row][col] != 0:
                                    pivot = row
                                    break
                            if pivot == -1:
                                valid = False
                                break
                            
                            # Swap
                            M[col], M[pivot] = M[pivot], M[col]
                            
                            # Normalize
                            pv = M[col][col]
                            for c in range(col, r + 1):
                                M[col][c] = M[col][c] / pv
                                
                            # Eliminate other rows
                            for row in range(m):
                                if row != col:
                                    factor = M[row][col]
                                    if factor != 0:
                                        for c in range(col, r + 1):
                                            M[row][c] -= factor * M[col][c]
                        
                        if not valid:
                            break
                            
                        # Solution is in the last column
                        # W[k*n+i, c] = M[c][r]
                        for c in range(r):
                            W[k * n + i, c] = float(M[c][r])
                            
                    except Exception:
                        valid = False
                        break
                        
                if not valid:
                    break
            
            if valid:
                # We have a valid decomposition with float W.
                # Now we need to convert U, V, W to integers (or half-integers).
                # U and V are already integers.
                # W is rational.
                # Find common denominator for W.
                
                # Convert W to Fractions
                W_frac = [[Fraction(W[i, c]).limit_denominator(10**6) for c in range(r)] for i in range(p * n)]
                
                # Find LCM of denominators
                denom_lcm = 1
                for i in range(p * n):
                    for c in range(r):
                        d = W_frac[i][c].denominator
                        denom_lcm = (denom_lcm * d) // np.gcd(denom_lcm, d)
                        
                # Scale W by denom_lcm
                W_int = [[int(W_frac[i][c] * denom_lcm) for c in range(r)] for i in range(p * n)]
                
                # U and V are already integers. 
                # However, the scaling of W doesn't require scaling U and V if we just return the matrices.
                # The decomposition is sum U_c V_c W_c = T.
                # If we scale W by D, we must scale U or V by 1/D? 
                # No, the product U*V*W must equal 1.
                # If we change W to D*W, the product becomes D times the original.
                # So we must scale U or V by 1/D.
                # But U and V must be integers/half-integers.
                # If D is an integer, 1/D is not integer.
                # So we should scale U and V by D, and keep W as integer?
                # Let's check:
                # Original: sum (U_c) (V_c) (W_c) = 1.
                # New: sum (D*U_c) (D*V_c) (W_c/D) = D * sum U_c V_c W_c = D.
                # This changes the target to D.
                # We need the target to be 1.
                # So we need sum (U'_c) (V'_c) (W'_c) = 1.
                # If we set U' = U, V' = V, W' = W (rational), it works.
                # To make W' integer, we can set W' = D*W (integer).
                # Then sum U V (D*W) = D.
                # To get 1, we can divide one of U or V by D.
                # If D divides all entries of U, we can do that.
                # Or, we can just return the rational numbers? 
                # The task says "integer/half-integer entries".
                # So we must ensure U, V, W have entries in Z or Z/2.
                
                # Strategy: 
                # 1. Get rational W.
                # 2. Let D be LCM of denominators of W.
                # 3. Let W_int = D * W.
                # 4. We need sum U V W_int = D.
                # 5. We can scale U by D? No, that makes it D^2.
                # 6. We can scale U by 1? No.
                # 7. We need to adjust U and V such that their product with W_int gives 1.
                # 8. Actually, we can just multiply U by D and V by 1? 
                #    sum (D*U) (V) (W_int/D) -> No.
                
                # Let's look at the equation:
                # sum_c U_{i,j,c} V_{j,k,c} W_{k,i,c} = 1
                # We found W such that this holds.
                # W has denominators. Let D be common denominator.
                # Let W' = D * W. Then W' is integer.
                # Then sum U V W' = D.
                # We want sum U' V' W' = 1.
                # We can set U' = U, V' = V, and divide the whole equation by D? No.
                # We can set U' = U, V' = V / D? No, V must be integer.
                
                # Alternative:
                # Scale U and V by D.
                # Let U' = D * U, V' = V, W' = W. (W is rational)
                # Let's make all integers.
                # Let W_int = D * W.
                # Then sum (U) (V) (W_int) = D.
                # If we set U_new = U, V_new = V, W_new = W_int, the result is D.
                # To get 1, we can divide U_new by D?
                # If D divides all entries of U, then U/D is integer.
                # We can try to choose U such that its entries are multiples of D?
                # Or, we can just multiply U by D and V by D?
                # sum (D*U) (D*V) (W/D^2) ...
                
                # Simplest:
                # We have a valid decomposition with rational W.
                # We can return the rational numbers if the evaluator accepts them?
                # "integer/half-integer entries".
                # So we must convert to integers.
                # 
                # Trick: 
                # Multiply U by D, V by 1, W by 1?
                # sum (D*U) (V) (W) = D.
                # Multiply U by 1, V by D, W by 1?
                # sum (U) (D*V) (W) = D.
                # Multiply U by D, V by D, W by 1/D^2?
                
                # Let's try to find a decomposition where W is integer directly.
                # We solved A W = 1.
                # If we solve A W = D, then W will be D times the previous W.
                # If D is the LCM of denominators of the previous W, then new W is integer.
                # So, we should solve A W = D for each block.
                # But D depends on the solution.
                
                # Iterative approach:
                # 1. Solve for W (rational).
                # 2. Find D (LCM of denominators).
                # 3. Scale U by D.
                # 4. Now sum (D*U) V W = D.
                # 5. We want sum U' V' W' = 1.
                # 6. Set U' = D*U, V' = V, W' = W. Result is D.
                # 7. Divide U' by D? That brings us back.
                
                # Correct Scaling:
                # We have sum U V W = 1.
                # Let W_int = D * W.
                # Then sum U V W_int = D.
                # Let U_int = U, V_int = V.
                # We need to divide the product by D.
                # We can divide U by D? Only if D divides U.
                # We can choose U to be multiples of D initially?
                # Let's initialize U with entries from {0, D, -D}.
                # Then U/D is integer.
                # So:
                # 1. Pick D (e.g., 1 initially, or a small number).
                # 2. Initialize U with entries in {0, D, -D}.
                # 3. Initialize V with entries in {0, 1, -1}.
                # 4. Solve for W.
                # 5. If W has denominators, let D_new be LCM.
                # 6. This is circular.
                
                # Better:
                # Just return the integer matrices U, V and the rational W?
                # No, "integer/half-integer entries".
                
                # Let's assume we can find a solution where W is integer.
                # If we solve A W = 1, and A is integer, W is rational.
                # If we multiply A by D (by scaling U or V), we get A' W = D.
                # If we scale U by D, A' = D A.
                # Then (D A) W = D => A W = 1. Same W.
                # So scaling U doesn't change W.
                
                # To make W integer, we need to solve A W = D such that the solution is integer.
                # This is not guaranteed.
                
                # However, we can just multiply the entire decomposition by D.
                # If sum U V W = 1, then sum (D U) V W = D.
                # And sum U (D V) W = D.
                # And sum U V (D W) = D.
                
                # If we set U' = D U, V' = V, W' = W, the value is D.
                # If the evaluator checks for value 1, this fails.
                
                # Wait, the tensor is fixed.
                # We must have sum U V W = 1.
                
                # So we MUST have W rational if U, V are integers.
                # Unless the system is such that W is integer.
                
                # If W is rational, we can write W = N/D.
                # Then sum U V N/D = 1 => sum U V N = D.
                # Let W_int = N.
                # Then sum U V W_int = D.
                # We can define U' = U, V' = V, W' = W_int.
                # The decomposition gives D.
                # To get 1, we can divide U' by D?
                # If D divides all U entries, U'/D is integer.
                # So we should initialize U with entries that are multiples of D.
                # But D is unknown until we solve.
                
                # Solution:
                # 1. Solve for W (rational).
                # 2. Compute D = LCM(denominators of W).
                # 3. Check if D divides all entries of U.
                #    If yes, let U_new = U / D. Then sum U_new V W_int = 1.
                #    U_new is integer. V is integer. W_int is integer.
                #    This works!
                # 4. If D does not divide U, we can try to find a different solution or scale V.
                #    Or, we can just multiply U by D and divide V by D? No.
                
                # So, the strategy is:
                # Initialize U with entries from {0, 1, -1}.
                # Solve for W.
                # Get D.
                # If D==1, we are done.
                # If D>1, we can't simply divide U.
                # BUT, we can just return U, V, W where W is rational?
                # The prompt says "integer/half-integer entries".
                # So we MUST convert.
                
                # If D does not divide U, we can multiply U by D and divide the target by D? No.
                
                # Actually, we can just multiply U and V by D, and W by 1/D^2?
                
                # Let's try a different initialization:
                # Initialize U with entries from {0, 2, -2} (even).
                # Then D might divide U?
                
                # Given the time constraint, I will return the integer U, V and the rational W 
                # converted to the closest half-integer? No, that breaks exactness.
                
                # I will implement the division check. If it fails, I retry with different random U.
                
                # For now, I'll just return the matrices. If D==1, it's perfect.
                # If D>1, I'll try to divide U by D. If it doesn't work, I'll retry.
                
                # Let's assume we find a case where D==1 or D divides U.
                
                # Convert to lists
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W_int # This is D*W, so it's integer.
                
                # If we use W_int, the sum is D.
                # We need sum to be 1.
                # So we must scale U or V by 1/D.
                # If D divides U, U_new = U / D.
                # Else, fail.
                
                # Let's check divisibility
                div_ok = True
                for i in range(n * m):
                    for c in range(r):
                        if U[i, c] % denom_lcm != 0:
                            div_ok = False
                            break
                    if not div_ok:
                        break
                            
                if div_ok:
                    U_list = [[U[i, c] // denom_lcm for c in range(r)] for i in range(n * m)]
                    # V stays same
                    # W is W_int
                    return {"U": U_list, "V": V_list, "W": W_list}
                else:
                    # Try to divide V
                    div_ok_v = True
                    for i in range(m * p):
                        for c in range(r):
                            if V[i, c] % denom_lcm != 0:
                                div_ok_v = False
                                break
                        if not div_ok_v:
                            break
                    if div_ok_v:
                        V_list = [[V[i, c] // denom_lcm for c in range(r)] for i in range(m * p)]
                        return {"U": U_list, "V": V_list, "W": W_list}
                    else:
                        # Fail, try next random
                        continue

    # Fallback: Schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
