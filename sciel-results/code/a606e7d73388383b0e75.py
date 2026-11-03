import numpy as np
import time
import itertools
from fractions import Fraction
import random

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    # 1. Schoolbook fallback (Rank 60)
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(nm)]
    V_sb = [[0] * rank_sb for _ in range(mp)]
    W_sb = [[0] * rank_sb for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = rank_sb
    
    def verify_decomposition(U, V, W, r):
        """Verify that the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(r):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(s - 1) > 1e-6:
                        return False
        return True

    def is_nice(w_vec, tol=1e-8):
        """Check if w_vec entries are close to integers or half-integers."""
        w2 = [2 * x for x in w_vec]
        for x in w2:
            if abs(x - round(x)) > tol:
                return False
        return True
        
    def round_to_half(x):
        """Round to nearest half-integer."""
        return round(x * 2) / 2.0

    def try_random_rank(r, max_iters=1000):
        """
        Try to find a rank-r decomposition by random sampling of W.
        For each random W, we check if there exist U, V such that the tensor equation holds.
        """
        if time.time() > deadline:
            return None
        
        # The tensor equation is: for all (i,j,k), sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
        # This is bilinear in U and V for fixed W.
        # We can fix W and try to solve for U and V.
        
        # Strategy: Randomly generate W with entries in {-1, -0.5, 0, 0.5, 1}
        # Then set up a system of equations.
        
        # For each (i,k), let w_ik = W[k*n+i, :] (length r)
        # The constraint is: sum_{j,c} U[i*m+j,c] * V[j*p+k,c] * w_ik[c] = 1
        
        # This is complex. Instead, let's use a simpler heuristic:
        # Assume W is sparse - each row has exactly 1 non-zero entry (a permutation-like structure)
        # Then for each (i,k), W[k*n+i, c_ik] = 1 for some c_ik
        # Then we need: sum_j U[i*m+j, c_ik] * V[j*p+k, c_ik] = 1
        
        # Let's try: W[k*n+i, c] = 1 if c == (i*p + k) % r, else 0
        # This gives a structured W. Then solve for U and V.
        
        # Actually, let's try a different approach: 
        # Use the fact that if we set W[k*n+i, c] = delta_{c, f(i,k)} for some function f,
        # then we need U[i*m+j, f(i,k)] * V[j*p+k, f(i,k)] = 1/m for all j.
        # This is hard to satisfy with half-integers.
        
        # Let's try random W and use least squares to find U and V.
        
        for it in range(max_iters):
            if time.time() > deadline:
                return None
            
            # Generate random W with half-integer entries
            W_rand = []
            for row in range(pn):
                row_data = []
                for col in range(r):
                    # Sparse: 70% zero, 30% non-zero
                    if random.random() < 0.7:
                        row_data.append(0)
                    else:
                        # Choose from {-1, -0.5, 0.5, 1}
                        row_data.append(random.choice([-1, -0.5, 0.5, 1]))
                W_rand.append(row_data)
            
            # Now try to solve for U and V.
            # The equation is: for all (i,j,k), sum_c U[i*m+j,c] * V[j*p+k,c] * W_rand[k*n+i,c] = 1
            # This is a system of n*m*p = 60 equations.
            # Unknowns: U has nm*r = 12*r entries, V has mp*r = 20*r entries.
            # Total unknowns: 32*r. For r >= 2, we have more unknowns than equations.
            
            # We can set up a linear system by fixing one of U or V.
            # Let's fix V randomly and solve for U, then check consistency.
            
            # Actually, this is a bilinear system. Let's use a different approach:
            # Iterate: fix V, solve for U; fix U, solve for V.
            
            # Initialize U and V randomly
            U_curr = [[random.choice([0, 0.5, 1]) for _ in range(r)] for _ in range(nm)]
            V_curr = [[random.choice([0, 0.5, 1]) for _ in range(r)] for _ in range(mp)]
            
            # Iterate
            converged = False
            for inner_it in range(20):
                if time.time() > deadline:
                    return None
                
                # Fix V, solve for U. For each (i,j), we need:
                # sum_k sum_c U[i*m+j,c] * V[j*p+k,c] * W_rand[k*n+i,c] = 1
                # This is: sum_c U[i*m+j,c] * (sum_k V[j*p+k,c] * W_rand[k*n+i,c]) = 1
                # Let A_ikc = V[j*p+k,c] * W_rand[k*n+i,c]
                # Then sum_c U[i*m+j,c] * (sum_k A_ikc) = 1... wait, this isn't quite right.
                
                # The equation is: for each (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W_rand[k*n+i,c] = 1
                # This must hold for ALL (i,j,k). So we have 60 equations.
                
                # Let's set up the full linear system.
                # Variables: U[i*m+j,c] for all i,j,c and V[j*p+k,c] for all j,k,c.
                # But the equations are bilinear, not linear.
                
                # Alternative: Fix W and use a tensor decomposition approach.
                # This is getting complex. Let's try a simpler heuristic.
                
                # Heuristic: Set W[k*n+i, c] = 1 for c = (i * p + k) % r, else 0.
                # Then for each (i,k), we need sum_j U[i*m+j, (i*p+k)%r] * V[j*p+k, (i*p+k)%r] = 1.
                # Let's try U[i*m+j, c] = 1/m and V[j*p+k, c] = 1 for the appropriate c.
                # But 1/m = 0.25 is not a half-integer.
                
                # Try U[i*m+j, c] = 0.5 and V[j*p+k, c] = 2/m = 0.5 for m=4? No, 2/4 = 0.5.
                # sum_j 0.5 * 0.5 = m * 0.25 = 1 for m=4. Yes!
                
                # So if we set W[k*n+i, c] = 1 for c = f(i,k), U[i*m+j, c] = 0.5, V[j*p+k, c] = 0.5,
                # then sum_j U*V = m * 0.25 = 1 for m=4.
                
                # But we need different c for different (i,k) to avoid interference.
                # If f(i,k) is injective, we need r >= n*p = 15.
                
                # Let's try this structured approach!
                if it == 0:  # Only try structured approach on first iteration
                    # f(i,k) = i * p + k, which is injective for i in [0,n), k in [0,p)
                    # So we need r >= n * p = 15
                    if r >= n * p:
                        U_struct = [[0] * r for _ in range(nm)]
                        V_struct = [[0] * r for _ in range(mp)]
                        W_struct = [[0] * r for _ in range(pn)]
                        
                        for i in range(n):
                            for k in range(p):
                                c = i * p + k
                                W_struct[k * n + i][c] = 1
                                for j in range(m):
                                    U_struct[i * m + j][c] = 0.5
                                    V_struct[j * p + k][c] = 0.5
                        
                        # Verify
                        if verify_decomposition(U_struct, V_struct, W_struct, r):
                            return {"U": U_struct, "V": V_struct, "W": W_struct}
            
            # If structured didn't work, try random search with a simpler model
            # Let's try: W is a permutation matrix (one 1 per row)
            # Then for each (i,k), there's exactly one c with W[k*n+i,c] = 1
            # And we need sum_j U[i*m+j,c] * V[j*p+k,c] = 1
            
            perm = list(range(r))
            random.shuffle(perm)
            
            W_perm = [[0] * r for _ in range(pn)]
            for idx in range(pn):
                W_perm[idx][perm[idx]] = 1
            
            # Now solve for U and V
            # For each (i,k), let c_ik = perm[k*n+i]
            # We need sum_j U[i*m+j, c_ik] * V[j*p+k, c_ik] = 1
            
            # Let's set U[i*m+j, c] = a_{i,j} and V[j*p+k, c] = b_{j,k} where c = c_ik
            # Then sum_j a_{i,j} * b_{j,k} = 1 for all i,k
            
            # This is a matrix equation: A * B = J (all-ones matrix) where A is n x m, B is m x p
            # We need A and B with half-integer entries.
            
            # Try A = (1/m) * J, B = J. Then A*B = (1/m) * m * J = J. 
            # But 1/m = 0.25 is not half-integer.
            
            # Try A = 0.5 * J, B = 2/m * J = 0.5 * J for m=4. Then A*B = 0.25 * m * J = J. Yes!
            # So A[i,j] = 0.5, B[j,k] = 0.5 for all i,j,k.
            
            U_perm = [[0] * r for _ in range(nm)]
            V_perm = [[0] * r for _ in range(mp)]
            
            for i in range(n):
                for k in range(p):
                    c = perm[k * n + i]
                    for j in range(m):
                        U_perm[i * m + j][c] = 0.5
                        V_perm[j * p + k][c] = 0.5
            
            if verify_decomposition(U_perm, V_perm, W_perm, r):
                return {"U": U_perm, "V": V_perm, "W": W_perm}
        
        return None

    # Try ranks from minimum possible up to schoolbook rank
    # Minimum rank is max(n*m, m*p, p*n) = max(12, 20, 15) = 20
    min_rank = max(n * m, m * p, p * n)
    
    # Try a few ranks with random search
    for r in range(min_rank, best_rank):
        if time.time() > deadline - 5:
            break
        
        result = try_random_rank(r, max_iters=50)
        if result is not None:
            best_result = result
            best_rank = r
            break
    
    return best_result
# EVOLVE-BLOCK-END
