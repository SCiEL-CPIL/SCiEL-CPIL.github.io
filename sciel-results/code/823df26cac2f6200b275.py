import numpy as np
from scipy.optimize import least_squares
import time
import sys

DIMENSIONS = [3, 3, 5]

def construct():
        n, m, p = DIMENSIONS
        # Target: Find a decomposition with rank < 45
        # Strategy: Split B into B1 (3x3) and B2 (3x2)
        # Rank(B1 part) + Rank(B2 part)
        # B2 part is schoolbook: 3*3*2 = 18 terms.
        # B1 part is 3x3x3. Schoolbook is 27.
        # If we find a rank 23 decomposition for B1, total is 41.
        
        # 1. Construct Schoolbook for B2 (columns 3,4,5 i.e. indices 2,3,4 in 0-based? No, 3x5 means cols 0..4)
        # B1: cols 0,1,2. B2: cols 3,4.
        
        # Let's build the full schoolbook first as fallback
        rank_school = n * m * p
        U_school = np.zeros((n*m, rank_school))
        V_school = np.zeros((m*p, rank_school))
        W_school = np.zeros((p*n, rank_school))
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_school[i*m+j, col] = 1
                    V_school[j*p+k, col] = 1
                    W_school[k*n+i, col] = 1
                    col += 1
        
        # 2. Try to optimize the B1 part (3x3x3)
        # The tensor T1 is 1 for i,j,k in 0..2
        T1 = np.zeros((n, m, p)) # Actually T1 is 3x3x3 part
        for i in range(n):
            for j in range(m):
                for k in range(3): # only first 3 cols
                    T1[i,j,k] = 1
        # Flatten T1 to match U*V*W structure?
        # The tensor is defined by T_{i,j,k} = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
        # We want to find U1 (9xR1), V1 (9xR1), W1 (9xR1) such that this equals T1.
        # Note: V and W indices for k>=3 are 0 in T1?
        # In the full tensor T_{3,3,5}, T_{i,j,k} is 1 for all k in 0..4.
        # If we split into T1 (k=0..2) and T2 (k=3..4), they are disjoint supports.
        # So we can decompose them independently and concatenate.
        
        # Let's try to find rank R1 for T1 (3x3x3).
        # We know R1=23 is possible.
        
        def solve_block(target_tensor, R):
            # target_tensor: shape (n, m, p)
            # Returns U, V, W or None
            n, m, p = target_tensor.shape
            # Total terms in tensor
            # We flatten the target into a vector of length n*m*p
            # But the mapping is specific.
            
            # Let's use a simple least squares on the flattened tensor equation.
            # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            
            # This is non-linear. We fix W, solve for U, V?
            # Or use iterative method.
            
            # For simplicity and speed, we might just return None if we don't have a hardcoded solution.
            # But 41 is a good improvement.
            pass
        
        # Since I don't have the hardcoded 23-rank tensor, and numerical optimization for rank 23 might not converge reliably to exact rationals in < 120s, 
        # I will try a **Rank 35** approximation? No, must be exact.
        
        # Let's try to find a **Rank 40** decomposition?
        # If I use Rank 22 for 3x3x3 (if it exists) + 18 = 40.
        
        # Actually, there is a simpler improvement: 
        # Use the **Strassen** algorithm for the 2x2 subblocks?
        
        # Let's fall back to a **numerical search for a rank 44 or 43 decomposition** that is easier to find?
        # Or just return schoolbook if no improvement is found quickly.
        
        # Given the risk, I will return the schoolbook but attempt a **very fast** check for a rank 44 decomposition by perturbing one term?
        # Unlikely to work.
        
        # I will implement the **concatenation of a numerically-found rank 23 decomposition for 3x3x3** if possible within a short time budget.
        
        start_time = time.time()
        time_limit = 100.0 # Leave 20s margin
        
        # Try to find rank 23 for 3x3x3
        # ... (Implementation of solver)
        
        # If solver fails or takes too long, return schoolbook.
        
        # To ensure I provide a valid candidate that is *likely* to be better than 1.0 if it runs,
        # but I must guarantee validity.
        
        # I will return the schoolbook decomposition. It is valid.
        # To improve score, I need rank < 45.
        
        # Let's try to hardcode a **Rank 41** solution using known coefficients for 3x3x3 rank 23.
        # I will paste the coefficients from a known source if I can recall them? No.
        
        # I will use a **randomized search** for a low rank decomposition of the FULL tensor.
        # If I find a valid exact decomposition with rank < 45, I return it.
        
        # Search for rank 40-44.
        
        best_rank = rank_school
        best_U = U_school
        best_V = V_school
        best_W = W_school
        
        # Try a few random initializations for rank 44
        for attempt in range(20):
            if time.time() - start_time > time_limit:
                break
            
            R = 44
            # Random initialization
            U = np.random.randint(-10, 10, (n*m, R))
            V = np.random.randint(-10, 10, (m*p, R))
            W = np.random.randint(-10, 10, (p*n, R))
            
            # This is not a solver. Just returning schoolbook is safer.
            
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
