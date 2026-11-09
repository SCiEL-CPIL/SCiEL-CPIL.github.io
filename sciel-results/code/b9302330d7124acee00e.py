"""Matrix multiplication 3x3x7 rank minimization."""
import time
import itertools
import numpy as np
from collections import defaultdict

DIMENSIONS = [3, 3, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    n, m, p = 3, 3, 7
    # The multiplication tensor T[i,j,k, a,b,c] = 1 if i=a, j=b, k=c
    # We want to decompose: T = sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r]
    # where U is (n*m) x rank, V is (m*p) x rank, W is (p*n) x rank
    # T[i*m+j, k*p+..., ] - actually the tensor is over (i,j) for first, (j,k) for second, (k,i) for third
    
    # Actually the tensor T has indices (i, j, k) where:
    # First factor: (i, j) -> index i*m + j in U
    # Second factor: (j, k) -> index j*p + k in V  
    # Third factor: (k, i) -> index k*n + i in W
    # T[i*m+j, j*p+k, k*n+i] = 1
    
    # Rank-1 term r: U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Sum over r should equal T[i*m+j, j*p+k, k*n+i]
    
    # For each (i,j,k), we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # And for "cross terms" where the indices don't align properly, we need 0.
    
    # Actually, the tensor is only non-zero when the indices are consistent:
    # T is a 3x3x7 x 3x7x3 x 7x3x3 tensor? No...
    
    # Let me re-read: U shape (n*m, rank) = (9, rank), V shape (m*p, rank) = (21, rank), W shape (p*n, rank) = (21, rank)
    # The multiplication: C = A @ B where A is 3x3, B is 3x7, C is 3x7
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # Tensor: T[i,j,k] = 1 (the bilinear form)
    # Decomposition: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]... 
    
    # Wait, W has shape (p*n, rank) = (21, rank), indexed by k*n + i
    # So the bilinear form is: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # This should equal 1 if the "multiplication" gives C[i,k] from A[i,j]*B[j,k]
    
    # The target: for the matrix multiplication, the coefficient of A[i,j]*B[j,k] in C[i,k] is 1.
    # So we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    # But wait, that's not right either. The product A[i,j]*B[j,k] contributes to C[i,k].
    # So the tensor T has T[i,j,k] = 1 for all i in [0,3), j in [0,3), k in [0,7)
    # And the decomposition is: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Hmm, but W is indexed by (k,i) with k in [0,7), i in [0,3). 
    # So W[k*n+i, r] is the coefficient for the (k,i) pair in the third factor.
    
    # Actually I think the convention is:
    # U[i*m + j, r] - coefficient for A[i,j]
    # V[j*p + k, r] - coefficient for B[j,k]  
    # W[k*n + i, r] - coefficient for C[i,k] (but indexed as k*n+i, so k is "row" in a 7x3 sense?)
    
    # The bilinear form: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] should give the coefficient
    # of A[i,j]*B[j,k] in C[i,k], which is 1.
    
    # So the constraint is: for all i in [0,3), j in [0,3), k in [0,7):
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # This is a system of 3*3*7 = 63 equations.
    
    # Strategy: Try small ranks with structured patterns.
    # Known: 3x3x3 has rank 23. For 3x3x7, we can potentially do better than 63.
    
    # Let's try a systematic approach:
    # 1. Start with schoolbook rank 63
    # 2. Try to reduce by finding linear dependencies / using Strassen-like tricks
    
    # Actually, let me try a direct search with small integer entries.
    # For rank R, we have 9*R + 21*R + 21*R = 51*R variables, with 63 constraints.
    # For R=23: 1173 variables, 63 constraints - very underdetermined.
    
    # But the constraints are cubic (product of three variables), so it's nonlinear.
    
    # Let me try a different approach: use the fact that we can combine terms.
    # For 3x3x3, the optimal is 23. For 3x3x7, maybe we can achieve something like 23*7/3 ≈ 54?
    # Or maybe there's a better structure.
    
    # Let me try to construct a decomposition by extending the 3x3x3 result.
    
    # First, let's get a 3x3x3 decomposition of rank 23.
    # I'll use a known construction or search for one.
    
    def find_3x3x3_decomposition(max_rank=23, time_limit=30.0):
        """Find a rank decomposition for 3x3x3 matrix multiplication."""
        n3, m3, p3 = 3, 3, 3
        start = time.time()
        
        # Try rank 23 with small integer entries
        # Use a greedy/random search with integer entries in {-1, 0, 1, 2}
        
        import random
        random.seed(42)
        
        best_rank = None
        best_decomp = None
        
        for rank in range(23, 27):
            if time.time() - start > time_limit:
                break
            
            # Try random integer decompositions
            for attempt in range(20000):
                if time.time() - start > time_limit:
                    break
                
                U = np.zeros((9, rank), dtype=float)
                V = np.zeros((9, rank), dtype=float)
                W = np.zeros((9, rank), dtype=float)
                
                # Fill with small integers
                for r in range(rank):
                    for idx in range(9):
                        if random.random() < 0.5:
                            U[idx, r] = random.choice([-1, 1, 2])
                        if random.random() < 0.5:
                            V[idx, r] = random.choice([-1, 1, 2])
                        if random.random() < 0.5:
                            W[idx, r] = random.choice([-1, 1, 2])
                
                # Check if this decomposes the tensor
                valid = True
                for i in range(3):
                    for j in range(3):
                        for k in range(3):
                            val = np.sum(U[i*3+j, :] * V[j*3+k, :] * W[k*3+i, :])
                            if abs(val - 1.0) > 1e-9:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                if valid:
                    best_rank = rank
                    best_decomp = (U, V, W)
                    break
            
            if best_rank is not None:
                break
        
        return best_rank, best_decomp
    
    # For 3x3x7, let's try a structured approach.
    # Idea: Use the 3x3x3 decomposition as a building block.
    # For each of the 7 columns of B, we can use a 3x3x3-like decomposition.
    # But that would give rank 23*7 = 161, which is worse than schoolbook 63.
    
    # Better idea: The tensor for 3x3x7 is:
    # T[i,j,k] = 1 for i in [0,3), j in [0,3), k in [0,7)
    # We need sum_r U[i*3+j, r] * V[j*7+k, r] * W[k*3+i, r] = 1
    
    # Note that V has 21 entries (3*7) and W has 21 entries (7*3).
    
    # Let me try a direct ILP-like search with small ranks.
    # Start with rank 63 (schoolbook) and try to reduce.
    
    # Actually, let me just implement the schoolbook and see if we can find improvements
    # through a local search.
    
    # Schoolbook:
    rank_school = 63
    U_school = np.zeros((9, rank_school))
    V_school = np.zeros((21, rank_school))
    W_school = np.zeros((21, rank_school))
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(7):
                U_school[i*3+j, col] = 1
                V_school[j*7+k, col] = 1
                W_school[k*3+i, col] = 1
                col += 1
    
    # Verify schoolbook
    valid = True
    for i in range(3):
        for j in range(3):
            for k in range(7):
                val = np.sum(U_school[i*3+j, :] * V_school[j*7+k, :] * W_school[k*3+i, :])
                if abs(val - 1.0) > 1e-9:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    # Now try to reduce the rank using local search.
    # Idea: Try to replace pairs of rank-1 terms with a single term.
    # Or use a more systematic approach.
    
    # Let me try a different strategy: use the fact that for fixed j, 
    # the terms involving A[i,j] and B[j,k] can be combined.
    
    # For a fixed j, we have 3*7 = 21 terms in schoolbook (one for each (i,k)).
    # Can we reduce these? The sub-tensor for fixed j is 3x7x3 (i,k, and the "j" is fixed).
    # This is like multiplying a 3x1 vector by a 1x7 vector, which has rank 3*7 = 21? 
    # No, it's an outer product, which has rank 1! But that's not quite right because
    # the decomposition needs to work simultaneously for all j.
    
    # Actually, for fixed j, the contribution is A[i,j] * B[j,k] for all i,k.
    # This is rank 1 in the (i,k) space for each j. But we need a joint decomposition.
    
    # Let me try a structured decomposition:
    # Use the 3x3x3 optimal (rank 23) for the "core" and extend to 7 columns.
    
    # Alternative: Try rank values between 23 and 63 and search for valid decompositions.
    
    def try_rank(R, time_budget=10.0):
        """Try to find a decomposition of rank R."""
        start = time.time()
        import random
        random.seed(123)
        
        U = np.zeros((9, R))
        V = np.zeros((21, R))
        W = np.zeros((21, R))
        
        # Initialize with schoolbook-like structure
        for r in range(min(R, 63)):
            i = r // 7
            j = (r // 7) % 3
            k = r % 7
            # Actually let me just do random initialization
            pass
        
        # Random initialization with small integers
        for r in range(R):
            for idx in range(9):
                if random.random() < 0.3:
                    U[idx, r] = random.choice([-1, 1, 2])
            for idx in range(21):
                if random.random() < 0.3:
                    V[idx, r] = random.choice([-1, 1, 2])
            for idx in range(21):
                if random.random() < 0.3:
                    W[idx, r] = random.choice([-1, 1, 2])
        
        # Gradient descent-like refinement
        for iteration in range(5000):
            if time.time() - start > time_budget:
                break
            
            # Compute error
            error = 0.0
            for i in range(3):
                for j in range(3):
                    for k in range(7):
                        val = np.sum(U[i*3+j, :] * V[j*7+k, :] * W[k*3+i, :])
                        error += (val - 1.0) ** 2
            
            if error < 1e-12:
                return True, U, V, W
            
            # Try random perturbations
            for _ in range(10):
                r = random.randint(0, R-1)
                mat = random.choice([U, V, W])
                idx = random.randint(0, mat.shape[0]-1)
                old_val = mat[idx, r]
                mat[idx, r] += random.choice([-1, 1, 0.5, -0.5])
                
                # Compute new error
                new_error = 0.0
                for i in range(3):
                    for j in range(3):
                        for k in range(7):
                            val = np.sum(U[i*3+j, :] * V[j*7+k, :] * W[k*3+i, :])
                            new_error += (val - 1.0) ** 2
                
                if new_error < error:
                    error = new_error
                else:
                    mat[idx, r] = old_val
        
        return False, U, V, W
    
    # Try various ranks
    best_rank = 63
    best_U, best_V, best_W = U_school.copy(), V_school.copy(), W_school.copy()
    
    for R in range(50, 64):
        if time.time() > deadline - 20:
            break
        success, U, V, W = try_rank(R, time_budget=5.0)
        if success:
            best_rank = R
            best_U, best_V, best_W = U, V, W
            break
    
    # Convert to list format
    U_list = [[int(best_U[i, r]) if abs(best_U[i, r] - round(best_U[i, r])) < 1e-9 else best_U[i, r] for r in range(best_rank)] for i in range(9)]
    V_list = [[int(best_V[i, r]) if abs(best_V[i, r] - round(best_V[i, r])) < 1e-9 else best_V[i, r] for r in range(best_rank)] for i in range(21)]
    W_list = [[int(best_W[i, r]) if abs(best_W[i, r] - round(best_W[i, r])) < 1e-9 else best_W[i, r] for r in range(best_rank)] for i in range(21)]
    
    return {"U": U_list, "V": V_list, "W": W_list}
