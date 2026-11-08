"""Matrix multiplication 2,3,8 - rank-minimal tensor decomposition."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction
from itertools import product

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    # Schoolbook decomposition as fallback (rank 48)
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_rank = rank_school
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    
    # The tensor condition: for all i in [0,n), j in [0,m), k in [0,p):
    # sum_s U[i*m+j, s] * V[j*p+k, s] * W[k*n+i, s] = 1
    #
    # Strategy: For small n=2, we can try to find structured decompositions.
    # Key insight: We can fix U and V columns and solve for W, or use algebraic methods.
    #
    # Let's try ranks from a lower bound up. The rank must be >= max(n*m, m*p, p*n) / something.
    # Actually, a better lower bound: rank >= n*p = 16 (each output element needs coverage)
    # But more tightly: the tensor has structure that may allow rank < 48.
    #
    # Approach: Try to find decompositions where U and V have sparse structure.
    # For each rank r, try random sparse initializations and check if W can be solved.
    
    # For a given U and V, the condition becomes:
    # For each (i,j,k): sum_s U[i*m+j, s] * V[j*p+k, s] * W[k*n+i, s] = 1
    # This is linear in W! For fixed U, V, we have a linear system in W.
    #
    # But W appears coupled: W[k*n+i, s] multiplies U[i*m+j, s]*V[j*p+k, s].
    # Let's define: for each s, let a_s(i,j) = U[i*m+j, s], b_s(j,k) = V[j*p+k, s], c_s(k,i) = W[k*n+i, s]
    # Then sum_s a_s(i,j) * b_s(j,k) * c_s(k,i) = 1
    #
    # This is NOT linear in W because W is multiplied by U and V which are also variables.
    # 
    # Alternative: Fix U and V, then the system is:
    # For each (i,j,k): sum_s [U[i*m+j, s] * V[j*p+k, s]] * W[k*n+i, s] = 1
    # Let M_{(i,j,k), s} = U[i*m+j, s] * V[j*p+k, s]
    # Then we need: M * W_flat = 1_vector
    # where W_flat is a vector of length p*n*r (reshaping W columns)
    #
    # Wait, that's not right either. W[k*n+i, s] is indexed by (k,i) and s.
    # Let me re-index: let w_{k,i,s} = W[k*n+i, s]
    # Then: sum_s M_{(i,j,k), s} * w_{k,i,s} = 1 for all (i,j,k)
    #
    # This IS linear in w! For fixed U and V, we have a linear system.
    # Number of equations: n*m*p = 48
    # Number of unknowns: p*n*r = 16*r
    # So we need 16*r >= 48, i.e., r >= 3 for the system to be potentially solvable.
    # But we also need the system to be consistent.
    
    # Let's try: for each rank r, generate random U and V with entries from a small set,
    # then solve the linear system for W. If a solution exists, we have a valid decomposition.
    
    # To make it more likely to work, use structured initializations.
    # Try entries from {-1, 0, 1} or {0, 1} or {-1, 0, 1, 2}.
    
    def try_decomposition(U, V, r):
        """Given U (nm x r) and V (mp x r), solve for W (pn x r) if possible."""
        # Build the linear system: for each (i,j,k), sum_s U[i*m+j, s] * V[j*p+k, s] * W[k*n+i, s] = 1
        # Unknowns: W[k*n+i, s] for k in [0,p), i in [0,n), s in [0,r)
        # Let's index unknowns by (k, i, s) -> flat index k*n*r + i*r + s
        
        num_eq = n * m * p  # 48
        num_unk = p * n * r  # 16 * r
        
        if num_unk < num_eq:
            return None  # Underdetermined in a bad way
        
        # Build matrix A and vector b
        # A[eq_idx, unk_idx] = U[i*m+j, s] * V[j*p+k, s] if unk_idx corresponds to (k,i,s)
        
        # For efficiency, build A as a sparse-like structure
        # eq_idx = i*m*p + j*p + k
        # unk_idx = k*n*r + i*r + s
        
        A = np.zeros((num_eq, num_unk))
        b = np.ones(num_eq)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    eq_idx = i * m * p + j * p + k
                    for s in range(r):
                        unk_idx = k * n * r + i * r + s
                        A[eq_idx, unk_idx] = U[i * m + j, s] * V[j * p + k, s]
        
        # Solve A * W_flat = b
        try:
            W_flat, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
        except:
            return None
        
        # Check if the solution is exact
        if not np.allclose(A @ W_flat, b, atol=1e-10):
            return None
        
        # Reshape W_flat to W (pn x r)
        W = W_flat.reshape(p * n, r)
        
        # Verify the solution
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U[i * m + j, s] * V[j * p + k, s] * W[k * n + i, s] for s in range(r))
                    if abs(val - 1) > 1e-10:
                        return None
        
        return W
    
    # Try different ranks and random initializations
    # Lower bound: rank >= ceil(n*m*p / (something))... let's just try from 16 up
    
    # For better results, try structured U and V
    # Key idea: Make U and V sparse to make the linear system easier to satisfy
    
    # Try ranks from 16 to 47
    for target_rank in range(16, rank_school):
        if time.time() > deadline - 15:
            break
        
        found = False
        # Try multiple random initializations
        max_attempts = 200 if target_rank < 30 else 50
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                break
            
            np.random.seed(12345 + attempt * 7 + target_rank)
            
            # Try different sparsity patterns
            sparsity = 0.3 if target_rank < 25 else 0.2
            U = np.zeros((n * m, target_rank))
            V = np.zeros((m * p, target_rank))
            
            # Random sparse entries from {-1, 0, 1}
            for idx in range(n * m):
                for s in range(target_rank):
                    if np.random.random() < sparsity:
                        U[idx, s] = np.random.choice([-1, 1])
            
            for idx in range(m * p):
                for s in range(target_rank):
                    if np.random.random() < sparsity:
                        V[idx, s] = np.random.choice([-1, 1])
            
            W = try_decomposition(U, V, target_rank)
            if W is not None:
                # Found a valid decomposition
                best_rank = target_rank
                best_result = {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
                found = True
                break
        
        if found:
            break
    
    # If we didn't find anything better, try a more structured approach
    # For 2x3x8, try to exploit the small n=2 dimension
    if best_rank == rank_school:
        # Try a structured approach: partition by i=0 and i=1
        # For each i, we have a 3x8 matrix multiplication contribution
        # C[i,k] = sum_j A[i,j] B[j,k]
        
        # Try rank 24: two separate 3x8 multiplications (one for each i)
        # But that's still 2*24=48... no, each 3x8 has rank 24, so total 48.
        
        # Try a smarter approach: share some terms between i=0 and i=1
        # Let's try rank 30 with a structured initialization
        
        for target_rank in [30, 28, 26, 24]:
            if time.time() > deadline - 10:
                break
            
            # Structured initialization: 
            # For s in [0, 12): handle i=0, j in [0,3), k in [0,4)
            # For s in [12, 24): handle i=1, j in [0,3), k in [0,4)
            # For s in [24, target_rank): handle i=0, j in [0,3), k in [4,8) and i=1, j in [0,3), k in [4,8)
            
            U = np.zeros((n * m, target_rank))
            V = np.zeros((m * p, target_rank))
            
            # First 12 terms: i=0, all j, k=0..3
            s = 0
            for j in range(m):
                for k in range(4):
                    U[0 * m + j, s] = 1
                    V[j * p + k, s] = 1
                    s += 1
            
            # Next 12 terms: i=1, all j, k=0..3
            for j in range(m):
                for k in range(4):
                    U[1 * m + j, s] = 1
                    V[j * p + k, s] = 1
                    s += 1
            
            # Remaining terms: i=0, k=4..7 and i=1, k=4..7
            # We need 3*4 + 3*4 = 24 more terms, but we only have target_rank - 24 left
            # So this structured approach only works for rank >= 48... not helpful.
            
            # Let's try a different structured approach
            # Use the fact that we can combine terms
            pass
        
        # Try random with different entry sets
        for target_rank in range(20, rank_school):
            if time.time() > deadline - 10:
                break
            
            for attempt in range(100):
                if time.time() > deadline - 5:
                    break
                
                np.random.seed(99999 + attempt * 13 + target_rank * 7)
                
                U = np.random.choice([-1, 0, 1], size=(n * m, target_rank))
                V = np.random.choice([-1, 0, 1], size=(m * p, target_rank))
                
                W = try_decomposition(U, V, target_rank)
                if W is not None:
                    best_rank = target_rank
                    best_result = {
                        "U": U.tolist(),
                        "V": V.tolist(),
                        "W": W.tolist()
                    }
                    break
            
            if best_rank < rank_school:
                break
    
    return best_result
# EVOLVE-BLOCK-END
