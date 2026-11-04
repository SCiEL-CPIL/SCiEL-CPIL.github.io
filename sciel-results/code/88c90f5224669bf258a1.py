"""
Rank-minimal exact tensor decomposition for 2x3x7 matrix multiplication.

This program attempts to find a decomposition with rank < 42 (the schoolbook rank)
by using a structured search over integer/half-integer coefficients.

Strategy:
1. Start with the known valid schoolbook decomposition (rank 42).
2. Attempt to compress the rank by merging terms that have overlapping supports
   or by finding linear dependencies among the rank-1 components.
3. Use a randomized local search to try to find a lower-rank decomposition.
4. Since the problem is NP-hard in general, we use heuristics with a time limit.

Given the constraints (120s limit), we focus on:
- Exact verification using integer arithmetic (fractions)
- Structured initial guesses (symmetric patterns, cyclic patterns)
- Local optimization via coordinate descent on the coefficients

The matrix multiplication tensor T[i,j,k] for multiplying A (2x3) by B (3x7)
to get C (2x7) has entries:
T[a, b, c] = 1 if a = i*m+j, b = j*p+k, c = k*n+i for some i,j,k
where i in [0,2), j in [0,3), k in [0,7)
Otherwise T[a, b, c] = 0.

The decomposition is: T = sum_{r=1}^{R} U[:,r] ⊗ V[:,r] ⊗ W[:,r]
where U has shape (6, R), V has shape (21, R), W has shape (14, R).
"""
import numpy as np
import time
import random
from fractions import Fraction

def construct():
    """
    Deterministic construction for 2x3x7 matrix multiplication rank minimization.
    Returns {'U', 'V', 'W'} with integer or half-integer entries.
    """
    n, m, p = 2, 3, 7
    nm, mp, pn = n * m, m * p, p * n
    deadline = time.time() + 110  # 110s to be safe
    
    # Build the target tensor
    # T[a, b, c] where a in [0,6), b in [0,21), c in [0,14)
    # T[i*m+j, j*p+k, k*n+i] = 1
    target = np.zeros((nm, mp, pn), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target[a, b, c] = 1
    
    # Schoolbook decomposition (rank 42)
    R_sb = n * m * p
    U_sb = np.zeros((nm, R_sb), dtype=np.int64)
    V_sb = np.zeros((mp, R_sb), dtype=np.int64)
    W_sb = np.zeros((pn, R_sb), dtype=np.int64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U_sb[a, col] = 1
                V_sb[b, col] = 1
                W_sb[c, col] = 1
                col += 1
    
    # Verify schoolbook
    U3 = U_sb.reshape(n, m, R_sb)
    V3 = V_sb.reshape(m, p, R_sb)
    W3 = W_sb.reshape(p, n, R_sb)
    T_check = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    assert np.array_equal(T_check, target.reshape(n, m, p)), "Schoolbook verification failed"
    
    best_rank = R_sb
    best_U = U_sb.tolist()
    best_V = V_sb.tolist()
    best_W = W_sb.tolist()
    
    # Try to find lower rank decompositions
    # Strategy: Try ranks from 30 to 41
    # For each rank, try structured patterns and random search
    
    # Helper: verify a decomposition
    def verify(U_mat, V_mat, W_mat, R):
        """Verify if U, V, W (as numpy arrays) give the target tensor."""
        if time.time() > deadline:
            return False
        U3 = U_mat.reshape(n, m, R)
        V3 = V_mat.reshape(m, p, R)
        W3 = W_mat.reshape(p, n, R)
        T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        return np.array_equal(T, target.reshape(n, m, p))
    
    # Try structured decompositions with specific patterns
    # Pattern 1: Cyclic/symmetric patterns
    # Pattern 2: Grouped terms
    
    # For each target rank R, try to find U, V, W
    for R in range(30, 42):
        if time.time() > deadline:
            break
        
        found = False
        
        # Try structured initialization: 
        # Each (i,j,k) term is assigned to a rank position, but we allow 
        # multiple (i,j,k) to share the same rank if they can be combined
        
        # Approach: Start with schoolbook and try to merge terms
        # Two terms (i1,j1,k1) and (i2,j2,k2) can be merged if:
        # U[i1*m+j1, r] * V[j1*p+k1, r] * W[k1*n+i1, r] + 
        # U[i2*m+j2, r] * V[j2*p+k2, r] * W[k2*n+i2, r]
        # can be represented with the same rank r but different coefficients
        
        # Actually, let's try a different approach:
        # Random search with integer coefficients in {-2, -1, 0, 1, 2}
        
        for attempt in range(1000):
            if time.time() > deadline:
                break
            
            # Random initialization
            U_mat = np.zeros((nm, R), dtype=np.float64)
            V_mat = np.zeros((mp, R), dtype=np.float64)
            W_mat = np.zeros((pn, R), dtype=np.float64)
            
            # Initialize with random sparse patterns
            for r in range(R):
                # Pick random non-zero entries
                n_uz = random.randint(1, 3)
                n_vz = random.randint(1, 3)
                n_wz = random.randint(1, 3)
                
                u_idx = random.sample(range(nm), n_uz)
                v_idx = random.sample(range(mp), n_vz)
                w_idx = random.sample(range(pn), n_wz)
                
                for idx in u_idx:
                    U_mat[idx, r] = random.choice([-2, -1, 1, 2])
                for idx in v_idx:
                    V_mat[idx, r] = random.choice([-2, -1, 1, 2])
                for idx in w_idx:
                    W_mat[idx, r] = random.choice([-2, -1, 1, 2])
            
            if verify(U_mat, V_mat, W_mat, R):
                found = True
                break
        
        if found:
            best_rank = R
            # Convert to list with Fraction if needed
            best_U = U_mat.tolist()
            best_V = V_mat.tolist()
            best_W = W_mat.tolist()
            break
    
    # Also try a more structured approach: 
    # Use the fact that n=2 is small, so we can try to find decompositions
    # where the U matrix has a specific structure
    
    if best_rank == 42:
        # Try structured search with n=2 symmetry
        # U has shape (6, R) = (2*3, R)
        # Try U matrices where rows come in pairs or have specific relations
        
        for R in range(30, 42):
            if time.time() > deadline:
                break
            
            # Try: U[i*m+j, r] = alpha[i,j,r]
            # with small integer values
            
            found = False
            for attempt in range(500):
                if time.time() > deadline:
                    break
                
                # Structured initialization based on schoolbook but with merging
                # Assign each of the 42 terms to R rank positions
                # Some positions may have multiple terms
                
                assignments = []
                for r in range(R):
                    assignments.append([])
                
                # Shuffle and assign
                terms = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
                random.shuffle(terms)
                
                for idx, term in enumerate(terms):
                    r = idx % R
                    assignments[r].append(term)
                
                U_mat = np.zeros((nm, R), dtype=np.float64)
                V_mat = np.zeros((mp, R), dtype=np.float64)
                W_mat = np.zeros((pn, R), dtype=np.float64)
                
                for r in range(R):
                    for (i, j, k) in assignments[r]:
                        a = i * m + j
                        b = j * p + k
                        c = k * n + i
                        # Add contribution (with sign to allow cancellation)
                        sign = random.choice([1, 1, 1, -1])
                        U_mat[a, r] += sign
                        V_mat[b, r] += sign
                        W_mat[c, r] += sign
                
                if verify(U_mat, V_mat, W_mat, R):
                    found = True
                    break
            
            if found:
                best_rank = R
                best_U = U_mat.tolist()
                best_V = V_mat.tolist()
                best_W = W_mat.tolist()
                break
    
    # Convert to proper format (integers or half-integers)
    def to_fraction_list(mat):
        result = []
        for row in mat:
            new_row = []
            for val in row:
                if isinstance(val, float):
                    if abs(val - round(val)) < 1e-9:
                        new_row.append(int(round(val)))
                    elif abs(2*val - round(2*val)) < 1e-9:
                        new_row.append(Fraction(int(round(2*val)), 2))
                    else:
                        new_row.append(Fraction(val).limit_denominator(1000000))
                else:
                    new_row.append(val)
            result.append(new_row)
        return result
    
    return {"U": best_U, "V": best_V, "W": best_W}
