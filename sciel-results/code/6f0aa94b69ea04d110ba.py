"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from scipy import linalg
from fractions import Fraction

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 135.0  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    nm = n * m  # 6
    mp = m * p  # 21
    pn = p * n  # 14
    
    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W
    
    def verify(U, V, W, tol=1e-6):
        """Check if the decomposition is exact and has integer/half-integer entries."""
        r = U.shape[1]
        target = np.zeros((nm, mp, pn))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    target[a, b, c] = 1.0
        
        R = np.zeros((nm, mp, pn))
        for c in range(r):
            R += np.einsum('a,b,c->abc', U[:, c], V[:, c], W[:, c])
        
        if np.max(np.abs(R - target)) > tol:
            return False
        
        # Check integer/half-integer
        for mat in [U, V, W]:
            for val in mat:
                if not (abs(val * 2 - round(val * 2)) < tol):
                    return False
        return True
    
    def round_to_half_integer(mat):
        """Round each entry to the nearest half-integer."""
        return np.round(mat * 2) / 2.0
    
    def winograd_fourier_decomposition():
        """
        Apply Winograd Fourier Transform Technique for m=3.
        Embed Z_3 into Z_5 using the 5-point DFT over Q(sqrt(5)).
        
        The 5-point DFT matrix F has entries F[ω^a, ω^b] = ω^{ab} where ω = e^{2πi/5}.
        Since 3 | 4 = 5-1, we can use this to create a structured decomposition.
        
        For the inner dimension m=3, we use the fact that:
        - The cyclic convolution of length 3 can be embedded in length 5
        - The 5-point DFT gives us a rank-5 decomposition for the m-dimension
        - But we only need rank-3 for the actual computation
        
        The key insight: use the 5-point DFT to create a rank-5 basis, then
        project down to rank-3 by exploiting the algebraic relations.
        
        For the full 2x3x7 tensor:
        - n=2: use rank-2 (trivial)
        - m=3: use rank-5 from DFT, but we can reduce to rank-3
        - p=7: use rank-7 (trivial)
        
        Total rank: 2 * 5 * 7 = 70 (too high)
        
        Better approach: use the structure more cleverly.
        
        Actually, let's use a different strategy:
        1. For m=3, use the fact that 3 = (5+1)/2, so we can use a rank-3 decomposition
           based on the 5-point DFT with specific projections.
        
        The 5-point DFT has the property that:
        F * F^* = 5 * I
        
        We can use this to create a decomposition where:
        - U has shape (6, 5*2*7) but we can compress it
        - Actually, let's think differently.
        
        New approach: use the tensor structure directly.
        The tensor T_{a,b,c} = 1 if a = i*m+j, b = j*p+k, c = k*n+i for some (i,j,k)
        
        We can write this as:
        T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k
        
        Now, use the 5-point DFT on the j-index (m=3 dimension):
        Let ω = e^{2πi/5}, and define:
        f_j = sum_{t=0}^{4} F[j,t] * x_t where F is the 5-point DFT matrix
        
        Since m=3 and 3|4, we can embed the 3-dimensional space into 5-dimensional
        using the first 3 basis vectors.
        
        Actually, let's try a simpler approach: use random structured matrices
        with the DFT as a starting point.
        """
        # Use the 5-point DFT structure for the m=3 dimension
        # ω = e^{2πi/5} = cos(2π/5) + i*sin(2π/5)
        # But we need real/rational entries, so use the real part: cos(2π/5) = (√5-1)/4
        
        # Actually, let's use a different approach: systematic search with structure
        # based on the algebraic properties of the tensor.
        
        # For now, return None and fall back to other methods
        return None
    
    def try_structured_rank(rank, time_limit=10.0):
        """
        Try to find a decomposition of given rank using structured random matrices.
        Use the algebraic structure of the tensor to guide the search.
        """
        if time.time() > start_time + time_limit:
            return None
        
        # Strategy: use random linear combinations of the basis tensors
        # T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k
        
        # We want to write T = sum_{c=0}^{rank-1} u_c ⊗ v_c ⊗ w_c
        
        # Use random projections with structure:
        # u_c = sum_{i,j} a_{c,i,j} * e_{i,j}
        # v_c = sum_{j,k} b_{c,j,k} * e_{j,k}  
        # w_c = sum_{k,i} d_{c,k,i} * e_{k,i}
        
        # The constraint is:
        # sum_c a_{c,i,j} * b_{c,j',k} * d_{c,k',i'} = 1 if (j=j', k=k', i=i') else 0
        
        # This is a system of equations in a, b, d.
        
        # For efficiency, fix a and b randomly (with structure), then solve for d.
        
        # Use the fact that the tensor has a "chain" structure:
        # (i,j) -> (j,k) -> (k,i)
        
        # Generate structured random a and b:
        a = np.zeros((rank, n, m))
        b = np.zeros((rank, m, p))
        
        for c in range(rank):
            # Each column has support on a few (i,j) and (j,k) pairs
            # Use random but structured choices
            for i in range(n):
                for j in range(m):
                    if random.random() < 0.4:  # 40% sparsity
                        a[c, i, j] = random.choice([-1, 1]) * random.choice([0.5, 1.0, 1.5])
            
            for j in range(m):
                for k in range(p):
                    if random.random() < 0.3:  # 30% sparsity
                        b[c, j, k] = random.choice([-1, 1]) * random.choice([0.5, 1.0, 1.5])
        
        # Now solve for d (which is w in the original notation)
        # The equation is:
        # sum_c a_{c,i,j} * b_{c,j,k} * d_{c,k,i} = 1 for all (i,j,k)
        
        # This is a linear system in d.
        # For each (k,i), d_{c,k,i} is unknown for each c.
        
        # Let's set up the system:
        # Unknowns: d_{c,k,i} for c=0..rank-1, k=0..p-1, i=0..n-1
        # Total unknowns: rank * p * n = rank * 14
        
        # Equations: for each (i,j,k), one equation
        # Total equations: n * m * p = 42
        
        # But we also need to ensure that the decomposition is zero elsewhere.
        # For (a,b,c) where a=i*m+j, b=j'*p+k, c=k'*n+i' with j≠j' or k≠k' or i≠i',
        # the value should be 0.
        
        # Actually, the tensor is only non-zero at the 42 valid positions,
        # and zero elsewhere. So we need to satisfy:
        # sum_c a_{c,i,j} * b_{c,j',k} * d_{c,k',i'} = δ_{j,j'} δ_{k,k'} δ_{i,i'}
        
        # Let's collect all equations:
        equations = []  # (row_vector, rhs)
        
        # First, the 42 non-zero entries:
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    row = np.zeros(rank * p * n)
                    for c in range(rank):
                        coeff = a[c, i, j] * b[c, j, k]
                        # d_{c,k,i} is at position c * (p*n) + k * n + i
                        row[c * (p * n) + k * n + i] = coeff
                    equations.append((row, 1.0))
        
        # Then, the zero constraints for all other combinations:
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for j2 in range(m):
                        for k2 in range(p):
                            for i2 in range(n):
                                if i == i2 and j == j2 and k == k2:
                                    continue
                                row = np.zeros(rank * p * n)
                                for c in range(rank):
                                    coeff = a[c, i, j] * b[c, j2, k2]
                                    row[c * (p * n) + k2 * n + i2] = coeff
                                equations.append((row, 0.0))
        
        # Build the system
        num_eqs = len(equations)
        A = np.zeros((num_eqs, rank * p * n))
        rhs = np.zeros(num_eqs)
        for idx, (row, val) in enumerate(equations):
            A[idx] = row
            rhs[idx] = val
        
        if time.time() > start_time + time_limit:
            return None
        
        try:
            d_flat, residuals, rank_sys, sv = np.linalg.lstsq(A, rhs, rcond=None)
        except Exception:
            return None
        
        if time.time() > start_time + time_limit:
            return None
        
        d = d_flat.reshape(rank, p, n)
        
        # Construct U, V, W from a, b, d
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        
        for c in range(rank):
            for i in range(n):
                for j in range(m):
                    U[i * m + j, c] = a[c, i, j]
            
            for j in range(m):
                for k in range(p):
                    V[j * p + k, c] = b[c, j, k]
            
            for k in range(p):
                for i in range(n):
                    W[k * n + i, c] = d[c, k, i]
        
        # Verify
        if verify(U, V, W, tol=1e-6):
            return (U, V, W)
        
        # Try rounding
        U_r = round_to_half_integer(U)
        V_r = round_to_half_integer(V)
        W_r = round_to_half_integer(W)
        if verify(U_r, V_r, W_r, tol=1e-6):
            return (U_r, V_r, W_r)
        
        return None
    
    def try_rank_with_support(U, V, rank, time_limit=5.0):
        """
        Given fixed U and V matrices (nm x rank, mp x rank), solve for W.
        """
        if time.time() > start_time + time_limit:
            return None
        
        equations = []
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c_idx = k * n + i
                    
                    row = np.zeros(pn * rank)
                    for c in range(rank):
                        coeff = U[a, c] * V[b, c]
                        row[c_idx * rank + c] = coeff
                    equations.append((row, 1.0))
        
        valid_set = set()
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c_idx = k * n + i
                    valid_set.add((a, b, c_idx))
        
        for a in range(nm):
            for b in range(mp):
                for c_idx in range(pn):
                    if (a, b, c_idx) in valid_set:
                        continue
                    row = np.zeros(pn * rank)
                    for c in range(rank):
                        coeff = U[a, c] * V[b, c]
                        row[c_idx * rank + c] = coeff
                    equations.append((row, 0.0))
        
        num_eqs = len(equations)
        A = np.zeros((num_eqs, pn * rank))
        rhs = np.zeros(num_eqs)
        for idx, (row, val) in enumerate(equations):
            A[idx] = row
            rhs[idx] = val
        
        if time.time() > start_time + time_limit:
            return None
        
        try:
            W_flat, residuals, rank_sys, sv = np.linalg.lstsq(A, rhs, rcond=None)
        except Exception:
            return None
        
        if time.time() > start_time + time_limit:
            return None
        
        W = W_flat.reshape(pn, rank)
        
        if verify(U, V, W, tol=1e-6):
            return W
        
        W_rounded = round_to_half_integer(W)
        if verify(U, V, W_rounded, tol=1e-6):
            return W_rounded
        
        return None
    
    # Try structured approach first
    best_result = None
    best_rank = 42
    
    # Try ranks from 36 down to 30 (targeting Winograd-style rank)
    for rank in range(36, 29, -1):
        if time.time() > deadline - 15:
            break
        
        # Try multiple attempts with structured random matrices
        for attempt in range(15):
            if time.time() > deadline - 15:
                break
            
            result = try_structured_rank(rank, time_limit=5.0)
            if result is not None:
                best_result = result
                best_rank = rank
                break
        
        if best_result is not None:
            break
    
    # If structured approach fails, try the original random approach
    if best_result is None:
        for rank in range(30, 42):
            if time.time() > deadline - 10:
                break
            
            for attempt in range(10):
                if time.time() > deadline - 10:
                    break
                
                U = np.zeros((nm, rank))
                V = np.zeros((mp, rank))
                for c in range(rank):
                    num_nz_u = max(1, int(nm * 0.3))
                    nz_indices = np.random.choice(nm, size=num_nz_u, replace=False)
                    for idx in nz_indices:
                        U[idx, c] = random.choice([-1, 1]) * random.choice([0.5, 1.0, 1.5, 2.0])
                    
                    num_nz_v = max(1, int(mp * 0.2))
                    nz_indices = np.random.choice(mp, size=num_nz_v, replace=False)
                    for idx in nz_indices:
                        V[idx, c] = random.choice([-1, 1]) * random.choice([0.5, 1.0, 1.5, 2.0])
                
                W = try_rank_with_support(U, V, rank, time_limit=3.0)
                if W is not None:
                    best_result = (U, V, W)
                    best_rank = rank
                    break
            
            if best_result is not None:
                break
    
    if best_result is None:
        U, V, W = schoolbook()
        best_result = (U, V, W)
        best_rank = 42
    
    U, V, W = best_result
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
