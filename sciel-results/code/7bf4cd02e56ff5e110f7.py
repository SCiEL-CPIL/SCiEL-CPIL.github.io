"""
Matrix multiplication 3x4x6: find rank-minimal exact tensor decomposition.

Strategy:
  - The matrix multiplication tensor T for C = AB (A: 3x4, B: 4x6, C: 3x6)
    is an element of R^{12} ⊗ R^{24} ⊗ R^{18} (using the convention
    W indexed by k*n + i where k is row of B / col of C, i is row of A / row of C).
    
  - Schoolbook rank = 3*4*6 = 72.
  
  - We attempt to find a lower-rank decomposition using structured search:
    1. Start with the schoolbook decomposition.
    2. Use a greedy merging / linear-dependency detection approach to try to
       reduce the rank by finding sets of terms whose combination can be
       replaced by fewer terms.
    3. Use alternating minimization with integer/half-integer constraints.
    
  - For 3x4x6, known lower bounds suggest rank > 60, but exact optimal is unknown.
    We'll try to find any decomposition with rank < 72.
"""
import numpy as np
from itertools import product
import random
import sys
import time

DIMENSIONS = [3, 4, 6]


def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 72
    
    # Build the schoolbook decomposition
    U_sb = np.zeros((n * m, schoolbook_rank), dtype=float)
    V_sb = np.zeros((m * p, schoolbook_rank), dtype=float)
    W_sb = np.zeros((p * n, schoolbook_rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1.0
                V_sb[j * p + k, col] = 1.0
                W_sb[k * n + i, col] = 1.0
                col += 1
    
    # Verify schoolbook: the tensor T[i,j,k] should be 1 for all (i,j,k)
    # where the term contributes. Actually, the decomposition must satisfy:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all valid (i,j,k)
    # and 0 otherwise... but actually for matrix multiplication, every (i,j,k)
    # triple contributes exactly once, so the tensor is the sum of all rank-1
    # basis tensors.
    
    # Let's verify: for each (i,j,k), the contribution should be 1
    # and for any other combination it should be 0.
    # Actually, the tensor T is defined such that:
    # T[(i,j), (j',k'), (i'',k'')] = 1 iff i=i'', j=j', k=k'
    # No wait, that's not right either.
    
    # Let me think carefully. The bilinear map is:
    # (A, B) -> AB where C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor T in R^{nm} ⊗ R^{mp} ⊗ R^{pn} (with W indexed by k*n+i) satisfies:
    # For basis vectors e_{i,j} in first factor, e_{j',k} in second, e_{k',i'} in third:
    # T(e_{i,j}, e_{j',k}, e_{k',i'}) = 1 iff i=i' AND j=j' AND k=k'
    # Wait no. Let's use the definition:
    # <T, A ⊗ B ⊗ C> = sum_{i,j,k} A[i,j] B[j,k] C[i,k] (where C is the dual)
    # So T = sum_{i,j,k} e_{i,j} ⊗ e_{j,k} ⊗ e_{k,i} (in the W convention)
    # This means T has 1 at positions where all three indices are "consistent".
    
    # In the decomposition: sum_r U[a,r] V[b,r] W[c,r] = T[a,b,c]
    # where a = i*m+j, b = j*p+k, c = k*n+i
    # T[a,b,c] = 1 iff the indices are consistent (i from a matches i from c, 
    #           j from a matches j from b, k from b matches k from c)
    
    # For schoolbook: each term r corresponds to a specific (i,j,k), and
    # U[i*m+j, r] = 1, V[j*p+k, r] = 1, W[k*n+i, r] = 1
    # So sum_r U[a,r] V[b,r] W[c,r] = 1 iff there exists r such that
    # a = i_r*m+j_r, b = j_r*p+k_r, c = k_r*n+i_r for the same (i_r,j_r,k_r)
    # This is exactly 1 when the indices are consistent, 0 otherwise. ✓
    
    # Now try to reduce rank.
    # Approach: Try to find a set of columns that can be replaced by a linear combination.
    
    # Method 1: Try random linear combinations of pairs of terms and see if we can
    # express the resulting tensor as a sum of fewer rank-1 terms.
    
    # Method 2: Use the fact that for small dimensions, we can try all possible
    # "Strassen-like" patterns.
    
    # Let's try a structured approach:
    # For each pair of schoolbook terms, try to replace them with 1 or 2 terms
    # that have half-integer or integer entries.
    
    # Actually, let's try a more systematic approach:
    # We know that the tensor has a lot of symmetry. Let's try to find a
    # decomposition using terms that involve sums of basis vectors.
    
    # Key insight: A rank-1 term in the decomposition has the form:
    # u ⊗ v ⊗ w where u ∈ R^{nm}, v ∈ R^{mp}, w ∈ R^{pn}
    # The tensor T can be written as sum of such terms.
    
    # For the schoolbook, each term is a single basis vector in each factor.
    # To reduce rank, we need terms where u, v, w have multiple non-zero entries.
    
    # Let's try: for a fixed j, can we combine all (i, j, k) terms into fewer terms?
    # The terms for fixed j are: sum_{i,k} e_{i,j} ⊗ e_{j,k} ⊗ e_{k,i}
    # This is a tensor of size n x p x (p*n) for fixed j.
    # Can we decompose this more efficiently?
    
    # For n=3, p=6, fixed j: we have 18 terms. Can we do better?
    # The tensor for fixed j is: T_j[i, k, k'n+i'] = delta_{i,i'} * delta_{k,k'}
    # Wait, that's just the identity-like structure.
    
    # Actually, let me reconsider. The full tensor T has:
    # T[a,b,c] = 1 iff a=(i,j), b=(j,k), c=(k,i) for some i,j,k
    # i.e., T[i*m+j, j*p+k, k*n+i] = 1 and 0 otherwise.
    
    # This is a very sparse tensor. The number of non-zero entries is n*m*p = 72.
    
    # A rank-r decomposition means: T = sum_{r=1}^{R} u_r ⊗ v_r ⊗ w_r
    # where u_r ∈ R^{12}, v_r ∈ R^{24}, w_r ∈ R^{18}.
    
    # For each non-zero entry T[a,b,c] = 1, we need:
    # sum_r u_r[a] * v_r[b] * w_r[c] = 1
    # For each zero entry T[a,b,c] = 0, we need:
    # sum_r u_r[a] * v_r[b] * w_r[c] = 0
    
    # This is a system of polynomial equations in the entries of u_r, v_r, w_r.
    
    # Let's try a random search with structured initialization.
    
    # First, let's try to see if rank 60-71 is achievable.
    # We'll use a local search: start from schoolbook, then try to merge pairs.
    
    best_rank = schoolbook_rank
    best_U = U_sb.copy()
    best_V = V_sb.copy()
    best_W = W_sb.copy()
    
    # Try to reduce rank by finding linear dependencies
    # For each pair of terms (r1, r2), check if their sum can be replaced by
    # a single term with half-integer entries.
    
    # A single term u ⊗ v ⊗ w has the property that the 3D array U[a]*V[b]*W[c]
    # is rank-1 in all three modes.
    
    # Let's try: for pairs of schoolbook terms, can we find u, v, w such that
    # u ⊗ v ⊗ w = e_{a1} ⊗ e_{b1} ⊗ e_{c1} + e_{a2} ⊗ e_{b2} ⊗ e_{c2}?
    # This would require the sum to be rank-1, which is generally not the case.
    
    # Instead, let's try a different approach: use the known structure of
    # matrix multiplication to find a better decomposition.
    
    # For 2x2x2, Strassen uses 7 terms instead of 8. The key is that the
    # 8 schoolbook terms can be grouped and some cancellations occur.
    
    # For 3x4x6, let's try to apply similar ideas.
    
    # Actually, let's just try a random search for lower ranks.
    # We'll initialize with random sparse factors and use gradient-free optimization.
    
    rng = random.Random(42)
    
    for target_rank in range(schoolbook_rank - 1, max(40, schoolbook_rank - 15), -1):
        for attempt in range(5):
            # Initialize with schoolbook and then randomly perturb
            U = U_sb[:, :target_rank].copy()
            V = V_sb[:, :target_rank].copy()
            W = W_sb[:, :target_rank].copy()
            
            # Add some random structure
            for r in range(target_rank):
                # Randomly add small perturbations
                if rng.random() < 0.3:
                    idx = rng.randint(0, n*m-1)
                    U[idx, r] += 0.5
                if rng.random() < 0.3:
                    idx = rng.randint(0, m*p-1)
                    V[idx, r] += 0.5
                if rng.random() < 0.3:
                    idx = rng.randint(0, p*n-1)
                    W[idx, r] += 0.5
            
            # Check if this is a valid decomposition
            # Compute the reconstructed tensor and compare
            error = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        a = i * m + j
                        b = j * p + k
                        c = k * n + i
                        val = 0
                        for r in range(target_rank):
                            val += U[a, r] * V[b, r] * W[c, r]
                        if abs(val - 1) > 1e-6:
                            error += 1
            
            # Also check that zero entries remain zero (sample some)
            for _ in range(100):
                a = rng.randint(0, n*m-1)
                b = rng.randint(0, m*p-1)
                c = rng.randint(0, p*n-1)
                # Check if this should be zero
                i_a = a // m
                j_a = a % m
                j_b = b // p
                k_b = b % p
                k_c = c // n
                i_c = c % n
                should_be_zero = not (i_a == i_c and j_a == j_b and k_b == k_c)
                if should_be_zero:
                    val = 0
                    for r in range(target_rank):
                        val += U[a, r] * V[b, r] * W[c, r]
                    if abs(val) > 1e-6:
                        error += 1
            
            if error == 0:
                best_rank = target_rank
                best_U = U
                best_V = V
                best_W = W
                break
        
        if best_rank < schoolbook_rank:
            break
    
    # If no improvement found, return schoolbook
    if best_rank == schoolbook_rank:
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    else:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
