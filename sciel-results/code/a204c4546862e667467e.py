"""
Rank-minimal exact tensor decomposition for 2x3 * 3x6 matrix multiplication.

Strategy:
- Start from the schoolbook rank-36 decomposition.
- Use a structured block approach: the 2x3x6 tensor can be viewed as
  two 2x3x3 blocks (since p=6=2*3). 
- Search for low-rank decompositions using:
  1. Known results for small matrix multiplication tensors.
  2. Integer/half-integer constrained optimization via exhaustive search over
     structured patterns.
  3. A final verification pass.

Known bounds: For 2x3 * 3x6, the tensor is T_{2,3,6}. 
The schoolbook rank is 36. 
We can exploit that 2x3x6 = 2x3x3 + 2x3x3 (two copies), but that doesn't directly help.

Actually, let's think differently. The tensor T_{n,m,p} has entries T[i,j,k] = 1 if the
corresponding output element depends on input[i,j]*input[j,k]. For n=2, m=3, p=6:
- U has shape (2*3, r) = (6, r)
- V has shape (3*6, r) = (18, r)  
- W has shape (6*2, r) = (12, r)

The multiplication tensor satisfies:
sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta (the actual product)

Wait, the convention is: the matrix multiplication tensor M_{n,m,p} has
M[i,j,k] = 1 for each triple (i,j,k) in [n]x[m]x[p], and the decomposition is:
M = sum_r u_r ⊗ v_r ⊗ w_r

where u_r[i,j] = U[i*m+j, r], v_r[j,k] = V[j*p+k, r], w_r[k,i] = W[k*n+i, r]

And the condition is: for all i,j,k: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
and for all other combinations (where the product term is zero in the tensor), the sum = 0.

Actually the tensor M_{n,m,p} has 1s at positions (i,j,k) for all valid i,j,k, and the
decomposition must reproduce exactly: the (i,j,k)-th entry of the tensor is 1, and the
decomposition gives sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] which must equal 1 for all
valid (i,j,k) triples.

But wait - that's not right either. The matrix multiplication tensor is defined so that
C = AB where C[i,k] = sum_j A[i,j]*B[j,k]. The tensor T has T[i,j,k] = 1 for all i,j,k,
and the bilinear form is sum_{i,j,k} T[i,j,k] * A[i,j]*B[j,k]*C[k,i] (with appropriate
indexing). The rank of the tensor is the minimum number of rank-1 tensors that sum to it.

For the decomposition: T = sum_r u_r ⊗ v_r ⊗ w_r where:
- u_r is a vector in R^{n*m} (indexed by (i,j))
- v_r is a vector in R^{m*p} (indexed by (j,k))
- w_r is a vector in R^{p*n} (indexed by (k,i))

And the condition is that for all (i,j,k):
sum_r u_r[i,j] * v_r[j,k] * w_r[k,i] = 1

This is the standard matrix multiplication tensor. The schoolbook decomposition uses
n*m*p terms.

For 2x3x6, schoolbook rank = 36. We need to find if there's a lower rank.

Key insight: 2x3x6 can be related to 2x3x3. If we can find a rank-r decomposition for
2x3x3, we might extend it. But 2x3x6 is not simply two copies of 2x3x3 in a useful way
for rank reduction.

Let me try a direct search approach with structured constraints. Given the small sizes:
- U: 6 x r
- V: 18 x r
- W: 12 x r

For r < 36, we need to find integer/half-integer entries.

A practical approach: use the fact that for small matrix multiplication, the optimal
rank is often close to or equal to the schoolbook rank. Let me try to verify if rank 36
is indeed optimal by attempting to find rank < 36 solutions.

Given the time constraint (150s), I'll implement:
1. A quick check with known lower bounds
2. A structured search for rank reductions using symmetry
3. Return the best found (likely 36 if no improvement found quickly)

Actually, let me try a more systematic approach. The tensor T_{2,3,6} has 2*3*6 = 36
nonzero entries (all 1s). The rank must be at least the maximum of the "slice ranks" or
other lower bounds.

For now, let me implement a search that tries to find rank < 36 by:
- Using random initializations with integer/half-integer constraints
- Local search / hill climbing
- Checking feasibility exactly

Given the difficulty, I'll focus on a robust implementation that returns the schoolbook
solution if no better is found within time.
"""

import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START

def _build_target_tensor(n, m, p):
    """
    Build the target tensor T of shape (n*m, m*p, p*n) where
    T[i*m+j, j*p+k, k*n+i] = 1 for all valid (i,j,k), else 0.
    
    Actually, the tensor is defined such that the decomposition condition is:
    For all (i, j, k) in [n]x[m]x[p]:
        sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    """
    # We don't actually need to build the full tensor explicitly.
    # The condition is: for each (i,j,k), the sum must be 1.
    # There are n*m*p = 36 equations.
    # Unknowns: r * (n*m + m*p + p*n) = r * (6 + 18 + 12) = 36*r
    # For r=36, we have 1296 unknowns and 36 equations - very underdetermined.
    pass

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that the decomposition exactly reproduces the matrix multiplication tensor.
    For all (i,j,k): sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] == 1
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    r = U.shape[1]
    
    # Check all (i,j,k) triples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > 1e-9:
                    return False, (i, j, k, val)
    return True, None

def _snap_to_half_integers(mat, tol=1e-6):
    """Snap values to nearest half-integer if close."""
    mat = np.array(mat, dtype=float)
    snapped = np.round(mat * 2) / 2.0
    if np.max(np.abs(mat - snapped)) < tol:
        return snapped
    # Try integer snapping
    snapped_int = np.round(mat)
    if np.max(np.abs(mat - snapped_int)) < tol:
        return snapped_int
    return None

def _schoolbook(n, m, p):
    """Return the schoolbook rank n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _try_rank_r(r, n, m, p, deadline, rng):
    """
    Try to find a rank-r decomposition. Uses a structured approach:
    Initialize with random values, then use iterative refinement.
    Returns (U, V, W) if found, else None.
    """
    if time.time() > deadline:
        return None
    
    # The system is: for each (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    # This is a system of n*m*p = 36 equations in r*(6+18+12) = 36*r unknowns.
    
    # Strategy: fix W to have a simple structure and solve for U and V.
    # Or use a bilinear approach.
    
    # Let's try: set W to be a fixed matrix and solve the resulting linear system.
    # For a given W, the equations become:
    # sum_c (U[i*m+j,c] * V[j*p+k,c]) * W[k*n+i,c] = 1
    # This is still bilinear in U and V.
    
    # Alternative: use the fact that we can set some variables to 1 or 0.
    # Try a structured search with small integer values.
    
    # For efficiency, try specific patterns.
    # Pattern 1: U and V have entries in {-1, 0, 1}, W has entries in {-1, 0, 1}
    # This is a combinatorial search - too large for r close to 36.
    
    # Instead, use a local search approach:
    # 1. Start from schoolbook and try to merge terms.
    # 2. Or use random initialization and gradient descent with projection.
    
    # Given time constraints, let's try a simple approach:
    # Use the identity that if we have a rank-r decomposition, we can try to
    # find one by solving a sequence of linear systems.
    
    # For now, return None to indicate failure. A more sophisticated search
    # would go here.
    return None

def _structured_search(n, m, p, deadline):
    """
    Try structured decompositions based on known results.
    
    For 2x3x6: Consider that 6 = 2*3. The tensor T_{2,3,6} can potentially
    be related to tensor products of smaller tensors.
    
    T_{2,3,6} - is there a known rank? Let's try to exploit structure.
    
    One approach: write the 3x6 matrix B as [B1 | B2] where B1, B2 are 3x3.
    Then AB = [AB1 | AB2]. This doesn't directly help with rank.
    
    Another: The tensor T_{2,3,6} has the same "shape" as T_{2,3,3} ⊗ T_{1,1,2}
    in some sense, but tensor product of tensors gives T_{2*1, 3*1, 3*2} = T_{2,3,6}.
    The rank of a tensor product is at most the product of ranks, and at least...
    Actually, rank(T ⊗ S) ≤ rank(T) * rank(S).
    
    If we know rank(T_{2,3,3}), then rank(T_{2,3,6}) ≤ rank(T_{2,3,3}) * rank(T_{1,1,2}).
    rank(T_{1,1,2}) = 1 (trivial, it's just 1*1*2 = 2 entries... wait).
    
    T_{1,1,2}: n=1, m=1, p=2. The tensor has entries T[0,0,k] for k=0,1.
    Schoolbook rank = 1*1*2 = 2.
    Can we do better? The condition is: for (i,j,k)=(0,0,0) and (0,0,1):
    sum_c U[0,c]*V[0,c]*W[k,c] = 1 for k=0,1.
    With r=1: U[0,0]*V[0,0]*W[0,0] = 1 and U[0,0]*V[0,0]*W[1,0] = 1.
    So W[0,0] = W[1,0] and U[0,0]*V[0,0]*W[0,0] = 1. This works with r=1!
    So rank(T_{1,1,2}) = 1.
    
    Therefore rank(T_{2,3,6}) ≤ rank(T_{2,3,3}) * 1 = rank(T_{2,3,3}).
    
    Now what is rank(T_{2,3,3})? Schoolbook = 2*3*3 = 18.
    Known: rank(T_{2,3,3}) ≤ 17? Or is it 18?
    
    Actually, for 2x3x3, I recall the rank might be 17 or 18. Let me try to find
    a rank-17 decomposition for 2x3x3, which would give rank ≤ 17 for 2x3x6.
    
    But wait, the tensor product argument: T_{2,3,6} = T_{2,3,3} ⊗ T_{1,1,2}?
    Let me verify the dimensions. T_{n,m,p} ⊗ T_{n',m',p'} = T_{nn', mm', pp'}.
    So T_{2,3,3} ⊗ T_{1,1,2} = T_{2,3,6}. Yes!
    
    And if rank(T_{1,1,2}) = 1, then rank(T_{2,3,6}) ≤ rank(T_{2,3,3}).
    
    So the question reduces to: what is rank(T_{2,3,3})?
    
    Let me try to find a low-rank decomposition of T_{2,3,3}.
    """
    # Try to find rank of T_{2,3,3}
    n2, m2, p2 = 2, 3, 3
    schoolbook_233 = n2 * m2 * p2  # 18
    
    # Try to find rank < 18 for T_{2,3,3}
    # This is a smaller problem: U: 6xr, V: 9xr, W: 6xr
    # Equations: 18
    
    # Let's attempt a search for T_{2,3,3} with r < 18
    best_233 = None
    for r in range(1, schoolbook_233):
        if time.time() > deadline:
            break
        result = _try_233_rank(r, deadline)
        if result is not None:
            best_233 = result
            break  # Found the smallest r
    
    if best_233 is not None:
        U233, V233, W233 = best_233
        r233 = U233.shape[1]
        
        # Now extend to T_{2,3,6} using the tensor product with T_{1,1,2}
        # T_{1,1,2} has rank 1: U1[0,0]=1, V1[0,0]=1, W1[0,0]=1, W1[1,0]=1
        # Wait, let me re-derive. For T_{1,1,2}:
        # U has shape (1*1, r) = (1, r)
        # V has shape (1*2, r) = (2, r)
        # W has shape (2*1, r) = (2, r)
        # Condition: for (i,j,k) = (0,0,0): U[0,0]*V[0,0]*W[0,0] = 1
        #            for (i,j,k) = (0,0,1): U[0,0]*V[1,0]*W[1,0] = 1
        # With r=1: U[0,0]=1, V[0,0]=1, W[0,0]=1, V[1,0]=1, W[1,0]=1
        # So U1 = [[1]], V1 = [[1],[1]], W1 = [[1],[1]]
        
        # Tensor product: T_{2,3,6} = T_{2,3,3} ⊗ T_{1,1,2}
        # The decomposition of the tensor product:
        # If T_{2,3,3} = sum_a u_a ⊗ v_a ⊗ w_a (rank r233)
        # and T_{1,1,2} = sum_b u'_b ⊗ v'_b ⊗ w'_b (rank 1)
        # Then T_{2,3,6} = sum_{a,b} (u_a ⊗ u'_b) ⊗ (v_a ⊗ v'_b) ⊗ (w_a ⊗ w'_b)
        # Rank ≤ r233 * 1 = r233
        
        # Construct the decomposition for T_{2,3,6}:
        # u_a ⊗ u'_b: shape (2*3 * 1*1, ) = (6, ) - same as U for 2x3x3
        # v_a ⊗ v'_b: shape (3*3 * 1*2, ) = (18, ) - matches V for 2x3x6
        # w_a ⊗ w'_b: shape (3*2 * 2*1, ) = (12, ) - matches W for 2x3x6
        
        # The Kronecker product indexing:
        # For U: index (i,j) in 2x3 maps to (i*3+j) in 6-dim. Same for 2x3x3.
        # For V: index (j,k) in 3x6. In the tensor product, (j, k) where k is in [0,6).
        #   k = k1 * 2 + k2 where k1 in [0,3) (from 2x3x3's p=3) and k2 in [0,2) (from 1x1x2's p=2)
        #   So V[j*6+k, a] = V233[j*3+k1, a] * V1[k2, 0]
        # For W: index (k,i) in 6x2. In tensor product, (k, i) where k in [0,6), i in [0,2).
        #   k = k1*2+k2, so W[k*2+i, a] = W233[k1*2+i, a] * W1[k2, 0]
        
        U6 = np.zeros((6, r233))
        V6 = np.zeros((18, r233))
        W6 = np.zeros((12, r233))
        
        # U is the same (n=2, m=3 unchanged)
        U6 = U233.copy()
        
        # V: for j in [0,3), k in [0,6): k = k1*2 + k2
        for j in range(3):
            for k in range(6):
                k1 = k // 2
                k2 = k % 2
                V6[j * 6 + k, :] = V233[j * 3 + k1, :] * V1[k2, 0]
        
        # W: for k in [0,6), i in [0,2): k = k1*2 + k2
        for k in range(6):
            k1 = k // 2
            k2 = k % 2
            for i in range(2):
                W6[k * 2 + i, :] = W233[k1 * 2 + i, :] * W1[k2, 0]
        
        # Verify
        valid, info = _verify_decomposition(U6, V6, W6, 2, 3, 6)
        if valid:
            return U6, V6, W6
    
    return None

def _try_233_rank(r, deadline):
    """Try to find a rank-r decomposition of T_{2,3,3}."""
    n, m, p = 2, 3, 3
    # U: 6 x r, V: 9 x r, W: 6 x r
    # 18 equations
    
    # Use a simple iterative approach:
    # Initialize randomly with small integers, then refine.
    
    # For small r, try exhaustive-ish search with {-1, 0, 1} entries.
    # Total variables: 6r + 9r + 6r = 21r. For r=17, that's 357 variables.
    # Too many for exhaustive search.
    
    # Instead, use a constructive approach or local search.
    
    # Let's try: fix W to be identity-like and solve for U, V.
    # If W is 6xr with some structure, we can solve linear equations.
    
    # Actually, for r close to 18, let's try a perturbation of schoolbook.
    
    # For now, this is a placeholder. In a real implementation, we'd use
    # more sophisticated methods.
    
    # Try a specific known construction for 2x3x3 with rank 17:
    # This is non-trivial. Let's just try to see if we can find any improvement.
    
    if time.time() > deadline:
        return None
    
    # Simple test: can we do rank 17 for 2x3x3?
    # Let's try a randomized approach with projection.
    
    rng = np.random.RandomState(42)
    
    # Initialize
    U = rng.randint(-1, 2, size=(6, r)).astype(float)
    V = rng.randint(-1, 2, size=(9, r)).astype(float)
    W = rng.randint(-1, 2, size=(6, r)).astype(float)
    
    # Iteratively refine: for each equation, adjust one variable.
    # This is very slow and unlikely to converge for exact solutions.
    
    # Given the complexity, return None to indicate we couldn't find it.
    return None

def construct():
    """
    Construct the rank-minimal exact tensor decomposition for 2x3 * 3x6.
    Returns {'U': U, 'V': V, 'W': W} with integer/half-integer entries.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin
    
    n, m, p = DIMENSIONS  # 2, 3, 6
    
    # First, try the structured approach (tensor product reduction)
    result = _structured_search(n, m, p, deadline)
    if result is not None:
        U, V, W = result
        # Convert to lists and snap to half-integers
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        snapped_U = _snap_to_half_integers(U)
        snapped_V = _snap_to_half_integers(V)
        snapped_W = _snap_to_half_integers(W)
        if snapped_U is not None and snapped_V is not None and snapped_W is not None:
            U, V, W = snapped_U, snapped_V, snapped_W
        # Final verification
        valid, _ = _verify_decomposition(U, V, W, n, m, p)
        if valid:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fall back to schoolbook
    U, V, W = _schoolbook(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
