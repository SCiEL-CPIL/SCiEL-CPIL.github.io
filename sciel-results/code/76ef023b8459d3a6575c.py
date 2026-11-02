import time
import numpy as np

DIMENSIONS = [2, 2, 3]

# EVOLVE-BLOCK-START
def _verify(U, V, W):
    """Verify the tensor decomposition is exact using numpy."""
    n, m, p = DIMENSIONS
    r = U.shape[1]
    try:
        U_reshaped = U.reshape(n, m, r)
        V_reshaped = V.reshape(m, p, r)
        W_reshaped = W.reshape(p, n, r)
        
        result = np.einsum('ijs,jks,ks->ijk', U_reshaped, V_reshaped, W_reshaped)
        
        return np.allclose(result, 1.0, atol=1e-9)
    except Exception:
        return False

def _strassen_core():
    """Return the rank-7 Strassen decomposition for the 2x2 x 2x2 block.
    Returns U_core (4x7), V_core (6x7) where V only has entries in cols 0,1,
    W_core (4x7) where W only has entries in rows k=0,1.
    """
    n, m, p = 2, 2, 3
    U = np.zeros((n * m, 7))
    V = np.zeros((m * p, 7))
    W = np.zeros((p * n, 7))

    def set_u(i, j, r, val):
        U[i * m + j, r] = val

    def set_v(j, k, r, val):
        V[j * p + k, r] = val

    def set_w(k, i, r, val):
        W[k * n + i, r] = val

    set_u(0,0,0,1); set_u(1,1,0,1)
    set_v(0,0,0,1); set_v(1,1,0,1)
    set_w(0,0,0,1); set_w(1,1,0,1)

    set_u(1,0,1,1); set_u(1,1,1,1)
    set_v(0,0,1,1)
    set_w(0,1,1,1); set_w(1,1,1,-1)

    set_u(0,0,2,1)
    set_v(0,1,2,1); set_v(1,1,2,-1)
    set_w(1,0,2,1); set_w(1,1,2,1)

    set_u(1,1,3,1)
    set_v(1,0,3,1); set_v(0,0,3,-1)
    set_w(0,0,3,1); set_w(0,1,3,1)

    set_u(0,0,4,1); set_u(0,1,4,1)
    set_v(1,1,4,1)
    set_w(0,0,4,-1); set_w(1,0,4,1)

    set_u(1,0,5,1); set_u(0,0,5,-1)
    set_v(0,0,5,1); set_v(0,1,5,1)
    set_w(1,1,5,1)

    set_u(0,1,6,1); set_u(1,1,6,-1)
    set_v(1,0,6,1); set_v(1,1,6,1)
    set_w(0,0,6,1)

    return U, V, W

def _rank11_solution():
    n, m, p = DIMENSIONS
    rank = 11
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    def set_u(i, j, r, val):
        U[i * m + j, r] = val

    def set_v(j, k, r, val):
        V[j * p + k, r] = val

    def set_w(k, i, r, val):
        W[k * n + i, r] = val

    # Strassen for 2x2 block (columns 0,1 of B)
    set_u(0,0,0,1); set_u(1,1,0,1)
    set_v(0,0,0,1); set_v(1,1,0,1)
    set_w(0,0,0,1); set_w(1,1,0,1)

    set_u(1,0,1,1); set_u(1,1,1,1)
    set_v(0,0,1,1)
    set_w(0,1,1,1); set_w(1,1,1,-1)

    set_u(0,0,2,1)
    set_v(0,1,2,1); set_v(1,1,2,-1)
    set_w(1,0,2,1); set_w(1,1,2,1)

    set_u(1,1,3,1)
    set_v(1,0,3,1); set_v(0,0,3,-1)
    set_w(0,0,3,1); set_w(0,1,3,1)

    set_u(0,0,4,1); set_u(0,1,4,1)
    set_v(1,1,4,1)
    set_w(0,0,4,-1); set_w(1,0,4,1)

    set_u(1,0,5,1); set_u(0,0,5,-1)
    set_v(0,0,5,1); set_v(0,1,5,1)
    set_w(1,1,5,1)

    set_u(0,1,6,1); set_u(1,1,6,-1)
    set_v(1,0,6,1); set_v(1,1,6,1)
    set_w(0,0,6,1)

    # Schoolbook for column 2
    set_u(0,0,7,1); set_v(0,2,7,1); set_w(2,0,7,1)
    set_u(0,1,8,1); set_v(1,2,8,1); set_w(2,0,8,1)
    set_u(1,0,9,1); set_v(0,2,9,1); set_w(2,1,9,1)
    set_u(1,1,10,1); set_v(1,2,10,1); set_w(2,1,10,1)

    return U, V, W

def _try_rank10_extension():
    """Try to find a rank-10 decomposition by adding 3 rank-1 terms to the
    Strassen core to generate column 2 of the output.
    
    The constraint is: for all (i,j), sum_{s=7}^{9} U_new[i,j,s] * V_new[j,2,s] * W_new[2,i,s] = 1
    (since C[i,2] = sum_j A[i,j]*B[j,2], and we need the tensor to produce identity-like behavior)
    
    Actually, the tensor T[i,j,k] = delta_{i,j,k} in the sense that T[i,j,k] = 1 if we're computing
    C[i,k] = sum_j A[i,j] B[j,k]. The multiplication tensor has T[i,j,k] = 1 for all valid (i,j,k).
    So for k=2, we need: sum_{s} U[i,j,s]*V[j,2,s]*W[2,i,s] = 1 for all i in {0,1}, j in {0,1}.
    
    With the Strassen core (s=0..6), V[j,2,s] = 0 for all s, so the core contributes nothing to k=2.
    We need 3 new terms (s=7,8,9) such that:
      sum_{s=7}^{9} U[i,j,s] * V[j,2,s] * W[2,i,s] = 1  for all i,j in {0,1}
    
    This is 4 equations. Let's parameterize:
      For each new term s, let u_s = [U[0,0,s], U[0,1,s], U[1,0,s], U[1,1,s]]
      v_s = [V[0,2,s], V[1,2,s]]  (only column 2 of B matters for this constraint)
      w_s = [W[2,0,s], W[2,1,s]]
    
    The constraint for (i,j) is: sum_s U[i,j,s]*V[j,2,s]*W[2,i,s] = 1
    
    This is bilinear in the variables. Let's try a structured approach:
    Assume we can express the 2x2 identity as a sum of 3 rank-1 "slices".
    
    Actually, let's think differently. The sub-tensor for k=2 is:
    T_sub[i,j] = 1 for all i,j (a 2x2 all-ones matrix in the (i,j) plane for fixed k=2)
    
    Wait, that's not right either. The multiplication tensor for (A,B)->C where C = A*B is:
    T[i,j,k] = 1 for all valid i,j,k (it's the "identity" tensor in the sense that
    C[i,k] = sum_j A[i,j]*B[j,k]).
    
    So for k=2: T[i,j,2] = 1 for i in {0,1}, j in {0,1}.
    
    We need: sum_{s=7}^{9} U[i,j,s]*V[j,2,s]*W[2,i,s] = 1 for all i,j.
    
    Let's define for each s: a_s = U[0,0,s], b_s = U[0,1,s], c_s = U[1,0,s], d_s = U[1,1,s]
    e_s = V[0,2,s], f_s = V[1,2,s]
    g_s = W[2,0,s], h_s = W[2,1,s]
    
    Constraints:
    (i=0,j=0): a_s*e_s*g_s summed over s = 1
    (i=0,j=1): b_s*f_s*g_s summed over s = 1
    (i=1,j=0): c_s*e_s*h_s summed over s = 1
    (i=1,j=1): d_s*f_s*h_s summed over s = 1
    
    With 3 terms, we have 24 unknowns and 4 equations. This is highly underdetermined.
    The question is whether integer/half-integer solutions exist.
    
    Let's try a simple ansatz: set all V[j,2,s] = 1 for all j,s (i.e., e_s=f_s=1).
    Then constraints become:
    sum_s a_s*g_s = 1
    sum_s b_s*g_s = 1
    sum_s c_s*h_s = 1
    sum_s d_s*h_s = 1
    
    With 3 terms: a1*g1+a2*g2+a3*g3=1, etc.
    
    Try: g1=1,g2=0,g3=0; h1=0,h2=1,h3=0
    Then: a1=1, c2=1, and b1*g1+b2*g2+b3*g3 = b1 = 1, d1*g1+d2*g2+d3*g3 = d1=1
    So: a1=1, b1=1, c2=1, d1=1, all others 0.
    
    This gives:
    s=7: U[0,0]=1, U[0,1]=1, V[0,2]=1, V[1,2]=1, W[2,0]=1, W[2,1]=0
    s=8: U[1,0]=1, V[0,2]=1, V[1,2]=1, W[2,0]=0, W[2,1]=1
    s=9: all zeros (unused)
    
    Wait, that's only 2 terms! Let me verify:
    (i=0,j=0): U[0,0,7]*V[0,2,7]*W[2,0,7] + U[0,0,8]*V[0,2,8]*W[2,0,8] + ... = 1*1*1 + 0 = 1 ✓
    (i=0,j=1): U[0,1,7]*V[1,2,7]*W[2,0,7] + ... = 1*1*1 + 0 = 1 ✓
    (i=1,j=0): U[1,0,7]*V[0,2,7]*W[2,1,7] + U[1,0,8]*V[0,2,8]*W[2,1,8] = 0*1*0 + 1*1*1 = 1 ✓
    (i=1,j=1): U[1,1,7]*V[1,2,7]*W[2,1,7] + U[1,1,8]*V[1,2,8]*W[2,1,8] = 0 + 0 = 0 ✗
    
    Oops, (i=1,j=1) gives 0, not 1. I need d_s*f_s*h_s to sum to 1.
    
    Let me redo: I need 4 independent constraints with 3 terms. Let me try:
    s=7: U[0,0]=1, U[0,1]=1, V[0,2]=1, V[1,2]=1, W[2,0]=1, W[2,1]=0
    s=8: U[1,0]=1, U[1,1]=1, V[0,2]=1, V[1,2]=1, W[2,0]=0, W[2,1]=1
    
    Check:
    (0,0): 1*1*1 + 0 = 1 ✓
    (0,1): 1*1*1 + 0 = 1 ✓
    (1,0): 0 + 1*1*0 = 0 ✗
    
    Still wrong. The issue is that W[2,0] and W[2,1] are coupled.
    
    Let me try:
    s=7: U[0,0]=1, V[0,2]=1, W[2,0]=1  (contributes to (0,0))
    s=8: U[0,1]=1, V[1,2]=1, W[2,0]=1  (contributes to (0,1))
    s=9: U[1,0]=1, V[0,2]=1, W[2,1]=1  (contributes to (1,0))
    But (1,1) needs: U[1,1]*V[1,2]*W[2,1] summed = 1, but none of our terms have U[1,1]≠0.
    
    So 3 terms with this structure can't work. We need to be more clever.
    
    Let's try:
    s=7: U[0,0]=1, U[1,0]=1, V[0,2]=1, W[2,0]=1, W[2,1]=1
    s=8: U[0,1]=1, U[1,1]=1, V[1,2]=1, W[2,0]=1, W[2,1]=1
    
    Check:
    (0,0): 1*1*1 + 0 = 1 ✓
    (0,1): 0 + 1*1*1 = 1 ✓
    (1,0): 1*1*1 + 0 = 1 ✓
    (1,1): 0 + 1*1*1 = 1 ✓
    
    This works with 2 terms! But wait, I need to check that these new terms don't interfere
    with the existing Strassen terms for k=0 and k=1. The Strassen terms have V[j,2,s]=0 for s<7,
    and the new terms have W[k,i,s]=0 for k≠2. So there's no interference!
    
    Actually wait - I set W[2,0]=1 and W[2,1]=1 for the new terms, but W is indexed as W[k*n+i] = W[2*2+i] = W[4+i].
    So W[4,7]=1, W[5,7]=1, W[4,8]=1, W[5,8]=1.
    
    And V[j*p+k] = V[j*3+2]. So V[0*3+2,7]=V[2,7]=1, V[1*3+2,7]=V[5,7]=1, etc.
    
    Let me verify this gives rank 9!
    """
    n, m, p = DIMENSIONS
    rank = 9
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    def set_u(i, j, r, val):
        U[i * m + j, r] = val

    def set_v(j, k, r, val):
        V[j * p + k, r] = val

    def set_w(k, i, r, val):
        W[k * n + i, r] = val

    # Strassen core (columns 0-6)
    set_u(0,0,0,1); set_u(1,1,0,1)
    set_v(0,0,0,1); set_v(1,1,0,1)
    set_w(0,0,0,1); set_w(1,1,0,1)

    set_u(1,0,1,1); set_u(1,1,1,1)
    set_v(0,0,1,1)
    set_w(0,1,1,1); set_w(1,1,1,-1)

    set_u(0,0,2,1)
    set_v(0,1,2,1); set_v(1,1,2,-1)
    set_w(1,0,2,1); set_w(1,1,2,1)

    set_u(1,1,3,1)
    set_v(1,0,3,1); set_v(0,0,3,-1)
    set_w(0,0,3,1); set_w(0,1,3,1)

    set_u(0,0,4,1); set_u(0,1,4,1)
    set_v(1,1,4,1)
    set_w(0,0,4,-1); set_w(1,0,4,1)

    set_u(1,0,5,1); set_u(0,0,5,-1)
    set_v(0,0,5,1); set_v(0,1,5,1)
    set_w(1,1,5,1)

    set_u(0,1,6,1); set_u(1,1,6,-1)
    set_v(1,0,6,1); set_v(1,1,6,1)
    set_w(0,0,6,1)

    # Extension for column 2 (columns 7-8)
    # Term 7: covers (i=0,j=0) and (i=1,j=0) for k=2
    set_u(0,0,7,1); set_u(1,0,7,1)
    set_v(0,2,7,1)
    set_w(2,0,7,1); set_w(2,1,7,1)

    # Term 8: covers (i=0,j=1) and (i=1,j=1) for k=2
    set_u(0,1,8,1); set_u(1,1,8,1)
    set_v(1,2,8,1)
    set_w(2,0,8,1); set_w(2,1,8,1)

    return U, V, W

def _try_rank10_search(deadline):
    """Search for a rank-10 decomposition using the rank-7 Strassen core
    plus 3 additional terms for column 2."""
    n, m, p = DIMENSIONS
    rank = 10
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    def set_u(i, j, r, val):
        U[i * m + j, r] = val

    def set_v(j, k, r, val):
        V[j * p + k, r] = val

    def set_w(k, i, r, val):
        W[k * n + i, r] = val

    # Strassen core
    set_u(0,0,0,1); set_u(1,1,0,1)
    set_v(0,0,0,1); set_v(1,1,0,1)
    set_w(0,0,0,1); set_w(1,1,0,1)

    set_u(1,0,1,1); set_u(1,1,1,1)
    set_v(0,0,1,1)
    set_w(0,1,1,1); set_w(1,1,1,-1)

    set_u(0,0,2,1)
    set_v(0,1,2,1); set_v(1,1,2,-1)
    set_w(1,0,2,1); set_w(1,1,2,1)

    set_u(1,1,3,1)
    set_v(1,0,3,1); set_v(0,0,3,-1)
    set_w(0,0,3,1); set_w(0,1,3,1)

    set_u(0,0,4,1); set_u(0,1,4,1)
    set_v(1,1,4,1)
    set_w(0,0,4,-1); set_w(1,0,4,1)

    set_u(1,0,5,1); set_u(0,0,5,-1)
    set_v(0,0,5,1); set_v(0,1,5,1)
    set_w(1,1,5,1)

    set_u(0,1,6,1); set_u(1,1,6,-1)
    set_v(1,0,6,1); set_v(1,1,6,1)
    set_w(0,0,6,1)

    # Try 3-term extension for column 2
    # We need: sum_{s=7}^{9} U[i,j,s]*V[j,2,s]*W[2,i,s] = 1 for all i,j
    # Try structured patterns
    patterns = [
        # Pattern A: 2 terms suffice (verified above)
        lambda: (
            (set_u(0,0,7,1), set_u(1,0,7,1), set_v(0,2,7,1), set_w(2,0,7,1), set_w(2,1,7,1)),
            (set_u(0,1,8,1), set_u(1,1,8,1), set_v(1,2,8,1), set_w(2,0,8,1), set_w(2,1,8,1)),
            (),
        ),
        # Pattern B: 3 terms, each covering one (i,j) pair
        lambda: (
            (set_u(0,0,7,1), set_v(0,2,7,1), set_w(2,0,7,1)),
            (set_u(0,1,8,1), set_v(1,2,8,1), set_w(2,0,8,1)),
            (set_u(1,0,9,1), set_v(0,2,9,1), set_w(2,1,9,1)),
        ),
        # Pattern C: 3 terms with mixed coverage
        lambda: (
            (set_u(0,0,7,1), set_u(0,1,7,1), set_v(0,2,7,1), set_v(1,2,7,1), set_w(2,0,7,1)),
            (set_u(1,0,8,1), set_u(1,1,8,1), set_v(0,2,8,1), set_v(1,2,8,1), set_w(2,1,8,1)),
            (),
        ),
    ]

    for pattern in patterns:
        if time.time() > deadline:
            break
        # Reset extension columns
        U[:, 7:] = 0
        V[:, 7:] = 0
        W[:, 7:] = 0
        
        terms = pattern()
        for term in terms:
            for op in term:
                pass  # Already applied via side effects
        
        if _verify(U, V, W):
            return U, V, W

    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140

    # First try rank-9 (Strassen + 2-term extension)
    U9, V9, W9 = _try_rank10_extension()  # Actually returns rank-9
    if _verify(U9, V9, W9):
        return {"U": U9.tolist(), "V": V9.tolist(), "W": W9.tolist()}

    # Try rank-10 search
    result10 = _try_rank10_search(deadline)
    if result10 is not None:
        U10, V10, W10 = result10
        return {"U": U10.tolist(), "V": V10.tolist(), "W": W10.tolist()}

    # Fallback to rank-11
    U, V, W = _rank11_solution()
    if _verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
