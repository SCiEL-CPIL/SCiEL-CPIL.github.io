"""Find a low-rank tensor decomposition for 2x3 by 3x9 matrix multiplication."""

import time
import itertools
import random
from fractions import Fraction

try:
    import numpy as np
    HAVE_NUMPY = True
except ImportError:
    HAVE_NUMPY = False

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = t0 + 115  # safety margin
    
    n, m, p = 2, 3, 9
    nm, mp, pn = n*m, m*p, p*n  # 6, 27, 18
    
    # Build the multiplication tensor T as a list of (i,j,k) triples
    # T[i,j,k] = 1 if and only if the product uses A[i,j]*B[j,k]
    # In index form: output index (i,k), input1 index (i,j), input2 index (j,k)
    # Tensor T_{(i,k), (i,j), (j,k)} = 1
    
    # We need U of shape (nm, rank), V of shape (mp, rank), W of shape (pn, rank)
    # such that for all (i,k):
    #   sum_c U[i*m+k, c] * V[c, ?] ... 
    # Actually, the standard formulation:
    #   (A*B)_{i,k} = sum_j A_{i,j} * B_{j,k}
    #   = sum_c (sum_j U_{i,j,c} * A_{i,j}) * (sum_j V_{j,k,c} * B_{j,k}) * W_{k,i,c}
    # Wait, that's not right. Let me re-derive.
    #
    # The tensor decomposition says:
    #   (A*B)_{i,k} = sum_c sum_j U_{i,j,c} * V_{j,k,c} * A_{i,j} * B_{j,k}
    # No, that's not standard either.
    #
    # Standard: we have bilinear forms.
    #   (A*B)_{i,k} = sum_c C_c(A) * D_c(B)
    # where C_c(A) = sum_j U_{i,j,c} * A_{i,j} and D_c(B) = sum_j V_{j,k,c} * B_{j,k}
    # But C_c and D_c depend on i,k as well... 
    #
    # Actually the correct formulation for matrix multiplication tensor:
    # T_{(i,k), (i,j), (j,k)} = delta_{i,k output} uses A_{i,j} and B_{j,k}
    # We want: T_{(i,k), (i,j), (j,k)} = sum_c U_{(i,j), c} * V_{(j,k), c} * W_{(i,k), c}
    # where W is indexed by output (i,k) in the convention W[k*n+i, c]
    
    # So: for all i, j, k:
    #   delta_{i,j,k} = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # where delta = 1 iff the triple (i,j,k) corresponds to a valid product term
    # For matrix mult: (A*B)_{i,k} = sum_j A_{i,j} B_{j,k}
    # So the tensor has T[i*m+j, j*p+k, k*n+i] = 1 for all valid (i,j,k)
    # i.e., for i in [0,n), j in [0,m), k in [0,p)
    
    # Build list of all (u, v, w) triples where u = i*m+j, v = j*p+k, w = k*n+i
    # and T[u,v,w] = 1
    triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                triples.append((u, v, w))
    
    # The constraint is: for each (u,v,w) in triples:
    #   sum_c U[u,c] * V[v,c] * W[w,c] = 1
    # For all (u,v,w) NOT in triples (that are valid index combinations):
    #   sum_c U[u,c] * V[v,c] * W[w,c] = 0
    
    # Let's enumerate all valid (u,v,w) index combinations:
    # u = i*m + j where i in [0,n), j in [0,m)  -> u in [0, nm)
    # v = j*p + k where j in [0,m), k in [0,p)  -> v in [0, mp)
    # w = k*n + i where k in [0,p), i in [0,n)  -> w in [0, pn)
    # But not all combinations of (u,v,w) are valid - they must be consistent
    # i.e., the j from u and v must match, and the i from u and w must match, etc.
    
    # Actually, any (u,v,w) in [0,nm) x [0,mp) x [0,pn) is a valid index for the tensor.
    # The tensor is zero unless the indices are consistent.
    
    # Let's check consistency:
    # u = i*m + j => i = u // m, j = u % m
    # v = j'*p + k => j' = v // p, k = v % p
    # w = k'*n + i' => k' = w // n, i' = w % n
    # Consistent iff j == j' and i == i' and k == k'
    
    def is_consistent(u, v, w):
        i_u = u // m
        j_u = u % m
        j_v = v // p
        k_v = v % p
        k_w = w // n
        i_w = w % n
        return j_u == j_v and i_u == i_w and k_v == k_w
    
    # Build the full constraint system
    # We'll use a random search / optimization approach
    
    # Strategy: try to find a decomposition of rank r by:
    # 1. Start with random initialization
    # 2. Use alternating minimization or gradient descent on the constraint violations
    
    # For each rank r from 30 up to 54, try to find a valid decomposition
    # Use random restarts within the time budget
    
    best_rank = nm * mp * pn  # fallback is schoolbook (54)
    best_result = None
    
    # Actually, let's think about this more carefully.
    # The tensor rank for 2x3x3 is known to be 18 (schoolbook).
    # For 2x3x9, we could potentially do better.
    # But without known structure, let's try random search.
    
    # Alternative: use the fact that this is a small problem and try to solve it
    # using a mixed-integer approach or careful algebraic search.
    
    # Let's try a structured search:
    # Assume W has a particular form and solve for U and V.
    
    # For rank r, we have:
    # U: 6 x r, V: 27 x r, W: 18 x r
    # 54*27*18 = 26244 equations (many are 0=0 due to consistency)
    
    # Let's count the actual constraints:
    # Number of consistent (u,v,w) triples = n*m*p = 54 (the ones that should equal 1)
    # Total (u,v,w) triples = 6*27*18 = 2916
    # Inconsistent triples = 2916 - 54 = 2862 (these should equal 0)
    
    # So we have 2916 constraints, 54*3*r unknowns.
    # For r=54: 54*3*54 = 8748 unknowns, 2916 constraints -> underdetermined
    # For r=30: 54*3*30 = 4860 unknowns, 2916 constraints -> still underdetermined
    
    # The challenge is finding integer/half-integer solutions.
    
    # Let's try a different approach: use the known structure.
    # For 2x3x3, schoolbook gives rank 18.
    # For 2x3x9, we can think of it as 3 blocks of 2x3x3.
    # If the blocks were independent, we'd need 3*18 = 54.
    # But maybe we can share some factors.
    
    # Let me try to construct a rank < 54 solution.
    
    # Approach: parametrize W and solve for U, V.
    # For a given W (18 x r), the constraint is:
    # For each consistent (u,v,w): sum_c U[u,c]*V[v,c]*W[w,c] = 1
    # For each inconsistent (u,v,w): sum_c U[u,c]*V[v,c]*W[w,c] = 0
    
    # This is bilinear in U and V for fixed W.
    # We can fix W and solve for U and V.
    
    # Actually, let's just try the schoolbook and verify it works, then try to improve.
    
    # First, let's implement the schoolbook as a baseline:
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(nm)]
    V_school = [[0] * rank_school for _ in range(mp)]
    W_school = [[0] * rank_school for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j][col] = 1
                V_school[j*p+k][col] = 1
                W_school[k*n+i][col] = 1
                col += 1
    
    def verify(U, V, W, rank):
        """Check if the decomposition is valid."""
        # For each (u, v, w):
        #   sum_c U[u,c] * V[v,c] * W[w,c] should be 1 if consistent, 0 otherwise
        for u in range(nm):
            for v in range(mp):
                for w in range(pn):
                    s = sum(U[u][c] * V[v][c] * W[w][c] for c in range(rank))
                    expected = 1 if is_consistent(u, v, w) else 0
                    if abs(s - expected) > 1e-9:
                        return False
        return True
    
    # Verify schoolbook
    assert verify(U_school, V_school, W_school, rank_school), "Schoolbook should be valid"
    
    best_rank = rank_school
    best_U, best_V, best_W = U_school, V_school, W_school
    
    if time.time() > deadline:
        return {"U": best_U, "V": best_V, "W": best_W}
    
    # Now try to find a lower rank decomposition
    # Strategy: use random initialization and iterative improvement
    
    # For a given rank r, try to find U, V, W
    # We'll use a randomized approach:
    # 1. Initialize W randomly with small integers/halves
    # 2. Solve for U and V
    
    # Actually, the equations are cubic in the variables (U*V*W).
    # Let's try a different approach: fix two of the matrices and solve for the third.
    
    # Fix W and V, solve for U:
    # For each u: sum_c U[u,c] * V[v,c] * W[w,c] = expected(u,v,w)
    # This is linear in U[u,*] for each (u,v,w).
    # But V and W are also unknown...
    
    # Let me try a simpler heuristic: 
    # Use the structure that for 2x3x9, we might be able to use rank ~ 45 or so.
    
    # Actually, let's just try random search for various ranks.
    
    def try_rank(r, num_trials=50):
        """Try to find a decomposition of rank r."""
        t_start = time.time()
        
        for trial in range(num_trials):
            if time.time() > deadline:
                break
            
            # Random initialization with small integers and halves
            U = [[random.choice([-1, -0.5, 0, 0.5, 1]) for _ in range(r)] for _ in range(nm)]
            V = [[random.choice([-1, -0.5, 0, 0.5, 1]) for _ in range(r)] for _ in range(mp)]
            W = [[random.choice([-1, -0.5, 0, 0.5, 1]) for _ in range(r)] for _ in range(pn)]
            
            # Check if this random guess works
            if verify(U, V, W, r):
                return U, V, W
        
        return None
    
    # Try ranks from 30 to 53
    for r in range(30, rank_school):
        if time.time() > deadline:
            break
        
        result = try_rank(r, num_trials=20)
        if result is not None:
            U, V, W = result
            best_rank = r
            best_U, best_V, best_W = U, V, W
            break
    
    if time.time() > deadline:
        return {"U": best_U, "V": best_V, "W": best_W}
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
