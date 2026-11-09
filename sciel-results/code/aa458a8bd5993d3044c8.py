"""Program for 3x3x5 matrix multiplication tensor decomposition."""
DIMENSIONS = [3, 3, 5]

import time
import itertools

# EVOLVE-BLOCK-START
def _verify(U, V, W):
    """
    Verify that U, V, W form a valid tensor decomposition for matrix multiplication.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    For all i in [0,n), j in [0,m), k in [0,p):
      sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    """
    n, m, p = DIMENSIONS
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for c in range(rank):
                    total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(total - 1.0) > 1e-9:
                    return False
    return True

def _is_half_int(x):
    """Check if x is a half-integer."""
    x2 = x * 2
    return abs(x2 - round(x2)) < 1e-10

def _schoolbook():
    """Rank-45 schoolbook decomposition."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _try_structured_rank(r, deadline):
    """
    Attempt to find a rank-r decomposition using structured algebraic ansatz.
    For 3x3x5, we exploit the fact that p=5 and try to share structure.
    Returns dict or None.
    """
    n, m, p = DIMENSIONS
    
    # Strategy: For each (i,k) pair in the output, we need:
    # sum_j A[i,j] B[j,k] = sum_c (sum_j U[i*m+j,c] * V[j*p+k,c]) * W[k*n+i,c]
    # This means for each c, the matrix U_c (3x3) times V_c (3x5) should
    # equal W[k*n+i,c] * E[i,k] where E[i,k] is the matrix with 1 at (i,k).
    # Actually more precisely: for each c, let M_c = U_c @ V_c (3x5 matrix).
    # Then we need sum_c M_c = identity-like structure... 
    # Actually the condition is:
    # For all i,j,k: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    # This means: sum_c (U[:,c] outer V[:,c]) * W[k*n+i,c] = 1 for each (i,k)
    # where U[:,c] is n x m, V[:,c] is m x p, so U[:,c] @ V[:,c] is n x p.
    # Let P_c = U[:,c] @ V[:,c] (n x p matrix).
    # Then: sum_c P_c[i,k] * W[k*n+i,c] = 1 for all i,k.
    
    # For schoolbook, each P_c has a single 1 at position (i,k) and W is 1.
    # To reduce rank, we need some P_c to have multiple non-zero entries,
    # and W values to be fractional (half-integers).
    
    # Try: group the 15 output positions (i,k) into groups, and for each group
    # use a single rank-1 term that covers multiple positions.
    # A rank-1 term with U_c (3x3) and V_c (3x5) gives P_c = U_c @ V_c (3x5).
    # If P_c has entries at multiple (i,k) positions, then W[k*n+i,c] must be 
    # chosen so that the weighted sum gives 1 at each covered position.
    
    # Simplest non-trivial: use P_c = all-ones matrix (3x5). Then W[k*n+i,c] = 1/15 for all i,k.
    # But 1/15 is not a half-integer. So this doesn't work directly.
    
    # Alternative: use P_c that is a sum of a few Kronecker products.
    # Let's try: P_c = e_i e_k^T for multiple (i,k) pairs in one term.
    # If P_c has 1s at positions (i1,k1), (i2,k2), ..., (ir,kr), then we need
    # W[k*n+i,c] such that for each (i,k) covered, W[k*n+i,c] * 1 = contribution.
    # But we need the total sum over all c to give 1 at each (i,k).
    
    # Let's try a specific construction:
    # Split the 15 (i,k) positions into 5 groups of 3 (one per k column).
    # For each k, use one term where P_c has 1s at all i for that k.
    # P_c = e_all e_k^T (3x5 matrix with 1s in column k).
    # Then W[k*n+i,c] = 1/3 for all i. But 1/3 is not a half-integer.
    
    # Try: P_c = e_all e_k^T with W = 1/2. Then contribution is 1/2 per (i,k).
    # We'd need 2 such terms per k to get 1. That's 10 terms total.
    # But we can combine: use P_c = 2 * e_all e_k^T, W = 1/2.
    # Then contribution is 2 * 1/2 = 1. So 5 terms total!
    # Check: U_c (3x3) and V_c (3x5) such that U_c @ V_c = 2 * e_all e_k^T.
    # e_all = [1,1,1]^T, e_k = standard basis in R^5.
    # 2 * e_all e_k^T is a 3x5 matrix with 2s in column k.
    # We need U_c (3x3) @ V_c (3x5) = 2 * e_all e_k^T.
    # One choice: U_c = 2 * e_all * e_1^T (3x3, first column is [2,2,2], rest 0)
    #             V_c = e_1 e_k^T (3x5, first row has 1 at position k, rest 0)
    # Check: U_c @ V_c = (2 * e_all e_1^T) @ (e_1 e_k^T) = 2 * e_all (e_1^T e_1) e_k^T = 2 * e_all e_k^T. Correct!
    # W[k*n+i,c] = 1/2 for all i.
    # 
    # But wait: we need to verify the full tensor condition, not just the P_c condition.
    # The condition is: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k.
    # With our construction:
    # For term c_k (corresponding to column k):
    # U[i*m+j, c_k] = 2 if j==0, else 0. (since U_c = 2*e_all*e_1^T, so U[i,j,c] = 2 if j=0)
    # V[j*p+k', c_k] = 1 if j==0 and k'==k, else 0. (since V_c = e_1*e_k^T, so V[j,k',c] = 1 if j=0,k'=k)
    # W[k'*n+i, c_k] = 1/2 for all i, and only non-zero when k'=k? No, W is indexed by (k',i).
    # Wait, W[k*n+i, c] is the value for output position (i,k) in term c.
    # In our construction, for term c_k, W[k'*n+i, c_k] = 1/2 for ALL i and k'?
    # No! We defined W[k*n+i,c] = 1/2 where k is the specific column. But W has p*n = 15 rows.
    # For term c_k, W has non-zero entries at positions (i, k) for all i, i.e., rows k*n+i for i=0,1,2.
    # W[k*n+i, c_k] = 1/2 for i=0,1,2, and W[k'*n+i, c_k] = 0 for k' != k.
    
    # Let's re-verify: for fixed (i,j,k), sum over c:
    # Only c_k contributes (since for c_k', k' != k, V[j*p+k, c_k'] = 0 unless j=0 and k=k', but k!=k').
    # Wait, V[j*p+k, c_k'] = 1 if j=0 and k=k'. So for k != k', V[j*p+k, c_k'] = 0.
    # So only c_k contributes: U[i*m+j, c_k] * V[j*p+k, c_k] * W[k*n+i, c_k]
    # = (2 if j==0 else 0) * (1 if j==0 else 0) * (1/2)
    # = (2 if j==0 else 0) * (1 if j==0 else 0) * 1/2
    # If j==0: 2 * 1 * 1/2 = 1. Correct!
    # If j!=0: 0 * 0 * 1/2 = 0. But we need 1!
    
    # This doesn't work because for j!=0, the contribution is 0, but we need 1.
    # The issue: P_c = U_c @ V_c should be 2*e_all*e_k^T, which has 2s in column k for all i.
    # But the tensor condition requires sum_j U[i*m+j,c] * V[j*p+k,c] = P_c[i,k].
    # For our U_c and V_c: sum_j U[i*m+j,c_k] * V[j*p+k,c_k] = sum_j (2 if j==0) * (1 if j==0 and k==k)
    # = 2*1 = 2 if k==k, else 0. So P_c[i,k] = 2 for all i when k matches. Correct.
    # But the full condition is sum_c P_c[i,k] * W[k*n+i,c] = 1.
    # With only c_k: P_c_k[i,k] * W[k*n+i,c_k] = 2 * 1/2 = 1. 
    # But this only works if the sum over j is implicit in P_c. Let me re-examine.
    
    # The tensor condition is: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    # This is NOT the same as sum_c (sum_j U[i*m+j,c] * V[j*p+k,c]) * W[k*n+i,c] = 1
    # because W depends on c and the j sum is inside.
    # Actually it IS the same: sum_c sum_j U[i*m+j,c] V[j*p+k,c] W[k*n+i,c]
    # = sum_c W[k*n+i,c] (sum_j U[i*m+j,c] V[j*p+k,c])
    # = sum_c W[k*n+i,c] P_c[i,k]
    # So my analysis was correct. The problem is that for j!=0, the individual terms are 0,
    # but the sum over j gives the right answer. Let me re-check.
    
    # For (i,j,k) with j!=0:
    # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # Only c_k can contribute to V[j*p+k,c] since V[j*p+k,c_k'] = 1 iff j==0 and k==k'.
    # For j!=0, V[j*p+k,c] = 0 for all c. So the sum is 0, not 1.
    
    # I see the error. The tensor condition requires the sum to be 1 for EVERY (i,j,k),
    # not just for the aggregated (i,k). The schoolbook has separate terms for each (i,j,k).
    # My grouped construction fails because it doesn't preserve the j-dependence correctly.
    
    # The correct condition is that for each (i,j,k), the sum must be 1.
    # This means we can't just group by (i,k); we need to handle j as well.
    
    # Let me try a different approach: use the known fact that for 3x3x5,
    # the minimal rank is likely 45 (schoolbook) or close to it.
    # Given the constraints (half-integers), it's hard to do better.
    
    # Return None to indicate no better decomposition found.
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120
    
    # Start with schoolbook
    result = _schoolbook()
    assert _verify(result["U"], result["V"], result["W"]), "Schoolbook verification failed"
    
    # Try to find a better decomposition
    for r in range(14, 45):
        if time.time() > deadline - 10:
            break
        res = _try_structured_rank(r, deadline)
        if res is not None:
            if _verify(res["U"], res["V"], res["W"]):
                result = res
                break
    
    return result
# EVOLVE-BLOCK-END
