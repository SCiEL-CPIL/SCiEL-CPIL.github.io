"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
import time
import itertools
import numpy as np

_DEADLINE = 140.0  # safety margin before 150s kill
_START_TIME = time.time()

def _remaining():
    return _DEADLINE - (time.time() - _START_TIME)

def _verify(U, V, W, n, m, p, rank):
    """Check that U, V, W exactly decompose the matmul tensor.
    Tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    should equal 1 if (i,j,k) is a valid triple, else 0.
    Actually the matmul tensor is T[i,j,k] = 1 for all i,j,k (it's the identity
    bilinear form). Wait - the matmul tensor T_{ijk} satisfies:
    (A*B)_{i,k} = sum_j A_{i,j} B_{j,k}
    So T_{i,j,k} = 1 for all i,j,k in range.
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    # T[i,j,k] should equal sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # and this should be 1 for all i in [0,n), j in [0,m), k in [0,p)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i * m + j], V[j * p + k] * W[k * n + i])
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _try_rank_reduction(n, m, p, target_rank):
    """Try to find a decomposition of rank target_rank < n*m*p.
    
    Key insight: The matmul tensor T_{ijk}=1 for all i,j,k can be written as
    sum_c u_{i,j}^c * v_{j,k}^c * w_{k,i}^c = 1.
    
    For n=2, we can try to factor using the structure. One approach:
    If we can write 1 = sum_c a_i^c * b_j^c * c_k^c (rank-1 in each index),
    but that's not quite right since the indices are coupled.
    
    Alternative: Try structured decompositions where some terms share
    structure. For example, if we can find vectors u_i, v_j, w_k such that
    sum_c u_{i,c} v_{j,c} w_{k,c} = 1 for all i,j,k, with rank < n*m*p.
    
    This is equivalent to finding a rank-r decomposition of the all-ones
    3D tensor of shape (n,m,p), which has rank 1! The all-ones tensor
    T_{ijk} = 1 can be written as u_i * v_j * w_k with u_i=1, v_j=1, w_k=1.
    
    BUT WAIT - that's not the matmul tensor. The matmul tensor IS the all-ones
    tensor because T_{ijk} = 1 for all valid i,j,k. So rank 1 should work?
    
    No! The constraint is that U has shape (n*m, rank), V has shape (m*p, rank),
    W has shape (p*n, rank), and the decomposition must satisfy:
    sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k.
    
    If we set rank=1: U[i*m+j, 0]=1, V[j*p+k, 0]=1, W[k*n+i, 0]=1,
    then sum = 1*1*1 = 1. This works!
    
    But wait - is this a valid "tensor decomposition" in the sense intended?
    The problem says "exact tensor decomposition for multiplying a 2 by 5 matrix
    by a 5 by 5 matrix." The standard formulation requires that the bilinear map
    (A, B) -> A*B be represented as sum_c (u_c . A_flat) (v_c . B_flat) * w_c,
    where the output is assembled from w_c.
    
    In the standard matmul tensor decomposition:
    (A*B)_{i,k} = sum_c (sum_j U[i*m+j, c] * A_{i,j}) * (sum_j V[j*p+k, c] * B_{j,k})
    
    So the bilinear form for term c is:
    (sum_j U[i*m+j, c] * A_{i,j}) * (sum_j V[j*p+k, c] * B_{j,k})
    and this contributes to output position (i,k) with weight W[k*n+i, c].
    
    For the all-ones approach to work, we need:
    sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] * A_{i,j}) * (sum_j V[j*p+k, c] * B_{j,k}) = (A*B)_{i,k}
    
    If U[i*m+j, c] = 1 for all i,j,c and V[j*p+k, c] = 1 for all j,k,c and W[k*n+i, c] = 1:
    Then term c gives: 1 * (sum_j A_{i,j}) * (sum_j B_{j,k})
    This is NOT equal to (A*B)_{i,k} = sum_j A_{i,j} B_{j,k}.
    
    So the all-ones approach does NOT work! The tensor is not the all-ones tensor
    in the bilinear sense.
    
    Let me re-derive. The matmul tensor T is defined by:
    (A*B)_{i,k} = sum_{j} T_{i,j,k} * A_{i,j} * B_{j,k}
    where T_{i,j,k} = 1 for all i,j,k.
    
    A rank-r decomposition means:
    T_{i,j,k} = sum_{c=1}^{r} U_{i,j,c} * V_{j,k,c} * W_{i,k,c}
    
    Wait, but the output factor W has indices (k*n+i, c), i.e., W depends on (k,i) not (i,k).
    And the convention is W[k*n+i, r].
    
    So the decomposition is:
    T_{i,j,k} = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    And T_{i,j,k} = 1 for all i in [0,n), j in [0,m), k in [0,p).
    
    So we need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k.
    
    This IS the all-ones tensor! So rank 1 should work:
    U[i*m+j, 0] = 1, V[j*p+k, 0] = 1, W[k*n+i, 0] = 1.
    
    But this can't be right because the known lower bounds for matmul are much higher than 1.
    
    I think I'm confusing the tensor decomposition with the bilinear algorithm. Let me reconsider.
    
    In a bilinear algorithm for matmul, we compute:
    (A*B)_{i,k} = sum_c alpha_{i,k,c} * (sum_j beta_{i,j,c} * A_{i,j}) * (sum_j gamma_{j,k,c} * B_{j,k})
    
    This is equivalent to saying the matmul tensor can be written as:
    T_{i,j,k} = sum_c alpha_{i,k,c} * beta_{i,j,c} * gamma_{j,k,c}
    
    where alpha_{i,k,c} = W[k*n+i, c], beta_{i,j,c} = U[i*m+j, c], gamma_{j,k,c} = V[j*p+k, c].
    
    And T_{i,j,k} = 1 for all i,j,k.
    
    So we need: sum_c W[k*n+i, c] * U[i*m+j, c] * V[j*p+k, c] = 1 for all i,j,k.
    
    With rank 1: W[k*n+i, 0]=1, U[i*m+j, 0]=1, V[j*p+k, 0]=1 gives sum = 1. ✓
    
    But this means the "algorithm" is:
    (A*B)_{i,k} = 1 * (sum_j 1*A_{i,j}) * (sum_j 1*B_{j,k}) = (row_sum_i of A) * (col_sum_k of B)
    
    This is clearly NOT matrix multiplication! The issue is that in a proper bilinear algorithm,
    the same linear forms of A and B are used across all output entries, but here each output
    entry (i,k) uses different linear forms of A and B.
    
    I think the correct formulation is:
    (A*B)_{i,k} = sum_c W[k*n+i, c] * L_c(A) * M_c(B)
    where L_c(A) and M_c(B) are FIXED linear forms of A and B (not depending on i,k).
    
    In that case, L_c(A) = sum_{i,j} U[i*m+j, c] * A_{i,j} and M_c(B) = sum_{j,k} V[j*p+k, c] * B_{j,k}.
    
    Then: (A*B)_{i,k} = sum_c W[k*n+i, c] * (sum_{i',j} U[i'*m+j, c] * A_{i',j}) * (sum_{j,k'} V[j*p+k', c] * B_{j,k'})
    
    For this to equal sum_j A_{i,j} B_{j,k}, we need the tensor identity:
    sum_c W[k*n+i, c] * U[i'*m+j, c] * V[j*p+k', c] = delta_{i',i} * delta_{k',k}
    
    This is a much harder constraint! The U and V matrices define linear forms on the ENTIRE
    matrix A and B, not just row i or column k.
    
    OK so the correct tensor equation is:
    sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = delta_{i,j,k} ???
    
    No. Let me look at the schoolbook example. In schoolbook:
    U[i*m+j, (i,j,k)] = 1, V[j*p+k, (i,j,k)] = 1, W[k*n+i, (i,j,k)] = 1.
    
    So sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 when c=(i,j,k) matches, else 0.
    So the sum = 1 for all i,j,k. That means T_{i,j,k} = 1.
    
    And the bilinear algorithm is:
    (A*B)_{i,k} = sum_{c} W[k*n+i, c] * (sum_{i',j} U[i'*m+j, c] * A_{i',j}) * (sum_{j,k'} V[j*p+k', c] * B_{j,k'})
    
    For schoolbook, c=(i',j,k'), U[i'*m+j,c]=1 only when i'=i and j=j, so sum_{i',j} U[i'*m+j,c]*A_{i',j} = A_{i,j}.
    Similarly sum_{j,k'} V[j*p+k',c]*B_{j,k'} = B_{j,k}.
    And W[k*n+i,c] = 1 only when k=k and i=i, so the sum over c gives just the term c=(i,j,k).
    Result: (A*B)_{i,k} = sum_{j} A_{i,j} B_{j,k}. ✓
    
    So the tensor equation is indeed sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = 1 for all i,j,k.
    
    But this is the all-ones tensor, which has rank 1! The catch is that U, V, W must be
    consistent: U is a matrix of shape (n*m, r), V is (m*p, r), W is (p*n, r), and the
    decomposition must satisfy the equation for ALL i,j,k simultaneously.
    
    For rank 1: U[i*m+j,0]=1, V[j*p+k,0]=1, W[k*n+i,0]=1 satisfies the equation.
    But does it give a valid bilinear algorithm? Let's check:
    L(A) = sum_{i,j} U[i*m+j,0] * A_{i,j} = sum_{i,j} A_{i,j} (sum of all entries of A)
    M(B) = sum_{j,k} V[j*p+k,0] * B_{j,k} = sum_{j,k} B_{j,k} (sum of all entries of B)
    (A*B)_{i,k} = W[k*n+i,0] * L(A) * M(B) = 1 * (sum A) * (sum B)
    
    This is NOT matrix multiplication! So the rank-1 decomposition of the all-ones tensor
    does NOT correspond to a valid bilinear algorithm.
    
    The issue is that the tensor equation sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = 1
    is necessary but not sufficient. We also need the bilinear algorithm to actually compute
    the correct product, which requires that the linear forms L_c(A) and M_c(B) are chosen
    such that the reconstruction works.
    
    Actually, I think the tensor equation IS sufficient. If sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = 1
    for all i,j,k, then:
    sum_c W[k*n+i,c] * (sum_{i',j} U[i'*m+j,c] A_{i',j}) * (sum_{j,k'} V[j*p+k',c] B_{j,k'})
    = sum_c sum_{i',j} sum_{j,k'} W[k*n+i,c] U[i'*m+j,c] V[j*p+k',c] A_{i',j} B_{j,k'}
    = sum_{i',j,k'} A_{i',j} B_{j,k'} (sum_c W[k*n+i,c] U[i'*m+j,c] V[j*p+k',c])
    = sum_{i',j,k'} A_{i',j} B_{j,k'} * delta_{i',i} delta_{k',k}   [if the tensor eq holds with deltas]
    
    Wait, the tensor equation says sum_c ... = 1, not delta. So:
    = sum_{i',j,k'} A_{i',j} B_{j,k'} * 1 = (sum A)(sum B)
    
    So the tensor equation sum_c ... = 1 gives (sum A)(sum B), not matmul.
    
    For matmul, we need:
    sum_c W[k*n+i,c] U[i'*m+j,c] V[j*p+k',c] = delta_{i',i} delta_{k',k}
    
    This is a DIFFERENT tensor equation! The tensor is not all-ones; it's a more complex object.
    
    Let me re-examine the schoolbook. In schoolbook, c ranges over (i0,j0,k0).
    U[i*m+j, (i0,j0,k0)] = 1 if i=i0 and j=j0, else 0.
    V[j*p+k, (i0,j0,k0)] = 1 if j=j0 and k=k0, else 0.
    W[k*n+i, (i0,j0,k0)] = 1 if k=k0 and i=i0, else 0.
    
    sum_c W[k*n+i,c] U[i'*m+j,c] V[j*p+k',c]:
    c=(i0,j0,k0). W[k*n+i, c]=1 iff i0=i and k0=k. U[i'*m+j,c]=1 iff i0=i' and j0=j.
    V[j*p+k',c]=1 iff j0=j and k0=k'.
    So all three are 1 iff i0=i=i' and j0=j and k0=k=k'. So the sum = 1 iff i'=i and k'=k, else 0.
    = delta_{i',i} delta_{k',k}. ✓
    
    So the correct tensor equation is:
    sum_c W[k*n+i, c] * U[i'*m+j, c] * V[j*p+k', c] = delta_{i',i} * delta_{k',k}
    
    This is NOT the all-ones tensor. It's a much more structured tensor.
    
    So my verification function was WRONG. Let me fix it.
    """
    pass

def construct():
    n, m, p = DIMENSIONS
    schoolbook_result = _schoolbook(n, m, p)
    
    # The correct tensor equation for matmul is:
    # sum_c W[k*n+i, c] * U[i2*m+j, c] * V[j*p+k2, c] = delta_{i2,i} * delta_{k2,k}
    # for all i,k,i2,k2,j.
    
    # Let's implement a proper verification.
    def _verify_correct(U, V, W, n, m, p, rank):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        # For each (i, k, i2, k2, j):
        # sum_c W[k*n+i, c] * U[i2*m+j, c] * V[j*p+k2, c] should equal delta_{i2,i} * delta_{k2,k}
        for i in range(n):
            for k in range(p):
                for i2 in range(n):
                    for k2 in range(p):
                        for j in range(m):
                            expected = 1.0 if (i2 == i and k2 == k) else 0.0
                            val = 0.0
                            for c in range(rank):
                                val += W[k * n + i, c] * U[i2 * m + j, c] * V[j * p + k2, c]
                            if abs(val - expected) > 1e-9:
                                return False
        return True
    
    # Try to find a lower rank decomposition.
    # For 2x5 x 5x5, the schoolbook rank is 50.
    # Known results: the rank of 2x5 x 5x5 matmul is at most 50 (schoolbook).
    # Can we do better?
    
    # Strategy: Try rank 49, 48, etc. by modifying the schoolbook.
    # One approach: look for linear dependencies among the rank-1 tensors.
    
    # The rank-1 tensor for (i,j,k) is:
    # T_{i,j,k}[i2,j2,k2] = delta_{i2,i} * delta_{j2,j} * delta_{k2,k} ... no.
    # Actually, the rank-1 tensor in the bilinear algorithm sense is:
    # For term c=(i,j,k): U_c[i2*m+j2] = delta_{i2,i}*delta_{j2,j}, V_c[j2*p+k2] = delta_{j2,j}*delta_{k2,k},
    # W_c[k2*n+i2] = delta_{k2,k}*delta_{i2,i}.
    # The tensor is T_c[i2,j2,k2] = U_c[i2*m+j2] * V_c[j2*p+k2] * W_c[k2*n+i2]
    # = delta_{i2,i}*delta_{j2,j} * delta_{j2,j}*delta_{k2,k} * delta_{k2,k}*delta_{i2,i}
    # = delta_{i2,i}*delta_{j2,j}*delta_{k2,k} (since deltas are idempotent).
    
    # So each rank-1 term contributes delta_{i2,i}*delta_{j2,j}*delta_{k2,k} to the tensor.
    # The sum over all (i,j,k) gives delta_{i2,i}*delta_{j2,j}*delta_{k2,k} summed = 1 when i2=i, j2=j, k2=k.
    # Wait, that's just the standard basis.
    
    # Hmm, I'm getting confused. Let me just try a few structured modifications.
    
    # For now, return schoolbook. A proper search would be needed.
    return schoolbook_result

# EVOLVE-BLOCK-END
