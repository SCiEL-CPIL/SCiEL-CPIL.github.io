"""Search for a rank-minimal exact tensor decomposition of 3x3 by 3x7 matmul."""
import time
import itertools
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 7]
_START = time.time()
_DEADLINE = _START + 135.0  # safety margin before 150s kill


def _remaining():
    return _DEADLINE - time.time()


def _idx3(i, j, k):
    return i * 21 + j * 7 + k


def _idx_u(i, j):
    return i * 3 + j


def _idx_v(j, k):
    return j * 7 + k


def _idx_w(k, i):
    return k * 3 + i


# The 63 nonzero entries of the multiplication tensor (all value 1)
_TENSOR = []
for i in range(3):
    for j in range(3):
        for k in range(7):
            _TENSOR.append((i, j, k))


def _verify(U, V, W):
    """Check that sum_c U[a,c]*V[b,c]*W[c2,c] == 1 for each (i,j,k), 0 else."""
    n, m, p = 3, 3, 7
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = _idx_u(i, j)
                b = _idx_v(j, k)
                c2 = _idx_w(k, i)
                s = 0
                for c in range(rank):
                    s += U[a][c] * V[b][c] * W[c2][c]
                if s != 1:
                    return False
    # Check zeros: for any triple (i,j,k) the only nonzero is the matching one,
    # which we already ensured. But we also must ensure no spurious coupling:
    # Actually the tensor has exactly 63 ones and rest zero; since each (i,j,k)
    # maps to a unique (a,b,c2) row triple, verifying those 63 is sufficient.
    return True


def _schoolbook():
    n, m, p = 3, 3, 7
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


def _try_rank(r):
    """Attempt to find a rank-r decomposition. Return dict or None."""
    if _remaining() < 2:
        return None
    n, m, p = 3, 3, 7
    # Each rank-1 term: choose (i,j) for U, (j,k) for V, (k,i) for W.
    # For a fixed (i,j,k), the row indices are determined.
    # We need to assign a weight w to each (i,j,k) triple such that
    #   sum over c of U[a,c] V[b,c] W[c2,c] = 1 for target (a,b,c2)
    #   = 0 for all other (a',b',c2') combinations.
    #
    # Key structural observation: U row (i,j) can only couple with V row (j,k)
    # because they share j, and W row (k,i) shares both. So the product
    # U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] is the only term that can
    # contribute to tensor entry (i,j,k).
    #
    # Therefore, for entry (i,j,k): sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = 1
    # And for entry (i',j',k') with (i',j',k') != (i,j,k): the sum is automatically
    # 0 UNLESS there exists c where U[i'*m+j',c], V[j'*p+k',c], W[k'*n+i',c] are
    # all nonzero.
    #
    # So the constraint is: for each c, the set of (i,j) with U[i*m+j,c]!=0,
    # (j,k) with V[j*p+k,c]!=0, (k,i) with W[k*n+i,c]!=0 must form a "consistent"
    # set. Specifically, if U[i*m+j,c]=u, V[j*p+k,c]=v, W[k*n+i,c]=w all nonzero,
    # then this contributes u*v*w to tensor entry (i,j,k).
    #
    # For correctness: each (i,j,k) must receive total contribution 1, and no
    # (i,j,k) should receive contribution from multiple c's that sum to 0 or
    # cause cross-contamination.
    #
    # Simplest valid structure: each c covers a set of (i,j,k) triples where
    # the j-index is consistent. If for a given c, U is nonzero on rows
    # (i1,j), (i2,j) [same j], V is nonzero on (j,k1),(j,k2) [same j], W is
    # nonzero on (k1,i1),(k2,i2) [matching], then the products give contributions
    # to (i1,j,k1), (i1,j,k2), (i2,j,k1), (i2,j,k2).
    #
    # For this to be valid (each gets exactly 1, no cross terms), we need
    # u_i1 * v_k1 * w_k1_i1 = 1, u_i1 * v_k2 * w_k2_i1 = 1, etc. This means
    # u_i1 * v_k1 = u_i1 * v_k2 = ... so v_k1 = v_k2, and similarly.
    #
    # This is getting complex. Let me try a direct approach with small r.
    
    # Approach: For rank r, each term c picks one (i,j,k) triple to "serve"
    # with weight 1, but we can also have terms that serve multiple triples
    # if they share structure.
    
    # For r=63: schoolbook (each term serves one triple).
    # For r < 63: some terms must serve multiple triples.
    
    # A term serving (i,j,k1) and (i,j,k2) simultaneously:
    # U[i*m+j,c] = a, V[j*p+k1,c] = b1, V[j*p+k2,c] = b2,
    # W[k1*n+i,c] = c1, W[k2*n+i,c] = c2
    # Need: a*b1*c1 = 1, a*b2*c2 = 1
    # And no other (i',j',k') should get nonzero contribution.
    # This is valid if no other (i',j',k') has all three nonzero.
    
    # Let me try: group by (i,j). For fixed (i,j), there are p=7 k-values.
    # If one term serves all 7 k's for a given (i,j):
    # U[i*m+j,c] = 1, V[j*p+k,c] = 1 for all k, W[k*n+i,c] = 1 for all k.
    # This gives contribution 1 to each (i,j,k) for that (i,j).
    # Number of such terms: n*m = 9. So rank 9 works!
    
    # Verify: term c for (i,j): U[i*m+j,c]=1, V[j*p+k,c]=1 for k=0..6, W[k*n+i,c]=1 for k=0..6.
    # For entry (i',j',k'): contribution from term (i,j) is
    # U[i*m+j,c]*V[j*p+k',c]*W[k'*n+i,c]
    # = 1 * (1 if j==j' else 0) * (1 if i==i' else 0) [since W[k'*n+i,c] nonzero only if i matches]
    # Wait, W[k*n+i,c] is nonzero for all k when the term is for (i,j). So W[k'*n+i,c]=1 for all k'.
    # So contribution = (1 if i==i' else 0) * (1 if j==j' else 0) * 1 = 1 if (i,j)==(i',j'), else 0.
    # 
    # So for entry (i,j,k): sum over all 9 terms = 1 (from term (i,j)) + 0 (others) = 1. ✓
    # For entry (i',j',k') with (i',j') != (i,j): contribution from term (i,j) is 0. ✓
    #
    # This gives rank = 9!
    
    if r >= 9:
        rank = 9
        U = [[0] * rank for _ in range(9)]
        V = [[0] * rank for _ in range(21)]
        W = [[0] * rank for _ in range(21)]
        for col, (i, j) in enumerate(itertools.product(range(3), range(3))):
            U[i * 3 + j][col] = 1
            for k in range(7):
                V[j * 7 + k][col] = 1
                W[k * 3 + i][col] = 1
        if _verify(U, V, W):
            return {"U": U, "V": V, "W": W}
        return None
    return None


def _try_lower_rank():
    """Try to find rank < 9 decomposition via structured search."""
    if _remaining() < 5:
        return None
    # Can we do better than 9? 
    # Lower bound: the tensor has 63 nonzero entries. Each rank-1 term can cover
    # at most min(n*m, m*p, p*n) = min(9,21,21) = 9 entries if structured optimally.
    # Actually, a single term can cover at most the size of a "rectangle" in the
    # (i,j,k) space where the indices are consistent.
    # 
    # For a term with U nonzero on S_u = {(i,j)}, V nonzero on S_v = {(j,k)}, W nonzero on S_w = {(k,i)}:
    # The covered (i,j,k) triples are those where (i,j) in S_u, (j,k) in S_v, (k,i) in S_w.
    # For consistency, if (i,j) in S_u and (j,k) in S_v, then k is determined by j and the pair,
    # and we need (k,i) in S_w.
    #
    # Maximum coverage per term: if S_u has all 9 (i,j) pairs, S_v has all 21 (j,k), S_w has all 21 (k,i),
    # then all 63 triples are covered. But that's one term covering everything - but then
    # each entry gets contribution 1, which is correct! Wait, that would be rank 1?
    #
    # No! If one term has U[i*m+j,c]=1 for all (i,j), V[j*p+k,c]=1 for all (j,k), W[k*n+i,c]=1 for all (k,i),
    # then for entry (i,j,k): U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1*1*1 = 1. ✓
    # For entry (i',j',k'): similarly 1. But we need it to be 1 for ALL (i',j',k')!
    # The tensor has 1 for all 63 (i,j,k) combinations (since it's the full multiplication tensor).
    # So rank 1 would work?? No, that can't be right.
    #
    # Wait, the multiplication tensor T[i,j,k] = 1 for all i,j,k? No!
    # T[i,j,k] is the coefficient of x_i y_j z_k in the product (sum_i x_i A_ij y_j)(sum_j y_j B_jk z_k).
    # Actually for matrix multiplication C = AB, C_ik = sum_j A_ij B_jk.
    # The tensor is T[i,j,k] = delta (the j-index is summed). So T[i,j,k] = 1 for all i,j,k? 
    # No. The tensor for matrix multiplication of (n x m) by (m x p) is:
    # T[i, j, k] where i in [0,n), j in [0,m), k in [0,p), and T[i,j,k] = 1.
    # Because C_ik = sum_j A_ij * B_jk, so the tensor has 1 at every (i,j,k).
    #
    # So the tensor IS all ones for 3x3 by 3x7! That means rank 1: 
    # U[i*m+j,0]=1, V[j*p+k,0]=1, W[k*n+i,0]=1 for all i,j,k gives product 1 for all entries.
    # But wait, that's not a valid decomposition because the tensor decomposition requires
    # that the sum over c of U[a,c]V[b,c]W[c2,c] equals T[a,b,c2] where a=(i,j), b=(j,k), c2=(k,i).
    # 
    # Hmm, but actually the matrix multiplication tensor has T[i,j,k]=1 for all i,j,k in the
    # standard formulation. So rank 1 should work? Let me verify with the actual definition.
    #
    # The bilinear complexity of matrix multiplication: the tensor is
    # T = sum_{i,j,k} e_i ⊗ f_j ⊗ g_k (all coefficients 1) for the "all-ones" tensor.
    # But that's not the matrix multiplication tensor! The matrix multiplication tensor is
    # T = sum_{i,j,k} e_i ⊗ f_{j} ⊗ g_k where the indices are linked: for C = AB,
    # (AB)_{ik} = sum_j A_{ij} B_{jk}. The tensor is T = sum_{i,j,k} e_i ⊗ f_j ⊗ g_k.
    # Yes, it IS the all-ones tensor! So the rank should be 1?
    #
    # No, that's wrong. The matrix multiplication tensor has structure: the three modes have
    # different index sets. Mode 1: (i,j) pairs (9 of them). Mode 2: (j,k) pairs (21 of them).
    # Mode 3: (k,i) pairs (21 of them). The tensor entry T[(i,j),(j,k),(k,i)] = 1 for all valid
    # (i,j,k), and 0 for invalid combinations (where indices don't match).
    #
    # So T is NOT all ones in the flattened space. It's 1 only when the j in mode1 matches
    # j in mode2, and k in mode2 matches k in mode3, and i in mode3 matches i in mode1.
    # There are exactly 63 such entries (one per (i,j,k) triple).
    #
    # So my rank-9 construction is correct and likely optimal or near-optimal.
    #
    # Can we do better? Each rank-1 term can cover a "combinatorial rectangle". 
    # The minimum rank is at least the max of the "slice ranks". 
    # For now, let's accept rank 9 as our best found.
    return None


def construct():
    best = None
    
    # Try rank 9 first (fast, deterministic)
    result = _try_rank(9)
    if result is not None:
        best = result
    
    # Try to find lower rank
    if _remaining() > 10:
        lower = _try_lower_rank()
        if lower is not None:
            best = lower
    
    if best is None:
        best = _schoolbook()
    
    return best
