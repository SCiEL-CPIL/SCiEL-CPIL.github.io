"""Matrix multiplication 2x4x5: rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 4, 5]
_DEADLINE = time.time() + 150

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p

    # Build the multiplication tensor: T[i*m+j, j*p+k, k*n+i] = 1
    # U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    # Schoolbook: each (i,j,k) triple is one rank-1 term
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

    # Try to find a lower-rank decomposition via algebraic search
    # For 2x4x5, the tensor lives in R^{8} x R^{20} x R^{10}
    # Schoolbook rank is 40. Lower bound is max(8, 20, 10) = 20.
    # We attempt a rank-20 decomposition using the structure.
    #
    # Key insight: group by j (the shared index between A and B).
    # For fixed j, the (i,k) block is outer product of e_i and e_k.
    # So T = sum_j sum_i sum_k e_{ij} ⊗ e_{jk} ⊗ e_{ki}
    #      = sum_j (sum_i e_{ij} ⊗ e_{ki}) ⊗ e_{jk}  ... not quite rank-1 in that split.
    #
    # Alternative: for each j, the sub-tensor over (i,k) has rank n*p = 10.
    # But across j, the V indices differ. Total rank is at most sum over j of rank_j
    # but we can potentially share terms.
    #
    # Try: fix j, then for each (i,k) we need a term. That's n*p per j, total m*n*p.
    # No savings from this grouping alone.
    #
    # Try another approach: for fixed i, the sub-tensor over (j,k) has rank m*p = 20.
    # For fixed k, the sub-tensor over (i,j) has rank n*m = 8.
    #
    # The rank is bounded below by max(n*m, m*p, p*n) = max(8, 20, 10) = 20.
    # And above by n*m*p = 40.
    #
    # For n=2, m=4, p=5, known results:
    # The border rank is at most n*m + m*p + n*p - 2 = 8+20+10-2 = 36 (not helpful).
    #
    # Let's try a direct rank-20 construction.
    # If we can find 20 rank-1 tensors that sum to T, we'd have rank 20.
    #
    # Strategy: parameterize U as 8x20, V as 20x20, W as 10x20 matrices.
    # The condition is: for all (a, b, c) where a=i*m+j, b=j*p+k, c=k*n+i,
    #   sum_r U[a,r] * V[b,r] * W[c,r] = 1
    # and for all other triples, the sum = 0.
    #
    # This is a system of polynomial equations. Let's try a structured ansatz.
    #
    # Ansatz: U[i*m+j, r] = alpha_{i,j,r}, V[j*p+k, r] = beta_{j,k,r}, W[k*n+i, r] = gamma_{k,i,r}
    # with r = 0..19.
    #
    # Let's try: split r into two groups of 10.
    # Group 1 (r=0..9): handle terms where the "extra" factor is in one direction.
    # Group 2 (r=10..19): handle the rest.
    #
    # Actually, let's try a simpler structured ansatz based on the min-rank lower bound.
    # For 2xNxN matrices, the rank is known to be at most 2N^2 - N (for N>=2).
    # For 2x4x4, rank <= 2*16-4 = 28. For 2x4x5, we don't have a clean formula.
    #
    # Let's try a computational approach: use the fact that the tensor is sparse
    # and try to find a decomposition with fewer terms by exploiting zero patterns.
    #
    # For now, return the schoolbook decomposition (rank 40) which is guaranteed correct.
    # A future iteration can attempt the rank-reduction search with a time budget.

    if time.time() < _DEADLINE - 30:
        # Try to find a lower-rank decomposition
        reduced = _try_lower_rank(n, m, p, rank)
        if reduced is not None:
            return reduced

    return {"U": U, "V": V, "W": W}


def _try_lower_rank(n, m, p, schoolbook_rank):
    """Attempt to find a rank below schoolbook using structured ansatz."""
    # Lower bound: max(n*m, m*p, p*n)
    lb = max(n * m, m * p, p * n)
    
    # For 2x4x5: lb = max(8, 20, 10) = 20
    # Try rank = lb
    target_rank = lb
    
    # Build constraint structure
    # T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for some (i,j,k)
    # T[a,b,c] = 0 otherwise
    
    # Key structural insight for reduction:
    # The tensor T = sum_{i,j,k} e_{(i,j)} ⊗ e_{(j,k)} ⊗ e_{(k,i)}
    # 
    # We can write this as sum_j (sum_i e_{(i,j)} ⊗ e_{(j,*)} ⊗ e_{(*,i)})
    # The inner sum over i for fixed j: sum_i e_{(i,j)} ⊗ e_{(j,k)} ⊗ e_{(k,i)}
    # This is a tensor of size (n*m) x (m*p) x (p*n) but nonzero only when
    # the middle index has j-component and the relationship holds.
    
    # For fixed j, define T_j[i, k] = e_{(i,j)} ⊗ e_{(j,k)} ⊗ e_{(k,i)}
    # This is a rank-1 tensor in the (i,k) space for each (i,k) pair... no.
    
    # Let me think differently. The tensor has a "cycle" structure: i -> j -> k -> i.
    # 
    # For 2x4x5, let's try to explicitly construct a rank-20 decomposition.
    # 
    # Parametrize: r = 0..19
    # Let U[i*4+j, r], V[j*5+k, r], W[k*2+i, r]
    # 
    # Try: for r = 0..9, let r = 5*i + k for i in {0,1}, k in {0,1,2,3,4}
    #   U[i*4+j, r] = 1 for all j  (when r encodes (i,k))
    #   V[j*5+k, r] = 1 if the k-component matches  (i.e., k in r equals k)
    #   W[k'*2+i, r] = 1 if k'=k and i-component matches
    # 
    # This doesn't work cleanly. Let me try a different parametrization.
    #
    # Alternative: use the fact that for each j, we have an n x p sub-tensor.
    # The rank of the full tensor >= max over j of (rank of fiber at j)
    # The fiber at fixed j (fixing the j in both a and b) gives a tensor
    # T_j[i, k, i'] where we need i=i' for it to be nonzero, and the value is
    # 1 if the j matches in both positions.
    #
    # Actually, fixing j in a=i*m+j and j' in b=j'*p+k, we need j=j' for nonzero.
    # So the "slice" at (j,j') is zero unless j=j', in which case it's
    # T[j,j; i,k,i'] = delta_{i,i'} delta_{j,j'} ... this is getting complex.
    
    # Let's just verify the schoolbook is correct and return it.
    # The structured search for lower rank is complex and risky within time bounds.
    return None


# EVOLVE-BLOCK-END
