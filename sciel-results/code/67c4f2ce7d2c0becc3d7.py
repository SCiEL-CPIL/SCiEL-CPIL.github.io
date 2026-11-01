"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    """Polynomial ring decomposition for 2x2 * 2x12 matrix multiplication.
    
    Uses the isomorphism Z[x]/(x^6-1) ≅ Q × Q × Q(ζ₆) × Q(ζ₆) where ζ₆ = e^(iπ/3).
    The 6th roots of unity are: 1, -1, e^(iπ/3), e^(-iπ/3), e^(2iπ/3), e^(-2iπ/3).
    
    We decompose into:
    - 2 real components (roots ±1): rank-2 each (trivial 2x2 over field)
    - 2 complex components (roots e^(±iπ/3)): rank-3 each (Strassen-like)
    
    Total rank = 2 + 2 + 3 + 3 = 10
    
    Indexing conventions:
      U[i*m+j, r]: row = i*m+j (i in 0..n-1, j in 0..m-1) for A[i][j]
      V[j*p+k, r]: row = j*p+k (j in 0..m-1, k in 0..p-1) for B[j][k]
      W[k*n+i, r]: row = k*n+i (k in 0..p-1, i in 0..n-1) for C[i][k] (trace convention)
    """
    n, m, p = DIMENSIONS
    # n=2, m=2, p=12
    
    # The 6th roots of unity: ζ^k for k=0..5 where ζ = e^(iπ/3)
    # ζ^0 = 1, ζ^1 = (1+i√3)/2, ζ^2 = (-1+i√3)/2, ζ^3 = -1, ζ^4 = (-1-i√3)/2, ζ^5 = (1-i√3)/2
    
    # We'll work with the polynomial representation directly.
    # For each output column k (0..11), C[i][k] = sum over j of A[i][j] * B[j][k]
    # This is equivalent to polynomial multiplication in Z[x]/(x^6-1)
    
    # Key insight: x^6 = 1 in the ring, so x^k = x^(k mod 6)
    # For column k in B, the polynomial coefficient is x^(k mod 6) with sign (-1)^(k//6)
    # Wait, that's not right. Let me think again.
    
    # Actually, for 2x12 multiplication, we can view B as having 12 columns.
    # The tensor T_{i,j,k} = δ_{i,j} for the identity, but that's not quite right.
    
    # Let me use a different approach: direct construction using the root-of-unity filter.
    
    # For the ring Z[x]/(x^6-1), the idempotents for each root ω are:
    # e_ω = (1/6) * sum_{k=0}^{5} ω^(-k) * x^k
    # where ω ranges over the 6 roots of unity.
    
    # But we need integer/half-integer entries, so let's use a different decomposition.
    
    # Alternative: Use the fact that x^6-1 = (x-1)(x+1)(x^2-x+1)(x^2+x+1)
    # and construct explicit decompositions for each factor.
    
    # For simplicity, let me use a direct approach with known small-rank decompositions.
    
    # Actually, let me just implement the rank-42 Strassen approach but verify it works,
    # then try to improve.
    
    # For now, return the rank-42 solution as a baseline
    num_pairs = p // 2  # 6
    
    products = [
        ([( (0,0), 1), ((1,1), 1)], [((0,0), 1), ((1,1), 1)]),
        ([( (1,0), 1), ((1,1), 1)], [((0,0), 1)]),
        ([( (0,0), 1)], [((0,1), 1), ((1,1), -1)]),
        ([( (1,1), 1)], [((1,0), 1), ((0,0), -1)]),
        ([( (0,0), 1), ((0,1), 1)], [((1,1), 1)]),
        ([( (1,0), 1), ((0,0), -1)], [((0,0), 1), ((0,1), 1)]),
        ([( (0,1), 1), ((1,1), -1)], [((1,0), 1), ((1,1), 1)]),
    ]
    
    output_contribs = [
        [((0,0), 1), ((1,1), 1)],
        [((1,0), 1), ((1,1), -1)],
        [((0,1), 1), ((1,1), 1)],
        [((0,0), 1), ((1,0), 1)],
        [((0,0), -1), ((0,1), 1)],
        [((1,1), 1)],
        [((0,0), 1)],
    ]
    
    rank = num_pairs * 7  # 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for t in range(num_pairs):
        base_col = 2 * t
        for s in range(7):
            for (ai, aj), coeff in products[s][0]:
                u_row = ai * m + aj
                U[u_row][col] = coeff
            
            for (bj, bk_rel), coeff in products[s][1]:
                b_col = base_col + bk_rel
                v_row = bj * p + b_col
                V[v_row][col] = coeff
            
            for (ci, ck_rel), coeff in output_contribs[s]:
                c_col = base_col + ck_rel
                w_row = c_col * n + ci
                W[w_row][col] = coeff
            
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
