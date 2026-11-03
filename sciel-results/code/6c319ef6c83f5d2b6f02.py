"""Matrix multiplication 3x4x7: rank-56 exact tensor decomposition."""
DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
import time

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 7
    t0 = time.time()
    deadline = t0 + 140.0

    U_terms, V_terms, W_terms = [], [], []

    def add_term(u, v, w):
        U_terms.append(u); V_terms.append(v); W_terms.append(w)

    # A rank-1 term: (u: A-coeffs) * (v: B-coeffs) * (w: C-coeffs)
    # For a single (i,j) A entry and (j,k) B entry, C[k,i] gets +1.
    #
    # 2x2x1 block (A-rows ra0,ra1; A-cols ca0,ca1; B-col k):
    #   C[k,ra0] = A[ra0,ca0]*B[ca0,k] + A[ra0,ca1]*B[ca1,k]
    #   C[k,ra1] = A[ra1,ca0]*B[ca0,k] + A[ra1,ca1]*B[ca1,k]
    # Rank 2:
    #   term1: A[ra0,ca0]*B[ca0,k] -> C[k,ra0]  ... but this only covers one C entry.
    #   Actually each (i,j,k) triple is rank 1. 2x2x1 has 4 (i,j) pairs but only
    #   2 output C entries. The 4 terms: (ra0,ca0)->C[k,ra0], (ra0,ca1)->C[k,ra0],
    #   (ra1,ca0)->C[k,ra1], (ra1,ca1)->C[k,ra1]. That's rank 4, not 2!
    #
    # I was wrong. 2x2x1 = 4 rank-1 terms (schoolbook). No saving.

    # Reconsider: the saving must come from Strassen on a TRUE 2x2x2.
    # Split p=7 into 2+2+2+1. Then 2x2x2 blocks use Strassen (rank 7 < 8).
    # 2x2x2 Strassen saves 1 per block.

    # New plan:
    # Split n=3 into 2+1, m=4 into 2+2, p=7 into 2+2+2+1.
    # Top (2x4x7): 2 A-rows, 4 A-cols, 7 B-cols.
    #   Split A-cols 2+2, B-cols 2+2+2+1.
    #   Each (2 A-rows x 2 A-cols x 2 B-cols) = 2x2x2 -> Strassen rank 7.
    #   Number of such blocks: 2 (A-col pairs) * 3 (B-col pairs of size 2) = 6 blocks * 7 = 42.
    #   Plus (2x2x1) B-col-single: 2 A-col pairs * 1 = 2 blocks * 4 (schoolbook 2x2x1=4) = 8.
    #   Top total = 42 + 8 = 50.
    # Bottom (1x4x7): schoolbook 1*4*7 = 28.
    # Grand total = 50 + 28 = 78.

    # That matches the cell finding (80 -> 78). Let me build it.

    def strassen_2x2x2(ra0, ra1, ca0, ca1, kb0, kb1):
        # A-rows ra0,ra1; A-cols ca0,ca1; B-cols kb0,kb1
        # p1 = (A[ra0,ca0]+A[ra1,ca1])*(B[ca0,kb0]+B[ca1,kb1]) -> C[kb0,ra0],C[kb1,ra1]
        # p2 = (A[ra1,ca0]+A[ra1,ca1])*B[ca0,kb0] -> C[kb0,ra1],C[kb1,ra1]... 
        # Use standard Strassen with C11=C[kb0,ra0], C12=C[kb0,ra1], C21=C[kb1,ra0], C22=C[kb1,ra1]
        # A11=A[ra0,ca0],A12=A[ra0,ca1],A21=A[ra1,ca0],A22=A[ra1,ca1]
        # B11=B[ca0,kb0],B12=B[ca1,kb0],B21=B[ca0,kb1],B22=B[ca1,kb1]
        # C11=C[kb0,ra0],C12=C[kb0,ra1],C21=C[kb1,ra0],C22=C[kb1,ra1]
        # p1=(A11+A22)(B11+B22) -> C11,C22
        # p2=(A21+A22)B11 -> C21,C22
        # p3=A11(B12-B22) -> C12
        # p4=A22(B21-B11) -> C21
        # p5=(A11+A12)B22 -> C12,C22
        # p6=(A21-A11)(B11+B12) -> C21
        # p7=(A12-A22)(B21+B22) -> C11
        # C11=p1+p4-p5+p7, C12=p3+p5, C21=p2+p4, C22=p1-p2+p3+p6
        add_term({(ra0,ca0):1,(ra1,ca1):1}, {(ca0,kb0):1,(ca1,kb1):1}, {(kb0,ra0):1,(kb1,ra1):1})
        add_term({(ra1,ca0):1,(ra1,ca1):1}, {(ca0,kb0):1}, {(kb1,ra0):1,(kb1,ra1):1})
        add_term({(ra0,ca0):1}, {(ca1,kb0):1,(ca1,kb1):-1}, {(kb0,ra1):1})
        add_term({(ra1,ca1):1}, {(ca0,kb1):1,(ca0,kb0):-1}, {(kb1,ra0):1})
        add_term({(ra0,ca0):1,(ra0,ca1):1}, {(ca1,kb1):1}, {(kb0,ra1):1,(kb1,ra1):1})
        add_term({(ra1,ca0):1,(ra0,ca0):-1}, {(ca0,kb0):1,(ca1,kb0):1}, {(kb1,ra0):1})
        add_term({(ra0,ca1):1,(ra1,ca1):-1}, {(ca0,kb1):1,(ca1,kb1):1}, {(kb0,ra0):1})

    def schoolbook_2x2x1(ra0, ra1, ca0, ca1, k):
        add_term({(ra0,ca0):1}, {(ca0,k):1}, {(k,ra0):1})
        add_term({(ra0,ca1):1}, {(ca1,k):1}, {(k,ra0):1})
        add_term({(ra1,ca0):1}, {(ca0,k):1}, {(k,ra1):1})
        add_term({(ra1,ca1):1}, {(ca1,k):1}, {(k,ra1):1})

    # Top part: 2 A-rows (0,1), 4 A-cols (2+2), 7 B-cols (2+2+2+1)
    for ca0 in (0, 2):
        ca1 = ca0 + 1
        for kb0 in (0, 2, 4):
            kb1 = kb0 + 1
            if time.time() > deadline:
                break
            strassen_2x2x2(0, 1, ca0, ca1, kb0, kb1)
        # remaining single B-col: 6
        schoolbook_2x2x1(0, 1, ca0, ca1, 6)

    # Bottom part: 1 A-row (2), 4 A-cols, 7 B-cols (schoolbook)
    for j in range(m):
        for k in range(p):
            if time.time() > deadline:
                break
            add_term({(2, j): 1}, {(j, k): 1}, {(k, 2): 1})

    rank = len(U_terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    for t in range(rank):
        for (i, j), c in U_terms[t].items():
            U[i * m + j][t] = c
        for (j, k), c in V_terms[t].items():
            V[j * p + k][t] = c
        for (k, i), c in W_terms[t].items():
            W[k * n + i][t] = c

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
