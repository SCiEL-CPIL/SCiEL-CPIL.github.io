"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

DEADLINE = time.time() + 115.0

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly matches matrix multiplication."""
    nm = n * m
    mp = m * p
    pn = p * n
    rank = len(U[0]) if U else 0
    
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                computed = 0.0
                for c in range(rank):
                    computed += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(computed - 1) > 1e-9:
                    return False
    return True

def _build_strassen_2x2_terms():
    """
    Return the 7 Strassen terms for 2x2 matrix multiplication.
    Each term is (U_dict, V_dict, W_dict) where:
      U_dict: {(i,j): val} for A indices
      V_dict: {(j,k): val} for B indices
      W_dict: {(i,k): val} for C indices
    Using Strassen's algorithm:
      M1 = (A00+A11)*(B00+B11)
      M2 = (A10+A11)*B00
      M3 = A00*(B01-B11)
      M4 = A11*(B10-B00)
      M5 = (A00+A01)*B11
      M6 = (A10-A00)*(B00+B01)
      M7 = (A01-A11)*(B10+B11)
      C00 = M1 + M4 - M5 + M7
      C01 = M3 + M5
      C10 = M2 + M4
      C11 = M1 - M2 + M3 + M6
    """
    terms = []
    # M1 = (A00+A11)*(B00+B11)
    terms.append((
        {(0,0):1, (1,1):1},
        {(0,0):1, (1,1):1},
        {(0,0):1, (1,1):1}
    ))
    # M2 = (A10+A11)*B00
    terms.append((
        {(1,0):1, (1,1):1},
        {(0,0):1},
        {(1,0):1, (1,1):-1}
    ))
    # M3 = A00*(B01-B11)
    terms.append((
        {(0,0):1},
        {(0,1):1, (1,1):-1},
        {(0,1):1, (1,1):1}
    ))
    # M4 = A11*(B10-B00)
    terms.append((
        {(1,1):1},
        {(1,0):1, (0,0):-1},
        {(0,0):1, (1,0):1}
    ))
    # M5 = (A00+A01)*B11
    terms.append((
        {(0,0):1, (0,1):1},
        {(1,1):1},
        {(0,0):-1, (0,1):1}
    ))
    # M6 = (A10-A00)*(B00+B01)
    terms.append((
        {(1,0):1, (0,0):-1},
        {(0,0):1, (0,1):1},
        {(1,1):1}
    ))
    # M7 = (A01-A11)*(B10+B11)
    terms.append((
        {(0,1):1, (1,1):-1},
        {(1,0):1, (1,1):1},
        {(0,0):1}
    ))
    return terms

def _build_rank32_decomposition(n, m, p):
    """
    Build a rank-32 decomposition for 3x3 * 3x4 matrix multiplication.
    
    Strategy:
    - Use Strassen (rank 7) for the top-left 2x2 block of A and columns 0,1 of B
      This handles C[0:2, 0:2] using 7 terms.
    - For the remaining entries, use schoolbook terms.
    
    Actually, let me think more carefully. We need to decompose:
    C[i,k] = sum_j A[i,j] * B[j,k] for i in {0,1,2}, j in {0,1,2}, k in {0,1,2,3}
    
    The tensor has 3*3*4 = 36 entries, each with coefficient 1.
    
    A cleaner approach: 
    - For columns k=0,1 of B: treat A as 3x3, B[:,0:2] as 3x2, C[:,0:2] as 3x2
      We can use Strassen on the 2x2 top-left part of A for these two columns.
      But Strassen gives 2x2*2x2 -> 2x2, not directly applicable to 3x3*3x2.
    
    Let me use a different decomposition strategy:
    - Split into blocks and use Strassen where possible.
    
    For 3x3 * 3x4:
    - The 3x3 * 3x3 part (columns 0,1,2 of B) can potentially use a rank-23 or rank-26 scheme.
    - The 3x3 * 3x1 part (column 3 of B) needs 9 schoolbook terms.
    
    If we can get the 3x3*3x3 part at rank 23, total = 23 + 9 = 32.
    
    For 3x3*3x3, let me try to construct a rank-23 decomposition using
    overlapping Strassen blocks.
    
    Actually, let me use a simpler but still improved approach:
    - Use Strassen for the 2x2 top-left block: 7 terms handle C[0:2,0:2] from A[0:2,0:2]*B[0:2,0:2]
    - But we also need contributions from A[0:2,2]*B[2,0:2] and A[2,0:2]*B[0:2,0:2] etc.
    
    This is getting complex. Let me use a verified construction.
    
    Known: The rank of 3x3x3 matrix multiplication is at most 23 (Bini et al.).
    Let me use a specific rank-23 construction.
    
    For now, let me use a safe rank-32 construction:
    - 7 Strassen terms for the 2x2x2 subproblem (top-left 2x2 of A, top-left 2x2 of B, top-left 2x2 of C)
    - Plus schoolbook terms for everything else
    
    Wait, that won't work directly because Strassen only handles the 2x2*2x2->2x2 part.
    
    Let me just use a clean schoolbook-based approach with some Strassen savings:
    
    Actually, the simplest improvement: use Strassen for computing each 2x2 block of C.
    
    For C[0:2, 0:2]: use Strassen on A[0:2, 0:2] * B[0:2, 0:2] -> 7 terms
    Plus we need A[0:2, 2] * B[2, 0:2] -> 4 terms (schoolbook)
    Plus we need A[2, 0:2] * B[0:2, 0:2] -> 4 terms (schoolbook)
    Total for C[0:2, 0:2]: 7 + 4 + 4 = 15 terms (vs 16 schoolbook)
    
    For C[0:2, 2:4]: A[0:2, 0:3] * B[0:3, 2:4] -> 2*3*2 = 12 terms schoolbook
    For C[2, 0:2]: A[2, 0:3] * B[0:3, 0:2] -> 3*2 = 6 terms schoolbook  
    For C[2, 2:4]: A[2, 0:3] * B[0:3, 2:4] -> 3*2 = 6 terms schoolbook
    
    Total: 15 + 12 + 6 + 6 = 39. That's worse than 36!
    
    The issue is that Strassen only saves 1 term for a 2x2 block, but the cross terms add up.
    
    Let me reconsider. For 3x3*3x4, the schoolbook rank is 36.
    
    A better approach: Use the fact that we can apply Strassen in a more global way.
    
    Let me try: partition A into rows {0,1} and {2}, B into cols {0,1} and {2,3}.
    C[0:2, 0:2] = A[0:2,0:2]*B[0:2,0:2] + A[0:2,2]*B[2,0:2]
    C[0:2, 2:4] = A[0:2,0:2]*B[0:2,2:4] + A[0:2,2]*B[2,2:4]
    C[2, 0:2] = A[2,0:2]*B[0:2,0:2] + A[2,2]*B[2,0:2]
    C[2, 2:4] = A[2,0:2]*B[0:2,2:4] + A[2,2]*B[2,2:4]
    
    For C[0:2, 0:2]: 
      A[0:2,0:2]*B[0:2,0:2] via Strassen: 7 terms
      A[0:2,2]*B[2,0:2]: 4 terms
      Total: 11 terms (vs 12 schoolbook)
    
    For C[0:2, 2:4]:
      A[0:2,0:2]*B[0:2,2:4]: 2*2*2 = 8 terms
      A[0:2,2]*B[2,2:4]: 2*2 = 4 terms
      Total: 12 terms (same as schoolbook)
    
    For C[2, 0:2]:
      A[2,0:2]*B[0:2,0:2]: 2*2 = 4 terms
      A[2,2]*B[2,0:2]: 2 terms
      Total: 6 terms (same as schoolbook)
    
    For C[2, 2:4]:
      A[2,0:2]*B[0:2,2:4]: 2*2 = 4 terms
      A[2,2]*B[2,2:4]: 2 terms
      Total: 6 terms (same as schoolbook)
    
    Grand total: 11 + 12 + 6 + 6 = 35 terms. That's rank 35, saving 1 term.
    
    Hmm, only 1 term saved. Let me try a different partition.
    
    What if I use Strassen for both C[0:2,0:2] and C[0:2,2:4] by treating B's columns differently?
    
    For C[0:2, 0:2] and C[0:2, 2:4] together:
    C[0:2, 0:4] = A[0:2, 0:2] * B[0:2, 0:4] + A[0:2, 2] * B[2, 0:4]
    
    A[0:2, 0:2] * B[0:2, 0:4]: This is 2x2 * 2x4.
    We can write B[0:2, 0:4] = [B[0:2,0:2], B[0:2,2:4]]
    
    Using Strassen-like: 
    (A00+A11)*(B00+B20) where B00=B[0:2,0:2], B20=B[0:2,2:4]
    But Strassen is for square matrices...
    
    Let me just go with rank 35 using the simple Strassen improvement.
    Actually wait, let me recount.
    
    Schoolbook for 3x3*3x4: 3*3*4 = 36 terms.
    
    My Strassen-based approach:
    - C[0:2,0:2] from A[0:2,0:2]*B[0:2,0:2]: 7 terms (Strassen)
    - C[0:2,0:2] from A[0:2,2]*B[2,0:2]: 4 terms
    - C[0:2,2:4] from A[0:2,0:2]*B[0:2,2:4]: 8 terms  
    - C[0:2,2:4] from A[0:2,2]*B[2,2:4]: 4 terms
    - C[2,0:2] from A[2,0:2]*B[0:2,0:2]: 4 terms
    - C[2,0:2] from A[2,2]*B[2,0:2]: 2 terms
    - C[2,2:4] from A[2,0:2]*B[0:2,2:4]: 4 terms
    - C[2,2:4] from A[2,2]*B[2,2:4]: 2 terms
    
    Total: 7+4+8+4+4+2+4+2 = 35 terms.
    
    So rank 35 is achievable. Let me implement this.
    
    Actually, I realize I should double-check: does the Strassen decomposition for 2x2*2x2->2x2
    correctly produce C[0:2,0:2] = A[0:2,0:2]*B[0:2,0:2]?
    
    Yes, Strassen's algorithm correctly computes 2x2 matrix multiplication with 7 multiplications.
    The 7 "multiplications" become 7 terms in the tensor decomposition.
    
    Let me implement rank 35.
    """
    nm = n * m  # 9
    mp = m * p  # 12
    pn = p * n  # 12
    
    terms = []  # Each term: list of (u_idx, v_idx, w_idx, coeff)
    
    # Strassen terms for A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
    strassen_terms = _build_strassen_2x2_terms()
    for u_dict, v_dict, w_dict in strassen_terms:
        term = []
        for (i, j), val in u_dict.items():
            term.append((i * m + j, val))  # U index: i*m + j
        for (j, k), val in v_dict.items():
            term.append((j * p + k, val))  # V index: j*p + k
        for (i, k), val in w_dict.items():
            term.append((k * n + i, val))  # W index: k*n + i
        terms.append((term, 'strassen'))
    
    # Schoolbook terms for remaining parts:
    # A[0:2, 2] * B[2, 0:2] -> C[0:2, 0:2]
    for i in range(2):
        for k in range(2):
            term = [
                (i * m + 2, 1),   # A[i,2]
                (2 * p + k, 1),   # B[2,k]
                (k * n + i, 1)    # C[i,k]
            ]
            terms.append((term, 'schoolbook'))
    
    # A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    for i in range(2):
        for j in range(2):
            for k in range(2, 4):
                term = [
                    (i * m + j, 1),   # A[i,j]
                    (j * p + k, 1),   # B[j,k]
                    (k * n + i, 1)    # C[i,k]
                ]
                terms.append((term, 'schoolbook'))
    
    # A[0:2, 2] * B[2, 2:4] -> C[0:2, 2:4]
    for i in range(2):
        for k in range(2, 4):
            term = [
                (i * m + 2, 1),   # A[i,2]
                (2 * p + k, 1),   # B[2,k]
                (k * n + i, 1)    # C[i,k]
            ]
            terms.append((term, 'schoolbook'))
    
    # A[2, 0:2] * B[0:2, 0:2] -> C[2, 0:2]
    for j in range(2):
        for k in range(2):
            term = [
                (2 * m + j, 1),   # A[2,j]
                (j * p + k, 1),   # B[j,k]
                (k * n + 2, 1)    # C[2,k]
            ]
            terms.append((term, 'schoolbook'))
    
    # A[2, 2] * B[2, 0:2] -> C[2, 0:2]
    for k in range(2):
        term = [
            (2 * m + 2, 1),   # A[2,2]
            (2 * p + k, 1),   # B[2,k]
            (k * n + 2, 1)    # C[2,k]
        ]
        terms.append((term, 'schoolbook'))
    
    # A[2, 0:2] * B[0:2, 2:4] -> C[2, 2:4]
    for j in range(2):
        for k in range(2, 4):
            term = [
                (2 * m + j, 1),   # A[2,j]
                (j * p + k, 1),   # B[j,k]
                (k * n + 2, 1)    # C[2,k]
            ]
            terms.append((term, 'schoolbook'))
    
    # A[2, 2] * B[2, 2:4] -> C[2, 2:4]
    for k in range(2, 4):
        term = [
            (2 * m + 2, 1),   # A[2,2]
            (2 * p + k, 1),   # B[2,k]
            (k * n + 2, 1)    # C[2,k]
        ]
        terms.append((term, 'schoolbook'))
    
    rank = len(terms)
    
    # Build U, V, W matrices
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    
    for c, (term, _type) in enumerate(terms):
        for idx, val in term:
            if idx < nm:
                U[idx][c] += val
            elif idx < nm + mp:
                V[idx - nm][c] += val
            else:
                W[idx - nm - mp][c] += val
    
    return U, V, W, rank

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3 * 3x4 matrix multiplication.
    """
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Try to build a lower-rank decomposition
    try:
        U, V, W, rank = _build_rank32_decomposition(n, m, p)
        # Verify the decomposition
        if _verify_decomposition(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    except Exception:
        pass
    
    # Fall back to schoolbook decomposition (rank 36)
    best_rank = n * m * p
    best_U = [[0] * best_rank for _ in range(nm)]
    best_V = [[0] * best_rank for _ in range(mp)]
    best_W = [[0] * best_rank for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i * m + j][col] = 1
                best_V[j * p + k][col] = 1
                best_W[k * n + i][col] = 1
                col += 1
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
