"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 5
    start_time = time.time()
    deadline = start_time + 135  # safety margin before 150s kill

    def _verify(U, V, W, n, m, p):
        """Verify that the tensor decomposition exactly reproduces the multiplication tensor."""
        rank = U.shape[1]
        # Build expected tensor: T[i,j,k] = 1 for all (i,j,k)
        # Actually the multiplication tensor has T[i,j,k] = delta_{i,k} ... no.
        # The matrix multiplication tensor M_{n,m,p} has entries M[i,j,k] = 1 for all i,j,k
        # because (AB)_{ik} = sum_j A_{ij} B_{jk}, so the tensor is the identity in the
        # sense that each (i,j,k) triple contributes A_{ij}*B_{jk} to C_{ik}.
        # The tensor T[i,j,k] = 1 for all (i,j,k) is the "all-ones" tensor which is
        # NOT the multiplication tensor.
        #
        # The multiplication tensor is: for each (i,j,k), the coefficient of
        # A_{ij} * B_{jk} in C_{ik} is 1. So T[i,j,k] = 1 for all valid (i,j,k).
        # Wait, that IS the all-ones tensor in the (i,j,k) indexing.
        # Let me re-examine: C_{ik} = sum_j A_{ij} B_{jk}
        # So the tensor T has T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
        # Yes, it's the all-ones tensor.
        
        T = np.ones((n, m, p))
        result = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    result[i, j, k] = val
        return np.allclose(result, T, atol=1e-9)

    def _is_half_integer(M):
        for val in M.flatten():
            if abs(val * 2 - round(val * 2)) > 1e-9:
                return False
        return True

    def _rank_of(U, V, W):
        return U.shape[1]

    def _build_decomposition(terms, n, m, p):
        """Build U, V, W matrices from a list of (u_vec, v_vec, w_vec) terms."""
        rank = len(terms)
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        for r in range(rank):
            U[:, r] = terms[r][0]
            V[:, r] = terms[r][1]
            W[:, r] = terms[r][2]
        return U, V, W

    def _try_decomp(terms, n, m, p):
        U, V, W = _build_decomposition(terms, n, m, p)
        if _verify(U, V, W, n, m, p) and _is_half_integer(U) and _is_half_integer(V) and _is_half_integer(W):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        return None

    def zeros_nm():
        return [0] * (n * m)

    def zeros_mp():
        return [0] * (m * p)

    def zeros_pn():
        return [0] * (p * n)

    def setA(u, i, j, val):
        u[i * m + j] += val

    def setB(v, j, k, val):
        v[j * p + k] += val

    def setC(w, i, k, val):
        w[k * n + i] += val

    # Known low-rank decompositions for small blocks:
    # We'll define them as functions that return lists of (u, v, w) vectors
    # for the sub-block specified by row indices (i subset), col indices (j subset), 
    # and output indices (k subset).
    
    def schoolbook_block(i_list, j_list, k_list):
        """Schoolbook decomposition for a sub-block: rank = |i_list|*|j_list|*|k_list|"""
        terms = []
        for i in i_list:
            for j in j_list:
                for k in k_list:
                    u = zeros_nm(); v = zeros_mp(); w = zeros_pn()
                    setA(u, i, j, 1)
                    setB(v, j, k, 1)
                    setC(w, i, k, 1)
                    terms.append((u, v, w))
        return terms

    def strassen_2x2x2(i_list, j_list, k_list):
        """
        Strassen decomposition for a 2x2x2 block.
        i_list = [i0, i1], j_list = [j0, j1], k_list = [k0, k1]
        Returns 7 terms.
        """
        i0, i1 = i_list
        j0, j1 = j_list
        k0, k1 = k_list
        
        terms = []
        
        # M1 = (A[i0,j0]+A[i1,j1]) * (B[j0,k0]+B[j1,k1]) -> C[i0,k0] += 1, C[i1,k1] += 1
        u = zeros_nm(); v = zeros_mp(); w = zeros_pn()
        setA(u, i0, j0, 1); setA(u, i1, j1, 1)
        setB(v, j0, k0, 1); setB(v, j1, k1, 1)
        setC(w, i0, k0, 1); setC(w, i1, k1, 1)
        terms.append((u, v, w))
        
        # M2 = (A[i1,j0]+A[i1,j1]) * B[j0,k0] -> C[i1,k0] += 1
        u = zeros_nm(); v = zeros_mp(); w = zeros_pn()
        setA(u, i1, j0, 1); setA(u, i1, j1, 1)
        setB(v, j0, k0, 1)
        setC(w, i1, k0, 1)
        terms.append((u, v, w))
        
        # M3 = A[i0,j0] * (B[j1,k0]-B[j1,k1]) -> C[i0,k1] += 1, C[i1,k1] += 1
        u = zeros_nm(); v = zeros_mp(); w = zeros_pn()
        setA(u, i0, j0, 1)
        setB(v, j1, k0, 1); setB(v, j1, k1, -1)
        setC(w, i0, k1, 1); setC(w, i1, k1, 1)
        terms.append((u, v, w))
        
        # M4 = A[i1,j1] * (B[j0,k1]-B[j0,k0]) -> C[i0,k0] += 1, C[i1,k0] += 1
        u = zeros_nm(); v = zeros_mp(); w = zeros_pn()
        setA(u, i1, j1, 1)
        setB(v, j0, k1, 1); setB(v, j0, k0, -1)
        setC(w, i0, k0, 1); setC(w, i1, k0, 1)
        terms.append((u, v, w))
        
        # M5 = (A[i0,j0]+A[i0,j1]) * B[j1,k1] -> C[i0,k0] -= 1, C[i0,k1] += 1
        u = zeros_nm(); v = zeros_mp(); w = zeros_pn()
        setA(u, i0, j0, 1); setA(u, i0, j1, 1)
        setB(v, j1, k1, 1)
        setC(w, i0, k0, -1); setC(w, i0, k1, 1)
        terms.append((u, v, w))
        
        # M6 = (A[i1,j0]-A[i0,j0]) * (B[j0,k0]+B[j0,k1]) -> C[i1,k1] += 1
        u = zeros_nm(); v = zeros_mp(); w = zeros_pn()
        setA(u, i1, j0, 1); setA(u, i0, j0, -1)
        setB(v, j0, k0, 1); setB(v, j0, k1, 1)
        setC(w, i1, k1, 1)
        terms.append((u, v, w))
        
        # M7 = (A[i0,j1]-A[i1,j1]) * (B[j1,k0]+B[j1,k1]) -> C[i0,k0] += 1
        u = zeros_nm(); v = zeros_mp(); w = zeros_pn()
        setA(u, i0, j1, 1); setA(u, i1, j1, -1)
        setB(v, j1, k0, 1); setB(v, j1, k1, 1)
        setC(w, i0, k0, 1)
        terms.append((u, v, w))
        
        return terms

    def winograd_2x2x3(i_list, j_list, k_list):
        """
        Winograd-style decomposition for 2x2x3 block.
        i_list = [i0, i1], j_list = [j0, j1], k_list = [k0, k1, k2]
        Known rank for 2x2x3 is 10.
        We'll use a known rank-10 decomposition.
        
        For 2x2xn, Winograd gives rank 2n+2 for n>=2 when using the standard method.
        For n=3: rank = 8? But it's known that rank(2x2x3) = 10.
        Let me use a verified rank-10 construction.
        
        Actually, let me use the following approach:
        C[i0,k] = A[i0,j0]*B[j0,k] + A[i0,j1]*B[j1,k]
        C[i1,k] = A[i1,j0]*B[j0,k] + A[i1,j1]*B[j1,k]
        
        For each k, we have a 2x2x1 multiplication.
        We can use: for each k, terms:
          t1 = A[i0,j0]*B[j0,k]
          t2 = A[i0,j1]*B[j1,k]
          t3 = A[i1,j0]*B[j0,k]
          t4 = A[i1,j1]*B[k1]
        That's 4*3 = 12 terms (schoolbook).
        
        Winograd for 2x2xn: 
        Let a=A[i0,j0], b=A[i0,j1], c=A[i1,j0], d=A[i1,j1]
        For each k: e_k=B[j0,k], f_k=B[j1,k]
        C[i0,k] = a*e_k + b*f_k
        C[i1,k] = c*e_k + d*f_k
        
        Winograd: 
        m1 = (a+b)*f_k, m2 = a*(e_k-f_k), m3 = (c+d)*e_k, m4 = c*(f_k-e_k)
        C[i0,k] = m1 + m2
        C[i1,k] = m3 + m4
        That's 4 terms per k, total 4*3 = 12. Not better.
        
        The true Winograd for 2x2xn uses:
        For the A matrix: a, b, c, d, a+b, c+d, a-b, c-d (8 values)
        For the B matrix: for each k, e_k+f_k, e_k-f_k (2 values per k, total 2n)
        Then:
        C[i0,k] = a*(e_k) + b*(f_k) = ((a+b)+(a-b))/2 * ((e_k+f_k)+(e_k-f_k))/2 + ((a+b)-(a-b))/2 * ((e_k+f_k)-(e_k-f_k))/2
        This gets complicated. Let me just use a verified rank-10 decomposition.
        
        Based on literature, rank(2x2x3)=10. Let me construct it:
        
        Terms 1-5: For k=0,1,2:
          u: a+b, v: e_k+f_k, w: contributes to both C[i0,k] and C[i1,k]
        Terms 6-7: a*f_k type corrections
        Terms 8-9: b*e_k type corrections  
        Term 10: additional correction
        
        This is getting complex. Let me just use schoolbook for 2x2x3 (rank 12) 
        and rely on the search to find better combinations.
        """
        return schoolbook_block(i_list, j_list, k_list)

    # Now the main search strategy:
    # We have m=3 columns (j=0,1,2) and p=5 rows (k=0..4) of B.
    # The full problem is 2x3x5.
    # 
    # Strategy: Try all ways to partition the (j,k) pairs into blocks,
    # where each block is either:
    #   - A 2x1xh block (single j, multiple k): rank = 2*h
    #   - A 2x2x2 block: rank = 7 (Strassen)
    #   - A 2x2xh block: use best known
    #   - A 2x3xh block: schoolbook or better
    #
    # Actually, let's think about it differently.
    # The multiplication is: C[i,k] = sum_j A[i,j] * B[j,k]
    # We can group terms by j. For each j, we multiply a 2x1 vector A[:,j] 
    # by a 1x5 row B[j,:] to get a 2x5 matrix contribution.
    # 
    # Alternative: group by k. For each k, we multiply 2x3 by 3x1.
    # 
    # Let's try a different structural approach:
    # Use the fact that rank(2x3x5) <= min(rank(2x3x3)+rank(2x3x2), rank(2x2x5)+rank(2x1x5), ...)
    #
    # Known bounds:
    # rank(2x2x2) = 7
    # rank(2x2x3) = 10
    # rank(2x2x4) = ? (likely 14 or less)
    # rank(2x3x3) = ?
    # rank(2x3x5) = ?
    #
    # Lower bound: rank >= max(n*p, m*p, n*m) = max(10, 15, 6) = 15
    # Also rank >= 2*min(n*m, m*p, n*p) ... no, that's not a standard bound.
    #
    # Let's try: split j into {0,1} and {2}
    #   2x2x5 + 2x1x5
    #   rank(2x2x5) + rank(2x1x5) = rank(2x2x5) + 10
    #   If rank(2x2x5) <= 14, total <= 24
    #   If rank(2x2x5) <= 12, total <= 22
    #
    # Split k into {0,1}, {2,3}, {4}:
    #   2x3x2 + 2x3x2 + 2x3x1
    #   rank(2x3x2) + rank(2x3x2) + rank(2x3x1)
    #   rank(2x3x1) = 6 (each of 3 j's times 2 i's)
    #   rank(2x3x2) = ? At most 12 (schoolbook), maybe less.
    #
    # Let's just try several explicit constructions and pick the best valid one.
    
    best_result = None
    best_rank = float('inf')
    
    # Construction 1: Split j={0,1} and j={2}
    # 2x2x5 block + 2x1x5 block
    # For 2x2x5: split k into {0,1},{2,3},{4}
    #   Strassen(2x2x2) + Strassen(2x2x2) + schoolbook(2x2x1)
    #   = 7 + 7 + 4 = 18
    # 2x1x5: rank 10
    # Total: 28. Worse than current 23.
    
    # Construction 2: Current approach but verify it works
    # 2x2x2 Strassen (j=0,1; k=0,1) + 2x2x2 Strassen (j=0,1; k=2,3) + 2x2x1 schoolbook (j=0,1; k=4) + 2x1x5 schoolbook (j=2)
    # = 7 + 7 + 4 + 10 = 28. Wait, 2x1x5 schoolbook is 2*1*5=10, not 5.
    # The current code has 5 terms for j=2 but each term handles one (i,k) pair... 
    # No, looking at the current code, Part D adds 2 terms per k (one for i=0, one for i=1),
    # so 5 k's * 2 = 10 terms. Total = 7+7+4+10 = 28.
    # But the current code claims rank 23. Let me recount.
    # Actually the current code's Part D:
    #   for k in range(5):
    #       add term for i=0
    #       add term for i=1
    #   That's 10 terms.
    # Total: 7+7+4+10 = 28.
    # But the comment says 23. There's a discrepancy. Let me just build it and check.
    
    # Let me try a cleaner approach: build several candidate decompositions
    # and return the best valid one.
    
    candidates = []
    
    # Candidate A: Split j={0,1}, j={2}
    #   j={0,1}: split k={0,1},{2,3},{4}
    #     Strassen(2x2x2) x2 + schoolbook(2x2x1) = 7+7+4 = 18
    #   j={2}: schoolbook(2x1x5) = 10
    # Total: 28
    
    terms_A = []
    terms_A += strassen_2x2x2([0,1], [0,1], [0,1])
    terms_A += strassen_2x2x2([0,1], [0,1], [2,3])
    terms_A += schoolbook_block([0,1], [0,1], [4])
    terms_A += schoolbook_block([0,1], [2], list(range(5)))
    candidates.append(terms_A)
    
    # Candidate B: Split k={0,1},{2,3},{4}, each with full j={0,1,2}
    #   2x3x2 + 2x3x2 + 2x3x1
    #   For 2x3x2: we can split j={0,1},{2}
    #     2x2x2 (Strassen, rank 7) + 2x1x2 (schoolbook, rank 4) = 11
    #   For 2x3x1: schoolbook rank 6
    # Total: 11 + 11 + 6 = 28
    
    terms_B = []
    # k={0,1}:
    terms_B += strassen_2x2x2([0,1], [0,1], [0,1])
    terms_B += schoolbook_block([0,1], [2], [0,1])
    # k={2,3}:
    terms_B += strassen_2x2x2([0,1], [0,1], [2,3])
    terms_B += schoolbook_block([0,1], [2], [2,3])
    # k={4}:
    terms_B += schoolbook_block([0,1], [0,1,2], [4])
    candidates.append(terms_B)
    
    # Candidate C: Try to use Winograd-like for 2x2x3
    # Split k={0,1,2},{3,4}
    #   2x3x3 + 2x3x2
    #   For 2x3x3: split j={0,1},{2}
    #     2x2x3 (rank 10 if we can find it, else 12) + 2x1x3 (rank 6) = 16 or 18
    #   For 2x3x2: split j={0,1},{2}
    #     2x2x2 (rank 7) + 2x1x2 (rank 4) = 11
    # Total: 16+11 = 27 or 18+11 = 29
    
    # Let's try with schoolbook for 2x2x3 (rank 12):
    terms_C = []
    terms_C += schoolbook_block([0,1], [0,1], [0,1,2])  # 2x2x3, rank 12
    terms_C += schoolbook_block([0,1], [2], [0,1,2])     # 2x1x3, rank 6
    terms_C += strassen_2x2x2([0,1], [0,1], [3,4])       # 2x2x2, rank 7
    terms_C += schoolbook_block([0,1], [2], [3,4])       # 2x1x2, rank 4
    candidates.append(terms_C)
    
    # Candidate D: The original approach from the seed program
    terms_D = []
    terms_D += strassen_2x2x2([0,1], [0,1], [0,1])
    terms_D += strassen_2x2x2([0,1], [0,1], [2,3])
    terms_D += schoolbook_block([0,1], [0,1], [4])
    terms_D += schoolbook_block([0,1], [2], list(range(5)))
    candidates.append(terms_D)
    
    # Candidate E: Try a different split
    # Split j={0},{1},{2} and k into groups
    # Each j is 2x1x5, rank 10 each, total 30. Too high.
    
    # Candidate F: Use the fact that we can share terms across j
    # For a given k, C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k]
    # This is a 2x3x1 multiplication, rank 6.
    # For 5 k's: 5*6 = 30. Schoolbook.
    
    # Let's try to be smarter. The key insight is that Strassen saves 1 multiplication
    # compared to schoolbook for 2x2x2 (7 vs 8).
    # 
    # Best known: rank(2x3x5) is not known exactly, but upper bounds exist.
    # Let's just try all candidates and pick the best valid one.
    
    for idx, terms in enumerate(candidates):
        if time.time() > deadline:
            break
        result = _try_decomp(terms, n, m, p)
        if result is not None:
            r = len(terms)
            if r < best_rank:
                best_rank = r
                best_result = result
    
    # If no candidate worked, fall back to schoolbook
    if best_result is None:
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
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return best_result
# EVOLVE-BLOCK-END
