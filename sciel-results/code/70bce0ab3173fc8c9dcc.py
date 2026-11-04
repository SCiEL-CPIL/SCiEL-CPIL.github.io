import numpy as np
import time

DIMENSIONS = [2, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 120  # 120s limit
    
    n, m, p = DIMENSIONS
    
    # We'll build the decomposition recursively
    # For 2x4x8, we split p into 4+4, giving two 2x4x4 subproblems
    # For 2x4x4, we split m into 2+2, giving two 2x2x4 subproblems
    # For 2x2x4, we split p into 2+2, giving two 2x2x2 subproblems
    # For 2x2x2, we use Strassen (rank 7)
    
    # Each subproblem contributes rank-1 terms (u, v, w)
    # We'll collect all terms and then assemble U, V, W
    
    terms = []  # list of (u_indices, v_indices, w_indices) where each is a dict or list
    
    # Actually, let me think about this differently.
    # Each rank-1 term is (a, b, c) where a is a vector in C^{n*m}, b in C^{m*p}, c in C^{p*n}
    # The term contributes a[i]*b[j]*c[k] to T[i,j,k]
    
    # For Strassen 2x2x2, the 7 terms are:
    # T1: A[0,0]+A[1,1] times B[0,0]+B[1,1]
    # T2: A[1,0]+A[1,1] times B[0,0]
    # T3: A[0,0] times B[1,0]-B[1,1]
    # T4: A[1,1] times B[0,1]-B[1,0]
    # T5: A[0,0]+A[0,1] times B[1,1]
    # T6: A[0,1]-A[0,0] times B[0,0]+B[0,1]
    # T7: A[1,1]-A[1,0] times B[0,1]+B[1,1]
    
    # And the output is:
    # C[0,0] = T1 - T3 + T5 - T6
    # C[0,1] = T3 + T4
    # C[1,0] = T2 + T4
    # C[1,1] = T1 - T2 + T5 + T7
    
    # Let me define the Strassen terms more carefully.
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-f)
    # m5 = (a+b)h
    # m6 = (b-a)(e+f)
    # m7 = (d-c)(f+h)
    # C[0,0] = m1 - m3 + m5 - m6
    # C[0,1] = m3 + m4
    # C[1,0] = m2 + m4
    # C[1,1] = m1 - m2 + m5 + m7
    
    # For each Strassen term, the "A-side" is a linear combination of A entries,
    # the "B-side" is a linear combination of B entries,
    # and the "C-side" coefficients tell us how to combine into C entries.
    
    # Let me represent each term as:
    # (a_vec, b_vec, c_coeffs)
    # where a_vec is a dict mapping A-index -> coefficient
    # b_vec is a dict mapping B-index -> coefficient
    # c_coeffs is a dict mapping C-index -> coefficient
    
    # For the 2x2x2 case with A indices (0,0),(0,1),(1,0),(1,1) and B indices (0,0),(0,1),(1,0),(1,1):
    
    strassen_222 = [
        # (a_coeffs, b_coeffs, c_coeffs)
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):-1}),  # m1
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1}),  # m2
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,0):-1, (0,1):1}),  # m3
        ({(1,1):1}, {(1,0):1, (0,1):-1}, {(0,1):1, (1,0):1}),  # m4
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):1, (1,1):1}),  # m5
        ({(0,1):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(0,0):-1}),  # m6
        ({(1,1):1, (1,0):-1}, {(0,1):1, (1,1):1}, {(1,1):1}),  # m7
    ]
    
    # Now I need to generalize this to the 2x4x8 case using recursion.
    # This is getting complex. Let me just hardcode the rank-56 decomposition.
    
    # Actually, let me take a step back. The recursive approach:
    # 2x4x8 -> two 2x4x4 (split p: j=0..3 and j=4..7)
    # 2x4x4 -> two 2x2x4 (split m: k=0,1 and k=2,3)
    # 2x2x4 -> two 2x2x2 (split p: j=0,1 and j=2,3)
    # 2x2x2 -> Strassen rank 7
    
    # Total: 2 * 2 * 2 * 7 = 56 terms
    
    # Let me build this recursively.
    
    def strassen_222_terms():
        """Return the 7 Strassen terms for 2x2x2.
        Each term is (a_dict, b_dict, c_dict) where:
        a_dict maps (i,k) -> coeff for A
        b_dict maps (k,j) -> coeff for B
        c_dict maps (i,j) -> coeff for C
        """
        return [
            ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):-1}),
            ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1}),
            ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,0):-1, (0,1):1}),
            ({(1,1):1}, {(1,0):1, (0,1):-1}, {(0,1):1, (1,0):1}),
            ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):1, (1,1):1}),
            ({(0,1):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(0,0):-1}),
            ({(1,1):1, (1,0):-1}, {(0,1):1, (1,1):1}, {(1,1):1}),
        ]
    
    def decompose(n, m, p, i_off=0, k_off=0, j_off=0):
        """Recursively decompose n x m x m x p multiplication.
        Returns list of terms, each (a_dict, b_dict, c_dict) with global indices.
        """
        if n == 2 and m == 2 and p == 2:
            base = strassen_222_terms()
            result = []
            for a, b, c in base:
                ga = {(i_off+i, k_off+k): v for (i,k), v in a.items()}
                gb = {(k_off+k, j_off+j): v for (k,j), v in b.items()}
                gc = {(i_off+i, j_off+j): v for (i,j), v in c.items()}
                result.append((ga, gb, gc))
            return result
        
        if p >= 4 and n == 2 and m == 4:
            # Split p into two halves
            p1 = p // 2
            terms = []
            terms.extend(decompose(n, m, p1, i_off, k_off, j_off))
            terms.extend(decompose(n, m, p1, i_off, k_off, j_off + p1))
            return terms
        
        if m >= 4 and n == 2 and p == 4:
            # Split m into two halves
            m1 = m // 2
            terms = []
            terms.extend(decompose(n, m1, p, i_off, k_off, j_off))
            terms.extend(decompose(n, m1, p, i_off, k_off + m1, j_off))
            return terms
        
        if p >= 4 and n == 2 and m == 2:
            # Split p into two halves
            p1 = p // 2
            terms = []
            terms.extend(decompose(n, m, p1, i_off, k_off, j_off))
            terms.extend(decompose(n, m, p1, i_off, k_off, j_off + p1))
            return terms
        
        raise ValueError(f"Unhandled case: {n}x{m}x{p}")
    
    all_terms = decompose(n, m, p)
    rank = len(all_terms)
    
    # Now build U, V, W matrices
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    for t, (a_dict, b_dict, c_dict) in enumerate(all_terms):
        for (i, k), coeff in a_dict.items():
            U[i*m + k][t] = coeff
        for (k, j), coeff in b_dict.items():
            V[k*p + j][t] = coeff
        for (i, j), coeff in c_dict.items():
            # W convention: W[j*n + i]
            W[j*n + i][t] = coeff
    
    return {"U": U, "V": V, "W": W}
