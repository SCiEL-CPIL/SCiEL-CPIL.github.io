import time
import numpy as np

DIMENSIONS = [2, 2, 12]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=12
    # U: (n*m, R) = (4, R)
    # V: (m*p, R) = (24, R)
    # W: (p*n, R) = (24, R)
    
    # We use 6 blocks of 2x2 matrices from B.
    # Each block uses Strassen's rank-7 algorithm.
    # Total rank R = 6 * 7 = 42.
    
    R = 42
    U = [[0.0] * R for _ in range(n * m)]
    V = [[0.0] * R for _ in range(m * p)]
    W = [[0.0] * R for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 multiplication
    # A = [[a00, a01], [a10, a11]]
    # B = [[b00, b01], [b10, b11]]
    # C = [[c00, c01], [c10, c11]]
    
    # M1 = (a00 - a11)(b01 + b11)
    # M2 = (a10 + a11)b00
    # M3 = (a00 + a01)b11
    # M4 = (a11 - a00)(b00 + b01)
    # M5 = (a00 + a10)b10
    # M6 = (a01 - a11)(b00 + b10)
    # M7 = (a00 - a10)(b01 - b11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 - M4 - M7
    # C10 = M2 - M1 + M5
    # C11 = M3 - M2 + M6
    
    # Indices for A (U):
    # a00: 0*2+0 = 0
    # a01: 0*2+1 = 1
    # a10: 1*2+0 = 2
    # a11: 1*2+1 = 3
    
    # Indices for B block j (V):
    # Block j corresponds to columns 2j, 2j+1 of B.
    # B row i, col 2j+k has V index i*p + (2j+k).
    # b00: 0*12 + 2j = 2j
    # b01: 0*12 + 2j+1 = 2j+1
    # b10: 1*12 + 2j = 12+2j
    # b11: 1*12 + 2j+1 = 13+2j
    
    # Indices for C block j (W):
    # C row 2j+k, col i has W index (2j+k)*n + i = (2j+k)*2 + i.
    # c00: (2j+0)*2 + 0 = 4j
    # c01: (2j+0)*2 + 1 = 4j+1
    # c10: (2j+1)*2 + 0 = 4j+2
    # c11: (2j+1)*2 + 1 = 4j+3
    
    # Define the 7 terms for one block
    # Each term is (u_coeffs, v_coeffs, w_coeffs)
    # u_coeffs: [u_a00, u_a01, u_a10, u_a11]
    # v_coeffs: [v_b00, v_b01, v_b10, v_b11]
    # w_coeffs: [w_c00, w_c01, w_c10, w_c11]
    
    terms = [
        # M1
        ([-1, 0, 0, 1], [0, 1, 0, 1], [1, 0, 0, 0]), # C00 += M1
        # Wait, the w_coeffs should represent the contribution to C.
        # The term is u * v * w.
        # So if M1 contributes +1 to C00, then w_c00 should be 1.
        # But M1 is a single product.
        # So for each M_k, we have ONE term in the sum?
        # No, M_k is a product of a linear form in A and a linear form in B.
        # So u and v are fixed for M_k.
        # But M_k contributes to multiple C entries with different coefficients.
        # So we need to split M_k into multiple terms if it contributes to multiple C entries?
        # No, the tensor decomposition is T = sum u_r v_r w_r.
        # If M_k = (u . A) (v . B), and C = sum_k gamma_k M_k,
        # then C_{ij} = sum_k gamma_{ij}^k (u_k . A) (v_k . B).
        # This can be written as sum_k (u_k . A) (v_k . B) (sum_{ij} gamma_{ij}^k C_{ij}^*)?
        # No, w_r is a linear form in C.
        # So w_r(C) = sum_{ij} w_{ij}^r C_{ij}.
        # We want sum_r u_r(A) v_r(B) w_r(C) = T(A,B,C).
        # T(A,B,C) = sum_{i,j,k} A_{ij} B_{jk} C_{ki} (dual pairing).
        # Actually, the tensor is T_{i,j,k} = sum_r U_{ij}^r V_{jk}^r W_{ki}^r.
        # Here W is indexed by k,i.
        # So for a given term r, U_r, V_r, W_r are vectors.
        # If we use Strassen, we have 7 products M_k.
        # Each M_k is (u_k . A) (v_k . B).
        # This M_k appears in the expression for C with coefficients.
        # C = sum_k M_k * Gamma_k, where Gamma_k is a 2x2 matrix of coefficients.
        # So the term in the tensor sum corresponding to M_k is:
        # (u_k . A) (v_k . B) (Gamma_k . C^*)?
        # No, the decomposition is additive over the rank terms.
        # If M_k contributes to C_{00} with coeff 1 and C_{11} with coeff 1,
        # we can write this as two terms:
        # Term 1: u_k, v_k, w_{k,00} (where w_{k,00} has 1 at C_{00} pos)
        # Term 2: u_k, v_k, w_{k,11} (where w_{k,11} has 1 at C_{11} pos)
        # This would increase the rank!
        # Rank 7 for 2x2x2 means we can do it with 7 terms.
        # How?
        # The tensor T_{2x2x2} has rank 7.
        # This means T = sum_{r=1}^7 u_r \otimes v_r \otimes w_r.
        # In this sum, each w_r is a vector in the C-space.
        # For Strassen, the 7 terms correspond to the 7 multiplications.
        # But each multiplication M_r contributes to multiple C entries.
        # Does this mean w_r is a linear combination of C entries?
        # Yes! w_r is the linear form in C that extracts the part of C "explained" by M_r.
        # But C is the output. The tensor is the multiplication map.
        # The identity is:
        # sum_{i,j,k} A_{ij} B_{jk} C_{ki} = sum_{r=1}^7 (u_r . A) (v_r . B) (w_r . C)
        # This is an identity in the variables A, B, C.
        # So for each r, w_r is a fixed vector (linear form on C-space).
        # And indeed, for Strassen, w_r is determined by the coefficients of M_r in the C formulas.
        # For example, if M_1 = C_{00} + C_{11} - ..., then w_1 is the vector with 1s at C_{00}, C_{11} etc.
        # So we do NOT split into multiple terms. We use ONE term per M_r, with w_r being the full linear form.
        
        # Let's re-verify this.
        # If T = sum u_r v_r w_r, then for any A, B, C:
        # <T, A \otimes B \otimes C> = sum_r <u_r, A> <v_r, B> <w_r, C>
        # The LHS is the standard matrix multiplication contraction.
        # The RHS is the Strassen decomposition.
        # So yes, w_r is the linear form corresponding to the coefficients of M_r in C.
        
        # So for each block j and each k=1..7, we have ONE term.
        # Total terms = 6 * 7 = 42.
    ]

    # Let's define the u, v, w vectors for the 7 Strassen terms.
    # u: [a00, a01, a10, a11]
    # v: [b00, b01, b10, b11]
    # w: [c00, c01, c10, c11]
    
    strassen_terms = [
        # k=1: M1 = (a00 - a11)(b01 + b11)
        # Contributes to C00 (+1), C10 (-1)
        # Check: C00 = M1 + M4 - M5 + M7. C10 = M2 - M1 + M5.
        # So w1 has +1 at c00, -1 at c10.
        # w1 = [1, 0, -1, 0]
        ([1, 0, 0, -1], [0, 1, 0, 1], [1, 0, -1, 0]),
        
        # k=2: M2 = (a10 + a11)b00
        # Contributes to C10 (+1), C11 (-1)
        # C10 = M2 - M1 + M5. C11 = M3 - M2 + M6.
        # w2 = [0, 0, 1, -1]
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        
        # k=3: M3 = (a00 + a01)b11
        # Contributes to C01 (+1), C11 (+1)
        # C01 = M3 - M4 - M7. C11 = M3 - M2 + M6.
        # w3 = [0, 1, 0, 1]
        ([1, 1, 0, 0], [0, 0, 0, 1], [0, 1, 0, 1]),
        
        # k=4: M4 = (a11 - a00)(b00 + b01)
        # Contributes to C00 (+1), C01 (-1)
        # C00 = M1 + M4 - M5 + M7. C01 = M3 - M4 - M7.
        # w4 = [1, -1, 0, 0]
        ([-1, 0, 0, 1], [1, 1, 0, 0], [1, -1, 0, 0]),
        
        # k=5: M5 = (a00 + a10)b10
        # Contributes to C00 (-1), C10 (+1)
        # C00 = M1 + M4 - M5 + M7. C10 = M2 - M1 + M5.
        # w5 = [-1, 0, 1, 0]
        ([1, 0, 1, 0], [0, 0, 1, 0], [-1, 0, 1, 0]),
        
        # k=6: M6 = (a01 - a11)(b00 + b10)
        # Contributes to C11 (+1)
        # C11 = M3 - M2 + M6.
        # w6 = [0, 0, 0, 1]
        ([0, 1, 0, -1], [1, 0, 1, 0], [0, 0, 0, 1]),
        
        # k=7: M7 = (a00 - a10)(b01 - b11)
        # Contributes to C00 (+1), C01 (-1)
        # C00 = M1 + M4 - M5 + M7. C01 = M3 - M4 - M7.
        # w7 = [1, -1, 0, 0]
        ([1, 0, -1, 0], [0, 1, 0, -1], [1, -1, 0, 0]),
    ]

    # Fill U, V, W
    for j in range(6):
        for k in range(7):
            r = j * 7 + k
            u_coeffs, v_coeffs, w_coeffs = strassen_terms[k]
            
            # U: A indices
            # a00: 0, a01: 1, a10: 2, a11: 3
            U[0][r] = u_coeffs[0]
            U[1][r] = u_coeffs[1]
            U[2][r] = u_coeffs[2]
            U[3][r] = u_coeffs[3]
            
            # V: B indices for block j
            # b00: 2j, b01: 2j+1, b10: 12+2j, b11: 13+2j
            V[2j][r] = v_coeffs[0]
            V[2j+1][r] = v_coeffs[1]
            V[12+2j][r] = v_coeffs[2]
            V[13+2j][r] = v_coeffs[3]
            
            # W: C indices for block j
            # c00: 4j, c01: 4j+1, c10: 4j+2, c11: 4j+3
            W[4j][r] = w_coeffs[0]
            W[4j+1][r] = w_coeffs[1]
            W[4j+2][r] = w_coeffs[2]
            W[4j+3][r] = w_coeffs[3]

    return {"U": U, "V": V, "W": W}
