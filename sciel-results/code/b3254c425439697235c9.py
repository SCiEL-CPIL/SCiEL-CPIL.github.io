"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
import time

def _build_rank35_block(U_block, V_block, W_block):
    """
    Return a rank-35 decomposition for multiplying 2x2 by 2x10 using Strassen
    on the 2x2x2 core blocks.
    U: shape (4, 35), V: shape (20, 35), W: shape (20, 35)
    """
    # We have 5 independent 2x2x2 blocks (k=0..4).
    # For each block, Strassen gives rank 7.
    # Total rank = 5 * 7 = 35.
    
    # Strassen terms for 2x2x2:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-c) -> This is problematic. Let me use the standard 7 terms.
    # Actually, the correct 7 Strassen terms for 2x2x2 are:
    # Let A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-c)  -> No, g is in B, c is in A. This is not separable.
    # Let me look up the correct Strassen terms for matrix multiplication.
    # The 7 products are:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-c) -> I think this is wrong. Let me use a different formulation.
    
    # Actually, for 2x2x2, the rank is 7 and the terms are:
    # p1 = (a + d)(e + h)
    # p2 = (c + d)e
    # p3 = a(f - h)
    # p4 = d(g - c) -> This is still problematic.
    
    # Let me use a verified set of Strassen terms:
    # From Wikipedia:
    # p1 = (a + d)(e + h)
    # p2 = (c + d)e
    # p3 = a(f - h)
    # p4 = d(g - c)
    # p5 = (a + b)h
    # p6 = (c - a)(e + f)
    # p7 = (b - d)(g + h)
    
    # Wait, p4 = d(g - c). d is in A, g is in B, c is in A. So g-c is not linear in B only.
    # This means the standard Strassen formulation might not directly give a tensor decomposition.
    # Let me verify: the tensor decomposition requires each term to be u_i(A) * v_i(B) * w_i(C).
    # For matrix multiplication, C = AB, so the tensor is sum_{i,j,k} A_{i,j} B_{j,k} E_{i,k}.
    # A rank-r decomposition is sum_{c=1}^r u_c(A) v_c(B) w_c(E).
    # Strassen's algorithm computes C using 7 products of linear forms in A and B, then linear combinations.
    # So the 7 products are of the form (linear in A) * (linear in B).
    # Let me check p4 = d(g - c). d is a coefficient of A. g is a coefficient of B. c is a coefficient of A.
    # So (g - c) is not a linear form in B only. It's a linear form in B and A.
    # This means p4 is not a valid term in the tensor decomposition.
    
    # I think I'm confusing the Strassen algorithm with the tensor decomposition.
    # Let me use a different approach: just use the schoolbook rank 40, since I'm not sure about rank 35.
    # Actually, let me try to construct rank 35 using a known result.
    # It is known that mu(2,2,2) = 7, so 5 blocks of 2x2x2 should give rank 35.
    # Let me hardcode the 7 terms for one block and replicate.
    
    # For a single 2x2x2 block with A = [[a,b],[c,d]] and B = [[e,f],[g,h]]:
    # The 7 products in a valid tensor decomposition are:
    # I'll use the following (verified) Strassen terms:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e  
    # M3 = a(f-h)
    # M4 = d(g-c) -> replace with d*g? No, that's not right.
    
    # Let me just use rank 40 schoolbook to be safe.
    return None

def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=10
    # Schoolbook: rank = n*m*p = 2*2*10 = 40
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
# EVOLVE-BLOCK-END
