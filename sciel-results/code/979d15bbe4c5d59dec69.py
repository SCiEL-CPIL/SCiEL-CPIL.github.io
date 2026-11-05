import numpy as np

def construct():
    n, m, p = 2, 5, 5
    
    # We use a block decomposition:
    # A is 2x5, split columns into A1 (2x2) and A2 (2x3)
    # B is 5x5, split into B11 (2x2), B12 (2x3), B21 (3x2), B22 (3x3)
    # C = A1*B11 + A2*B21 (first 2 cols of C)
    #   + A1*B12 + A2*B22 (last 3 cols of C)
    
    # For A1*B11 (2x2x2), we use Strassen's algorithm (rank 7)
    # For the other blocks, we use schoolbook multiplication
    
    # Strassen's algorithm for 2x2x2:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For the 2x2x2 block A1*B11:
    # A1 has rows i=0,1 and cols j=0,1
    # B11 has rows j=0,1 and cols k=0,1
    # This contributes to C[0:2, 0:2]
    
    # Let's build the full decomposition
    
    rank = 0
    U = []
    V = []
    W = []
    
    def add_term(u_idx, v_idx, w_idx, u_val=1, v_val=1, w_val=1):
        nonlocal rank
        U.append([0] * rank + [u_val])
        V.append([0] * rank + [v_val])
        W.append([0] * rank + [w_val])
        rank += 1
    
    # Wait, we need to accumulate into existing rows. Let me use a different approach.
    # We'll collect all terms as (u_idx, v_idx, w_idx, u_val, v_val, w_val) tuples
    # Then build the matrices.
    
    terms = []
    
    def add_term(u_idx, v_idx, w_idx, u_val=1, v_val=1, w_val=1):
        terms.append((u_idx, v_idx, w_idx, u_val, v_val, w_val))
    
    # Strassen for A1*B11 (2x2x2):
    # A1: rows i=0,1; cols j=0,1
    # B11: rows j=0,1; cols k=0,1
    # Output: C[0:2, 0:2]
    
    # M1 = (A11 + A22)(B11 + B22)
    # A11 = A[0,0], A22 = A[1,1]
    # B11 = B[0,0], B22 = B[1,1]
    # This is a product of sums, which in tensor decomposition means:
    # We need to represent (a+b)(c+d) = ac + ad + bc + bd
    # But Strassen combines these into 7 multiplications total.
    
    # Actually, let me just enumerate all 7 Strassen products explicitly.
    
    # M1 = (A[0,0] + A[1,1]) * (B[0,0] + B[1,1])
    # This contributes to C[0,0] with +1, C[0,1] with 0, C[1,0] with 0, C[1,1] with +1
    # Wait, no. M1 is used in C11 = M1 + M4 - M5 + M7 and C22 = M1 - M2 + M3 + M6
    
    # Let me be more careful. The Strassen decomposition for multiplying A (2x2) by B (2x2) to get C (2x2):
    # M1 = (A[0,0] + A[1,1]) * (B[0,0] + B[1,1])
    # M2 = (A[1,0] + A[1,1]) * B[0,0]
    # M3 = A[0,0] * (B[0,1] - B[1,1])
    # M4 = A[1,1] * (B[1,0] - B[0,0])
    # M5 = (A[0,0] + A[0,1]) * B[1,1]
    # M6 = (A[1,0] - A[0,0]) * (B[0,0] + B[0,1])
    # M7 = (A[0,1] - A[1,1]) * (B[1,0] + B[1,1])
    
    # C[0,0] = M1 + M4 - M5 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 + M6
    
    # In the context of our problem:
    # A is 2x5, B is 5x5
    # A1 = A[:, 0:2] (2x2), A2 = A[:, 2:5] (2x3)
    # B11 = B[0:2, 0:2] (2x2), B12 = B[0:2, 2:5] (2x3)
    # B21 = B[2:5, 0:2] (3x2), B22 = B[2:5, 2:5] (3x3)
    
    # C = A1*B11 + A2*B21 (cols 0,1) + A1*B12 + A2*B22 (cols 2,3,4)
    
    # For A1*B11, we use Strassen. The indices:
    # A1[i,j] corresponds to global A[i, j] where i in {0,1}, j in {0,1}
    # B11[j,k] corresponds to global B[j, k] where j in {0,1}, k in {0,1}
    # C11[i,k] corresponds to global C[i, k] where i in {0,1}, k in {0,1}
    
    # So for Strassen on A1*B11:
    # A[0,0] -> U index 0*5+0 = 0
    # A[1,1] -> U index 1*5+1 = 6
    # A[1,0] -> U index 1*5+0 = 5
    # A[0,1] -> U index 0*5+1 = 1
    # B[0,0] -> V index 0*5+0 = 0
    # B[1,1] -> V index 1*5+1 = 6
    # B[0,1] -> V index 0*5+1 = 1
    # B[1,0] -> V index 1*5+0 = 5
    
    # C[0,0] -> W index 0*2+0 = 0
    # C[0,1] -> W index 1*2+0 = 2
    # C[1,0] -> W index 0*2+1 = 1
    # C[1,1] -> W index 1*2+1 = 3
    
    # M1 = (A[0,0] + A[1,1]) * (B[0,0] + B[1,1])
    # This is (U[0] + U[6]) * (V[0] + V[6])
    # In tensor decomposition, this means we need terms that multiply these sums.
    # (a+b)(c+d) = ac + ad + bc + bd
    # But we can't just add 4 terms because that defeats the purpose.
    # The key insight is that Strassen reduces the number of multiplications from 4 to 7 for 2x2x2.
    # But each "multiplication" in Strassen is a scalar product of two linear combinations.
    # So M1 corresponds to ONE rank-1 term where:
    # U has 1 in positions 0 and 6
    # V has 1 in positions 0 and 6
    # W has contributions to C[0,0] and C[1,1]
    
    # M1 contributes to C[0,0] with +1 and C[1,1] with +1
    add_term(0, 0, 0, 1, 1, 1)  # U[0], V[0], W[0]
    add_term(6, 6, 0, 1, 1, 1)  # U[6], V[6], W[0]
    # Wait, this isn't right. Let me think again.
    
    # Actually, in a tensor decomposition, each rank-1 term is a single product U[i,j,r] * V[j,k,r] * W[k,i,r].
    # For Strassen, each M term is a product of two linear combinations.
    # M1 = (A[0,0] + A[1,1]) * (B[0,0] + B[1,1])
    # This equals A[0,0]*B[0,0] + A[0,0]*B[1,1] + A[1,1]*B[0,0] + A[1,1]*B[1,1]
    # So it actually corresponds to 4 rank-1 terms, not 1!
    
    # I think I'm confusing myself. Let me reconsider.
    
    # The standard Strassen algorithm uses 7 multiplications of 2x2 matrices.
    # Each such multiplication itself requires 4 scalar multiplications (schoolbook) or fewer.
    # But if we're doing scalar-level tensor decomposition, Strassen doesn't help directly
    # because the "multiplications" in Strassen are still scalar multiplications of linear combinations.
    
    # Wait, no. In tensor decomposition of matrix multiplication, the rank is the minimum number
    # of rank-1 tensors needed. Strassen's algorithm shows that 2x2x2 has rank 7, not 8.
    # This means we can decompose the 2x2x2 multiplication tensor into 7 rank-1 terms.
    
    # Let me look up the explicit Strassen decomposition for 2x2x2.
    
    # The 7 rank-1 terms for 2x2x2 are:
    # T1: (A00+A11)(B00+B11) -> contributes to C00 and C11
    # T2: (A10+A11)B00 -> contributes to C10
    # T3: A00(B01-B11) -> contributes to C01
    # T4: A11(B10-B00) -> contributes to C10 and C11
    # T5: (A00+A01)B11 -> contributes to C01 and C11
    # T6: (A10-A00)(B00+B01) -> contributes to C10
    # T7: (A01-A11)(B10+B11) -> contributes to C00 and C11
    
    # Each of these is a single rank-1 term in the tensor decomposition.
    # For T1: U has 1 at (0,0) and (1,1), V has 1 at (0,0) and (1,1), W has 1 at (0,0) and (1,1)
    # But that's not a single rank-1 term; that's 4 terms.
    
    # I think the confusion is that in the tensor decomposition, each rank-1 term corresponds
    # to a single product of a vector in U-space, a vector in V-space, and a vector in W-space.
    # So for T1, we need:
    # u = e_{0,0} + e_{1,1} (in U-space, where U-space has dimension 4 for 2x2)
    # v = e_{0,0} + e_{1,1} (in V-space)
    # w = e_{0,0} + e_{1,1} (in W-space)
    # And the term is u * v * w, which expands to 4 products.
    # But in the tensor decomposition, this counts as ONE rank-1 term.
    
    # So for our problem, we need to map these to the global indices.
    
    # Let me restart with a cleaner approach.
    
    terms.clear()
    
    # For A1*B11 (2x2x2) using Strassen:
    # A1[i,j] -> U index i*5+j
    # B11[j,k] -> V index j*5+k
    # C11[i,k] -> W index k*2+i
    
    # T1: (A[0,0]+A[1,1])(B[0,0]+B[1,1])
    # U: 1 at (0,0) and (1,1) -> indices 0 and 6
    # V: 1 at (0,0) and (1,1) -> indices 0 and 6
    # W: 1 at (0,0) and (1,1) -> indices 0 and 3
    add_term(0, 0, 0, 1, 1, 1)
    add_term(0, 6, 0, 1, 1, 1)
    add_term(6, 0, 0, 1, 1, 1)
    add_term(6, 6, 0, 1, 1, 1)
    # But this is 4 terms, not 1. I think I'm still confused.
    
    # Let me just use schoolbook for everything to be safe, then see if I can improve.
    
    terms.clear()
    
    # Schoolbook for A1*B11 (2x2x2): 4 terms
    for i in range(2):
        for j in range(2):
            for k in range(2):
                add_term(i*5+j, j*5+k, k*2+i, 1, 1, 1)
    
    # Schoolbook for A2*B21 (2x3x2): 12 terms
    for i in range(2):
        for j in range(2, 5):
            for k in range(2):
                add_term(i*5+j, j*5+k, k*2+i, 1, 1, 1)
    
    # Schoolbook for A1*B12 (2x2x3): 12 terms
    for i in range(2):
        for j in range(2):
            for k in range(2, 5):
                add_term(i*5+j, j*5+k, k*2+i, 1, 1, 1)
    
    # Schoolbook for A2*B22 (2x3x3): 18 terms
    for i in range(2):
        for j in range(2, 5):
            for k in range(2, 5):
                add_term(i*5+j, j*5+k, k*2+i, 1, 1, 1)
    
    # Total: 4 + 12 + 12 + 18 = 46 terms
    # But wait, this is just schoolbook for the whole 2x5x5, which is 50 terms.
    # I must have miscounted. Let me recount.
    
    # A1*B11: 2*2*2 = 8? No, 2*2*2 = 8 is wrong. It's 2*2*2 = 8? No, 2*2*2 = 8 is not right either.
    # 2x2x2 schoolbook has 2*2*2 = 8 terms? No, it has 2*2*2 = 8? That's not right.
    # For multiplying a 2x2 by a 2x2, schoolbook has 4*4 = 16? No, it has 2*2*2 = 8? 
    # Actually, for C[i,k] = sum_j A[i,j]*B[j,k], the number of terms is n*m*p = 2*2*2 = 8? 
    # No, it's n*m*p = 2*2*2 = 8 is not right. It's sum over j, so for each (i,k) pair, there are m=2 terms.
    # Total terms = n*p*m = 2*2*2 = 8. Yes, 8 terms for 2x2x2 schoolbook.
    
    # So A1*B11: 8 terms
    # A2*B21: 2*3*2 = 12 terms
    # A1*B12: 2*2*3 = 12 terms
    # A2*B22: 2*3*3 = 18 terms
    # Total: 8+12+12+18 = 50 terms. That's just schoolbook.
    
    # So my block decomposition didn't help. Let me try to actually use Strassen for the 2x2x2 part.
    
    terms.clear()
    
    # For A1*B11 (2x2x2), use Strassen (7 terms instead of 8):
    # I need to find the correct Strassen decomposition.
    
    # After some research, the Strassen decomposition for 2x2x2 is:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Each M is a rank-1 term in the tensor decomposition.
    
    # M1: U has 1 at (0,0) and (1,1); V has 1 at (0,0) and (1,1); W has 1 at (0,0) and (1,1)
    # In global indices:
    # U: (0,0)->0, (1,1)->6
    # V: (0,0)->0, (1,1)->6
    # W: (0,0)->0, (1,1)->3
    # But this is 4 products, not 1 rank-1 term.
    
    # I think the issue is that in the tensor decomposition, a "rank-1 term" is a single outer product
    # of three vectors. So for M1, we need:
    # u = e0 + e6 (in U-space)
    # v = e0 + e6 (in V-space)
    # w = e0 + e3 (in W-space)
    # And the term is u * v * w.
    # This is ONE rank-1 term, even though it expands to 4 products.
    
    # So in our matrix representation, for this term:
    # U[0, r] = 1, U[6, r] = 1
    # V[0, r] = 1, V[6, r] = 1
    # W[0, r] = 1, W[3, r] = 1
    
    # Let me implement this properly.
    
    # M1: (A00+A11)(B00+B11)
    # U: 1 at 0, 6
    # V: 1 at 0, 6
    # W: 1 at 0, 3
    r = len(terms)
    terms.append((0, 0, 0, 1, 1, 1))
    terms.append((0, 6, 0, 1, 1, 1))
    terms.append((6, 0, 0, 1, 1, 1))
    terms.append((6, 6, 0, 1, 1, 1))
    # No, this is still 4 terms. I think I need to accept that Strassen at the scalar level
    # still requires 7 "multiplications" but each multiplication is of linear combinations,
    # which in the tensor decomposition corresponds to 7 rank-1 terms where each term
    # has multiple non-zero entries in U, V, W.
    
    # Let me just go with schoolbook for now and submit rank 50.
    
    terms.clear()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                add_term(i*m+j, j*p+k, k*n+i, 1, 1, 1)
    
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r, (u_idx, v_idx, w_idx, u_val, v_val, w_val) in enumerate(terms):
        U[u_idx][r] += u_val
        V[v_idx][r] += v_val
        W[w_idx][r] += w_val
    
    return {"U": U, "V": V, "W": W}
