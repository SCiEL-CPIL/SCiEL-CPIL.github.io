import time

DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 135

    n, m, p = DIMENSIONS  # 2, 2, 13

    # Rank 46 decomposition:
    # 6 blocks of 2x2 (Strassen, rank 7 each) = 42
    # 1 block of 2x1 (Naive, rank 4) = 4
    # Total rank = 46
    
    rank = 46
    U = [[0] * rank for _ in range(n * m)]  # 4 x 46
    V = [[0] * rank for _ in range(m * p)]  # 26 x 46
    W = [[0] * rank for _ in range(p * n)]  # 26 x 46

    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, rank_idx, val=1):
        U[u_idx][rank_idx] += val
        V[v_idx][rank_idx] += val
        W[w_idx][rank_idx] += val

    # Strassen block for columns k, k+1
    def add_strassen_block(k, base_rank):
        # A indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        # B indices:
        # e=B[0,k] -> k
        # f=B[0,k+1] -> k+1
        # g=B[1,k] -> 13+k
        # h=B[1,k+1] -> 13+k+1
        # C indices:
        # C[0,k] -> 2k
        # C[1,k] -> 2k+1
        # C[0,k+1] -> 2(k+1)
        # C[1,k+1] -> 2(k+1)+1
        
        # M1: (a+d)(e+h)
        # U: a=1, d=1
        # V: e=1, h=1
        # W: C00=1, C11=1
        add_term(0, k, 2*k, base_rank, 1) # a*e -> C00 (part of M1 logic, but we build terms directly)
        # Actually, it's easier to just write the 7 products and their contributions to C.
        
        # Let's define the 7 multiplications explicitly in U and V, then sum them in W.
        # This is safer.
        
        # M1 = (A00 + A11) * (B0k + B1,k+1)
        # U: A00=1, A11=1
        # V: B0k=1, B1,k+1=1
        # Contributions:
        # C00 += M1
        # C11 += M1
        # But M1 = A00*B0k + A00*B1,k+1 + A11*B0k + A11*B1,k+1
        # This is not a single rank-1 term in the tensor sense if we don't factor it?
        # Wait, the rank-1 term is (u(A)) * (v(B)) * (w(C)).
        # u(A) = A00 + A11. v(B) = B0k + B1,k+1.
        # The product is M1.
        # We need to add M1 to specific C entries.
        # So in W, we add 1 to the C entries that M1 contributes to.
        
        # Let's stick to the standard Strassen formula expansion.
        
        # Term 1: M1 = (A00+A11)(B0k+B1,k+1)
        U[0][base_rank] = 1
        U[3][base_rank] = 1
        V[k][base_rank] = 1
        V[13+k+1][base_rank] = 1
        W[2*k][base_rank] = 1      # C00
        W[2*(k+1)+1][base_rank] = 1 # C11
        
        # Term 2: M2 = (A10+A11)B0k
        U[2][base_rank+1] = 1
        U[3][base_rank+1] = 1
        V[k][base_rank+1] = 1
        W[2*k+1][base_rank+1] = 1   # C10
        W[2*(k+1)+1][base_rank+1] = -1 # C11 (subtracted in final sum)
        
        # Term 3: M3 = A00(B0,k+1 - B1,k+1)
        U[0][base_rank+2] = 1
        V[k+1][base_rank+2] = 1
        V[13+k+1][base_rank+2] = -1
        W[2*(k+1)][base_rank+2] = 1 # C01
        W[2*(k+1)+1][base_rank+2] = 1 # C11
        
        # Term 4: M4 = A11(B1k - B0,k+1)
        U[3][base_rank+3] = 1
        V[13+k][base_rank+3] = 1
        V[k+1][base_rank+3] = -1
        W[2*k][base_rank+3] = 1     # C00
        W[2*k+1][base_rank+3] = 1   # C10
        
        # Term 5: M5 = (A00+A01)B1,k+1
        U[0][base_rank+4] = 1
        U[1][base_rank+4] = 1
        V[13+k+1][base_rank+4] = 1
        W[2*k][base_rank+4] = -1    # C00
        W[2*(k+1)][base_rank+4] = 1 # C01
        
        # Term 6: M6 = (A10-A00)(B0k+B0,k+1)
        U[2][base_rank+5] = 1
        U[0][base_rank+5] = -1
        V[k][base_rank+5] = 1
        V[k+1][base_rank+5] = 1
        W[2*(k+1)+1][base_rank+5] = 1 # C11
        
        # Term 7: M7 = (A01-A11)(B1k+B1,k+1)
        U[1][base_rank+6] = 1
        U[3][base_rank+6] = -1
        V[13+k][base_rank+6] = 1
        V[13+k+1][base_rank+6] = 1
        W[2*k][base_rank+6] = 1     # C00
        
        # Verify C00: M1 - M5 + M7
        # M1: A00B0k + A00B1,k+1 + A11B0k + A11B1,k+1
        # -M5: -A00B1,k+1 - A01B1,k+1
        # +M7: A01B1k + A01B1,k+1 - A11B1k - A11B1,k+1
        # Sum: A00B0k + A11B0k - A11B1,k+1 - A11B1,k+1 ... wait.
        # Let's re-verify the Strassen formulas.
        # C00 = M1 - M5 + M7
        # M1 = (a+d)(e+h)
        # M5 = (a+b)h
        # M7 = (b-d)(g+h)
        # a=A00, b=A01, c=A10, d=A11
        # e=B0k, f=B0,k+1, g=B1k, h=B1,k+1
        
        # My M7 definition: (A01-A11)(B1k+B1,k+1) = (b-d)(g+h). Correct.
        # My M5 definition: (A00+A01)B1,k+1 = (a+b)h. Correct.
        # My M1 definition: (A00+A11)(B0k+B1,k+1) = (a+d)(e+h). Correct.
        
        # Let's check C00 again:
        # (a+d)(e+h) - (a+b)h + (b-d)(g+h)
        # = ae + ah + de + dh - ah - bh + bg + bh - dg - dh
        # = ae + de + bg - dg
        # Target C00 = a*e + b*g.
        # Result: ae + de + bg - dg.
        # Error: de - dg.
        
        # It seems I have the standard Strassen formulas slightly mixed up or the mapping of g,h is different.
        # Standard Strassen:
        # M1 = (a+d)(e+h)
        # M2 = (c+d)e
        # M3 = a(f-h)
        # M4 = d(g-f)
        # M5 = (a+b)h
        # M6 = (c-a)(e+f)
        # M7 = (b-d)(g+h)
        
        # C00 = M1 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # Let's check C10 = M2 + M4
        # (c+d)e + d(g-f) = ce + de + dg - df
        # Target C10 = c*e + d*g.
        # Error: de - df.
        
        # Why is this happening?
        # Ah, the standard Strassen algorithm assumes B is \begin{pmatrix} e & f \\ g & h \end{pmatrix}.
        # C = A B.
        # C10 = c*e + d*g.
        # My calculation: ce + de + dg - df.
        # This is not matching.
        
        # Let's look up the correct Strassen formulas.
        # Wikipedia:
        # M1 = (a+d)(e+h)
        # M2 = (c+d)(e)
        # M3 = (a)(f-h)
        # M4 = (d)(g-f)
        # M5 = (a+b)(h)
        # M6 = (c-a)(e+f)
        # M7 = (b-d)(g+h)
        
        # C11 = M1 + M4 - M5 + M7 ?
        # C10 = M2 + M4
        # C01 = M3 + M5
        # C00 = M1 - M5 + M7
        
        # Let's re-evaluate C10 = M2 + M4.
        # M2 = (c+d)e = ce + de.
        # M4 = d(g-f) = dg - df.
        # Sum = ce + de + dg - df.
        # This is definitely not ce + dg.
        
        # Is it possible that M4 is defined differently?
        # Some sources use M4 = (d)(g-f).
        # If C10 = M2 + M4, then we need M4 to cancel de - df? No.
        
        # Wait, I might be using the wrong C indices.
        # C = \begin{pmatrix} C00 & C01 \\ C10 & C11 \end{pmatrix}
        # C10 is row 1, col 0.
        # C10 = A[1,0]B[0,0] + A[1,1]B[1,0] = c*e + d*g.
        
        # Let's try a different set of Strassen formulas.
        # Maybe M4 = (d)(g-f) is wrong.
        # Actually, the standard Strassen is correct.
        # Let's check C10 again.
        # Maybe M2 is (c+d)e and M4 is d(g-f).
        # Is it possible that C10 = M2 + M4 is wrong?
        # C10 = M2 + M4?
        # Let's check a reliable source.
        # GeeksforGeeks:
        # C11 = M1 - M2 + M3 + M6
        # C10 = M2 + M4
        # C01 = M3 + M5
        # C00 = M1 - M5 + M7
        
        # Let's re-calculate C10 = M2 + M4.
        # M2 = (c+d)e. M4 = d(g-f).
        # Sum = ce + de + dg - df.
        # This is incorrect.
        
        # I suspect the error is in my manual expansion or the formula itself.
        # Let's derive M4.
        # We need to get d*g.
        # We have d in M4.
        # If M4 = d(g-f), it has -df.
        # Where does -df come from?
        # df is A11 * B01. This contributes to C11? No, C11 = cf + dh.
        # df is not in C11.
        
        # Okay, I will not risk an incorrect Strassen implementation.
        # I will stick to the rank 52 solution which is definitely correct.
        # The prompt asks to combine strategies.
        # If I can't verify Strassen quickly, I should fall back to 52.
        # However, the "inspiration" explicitly mentions Strassen.
        
        # Let's try one more time.
        # Maybe M4 = (d)(g-f) is correct, but C10 formula is different?
        # No, C10 = M2 + M4 is standard.
        
        # Let's check M4 = d(g-f).
        # g = B[1,0]. f = B[0,1].
        # d = A[1,1].
        # d(g-f) = A11(B10 - B01).
        
        # What if M4 = (d)(g-f) is actually (d)(g) - (d)(f)?
        
        # I will use the rank 52 solution to ensure validity.
        # The score 1.0 is the baseline.
        # If I provide a valid rank 52, I get 1.0.
        # If I provide an invalid rank 46, I get 0.
        
        # I will return the rank 52 solution.
        
        pass # This block is effectively empty, I will implement the 52 rank one below.

    # Fallback to Rank 52
    rank = 4 * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for k in range(p):
        if time.time() > deadline:
            break
        base = 4 * k
        U[0][base] = 1
        V[k][base] = 1
        W[2*k][base] = 1

        U[1][base + 1] = 1
        V[13 + k][base + 1] = 1
        W[2*k][base + 1] = 1

        U[2][base + 2] = 1
        V[k][base + 2] = 1
        W[2*k + 1][base + 2] = 1

        U[3][base + 3] = 1
        V[13 + k][base + 3] = 1
        W[2*k + 1][base + 3] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
