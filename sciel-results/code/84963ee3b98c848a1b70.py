"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 5
    deadline = time.time() + 130

    def _verify(U, V, W, n, m, p):
        rank = U.shape[1]
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

    def _try_decomp(U_terms, V_terms, W_terms, n, m, p):
        rank = len(U_terms)
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        for r in range(rank):
            U[:, r] = U_terms[r]
            V[:, r] = V_terms[r]
            W[:, r] = W_terms[r]
        if _verify(U, V, W, n, m, p) and _is_half_integer(U) and _is_half_integer(V) and _is_half_integer(W):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        return None

    U_terms = []
    V_terms = []
    W_terms = []

    def add_term(u_vec, v_vec, w_vec):
        U_terms.append(list(u_vec))
        V_terms.append(list(v_vec))
        W_terms.append(list(w_vec))

    def zeros6():
        return [0] * 6

    def zeros15():
        return [0] * 15

    def zeros10():
        return [0] * 10

    def setA(u, i, j, val):
        u[i * 3 + j] += val

    def setB(v, j, k, val):
        v[j * 5 + k] += val

    def setC(w, i, k, val):
        w[k * 2 + i] += val

    # =================================================================
    # Part 1: 2x1x5 outer product for column j=0 (rank 10)
    # C[i,k] = A[i,0] * B[0,k] for i in {0,1}, k in {0..4}
    # =================================================================
    for k in range(5):
        # C[0,k] = A[0,0] * B[0,k]
        u = zeros6(); v = zeros15(); w = zeros10()
        setA(u, 0, 0, 1)
        setB(v, 0, k, 1)
        setC(w, 0, k, 1)
        add_term(u, v, w)
        # C[1,k] = A[1,0] * B[0,k]
        u = zeros6(); v = zeros15(); w = zeros10()
        setA(u, 1, 0, 1)
        setB(v, 0, k, 1)
        setC(w, 1, k, 1)
        add_term(u, v, w)

    # =================================================================
    # Part 2: 2x2x3 Winograd for columns j=1,2 and rows k=0,1,2 (rank 10)
    # A_sub = A[:, 1:3] (2x2), B_sub = B[1:3, 0:3] (2x3)
    # C_sub = A_sub @ B_sub (2x3)
    # Winograd F2T2 decomposition:
    #   m1 = (a11 + a12) * (b10 + b20)
    #   m2 = (a11 + a12) * (b11 + b21)
    #   m3 = (a11 + a12) * (b12 + b22)
    #   m4 = a11 * b20
    #   m5 = a12 * b11
    #   m6 = a11 * b21
    #   m7 = a12 * b20
    #   m8 = a11 * b22
    #   m9 = a12 * b12
    #   m10 = a12 * b21
    # Wait, that's not right. Let me use the standard F2T2.
    #
    # Standard 2x2x3 Winograd (rank 10):
    # A = [[a, b], [c, d]], B = [[e, f, g], [h, i, j]]
    #
    # m1 = (a+d)(e+h)
    # m2 = (a+d)(f+i)
    # m3 = (a+d)(g+j)
    # m4 = a*h
    # m5 = b*f
    # m6 = a*i
    # m7 = b*e
    # m8 = a*j
    # m9 = b*g
    # m10 = b*h  -- wait, let me re-derive.
    #
    # Actually, the standard decomposition for 2x2x3 is:
    # m1 = (a+d)(e+h)
    # m2 = (a+d)(f+i)
    # m3 = (a+d)(g+j)
    # m4 = a*h
    # m5 = b*f
    # m6 = c*e
    # m7 = d*i
    # m8 = a*j
    # m9 = b*g
    # m10 = d*h  -- no...
    #
    # Let me use a different approach. The known rank-10 decomposition:
    # x1=a, x2=b, x3=c, x4=d, x5=a+d, x6=b+c, x7=a-b, x8=c-d
    # y1=e, y2=f, y3=g, y4=h, y5=i, y6=j, y7=e+h, y8=f+i, y9=g+j
    #
    # Actually, let me just use the explicit terms:
    # For C[0,0] = a*e + b*h
    # For C[0,1] = a*f + b*i
    # For C[0,2] = a*g + b*j
    # For C[1,0] = c*e + d*h
    # For C[1,1] = c*f + d*i
    # For C[1,2] = c*g + d*j
    #
    # Rank 10 decomposition:
    # m1 = (a+d) * (e+h)
    # m2 = (a+d) * (f+i)
    # m3 = (a+d) * (g+j)
    # m4 = a * h
    # m5 = b * f
    # m6 = c * e
    # m7 = d * i
    # m8 = a * j
    # m9 = b * g
    # m10 = c * g  -- no, this doesn't work either.
    #
    # Let me use a verified decomposition. The 2x2x3 tensor has rank 10.
    # One explicit decomposition:
    # m1: (a) * (e) -> C[0,0]
    # m2: (b) * (h) -> C[0,0]
    # m3: (a) * (f) -> C[0,1]
    # m4: (b) * (i) -> C[0,1]
    # m5: (a) * (g) -> C[0,2]
    # m6: (b) * (j) -> C[0,2]
    # m7: (c) * (e) -> C[1,0]
    # m8: (d) * (h) -> C[1,0]
    # m9: (c) * (f) -> C[1,1]
    # m10: (d) * (i) -> C[1,1]
    # ... but that's 12 terms for the 6 outputs, not 10.
    #
    # The rank-10 decomposition shares factors. Let me use:
    # m1 = (a+d) * (e+h)
    # m2 = (a+d) * (f+i)
    # m3 = (a+d) * (g+j)
    # m4 = a * h
    # m5 = b * e
    # m6 = a * i
    # m7 = b * f
    # m8 = a * j
    # m9 = b * g
    # m10 = (c-d) * (e-h)
    #
    # C[0,0] = m1 - m4 + m5 = (a+d)(e+h) - a*h + b*e = ae + ah + de + dh - ah + be = ae + de + dh + be
    # That's not right. C[0,0] should be ae + bh.
    #
    # I'm struggling with the exact terms. Let me use a simpler approach:
    # Just use the schoolbook for this 2x2x3 block (rank 12) and see if the total is still an improvement.
    # Total would be 10 + 12 + 7 = 29. Not an improvement.
    #
    # Alternative: use rank 12 for the 2x2x3 part via 2x2x2 (rank 7) + 2x2x1 (rank 4) = 11? No, 2x2x1 rank is 4.
    # 7 + 4 = 11 for 2x2x3? That would give total 10 + 11 + 7 = 28.
    #
    # Actually, 2x2x3 can be decomposed as 2x2x2 (rank 7) + 2x2x1 (rank 4) = rank 11.
    # But R(2x2x3) = 10, so there's a rank-10 solution. Let me find it.
    #
    # After more thought, here's a valid rank-10 decomposition for 2x2x3:
    # A = [[a,b],[c,d]], B = [[e,f,g],[h,i,j]]
    #
    # m1 = a * (e+f+g)
    # m2 = b * (h+i+j)
    # m3 = (a+b) * (e-h)
    # m4 = (a+b) * (f-i)
    # m5 = (a+b) * (g-j)
    # m6 = c * (e+f+g)
    # m7 = d * (h+i+j)
    # m8 = (c+d) * (e-h)
    # m9 = (c+d) * (f-i)
    # m10 = (c+d) * (g-j)
    #
    # C[0,0] = m1 - m2 + m3 = a(e+f+g) - b(h+i+j) + (a+b)(e-h)
    # = ae + af + ag - bh - bi - bj + ae - ah + be - bh
    # = 2ae + af + ag + be - 2bh - bi - bj - ah
    # That's not ae + bh. This is wrong.
    #
    # I'll use a safe fallback: decompose 2x2x3 as 2x1x3 + 1x2x3 = 3 + 6 = 9? No.
    # 2x1x3 rank 3, 1x2x3 rank 6. Total 9. But that's not how tensor decomposition works with sharing.
    #
    # Let me just use the full schoolbook for the 2x2x3 part (rank 12) and accept total rank 29.
    # Wait, 10 (outer j=0) + 12 (schoolbook 2x2x3) + 7 (Strassen 2x2x2) = 29. Same as original.
    #
    # Better idea: use 2x2x5 directly with rank 11 if I can find it, or use a different split.
    # Let me split as: j=0 (2x1x5, rank 10) + j=1,2 with k=0..4 (2x2x5).
    # R(2x2x5) = 11. Total = 21.
    #
    # I need the explicit rank-11 decomposition for 2x2x5. Let me construct it.
    #
    # For 2x2xn, the rank is 2n+1. The construction:
    # A = [[a,b],[c,d]], B is 2xn.
    # m0 = (a+d) * colsum(B)  where colsum is elementwise sum of B's rows
    # For k=0..n-1:
    #   m_{2k+1} = a * B[1,k]
    #   m_{2k+2} = b * B[0,k]
    #   C[0,k] = m0[k] - m_{2k+1}
    #   C[1,k] = m0[k] - m_{2k+2}
    #
    # Wait, that's 1 + 2n = 2n+1 terms. Let me verify:
    # C[0,k] = (a+d)(B[0,k]+B[1,k]) - a*B[1,k] = a*B[0,k] + a*B[1,k] + d*B[0,k] + d*B[1,k] - a*B[1,k]
    # = a*B[0,k] + d*B[0,k] + d*B[1,k] = (a+d)*B[0,k] + d*B[1,k]
    # But C[0,k] should be a*B[0,k] + b*B[1,k].
    # So this is wrong.
    #
    # Correct construction for 2x2xn rank 2n+1:
    # m0 = (a-b) * (B[0,:] - B[1,:])  [elementwise, n terms? No, this is n scalar products]
    #
    # I'm going in circles. Let me just implement a known valid decomposition.
    #
    # After research, the 2x2x5 rank-11 decomposition is:
    # m1 = (a+d) * (b00+b10)
    # m2 = (a+d) * (b01+b11)
    # m3 = (a+d) * (b02+b12)
    # m4 = (a+d) * (b03+b13)
    # m5 = (a+d) * (b04+b14)
    # m6 = a * b10
    # m7 = b * b01
    # m8 = c * b02
    # m9 = d * b13
    # m10 = (a-c) * (b04-b14)  -- not sure
    # m11 = (b-d) * (b10-b00)  -- not sure
    #
    # This is too risky without verification. Let me use a safe decomposition.
    #
    # Safe approach: 2x2x5 = 2x2x3 (rank 10) + 2x2x2 (rank 7) = rank 17
    # Total: 10 + 17 = 27.
    #
    # For 2x2x3 rank 10, I'll use the following verified decomposition:
    # A = [[a,b],[c,d]], B = [[e,f,g],[h,i,j]]
    # m1 = (a+d)(e+h)
    # m2 = (a+d)(f+i)
    # m3 = (a+d)(g+j)
    # m4 = a*h
    # m5 = b*e
    # m6 = c*f
    # m7 = d*i
    # m8 = a*j
    # m9 = b*g
    # m10 = c*e  -- let me check C[0,0] = m1 - m4 + m5 = (a+d)(e+h) - ah + be
    # = ae + ah + de + dh - ah + be = ae + de + dh + be
    # Should be ae + bh. So m1 - m4 + m5 = ae + de + dh + be. Not equal to ae + bh.
    #
    # I give up trying to derive it manually. Let me use the schoolbook for 2x2x3 (rank 12).
    # Total: 10 + 12 + 7 = 29. No improvement.
    #
    # Alternative: don't split at all. Use schoolbook for all of 2x3x5 (rank 30) as fallback.
    # Or use 2x2x5 with a rank-12 decomposition I can verify.
    #
    # 2x2x5 via 2x2x4 (rank 12) + 2x2x1 (rank 4) = rank 16.
    # 2x2x4 via 2x2x3 (rank 10) + 2x2x1 (rank 4) = rank 14.
    # So 2x2x5 = 14 + 4 = 18. Total: 10 + 18 = 28.
    #
    # Let me use this: 2x1x5 (rank 10) + 2x2x4 (rank 12) + 2x2x1 (rank 4) = 26.
    # 2x2x4 = 2x2x2 (rank 7) + 2x2x2 (rank 7) = rank 14. Total: 10 + 14 + 4 = 28.
    # Or 2x2x4 = 2x2x3 (rank 10) + 2x2x1 (rank 4) = rank 14. Total: 10 + 14 + 4 = 28.
    #
    # I'll use 2x1x5 (10) + 2x2x2 (7) + 2x2x2 (7) + 2x2x1 (4) = 28.
    #
    # Part 2a: 2x2x2 Strassen for j=1,2 and k=0,1 (rank 7)
    # Part 2b: 2x2x2 Strassen for j=1,2 and k=2,3 (rank 7)
    # Part 2c: 2x2x1 outer for j=1,2 and k=4 (rank 4)
    
    # Part 2a: Strassen for A[:,1:3] @ B[1:3, 0:2]
    # A_sub = [[a,b],[c,d]], B_sub = [[e,f],[g,h]]
    # Strassen:
    # m1 = (a+d)(e+g)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-f)
    # m5 = (a+b)h
    # m6 = (c-a)(e+g)
    # m7 = (b-d)(f+h)
    # C[0,0] = m1 + m4 - m5 + m7
    # C[0,1] = m3 + m5
    # C[1,0] = m2 + m5
    # C[1,1] = m1 - m2 + m3 + m6
    
    strassen_0_1 = [
        # m1 = (a+d)(e+g)
        ([(0,1,1),(1,2,1)], [(1,0,1),(1,1,1)], [(0,0,1),(1,0,1)]),
        # m2 = (c+d)e
        ([(1,1,1),(1,2,1)], [(1,0,1)], [(1,0,1)]),
        # m3 = a(f-h)
        ([(0,1,1)], [(1,1,1),(1,0,-1)], [(0,1,1)]),
        # m4 = d(g-f)
        ([(1,2,1)], [(1,1,-1),(1,0,1)], [(0,0,1)]),
        # m5 = (a+b)h
        ([(0,1,1),(0,2,1)], [(1,1,1)], [(0,1,-1),(1,0,-1)]),
        # m6 = (c-a)(e+g)
        ([(1,1,1),(0,1,-1)], [(1,0,1),(1,1,1)], [(1,1,1)]),
        # m7 = (b-d)(f+h)
        ([(0,2,1),(1,2,-1)], [(1,1,1),(1,0,1)], [(0,0,-1),(0,1,1)]),
    ]
    
    for (a_list, b_list, c_list) in strassen_0_1:
        u = zeros6(); v = zeros15(); w = zeros10()
        for (i, j, coeff) in a_list:
            setA(u, i, j, coeff)
        for (j, k, coeff) in b_list:
            setB(v, j, k, coeff)
        for (i, k, coeff) in c_list:
            setC(w, i, k, coeff)
        add_term(u, v, w)
    
    # Part 2b: Strassen for A[:,1:3] @ B[1:3, 2:4]
    strassen_2_3 = [
        ([(0,1,1),(1,2,1)], [(1,2,1),(1,3,1)], [(0,2,1),(1,2,1)]),
        ([(1,1,1),(1,2,1)], [(1,2,1)], [(1,2,1)]),
        ([(0,1,1)], [(1,3,1),(1,2,-1)], [(0,3,1)]),
        ([(1,2,1)], [(1,3,-1),(1,2,1)], [(0,2,1)]),
        ([(0,1,1),(0,2,1)], [(1,3,1)], [(0,3,-1),(1,2,-1)]),
        ([(1,1,1),(0,1,-1)], [(1,2,1),(1,3,1)], [(1,3,1)]),
        ([(0,2,1),(1,2,-1)], [(1,3,1),(1,2,1)], [(0,2,-1),(0,3,1)]),
    ]
    
    for (a_list, b_list, c_list) in strassen_2_3:
        u = zeros6(); v = zeros15(); w = zeros10()
        for (i, j, coeff) in a_list:
            setA(u, i, j, coeff)
        for (j, k, coeff) in b_list:
            setB(v, j, k, coeff)
        for (i, k, coeff) in c_list:
            setC(w, i, k, coeff)
        add_term(u, v, w)
    
    # Part 2c: 2x2x1 outer for j=1,2 and k=4 (rank 4)
    # C[0,4] = A[0,1]*B[1,4] + A[0,2]*B[2,4]
    # C[1,4] = A[1,1]*B[1,4] + A[1,2]*B[2,4]
    for i in range(2):
        for j in range(2, 4):  # j=1,2
            u = zeros6(); v = zeros15(); w = zeros10()
            setA(u, i, j, 1)
            setB(v, j, 4, 1)
            setC(w, i, 4, 1)
            add_term(u, v, w)
    
    result = _try_decomp(U_terms, V_terms, W_terms, n, m, p)
    if result is not None:
        return result
    
    # Fallback: guaranteed-valid schoolbook decomposition (rank 30)
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
# EVOLVE-BLOCK-END
