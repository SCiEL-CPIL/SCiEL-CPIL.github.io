import time

DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
def construct():
    # Deadline guard: return a safe decomposition well before the kill switch.
    start_time = time.time()
    deadline = 12.0  # seconds
    
    n, m, p = DIMENSIONS  # 2, 2, 13
    
    # Matrix multiplication (n x m) * (m x p) -> (n x p)
    # A is 2x2, B is 2x13, C is 2x13
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # Indices for A: i*m + j = 2*i + j (i in 0..1, j in 0..1) -> 4 entries
    # Indices for B: j*p + k = 13*j + k (j in 0..1, k in 0..12) -> 26 entries
    # Indices for C: k*n + i = 2*k + i (k in 0..12, i in 0..1) -> 26 entries (official trace convention W[k*n+i, r])
    
    # Rank 40 construction based on Strassen extension.
    # Strassen for first 2 columns (k=0,1) uses rank 7.
    # Each subsequent column (k=2..12) adds 3 multiplications.
    # Total rank = 7 + 3 * (13 - 2) = 7 + 33 = 40.
    
    rank = 40
    U = [[0] * rank for _ in range(n * m)]  # 4 x 40
    V = [[0] * rank for _ in range(m * p)]  # 26 x 40
    W = [[0] * rank for _ in range(p * n)]  # 26 x 40
    
    # A indices:
    # 0: A[0,0]
    # 1: A[0,1]
    # 2: A[1,0]
    # 3: A[1,1]
    
    # B indices for k=0,1:
    # j=0, k=0 -> 0
    # j=1, k=0 -> 13
    # j=0, k=1 -> 1
    # j=1, k=1 -> 14
    
    # C indices for k=0,1:
    # k=0, i=0 -> 0
    # k=0, i=1 -> 1
    # k=1, i=0 -> 2
    # k=1, i=1 -> 3
    
    # Strassen decomposition for k=0,1:
    # C[0,0] = A[0,0]B[0,0] + A[0,1]B[1,0]
    # C[0,1] = A[0,0]B[0,1] + A[0,1]B[1,1]
    # C[1,0] = A[1,0]B[0,0] + A[1,1]B[1,0]
    # C[1,1] = A[1,0]B[0,1] + A[1,1]B[1,1]
    
    # M1: (A[0,0] + A[1,1]) * (B[0,0] + B[1,1])
    #     = A[0,0]B[0,0] + A[0,0]B[1,1] + A[1,1]B[0,0] + A[1,1]B[1,1]
    #     Contributes to C[0,0] (+1), C[0,1] (-1 via M3/M5 logic? No, let's use standard Strassen)
    
    # Standard Strassen for 2x2 * 2x2:
    # M1 = (a+d)(e+h) -> C00, C11
    # M2 = (c+d)e    -> C10, C11
    # M3 = a(f-h)    -> C01, C11
    # M4 = d(g-f)    -> C00, C10
    # M5 = (a+b)h    -> C00, C01
    # M6 = (c-a)(e+f)-> C11
    # M7 = (b-d)(g+h)-> C00
    
    # Mapping:
    # a=A[0,0] (idx 0), b=A[0,1] (idx 1), c=A[1,0] (idx 2), d=A[1,1] (idx 3)
    # e=B[0,0] (idx 0), f=B[0,1] (idx 1), g=B[1,0] (idx 13), h=B[1,1] (idx 14)
    # C00 (idx 0), C01 (idx 2), C10 (idx 1), C11 (idx 3)
    # Note: W index is k*n+i.
    # C[0,0] -> k=0, i=0 -> 0
    # C[0,1] -> k=1, i=0 -> 2
    # C[1,0] -> k=0, i=1 -> 1
    # C[1,1] -> k=1, i=1 -> 3
    
    # Let's verify the Strassen formulas against the indices:
    # C[0,0] = a*e + b*g
    # C[0,1] = a*f + b*h
    # C[1,0] = c*e + d*g
    # C[1,1] = c*f + d*h
    
    # M1: (a+d)(e+h) = ae + ah + de + dh
    #    W[0] += 1 (C00)
    #    W[3] += 1 (C11)
    #    Check: C00 gets ae (good). C11 gets dh (good).
    #    But we have extra terms ah and de.
    
    # M2: (c+d)e = ce + de
    #    W[1] += 1 (C10)
    #    W[3] += -1 (C11)
    #    Check: C10 gets ce (good). C11 gets -de (cancels de from M1).
    
    # M3: a(f-h) = af - ah
    #    W[2] += 1 (C01)
    #    W[3] += -1 (C11)
    #    Check: C01 gets af (good). C11 gets -ah (cancels ah from M1).
    
    # M4: d(g-f) = dg - df
    #    W[0] += 1 (C00)
    #    W[1] += -1 (C10)
    #    Check: C00 gets dg? No, C00 needs a*e + b*g. dg is not in C00.
    #    Wait, C[0,0] = a*e + b*g. C[1,0] = c*e + d*g.
    #    M4 contributes dg to C00? That's wrong.
    #    Let's re-verify standard Strassen.
    #    P1 = (a+d)(e+h)
    #    P2 = (c+d)e
    #    P3 = a(f-h)
    #    P4 = d(g-f)
    #    P5 = (a+b)h
    #    P6 = (c-a)(e+f)
    #    P7 = (b-d)(g+h)
    #
    #    C11 = P1 - P5 + P7  ? No.
    #    Standard:
    #    C11 = P1 + P4 - P5 + P7 ?
    #    Let's derive:
    #    C11 = c*f + d*h
    #    P1 = ae+ah+de+dh
    #    P2 = ce+de
    #    P3 = af-ah
    #    P4 = dg-df
    #    P5 = ah+bh
    #    P6 = ce+cf-ae-af
    #    P7 = bg+bh-dg-dh
    #
    #    C11 = P1 - P5 + P7 - P4 ?
    #    P1 - P5 + P7 - P4 = (ae+ah+de+dh) - (ah+bh) + (bg+bh-dg-dh) - (dg-df)
    #    = ae + de + dh - bh + bg + bh - dg - dh - dg + df
    #    = ae + de + bg - 2dg + df
    #    This doesn't look like cf+dh.
    
    #    Let's use the known correct formulas:
    #    C11 = P1 - P4 - P5 + P7
    #    P1 - P4 - P5 + P7 = (ae+ah+de+dh) - (dg-df) - (ah+bh) + (bg+bh-dg-dh)
    #    = ae + ah + de + dh - dg + df - ah - bh + bg + bh - dg - dh
    #    = ae + de + df + bg - 2dg
    #    Still not cf+dh.
    
    #    Okay, I will construct the decomposition explicitly for the first two columns using 7 terms that I know work, or just use 4 direct terms for k=0,1 and 4 for others.
    #    Rank 4 per column is safe and valid.
    #    Total rank = 4 * 13 = 52.
    #    The seed claimed 40 but was invalid. 52 is valid and likely better than 0.
    #    Can I do better?
    #    Strassen is rank 7 for 2x2*2x2.
    #    If I use Strassen for the first 2 columns (rank 7) and then 4 per column for the rest?
    #    Rank = 7 + 11*4 = 51.
    #    Let's try to fix the Strassen part.
    
    #    Correct Strassen for C = A*B:
    #    C[0,0] = P1 + P4 - P5 + P7
    #    C[0,1] = P3 + P5
    #    C[1,0] = P2 + P4
    #    C[1,1] = P1 - P2 + P3 + P6
    #
    #    Let's check C[1,1] with this:
    #    P1 - P2 + P3 + P6
    #    = (ae+ah+de+dh) - (ce+de) + (af-ah) + (ce+cf-ae-af)
    #    = ae+ah+de+dh -ce-de+af-ah+ce+cf-ae-af
    #    = dh + cf. Correct!
    #
    #    Let's check C[0,1] = P3 + P5:
    #    (af-ah) + (ah+bh) = af + bh. Correct!
    #
    #    Let's check C[1,0] = P2 + P4:
    #    (ce+de) + (dg-df) ... wait.
    #    P4 is d(g-f) = dg - df.
    #    C[1,0] = c*e + d*g.
    #    P2 + P4 = ce + de + dg - df. Incorrect.
    #
    #    Standard Strassen mapping:
    #    P1 = (a+d)(e+h)
    #    P2 = (c+d)(e)
    #    P3 = (a)(f-h)
    #    P4 = (d)(g-f)
    #    P5 = (a+b)(h)
    #    P6 = (c-a)(e+f)
    #    P7 = (b-d)(g+h)
    #
    #    C[0,0] = P1 - P5 + P7 - P4
    #    C[0,1] = P3 + P5
    #    C[1,0] = P2 + P4
    #    C[1,1] = P1 - P2 + P3 + P6
    #
    #    Let's re-evaluate C[1,0] = P2 + P4:
    #    P2 = (c+d)e = ce + de
    #    P4 = d(g-f) = dg - df
    #    Sum = ce + de + dg - df.
    #    Target C[1,0] = ce + dg.
    #    Mismatch: de - df.
    #
    #    It seems I am mixing up row/col conventions or the standard Strassen algorithm for matrix multiplication.
    #    Actually, for matrix multiplication, the Strassen algorithm is:
    #    M1 = (a+d)(e+h)
    #    M2 = (c+d)(e)
    #    M3 = (a)(f-h)
    #    M4 = (d)(g-f)
    #    M5 = (a+b)(h)
    #    M6 = (c-a)(e+f)
    #    M7 = (b-d)(g+h)
    #
    #    C[0,0] = M1 - M5 + M7
    #    C[0,1] = M3 + M5
    #    C[1,0] = M2 + M4
    #    C[1,1] = M1 - M2 + M3 + M6
    #
    #    Let's re-check C[1,0] = M2 + M4.
    #    M2 = (c+d)e = ce + de
    #    M4 = d(g-f) = dg - df
    #    Sum = ce + de + dg - df.
    #    Target = ce + dg.
    #    Error: de - df.
    #
    #    Why is this wrong?
    #    Ah, the standard Strassen algorithm splits the matrices into 2x2 blocks.
    #    Here, A and B are 2x2 and 2x2 (for the first two columns).
    #    a=A[0,0], b=A[0,1], c=A[1,0], d=A[1,1].
    #    e=B[0,0], f=B[0,1], g=B[1,0], h=B[1,1].
    #
    #    C[1,0] = A[1,0]B[0,0] + A[1,1]B[1,0] = c*e + d*g.
    #    M2 + M4 = (c+d)e + d(g-f) = ce + de + dg - df.
    #    This is definitely not equal to ce + dg unless de = df, which is not generally true.
    #
    #    I must have the Strassen formula wrong.
    #    Let's look up the correct Strassen formula.
    #    p1 = (a + d)(e + h)
    #    p2 = (c + d)(e)
    #    p3 = (a)(f - h)
    #    p4 = (d)(g - f)
    #    p5 = (a + b)(h)
    #    p6 = (c - a)(e + f)
    #    p7 = (b - d)(g + h)
    #
    #    C11 = p1 - p4 - p5 + p7
    #    C12 = p3 + p5
    #    C21 = p2 + p4
    #    C22 = p1 - p2 + p3 + p6
    #
    #    Wait, C21 in my notation is C[1,0].
    #    Let's check C21 = p2 + p4 again.
    #    p2 = (c+d)e
    #    p4 = d(g-f)
    #    Sum = ce + de + dg - df.
    #    Target C21 = ce + dg.
    #    It is still off by de - df.
    #
    #    Is it possible that p4 is defined as d(g-f) but contributes differently?
    #    No, the math is the math.
    #    Let's try a different set of formulas.
    #    Maybe I am misidentifying the variables.
    #    A = [[a, b], [c, d]]
    #    B = [[e, f], [g, h]]
    #    C = A B
    #    C[0,0] = ae + bg
    #    C[0,1] = af + bh
    #    C[1,0] = ce + dg
    #    C[1,1] = cf + dh
    #
    #    Let's test C[0,0] = p1 - p5 + p7.
    #    p1 = (a+d)(e+h) = ae + ah + de + dh
    #    p5 = (a+b)h = ah + bh
    #    p7 = (b-d)(g+h) = bg + bh - dg - dh
    #    p1 - p5 + p7 = ae + ah + de + dh - ah - bh + bg + bh - dg - dh
    #    = ae + de + bg - dg.
    #    Target: ae + bg.
    #    Off by de - dg.
    #
    #    It seems the standard Strassen algorithm I am recalling is for 2x2 matrix multiplication where the blocks are also 2x2 matrices, but here the "blocks" are scalars. The algebra should hold.
    #    Let's re-read a reliable source mentally.
    #    Ah! I think the issue is that Strassen's algorithm requires 7 multiplications for 2x2*2x2.
    #    My manual expansion shows errors, which means my formula memory is flawed.
    #    Given the time constraint and the risk of providing an invalid decomposition, I will use the direct 4-term method for all columns.
    #    Rank = 4 * 13 = 52.
    #    This is guaranteed to be correct.
    
    # Reset matrices
    rank = 4 * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    for k in range(p):
        # C[0,k] = A[0,0]B[0,k] + A[0,1]B[1,k]
        # C[1,k] = A[1,0]B[0,k] + A[1,1]B[1,k]
        
        # Term 1: A[0,0] * B[0,k] -> C[0,k]
        U[0][r] = 1       # A[0,0]
        V[k][r] = 1       # B[0,k] (j=0)
        W[2*k][r] = 1     # C[0,k] (i=0)
        r += 1
        
        # Term 2: A[0,1] * B[1,k] -> C[0,k]
        U[1][r] = 1       # A[0,1]
        V[13+k][r] = 1    # B[1,k] (j=1)
        W[2*k][r] = 1     # C[0,k] (i=0)
        r += 1
        
        # Term 3: A[1,0] * B[0,k] -> C[1,k]
        U[2][r] = 1       # A[1,0]
        V[k][r] = 1       # B[0,k] (j=0)
        W[2*k+1][r] = 1   # C[1,k] (i=1)
        r += 1
        
        # Term 4: A[1,1] * B[1,k] -> C[1,k]
        U[3][r] = 1       # A[1,1]
        V[13+k][r] = 1    # B[1,k] (j=1)
        W[2*k+1][r] = 1   # C[1,k] (i=1)
        r += 1
        
        if time.time() - start_time > deadline:
            break
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
