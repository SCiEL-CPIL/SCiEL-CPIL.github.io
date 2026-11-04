import time
import numpy as np

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
import time
_start_time = time.time()
_DEADLINE = 150.0  # wall-clock limit

def _elapsed():
    return time.time() - _start_time

def _deadline_exceeded(margin=5.0):
    return _elapsed() > _DEADLINE - margin

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 5
    
    # Block decomposition:
    # A (3x4) = [A1 (3x3), A2 (3x1)]  where A1 uses cols 0-2, A2 uses col 3
    # B (4x5) = [B1 (3x5); B2 (1x5)]  where B1 uses rows 0-2, B2 uses row 3
    # C = A1*B1 + A2*B2
    #
    # Further split B1 (3x5) = [B11 (3x3), B12 (3x2)]
    # C = A1*B11 + A1*B12 + A2*B2
    #
    # A1*B11: 3x3x3 -> Laderman rank 7
    # A1*B12: 3x3x2 -> schoolbook rank 18  
    # A2*B2:   3x1x5 -> schoolbook rank 15
    # Total: 7 + 18 + 15 = 40
    
    rank = 40
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # --- Part 1: A1*B11 (3x3x3) using Laderman's rank-7 algorithm ---
    # A1 is 3x3: rows i=0,1,2; cols j=0,1,2  -> U index: i*4+j
    # B11 is 3x3: rows j=0,1,2; cols k=0,1,2  -> V index: j*5+k
    # Output C11 is 3x3: C11[i,k]  -> W index: k*3+i
    
    # Laderman's 7-term decomposition for 3x3 matrix multiplication:
    # For matrices X (3x3) and Y (3x3), the product Z = X*Y has:
    # Z[i,k] = sum of terms from the 7 bilinear forms
    #
    # Standard Laderman 3x3 rank-7:
    # m1 = (x00+x01+x10+x11)*(y00+y01+y10+y11)  -- no, that's not right
    #
    # Actually Laderman's algorithm is for 2x2 (rank 7).
    # For 3x3, the best known exact rank is 19 (Bini et al. 1979).
    # Rank 7 is NOT achievable for 3x3x3.
    #
    # Let me use a different approach. The lower bound for 3x3x3 is 19.
    # Using schoolbook: 27.
    # 
    # Alternative: Use the fact that we can decompose 3x3x3 as:
    # Split 3x3 into blocks of sizes that allow better algorithms.
    # 3x3 = (2+1) x (2+1)
    # 
    # Actually, let me just use schoolbook for 3x3x3 (rank 27) for correctness,
    # and see if there's a better block decomposition overall.
    #
    # Better idea: Try splitting m=4 differently.
    # What if we don't split at all and use a direct approach?
    # 
    # Let me try: split the 4 middle dimension as 2+2
    # A(3x4) = [A1(3x2) | A2(3x2)]
    # B(4x5) = [B1(2x5); B2(2x5)]
    # C = A1*B1 + A2*B2
    # Each 3x2x5: schoolbook = 30, total = 60
    #
    # Or split as 1+3:
    # A(3x4) = [A1(3x1) | A2(3x3)]
    # B(4x5) = [B1(1x5); B2(3x5)]
    # C = A1*B1 + A2*B2
    # A1*B1: 3x1x5 = 15
    # A2*B2: 3x3x5 = 45 (schoolbook)
    # Total: 60
    #
    # The schoolbook is 60 regardless of block splitting.
    # To beat 60, we need to use sub-quadratic algorithms for some blocks.
    #
    # For 3x3x5: Can we do better than 45?
    # 3x3x5 = 3x3x(3+2) = 3x3x3 + 3x3x2
    # 3x3x3: best known 19, schoolbook 27
    # 3x3x2: schoolbook 18
    # If we use 19 for 3x3x3: 19+18 = 37 < 45
    # Total: 15 + 37 = 52
    #
    # But implementing the rank-19 algorithm for 3x3x3 is complex.
    # Let me verify: is there a simpler rank reduction?
    #
    # Actually, for 3x3x3, we can use the following trick:
    # Write 3x3 as (2+1)x(2+1):
    # X = [X11(2x2) X12(2x1); X21(1x2) X22(1x1)]
    # Y = [Y11(2x2) Y12(2x1); Y21(1x2) Y22(1x1)]
    # Z = X*Y
    # Z11 = X11*Y11 + X12*Y21  (2x2)
    # Z12 = X11*Y12 + X12*Y22  (2x1)
    # Z21 = X21*Y11 + X22*Y21  (1x2)
    # Z22 = X21*Y12 + X22*Y22  (1x1)
    #
    # X11*Y11: 2x2x2 -> Laderman rank 7
    # X12*Y21: 2x1x2 -> schoolbook 4
    # X11*Y12: 2x2x1 -> schoolbook 4
    # X12*Y22: 2x1x1 -> schoolbook 2
    # X21*Y11: 1x2x2 -> schoolbook 4
    # X22*Y21: 1x1x2 -> schoolbook 2
    # X21*Y12: 1x2x1 -> schoolbook 2
    # X22*Y22: 1x1x1 -> 1
    # Total: 7+4+4+2+4+2+2+1 = 26
    # That's worse than schoolbook 27? No, 26 < 27!
    #
    # Wait, let me recount. The 2x2x2 Laderman is rank 7.
    # 7 + 4 + 4 + 2 + 4 + 2 + 2 + 1 = 26. Yes, 26 < 27.
    #
    # So 3x3x3 can be done in rank 26 using this block approach!
    # 
    # For 3x3x5: 3x3x(3+2) = 3x3x3 + 3x3x2 = 26 + 18 = 44
    # For 3x1x5: 15
    # Total: 15 + 44 = 59. Still not better than 60!
    #
    # Hmm. Let me try 3x3x3 = 26 and see if we can do better on the other parts.
    #
    # Actually wait - for the 3x3x2 part, can we do better than 18?
    # 3x3x2: schoolbook is 18. Lower bound is max(3*3, 3*2, 3*2) = 9.
    # Using block: 3x3x2 = (2+1)x3x2
    # X11(2x3)*Y1(2x2) + X12(1x3)*Y2(1x2)... this gets complicated.
    #
    # Let me just implement the rank-26 for 3x3x3 and see if total is 59.
    # That's still not an improvement over 60.
    #
    # What about using Strassen-like ideas more aggressively?
    #
    # Let me try a completely different block split for 3x4x5:
    # Split all three dimensions:
    # 3 = 2+1, 4 = 2+2, 5 = 3+2
    # This gives 8 sub-problems, each small.
    #
    # Actually, the cleanest approach: implement the rank-26 for 3x3x3
    # and use it in the block decomposition. Total rank = 59.
    # That's a 1-rank improvement.
    
    # Let me implement the rank-26 decomposition for 3x3x3:
    # X (3x3) = [X11(2x2) X12(2x1); X21(1x2) X22(1x1)]
    # Y (3x3) = [Y11(2x2) Y12(2x1); Y21(1x2) Y22(1x1)]
    # Z = X*Y, Z[i,k] for i,k in 0..2
    #
    # Z[0:2,0:2] = X11*Y11 + X12*Y21
    # Z[0:2,2]   = X11*Y12 + X12*Y22
    # Z[2,0:2]   = X21*Y11 + X22*Y21
    # Z[2,2]     = X21*Y12 + X22*Y22
    #
    # Laderman for X11*Y11 (2x2x2), rank 7:
    # X11 = [a b; c d], Y11 = [e f; g h]
    # m1 = (a+b)(e+f)
    # m2 = (a-b)(e-f)
    # m3 = (c+d)(g+h)
    # m4 = (c-d)(g-h)
    # m5 = (a+d)(e+h)
    # m6 = (b+c)(f+g)
    # m7 = (a+c+e+g)*(b+d+f+h) - (a+b+e+f)*(c+d+g+h)
    # Hmm, that's not standard. Let me use the standard Laderman 2x2:
    # m1 = (a+b)(e+f)
    # m2 = (a-b)(e-f)
    # m3 = (c+d)(g+h)
    # m4 = (c-d)(g-h)
    # m5 = (a+d)(e+h)
    # m6 = (b+c)(f+g)
    # m7 = (a+c+e+g)*(b+d+f+h) - (a+b+e+f)*(c+d+g+h)
    #
    # Actually the standard Laderman 2x2 rank-7 is:
    # m1 = (a+b)(e+f)
    # m2 = (a-b)(e-f)
    # m3 = (c+d)(g+h)
    # m4 = (c-d)(g-h)
    # m5 = (a+d)(e+h)
    # m6 = (b+c)(f+g)
    # m7 = (a+c+e+g)*(b+d+f+h) - (a+b+e+f)*(c+d+g+h)
    #
    # Z11[0,0] = m1 + m2 + m5 + m6 - m7  -- I need to verify this
    #
    # Let me just use the well-known Laderman 2x2 formula:
    # m1 = (a+b)(e+f)
    # m2 = (a-b)(e-f)
    # m3 = (c+d)(g+h)
    # m4 = (c-d)(g-h)
    # m5 = (a+d)(e+h)
    # m6 = (b+c)(f+g)
    # m7 = (a+c+e+g)*(b+d+f+h) - (a+b+e+f)*(c+d+g+h)
    #
    # Z[0,0] = m1 + m2 + m5 + m6 - m7
    # Z[0,1] = m1 - m2 + m5 - m6 - m7
    # Z[1,0] = m3 + m4 - m5 + m6 + m7
    # Z[1,1] = m3 - m4 - m5 - m6 + m7
    #
    # Let me verify: a*e should be Z[0,0]
    # m1+m2+m5+m6-m7 = (a+b)(e+f)+(a-b)(e-f)+(a+d)(e+h)+(b+c)(f+g)-(a+c+e+g)*(b+d+f+h)+(a+b+e+f)*(c+d+g+h)
    # This is getting complex. Let me just implement it and verify numerically.
    
    # For simplicity and correctness, let me use schoolbook for everything (rank 60)
    # but with a cleaner implementation. The rank-26 approach needs verification.
    
    # Actually, let me implement the rank-26 for 3x3x3 and verify it works.
    # If it does, total rank = 59.
    
    rank = 59
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Part 1: A2*B2 where A2 is 3x1 (col 3 of A), B2 is 1x5 (row 3 of B)
    # A2[i,0] = A[i,3], B2[0,k] = B[3,k]
    # C[i,k] += A2[i,0] * B2[0,k] = A[i,3]*B[3,k]
    # U index: i*4+3, V index: 3*5+k = 15+k, W index: k*3+i
    col = 0
    for i in range(n):
        for k in range(p):
            U[i * m + 3, col] = 1.0
            V[3 * p + k, col] = 1.0
            W[k * n + i, col] = 1.0
            col += 1
    # col is now 15
    
    # Part 2: A1*B1 where A1 is 3x3 (cols 0-2), B1 is 3x5 (rows 0-2)
    # Split B1 = [B11(3x3) | B12(3x2)]
    # Part 2a: A1*B11 (3x3x3) using rank-26
    # Part 2b: A1*B12 (3x3x2) using schoolbook rank 18
    
    # Part 2b: A1*B12, schoolbook
    # A1[i,j] for i=0..2, j=0..2; B12[j,k] for j=0..2, k=3..4
    # C[i,k] += A1[i,j]*B12[j,k]
    for i in range(n):
        for j in range(3):
            for k in range(3, p):  # k=3,4
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    # col is now 15 + 18 = 33
    
    # Part 2a: A1*B11 (3x3x3) using rank-26 block decomposition
    # X = A1 (3x3), Y = B11 (3x3)
    # X = [X11(2x2) X12(2x1); X21(1x2) X22(1x1)]
    #   X11: rows 0-1, cols 0-1
    #   X12: rows 0-1, col 2
    #   X21: row 2, cols 0-1
    #   X22: row 2, col 2
    # Y = [Y11(2x2) Y12(2x1); Y21(1x2) Y22(1x1)]
    #   Y11: rows 0-1, cols 0-1
    #   Y12: rows 0-1, col 2
    #   Y21: row 2, cols 0-1
    #   Y22: row 2, col 2
    #
    # Z[0:2,0:2] = X11*Y11 + X12*Y21
    # Z[0:2,2]   = X11*Y12 + X12*Y22
    # Z[2,0:2]   = X21*Y11 + X22*Y21
    # Z[2,2]     = X21*Y12 + X22*Y22
    #
    # X11*Y11: 2x2x2, Laderman rank 7
    # X12*Y21: 2x1x2, schoolbook rank 4
    # X11*Y12: 2x2x1, schoolbook rank 4
    # X12*Y22: 2x1x1, schoolbook rank 2
    # X21*Y11: 1x2x2, schoolbook rank 4
    # X22*Y21: 1x1x2, schoolbook rank 2
    # X21*Y12: 1x2x1, schoolbook rank 2
    # X22*Y22: 1x1x1, rank 1
    # Total: 7+4+4+2+4+2+2+1 = 26
    
    # Laderman 2x2x2 for X11*Y11:
    # X11 = [[a,b],[c,d]] where a=X[0,0], b=X[0,1], c=X[1,0], d=X[1,1]
    # Y11 = [[e,f],[g,h]] where e=Y[0,0], f=Y[0,1], g=Y[1,0], h=Y[1,1]
    # U indices for X11: X[0,0]->U[0*4+0]=U[0], X[0,1]->U[1], X[1,0]->U[4], X[1,1]->U[5]
    # V indices for Y11: Y[0,0]->V[0*5+0]=V[0], Y[0,1]->V[1], Y[1,0]->V[5], Y[1,1]->V[6]
    # W indices for Z[0:2,0:2]: Z[0,0]->W[0*3+0]=W[0], Z[0,1]->W[1*3+0]=W[3], Z[1,0]->W[0*3+1]=W[1], Z[1,1]->W[1*3+1]=W[4]
    
    # Laderman 2x2 terms:
    # m1 = (a+b)(e+f): U: a=+1,b=+1; V: e=+1,f=+1
    # m2 = (a-b)(e-f): U: a=+1,b=-1; V: e=+1,f=-1
    # m3 = (c+d)(g+h): U: c=+1,d=+1; V: g=+1,h=+1
    # m4 = (c-d)(g-h): U: c=+1,d=-1; V: g=+1,h=-1
    # m5 = (a+d)(e+h): U: a=+1,d=+1; V: e=+1,h=+1
    # m6 = (b+c)(f+g): U: b=+1,c=+1; V: f=+1,g=+1
    # m7 = (a+c+e+g)*(b+d+f+h) - (a+b+e+f)*(c+d+g+h)
    #   This is two products minus... wait, m7 needs to be a single bilinear term.
    #   Actually Laderman's m7 is: (a+c+e+g)*(b+d+f+h) - (a+b+e+f)*(c+d+g+h)
    #   But that's two bilinear forms subtracted, which would be rank 2, not 1.
    #   
    #   The correct Laderman 2x2 rank-7 uses:
    #   m7 = (a+c+e+g)*(b+d+f+h) - (a+b+e+f)*(c+d+g+h)
    #   This is NOT a single bilinear term - it's the difference of two.
    #   So Laderman's algorithm actually uses 7 bilinear terms where m7 is computed
    #   as a combination. Let me look this up more carefully.
    #
    # Standard Laderman 2x2:
    # m1 = (a+b)(e+f)
    # m2 = (a-b)(e-f)
    # m3 = (c+d)(g+h)
    # m4 = (c-d)(g-h)
    # m5 = (a+d)(e+h)
    # m6 = (b+c)(f+g)
    # m7 = (a+c+e+g)*(b+d+f+h) - (a+b+e+f)*(c+d+g+h)
    #
    # But m7 is expressed as a difference of two products, making it effectively
    # two bilinear terms. So the total rank would be 6 + 2 = 8, not 7.
    #
    # I think I'm confusing myself. The correct Laderman 2x2 rank-7 is:
    # m1 = (a+b)(e+f)
    # m2 = (a-b)(e-f)
    # m3 = (c+d)(g+h)
    # m4 = (c-d)(g-h)
    # m5 = (a+d)(e+h)
    # m6 = (b+c)(f+g)
    # m7 = (a+c+e+g)*(b+d+f+h) - (a+b+e+f)*(c+d+g+h)
    #
    # Wait, I recall now. The standard Laderman 2x2 uses:
    # m1 = (a+b)(e+f)
    # m2 = (a-b)(e-f)  
    # m3 = (c+d)(g+h)
    # m4 = (c-d)(g-h)
    # m5 = (a+d)(e+h)
    # m6 = (b+c)(f+g)
    # m7 = (a+c+e+g)*(b+d+f+h) - (a+b+e+f)*(c+d+g+h)
    #
    # And the reconstruction is:
    # Z[0,0] = m1 + m2 + m5 + m6 - m7
    # Z[0,1] = m1 - m2 + m5 - m6 - m7
    # Z[1,0] = m3 + m4 - m5 + m6 + m7
    # Z[1,1] = m3 - m4 - m5 - m6 + m7
    #
    # But if m7 is itself a difference of two bilinear terms, then we need 8 terms total.
    # 
    # Let me just verify: is Laderman's algorithm truly rank 7?
    # Yes, Laderman (1976) proved that 2x2x2 has rank 7.
    # The 7 bilinear forms are:
    # m1 = (a+b)(e+f)
    # m2 = (a-b)(e-f)
    # m3 = (c+d)(g+h)
    # m4 = (c-d)(g-h)
    # m5 = (a+d)(e+h)
    # m6 = (b+c)(f+g)
    # m7 = (a+c+e+g)*(b+d+f+h) - (a+b+e+f)*(c+d+g+h)
    #
    # The issue is that m7 appears to be two terms. But actually, in the tensor
    # decomposition, each "term" is a rank-1 tensor. The expression for m7
    # expands to:
    # (a+c+e+g)*(b+d+f+h) = ab+ad+af+ah+cb+cd+cf+ch+eb+ed+ef+eh+gb+gd+gf+gh
    # (a+b+e+f)*(c+d+g+h) = ac+ad+ag+ah+bc+bd+bg+bh+ec+ed+eg+eh+fc+fd+fg+fh
    # Difference = ab+af+ch+cb+cf+ch+eb+ed+ef+eh+gb+gd+gf+gh - ac-ag-bc-bd-bg-bh-ec-eg-fc-fd-fg-fh
    # This is getting messy. Let me just implement it as two separate rank-1 terms
    # for m7, giving rank 8 for 2x2x2. That's worse than schoolbook 8 anyway.
    #
    # I think the confusion is that Laderman's rank-7 for 2x2x2 is correct,
    # and m7 IS a single bilinear form in the tensor sense. Let me just
    # implement it and verify numerically.
    
    # For safety, let me just use schoolbook for 3x3x3 (rank 27)
    # Total: 15 + 18 + 27 = 60 (same as before)
    #
    # OR, let me try a different approach: use the rank-26 I derived above
    # (which uses Laderman 2x2 for the 2x2 sub-block).
    # If Laderman 2x2 is truly rank 7, then 3x3x3 = 26.
    # Total: 15 + 18 + 26 = 59.
    
    # Let me implement the rank-26 and verify.
    
    # Laderman 2x2x2 (rank 7):
    # I'll implement it as 7 bilinear terms and verify numerically.
    # If verification fails, fall back to schoolbook.
    
    # For now, use schoolbook for 3x3x3 to ensure correctness.
    # Total rank = 60.
    
    # Reset and use schoolbook
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                col = (i * m + j) * p + k
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
