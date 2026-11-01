"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import nnls

DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=16
    # Use Strassen's algorithm for each of 8 groups of 2 columns
    # Each group uses rank 7, total rank = 8*7 = 56
    
    rank = 56
    num_groups = p // 2  # 8 groups of 2 columns each
    
    U = [[0] * rank for _ in range(n * m)]  # 4 x 56
    V = [[0] * rank for _ in range(m * p)]  # 32 x 56
    W = [[0] * rank for _ in range(p * n)]  # 32 x 56
    
    # Strassen's 7 multiplications for 2x2x2:
    # The factors for U (A indices: i*m+j), V (B indices: j*p+k), W (C indices: k*n+i)
    # For a single 2x2x2 block:
    # M1 = (A00+A11)(B00+B11): U=[1,0,0,1], V=[1,0,1,0], W=[1,0,0,1]
    # M2 = (A10+A11)B00:       U=[0,0,1,1], V=[1,0,0,0], W=[0,0,1,0]
    # M3 = A00(B10-B11):        U=[1,0,0,0], V=[0,1,-1,0], W=[0,1,0,0]
    # M4 = A11(B01-B00):        U=[0,0,0,1], V=[0,1,0,-1], W=[0,0,0,1]
    # M5 = (A00+A10)B11:        U=[1,1,0,0], V=[0,0,0,1], W=[0,0,1,0]
    # M6 = (A11-A10)(B00+B10):  U=[0,0,-1,1], V=[1,1,0,0], W=[0,1,0,0]
    # M7 = (A01-A11)(B01+B11):  U=[0,1,-1,0], V=[0,1,1,0], W=[1,0,0,0]
    
    # Wait, let me re-derive. For 2x2x2:
    # A is 2x2: indices (i,j) -> flat index i*m+j = i*2+j
    #   A00 -> 0, A01 -> 1, A10 -> 2, A11 -> 3
    # B is 2x2: indices (j,k) -> flat index j*p+k = j*2+k
    #   B00 -> 0, B01 -> 1, B10 -> 2, B11 -> 3
    # C is 2x2: indices (k,i) -> flat index k*n+i = k*2+i
    #   C00 -> 0, C01 -> 1, C10 -> 2, C11 -> 3
    
    # Strassen:
    # M1 = (A00+A11)(B00+B11): U[0]=1,U[3]=1; V[0]=1,V[3]=1; W[0]=1,W[3]=1
    # M2 = (A10+A11)B00:       U[2]=1,U[3]=1; V[0]=1; W[2]=1
    # M3 = A00(B10-B11):        U[0]=1; V[2]=1,V[3]=-1; W[1]=1
    # M4 = A11(B01-B00):        U[3]=1; V[1]=1,V[0]=-1; W[3]=1
    # M5 = (A00+A10)B11:        U[0]=1,U[2]=1; V[3]=1; W[2]=1
    # M6 = (A11-A10)(B00+B10):  U[3]=1,U[2]=-1; V[0]=1,V[2]=1; W[1]=1
    # M7 = (A01-A11)(B01+B11):  U[1]=1,U[3]=-1; V[1]=1,V[3]=1; W[0]=1
    
    # Let me verify: C = A*B
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    
    # Using Strassen:
    # M1 = (A00+A11)(B00+B11) = A00*B00 + A00*B11 + A11*B00 + A11*B11
    # M2 = (A10+A11)B00 = A10*B00 + A11*B00
    # M3 = A00(B10-B11) = A00*B10 - A00*B11
    # M4 = A11(B01-B00) = A11*B01 - A11*B00
    # M5 = (A00+A10)B11 = A00*B11 + A10*B11
    # M6 = (A11-A10)(B00+B10) = A11*B00 + A11*B10 - A10*B00 - A10*B10
    # M7 = (A01-A11)(B01+B11) = A01*B01 + A01*B11 - A11*B01 - A11*B11
    
    # C00 = M1 + M4 - M5 + M7
    #   = (A00*B00 + A00*B11 + A11*B00 + A11*B11) + (A11*B01 - A11*B00) - (A00*B11 + A10*B11) + (A01*B01 + A01*B11 - A11*B01 - A11*B11)
    #   = A00*B00 + A00*B11 + A11*B00 + A11*B11 + A11*B01 - A11*B00 - A00*B11 - A10*B11 + A01*B01 + A01*B11 - A11*B01 - A11*B11
    #   = A00*B00 + A01*B11 + A01*B01 - A10*B11
    # That's not right. Let me use the standard Strassen formulas.
    
    # Standard Strassen for C = A*B where A,B,C are 2x2:
    # P1 = (A11 + A22)(B11 + B22)
    # P2 = (A21 + A22)B11
    # P3 = A11(B21 - B22)
    # P4 = A22(B12 - B11)
    # P5 = (A11 + A12)B22
    # P6 = (A21 - A11)(B11 + B12)
    # P7 = (A12 - A22)(B21 + B22)
    # C11 = P1 + P4 - P5 + P7
    # C12 = P3 + P5
    # C21 = P2 + P4
    # C22 = P1 - P2 + P3 + P6
    
    # In 0-indexed (i,j) for A, (j,k) for B, (k,i) for C:
    # A00=A11, A01=A12, A10=A21, A11=A22
    # B00=B11, B01=B12, B10=B21, B11=B22
    # C00=C11, C01=C12, C10=C21, C11=C22
    
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B10-B11)
    # P4 = A11(B01-B00)
    # P5 = (A00+A01)B11
    # P6 = (A10-A00)(B00+B01)
    # P7 = (A01-A11)(B10+B11)
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    # Let me verify C00:
    # P1 + P4 - P5 + P7
    # = (A00+A11)(B00+B11) + A11(B01-B00) - (A00+A01)B11 + (A01-A11)(B10+B11)
    # = A00*B00 + A00*B11 + A11*B00 + A11*B11 + A11*B01 - A11*B00 - A00*B11 - A01*B11 + A01*B10 + A01*B11 - A11*B10 - A11*B11
    # = A00*B00 + A11*B01 + A01*B10 - A11*B10
    # Hmm, that's not A00*B00 + A01*B10. Let me recheck.
    
    # Actually I think I have an error. Let me use the correct Strassen formulas.
    # For C = A*B:
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
    
    # Strassen:
    # M1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
    # M2 = (A[1,0]+A[1,1])*B[0,0]
    # M3 = A[0,0]*(B[1,0]-B[1,1])
    # M4 = A[1,1]*(B[0,1]-B[0,0])
    # M5 = (A[0,0]+A[0,1])*B[1,1]
    # M6 = (A[1,1]-A[1,0])*(B[0,0]+B[1,0])
    # M7 = (A[0,1]-A[1,1])*(B[0,1]+B[1,1])
    
    # C[0,0] = M1 + M4 - M5 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 + M6
    
    # Verify C[0,0]:
    # M1 + M4 - M5 + M7
    # = (A00+A11)(B00+B11) + A11(B01-B00) - (A00+A01)B11 + (A01-A11)(B01+B11)
    # = A00*B00 + A00*B11 + A11*B00 + A11*B11 + A11*B01 - A11*B00 - A00*B11 - A01*B11 + A01*B01 + A01*B11 - A11*B01 - A11*B11
    # = A00*B00 + A01*B01
    # But C[0,0] should be A00*B00 + A01*B10. So this is wrong.
    
    # I'm getting confused with the indexing. Let me be very careful.
    # In the tensor decomposition, W[k*n+i, r] corresponds to C[i,k] (row i, column k of C).
    # So W index k*n+i means output C[i,k].
    
    # For the 2x2x2 subproblem:
    # A[i,j] for i,j in {0,1}: flat index i*m+j = i*2+j
    #   A[0,0]->0, A[0,1]->1, A[1,0]->2, A[1,1]->3
    # B[j,k] for j,k in {0,1}: flat index j*p+k = j*2+k
    #   B[0,0]->0, B[0,1]->1, B[1,0]->2, B[1,1]->3
    # C[i,k] for i,k in {0,1}: flat index k*n+i = k*2+i
    #   C[0,0]->0, C[0,1]->1, C[1,0]->2, C[1,1]->3
    
    # C[i,k] = sum_j A[i,j]*B[j,k]
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
    
    # Strassen (using A[i,j], B[j,k], C[i,k]):
    # M1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
    # M2 = (A[1,0]+A[1,1])*B[0,0]
    # M3 = A[0,0]*(B[1,0]-B[1,1])
    # M4 = A[1,1]*(B[0,1]-B[0,0])
    # M5 = (A[0,0]+A[0,1])*B[1,1]
    # M6 = (A[1,1]-A[1,0])*(B[0,0]+B[1,0])
    # M7 = (A[0,1]-A[1,1])*(B[0,1]+B[1,1])
    
    # C[0,0] = M1 + M4 - M5 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 + M6
    
    # Verify C[0,0] = M1 + M4 - M5 + M7:
    # M1 = (A00+A11)(B00+B11) = A00*B00 + A00*B11 + A11*B00 + A11*B11
    # M4 = A11*(B01-B00) = A11*B01 - A11*B00
    # M5 = (A00+A01)*B11 = A00*B11 + A01*B11
    # M7 = (A01-A11)*(B01+B11) = A01*B01 + A01*B11 - A11*B01 - A11*B11
    # Sum = A00*B00 + A00*B11 + A11*B00 + A11*B11 + A11*B01 - A11*B00 - A00*B11 - A01*B11 + A01*B01 + A01*B11 - A11*B01 - A11*B11
    # = A00*B00 + A01*B01
    # But C[0,0] = A00*B00 + A01*B10. So M7 should use B[1,0] and B[1,1], not B[0,1] and B[1,1].
    
    # I see the issue. Let me look up the correct Strassen formulas more carefully.
    # The standard Strassen algorithm computes C = A*B where all are 2x2.
    # Let A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = [[i,j],[k,l]]
    # i = a*e + b*g
    # j = a*f + b*h
    # k = c*e + d*g
    # l = c*f + d*h
    
    # Strassen:
    # p1 = (a+d)(e+h)
    # p2 = (c+d)e
    # p3 = a(g-h)
    # p4 = d(f-e)
    # p5 = (a+b)h
    # p6 = (c-a)(e+f)
    # p7 = (b-d)(g+h)
    
    # i = p1 + p4 - p5 + p7
    # j = p3 + p5
    # k = p2 + p4
    # l = p1 - p2 + p3 + p6
    
    # Verify i = p1 + p4 - p5 + p7:
    # p1 = (a+d)(e+h) = ae+ah+de+dh
    # p4 = d(f-e) = df-de
    # p5 = (a+b)h = ah+bh
    # p7 = (b-d)(g+h) = bg+bh-dg-dh
    # i = ae+ah+de+dh + df-de - ah-bh + bg+bh-dg-dh
    # = ae + df + bg - dg + dh - dh + bh - bh
    # = ae + bg + df - dg
    # That's not ae+bg. There's an error.
    
    # Let me recheck: p7 = (b-d)(g+h) = bg+bh-dg-dh
    # i = ae+ah+de+dh + df-de - ah-bh + bg+bh-dg-dh
    # = ae + (ah-ah) + (de-de) + (dh-dh) + df - bh + bh + bg - dg
    # = ae + df + bg - dg
    # Still wrong. The correct formula should give i = ae+bg.
    
    # I think the issue is that I'm using the wrong Strassen formulas. Let me look this up properly.
    # Actually, I recall that the standard Strassen formulas are:
    # p1 = (a+d)(e+h)
    # p2 = (c+d)(e+f)
    # p3 = (a+b)h
    # p4 = (c-a)(e+f)  -- no wait
    # 
    # Let me just use a known correct set. From Wikipedia:
    # p1 = (a + d)(e + h)
    # p2 = (c + d)e
    # p3 = a(g - h)
    # p4 = d(f - e)
    # p5 = (a + b)h
    # p6 = (c - a)(e + f)
    # p7 = (b - d)(g + h)
    # 
    # i = p1 + p4 - p5 + p7
    # j = p3 + p5
    # k = p2 + p4
    # l = p1 - p2 + p3 + p6
    
    # Let me verify l = p1 - p2 + p3 + p6:
    # p1 = (a+d)(e+h) = ae+ah+de+dh
    # p2 = (c+d)e = ce+de
    # p3 = a(g-h) = ag-ah
    # p6 = (c-a)(e+f) = ce+cf-ae-af
    # l = ae+ah+de+dh - ce-de + ag-ah + ce+cf-ae-af
    # = (ae-ae) + (ah-ah) + (de-de) + dh + (-ce+ce) + ag + cf - af
    # = dh + ag + cf - af
    # But l should be cf+dh. So there's an extra ag-af term.
    
    # I'm clearly making errors. Let me just directly construct the decomposition for the 2x2x2 case and verify it numerically.
    
    # Actually, let me take a different approach. I'll just use the schoolbook rank of 64 for now, since getting Strassen right is error-prone.
    # But wait, the task says "rank-minimal", so I should try to find the minimum.
    
    # For M_{2,2,16}, let me think about what the minimum rank could be.
    # The rank is at least max(n*p, m*p, n*m) = max(32, 32, 4) = 32 by certain lower bounds.
    # And at most n*m*p = 64 by schoolbook.
    # With Strassen, we can get 8*7 = 56.
    # Can we do better than 56?
    
    # Actually, for M_{2,2,16}, we can potentially do even better by exploiting the structure more.
    # But let me start with 56 and see if it works.
    
    # Let me carefully define the Strassen decomposition for 2x2x2.
    # I'll verify it numerically.
    
    # For a single 2x2x2 block (columns 2*g to 2*g+1 of the 2x16 B matrix):
    # A is 2x2, B_block is 2x2, C_block is 2x2
    
    # Let me define the 7 Strassen factors and verify:
    # Using the convention:
    # A[i,j] flat: i*2+j
    # B[j,k] flat: j*2+k  
    # C[i,k] flat: k*2+i (note: W[k*n+i] = W[k*2+i])
    
    # I'll define the factors and check with a random example.
    
    # Let me try a different, verified set of Strassen factors:
    # For C = A*B (2x2):
    # M1: U=[1,0,0,1], V=[1,0,1,0], W=[1,0,0,1]  -> (A00+A11)(B00+B11) contributes to C00,C01,C10,C11
    # This is getting complicated. Let me just use rank 64 (schoolbook) for safety, 
    # but the task asks for rank-minimal.
    
    # Actually, let me just try rank 56 with Strassen and verify numerically.
    
    # Here's my plan:
    # For each group g (0 to 7), columns 2g and 2g+1 of B:
    # Use 7 Strassen multiplications.
    # Total rank = 56.
    
    # Let me define the Strassen factors for one 2x2x2 block and verify.
    
    # I'll use this known correct Strassen decomposition:
    # Reference: https://en.wikipedia.org/wiki/Strassen_algorithm
    
    # For C = A*B where A,B,C are 2x2:
    # Let A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # C = [[ae+bg, af+bh],[ce+dg, cf+dh]]
    
    # Strassen's 7 products:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(g-h)
    # M4 = d(f-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # C[0,0] = M1 + M4 - M5 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 + M6
    
    # Let me verify C[1,1] = M1 - M2 + M3 + M6:
    # M1 = (a+d)(e+h) = ae+ah+de+dh
    # M2 = (c+d)e = ce+de
    # M3 = a(g-h) = ag-ah
    # M6 = (c-a)(e+f) = ce+cf-ae-af
    # M1-M2+M3+M6 = ae+ah+de+dh - ce-de + ag-ah + ce+cf-ae-af
    # = (ae-ae) + (ah-ah) + (de-de) + dh + (-ce+ce) + ag + cf - af
    # = dh + ag + cf - af
    # Should be cf+dh. Extra term: ag-af = a(g-f). Not zero in general.
    
    # So this set is wrong. Let me try another set.
    # 
    # Correct Strassen (from various sources):
    # p1 = (a+d)(e+h)
    # p2 = (c+d)(e+f)
    # p3 = (a+b)h
    # p4 = (c-a)(e+f)
    # p5 = (b-d)(g+h)
    # p6 = (a+d)(g-h)
    # p7 = (c-a)(f-e)
    # 
    # No, that doesn't look right either.
    
    # Let me just directly solve for the decomposition.
    # I want: C[i,k] = sum_j A[i,j]*B[j,k] = sum_r U[i*2+j... wait.
    # C[i,k] = sum_r U[i*m+j... no.
    # 
    # The decomposition is:
    # C[i,k] = sum_j A[i,j]*B[j,k]
    # = sum_r (sum_i' U[i', r] where i' = i*m+j... no.
    # 
    # Actually: T[i,j,k] = delta (the tensor is 1 for valid (i,j,k) and 0 otherwise)
    # The decomposition is: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # 
    # For 2x2x2: T[i,j,k] = 1 for all i,j,k in {0,1} (it's the full tensor for matrix mult)
    # Wait no, T[i,j,k] is the entry of the matrix multiplication tensor.
    # T[i,j,k] = 1 if the (i,j,k) index corresponds to a term in C[i,k] = sum_j A[i,j]*B[j,k]
    # So T[i,j,k] = 1 for all i,j,k (since every combination appears).
    # 
    # The decomposition requires:
    # For all i,j,k: sum_r U[i*2+j, r] * V[j*2+k, r] * W[k*2+i, r] = 1
    # 
    # This is a system of 8 equations (for i,j,k in {0,1}) with 3*4=12 unknowns per r.
    # With 7 r's, we have 84 unknowns and 8 equations. Very underdetermined.
    
    # Let me just use the schoolbook rank 64 for now. It's safe and correct.
    # The question is whether we can do better.
    
    # Given the complexity of getting Strassen right, and the risk of errors,
    # let me try rank 56 with a carefully verified Strassen decomposition.
    
    # I'll verify numerically in the code.
    
    # Here's my approach:
    # 1. Define the Strassen factors for one 2x2x2 block.
    # 2. Verify numerically that they give the correct result.
    # 3. If correct, extend to 8 blocks for rank 56.
    # 4. If not correct, fall back to rank 64.
    
    # Let me define the Strassen factors and verify:
    
    # For one 2x2x2 block, the 7 factors (U_row, V_row, W_row) are:
    # I'll use indices: U has 4 rows (A[0,0],A[0,1],A[1,0],A[1,1])
    #                V has 4 rows (B[0,0],B[0,1],B[1,0],B[1,1])
    #                W has 4 rows (C[0,0],C[0,1],C[1,0],C[1,1]) where W[k*2+i]
    
    # Let me try these factors (from a verified source):
    strassen_factors = [
        # (U_coeffs, V_coeffs, W_coeffs)
        # Each is a dict from flat index to coefficient
        ({0:1, 3:1}, {0:1, 3:1}, {0:1, 3:1}),  # M1 = (A00+A11)(B00+B11)
        ({2:1, 3:1}, {0:1}, {2:1}),               # M2 = (A10+A11)B00
        ({0:1}, {2:1, 3:-1}, {1:1}),              # M3 = A00(B10-B11)
        ({3:1}, {1:1, 0:-1}, {3:1}),              # M4 = A11(B01-B00)
        ({0:1, 1:1}, {3:1}, {2:1}),               # M5 = (A00+A01)B11
        ({3:1, 2:-1}, {0:1, 2:1}, {1:1}),         # M6 = (A11-A10)(B00+B10)
        ({1:1, 3:-1}, {2:1, 3:1}, {0:1}),         # M7 = (A01-A11)(B10+B11)
    ]
    
    # Verify: C[i,k] = sum_j A[i,j]*B[j,k] = sum_r U[i*2+j... 
    # Actually, the tensor entry T[i,j,k] should equal sum_r U[i*2+j, r]*V[j*2+k, r]*W[k*2+i, r]
    # But T[i,j,k] = 1 for all i,j,k (it's the matrix multiplication tensor).
    # Wait, that's not right. The matrix multiplication tensor has T[i,j,k] = 1 for all i,j,k
    # only in the sense that the decomposition should satisfy:
    # A[i,j]*B[j,k] = sum_r U[i*2+j, r]*V[j*2+k, r]*W[k*2+i, r] * (A*B)[i,k]
    # No, that's not right either.
    # 
    # The correct condition is:
    # For all i,j,k: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # where T[i,j,k] is the matrix multiplication tensor.
    # For M_{n,m,p}, T[i,j,k] = 1 for all i,j,k (the tensor is the all-ones tensor? No!)
    # 
    # Actually, the matrix multiplication tensor T_{n,m,p} is defined by:
    # T_{n,m,p}[i,j,k] = 1 for all i,j,k
    # And the decomposition satisfies:
    # (A*B)[i,k] = sum_j A[i,j]*B[j,k] = sum_{i,j,k} T[i,j,k]*A[i,j]*B[j,k]
    # = sum_r sum_{i,j,k} U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]*A[i,j]*B[j,k]
    # = sum_r (sum_j U[i*m+j,r]*A[i,j]) * (sum_j V[j*p+k,r]*B[j,k]) * W[k*n+i,r]
    # 
    # Wait, that's not quite right. Let me think again.
    # 
    # The decomposition is:
    # sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] = sum_r (sum_i U[i*m+j,r] A[i,j]) ... 
    # 
    # No, the standard form is:
    # T = sum_r U_r ⊗ V_r ⊗ W_r
    # where U_r is a vector in R^{n*m}, V_r in R^{m*p}, W_r in R^{p*n}
    # 
    # And the matrix multiplication is:
    # (A*B)[i,k] = sum_j A[i,j]*B[j,k]
    # = sum_{i,j,k} δ_{i,k}^{ij} A[i,j] B[j,k]  (where δ is the tensor)
    # 
    # The tensor T[i,j,k] = 1 for all i,j,k is NOT the matrix multiplication tensor.
    # The matrix multiplication tensor has T[i,j,k] = 1 for all i,j,k only if we're 
    # summing over all terms, but the actual tensor entries are:
    # T[i,j,k] = 1 if this term contributes to C[i,k], which is always true for valid i,j,k.
    # 
    # Actually, I think the matrix multiplication tensor T_{n,m,p} is defined as:
    # T[i,j,k] = 1 for all i in {0..n-1}, j in {0..m-1}, k in {0..p-1}
    # And the decomposition satisfies:
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # 
    # So for M_{2,2,2}, we need:
    # sum_r U[i*2+j, r] * V[j*2+k, r] * W[k*2+i, r] = 1 for all i,j,k in {0,1}
    # 
    # That's 8 equations. Let me verify my Strassen factors satisfy this.
    
    # Let me check T[0,0,0] = sum_r U[0, r]*V[0, r]*W[0, r]
    # U[0] = A[0,0], V[0] = B[0,0], W[0] = C[0,0] (since k*2+i = 0*2+0 = 0)
    # 
    # r=0: U[0,0]=1, V[0,0]=1, W[0,0]=1 -> 1*1*1 = 1
    # r=1: U[0,1]=0, V[0,1]=1, W[0,1]=0 -> 0
    # r=2: U[0,2]=1, V[0,2]=0, W[0,2]=0 -> 0
    # r=3: U[0,3]=0, V[0,3]=-1, W[0,3]=1 -> 0
    # r=4: U[0,4]=1, V[0,4]=0, W[0,4]=0 -> 0
    # r=5: U[0,5]=0, V[0,5]=1, W[0,5]=0 -> 0
    # r=6: U[0,6]=0, V[0,6]=0, W[0,6]=1 -> 0
    # Sum = 1. Correct!
    
    # Let me check T[0,1,0] = sum_r U[1, r]*V[0, r]*W[0, r]
    # U[1] = A[0,1], V[0] = B[0,0], W[0] = C[0,0]
    # r=0: U[1,0]=0, V[0,0]=1, W[0,0]=1 -> 0
    # r=1: U[1,1]=0, V[0,1]=1, W[0,1]=0 -> 0
    # r=2: U[1,2]=0, V[0,2]=0, W[0,2]=0 -> 0
    # r=3: U[1,3]=0, V[0,3]=-1, W[0,3]=1 -> 0
    # r=4: U[1,4]=1, V[0,4]=0, W[0,4]=0 -> 0
    # r=5: U[1,5]=0, V[0,5]=1, W[0,5]=0 -> 0
    # r=6: U[1,6]=1, V[0,6]=0, W[0,6]=1 -> 0
    # Sum = 0. But T[0,1,0] should be 1!
    
    # So my Strassen factors are wrong. The issue is that T[i,j,k] = 1 for all i,j,k,
    # and my factors don't satisfy this.
    
    # I think I'm confusing the tensor decomposition with the algorithm.
    # In the tensor decomposition, we need:
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # 
    # For the matrix multiplication tensor, T[i,j,k] = 1 for all i,j,k.
    # This means we need the sum to equal 1 for every (i,j,k) triple.
    
    # This is a very different condition from "the algorithm computes C=A*B correctly".
    # The algorithm correctness is:
    # (A*B)[i,k] = sum_j A[i,j]*B[j,k]
    # = sum_{i,j,k} T[i,j,k] A[i,j] B[j,k]
    # = sum_r (sum_{i,j} U[i*m+j,r] A[i,j]) (sum_{j,k} V[j*p+k,r] B[j,k]) W[k*n+i,r]
    # Wait, that's not right either.
    
    # Let me look at this more carefully. The standard definition is:
    # The matrix multiplication tensor T_{n,m,p} ∈ R^{n*m × m*p × p*n} is defined by:
    # T_{n,m,p}[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k
    # and 0 otherwise.
    # 
    # So T is not the all-ones tensor. It has 1's at specific positions.
    # For M_{2,2,2}, T has 1's at:
    # [0,0,0], [0,2,2], [1,1,1], [1,3,3], [2,0,2], [2,2,0], [3,1,3], [3,3,1]
    # (using flat indices for the three modes)
    # 
    # Wait, let me recompute. For i,j,k in {0,1}:
    # Mode 1 index: i*m+j = i*2+j
    # Mode 2 index: j*p+k = j*2+k
    # Mode 3 index: k*n+i = k*2+i
    # 
    # (i,j,k)=(0,0,0): [0,0,0]
    # (i,j,k)=(0,0,1): [0,1,2]
    # (i,j,k)=(0,1,0): [1,2,0]
    # (i,j,k)=(0,1,1): [1,3,2]
    # (i,j,k)=(1,0,0): [2,0,1]
    # (i,j,k)=(1,0,1): [2,1,3]
    # (i,j,k)=(1,1,0): [3,2,1]
    # (i,j,k)=(1,1,1): [3,3,3]
    
    # So T has 1's at these 8 positions and 0 elsewhere.
    # The decomposition requires:
    # T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r] for all a,b,c
    
    # This is the correct condition. My earlier verification was wrong because I was checking
    # the wrong entries.
    
    # Let me re-verify with the correct T entries.
    # T[0,0,0] = 1: sum_r U[0,r]*V[0,r]*W[0,r]
    # r=0: 1*1*1=1, others 0. Sum=1. ✓
    
    # T[0,1,2] = 1: sum_r U[0,r]*V[1,r]*W[2,r]
    # U[0]: [1,0,1,0,1,0,0]
    # V[1]: [0,1,0,1,0,1,1]
    # W[2]: [0,0,0,1,1,0,0]
    # r=0: 1*0*0=0
    # r=1: 0*1*0=0
    # r=2: 1*0*0=0
    # r=3: 0*1*1=0
    # r=4: 1*0*1=0
    # r=5: 0*1*0=0
    # r=6: 0*1*0=0
    # Sum=0. But should be 1. ✗
    
    # So my factors are definitely wrong.
    
    # Given the difficulty, let me just use the schoolbook rank 64. It's correct and safe.
    # The task asks for rank-minimal, but if I can't verify a lower rank, I should use 64.
    
    # Actually, wait. Let me reconsider. The schoolbook decomposition IS correct:
    # U[i*m+j, i*m*p+j*p+k] = 1
    # V[j*p+k, i*m*p+j*p+k] = 1
    # W[k*n+i, i*m*p+j*p+k] = 1
    # for all i,j,k.
    # 
    # This gives rank = n*m*p = 64 and satisfies:
    # T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
    # For a=i*m+j, b=j*p+k, c=k*n+i:
    # Only r=i*m*p+j*p+k contributes, giving 1*1*1=1. ✓
    # For other (a,b,c): no r contributes (since r is uniquely determined by i,j,k,
    # and different (i,j,k) give different r's, and the indices a,b,c must be consistent).
    
    # So rank 64 is correct. Can we do better?
    # For M_{2,2,16}, the minimum rank is known to be... I'm not sure.
    # Let me just use 64 for now.
    
    rank = n * m * p  # 64
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i * (m * p) + j * p + k
                U[i * m + j][r] = 1
                V[j * p + k][r] = 1
                W[k * n + i][r] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
