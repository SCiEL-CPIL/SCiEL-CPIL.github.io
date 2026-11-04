"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
import time
import numpy as np

_T0 = time.time()
_DEADLINE = 140.0

def _verify(U, V, W, tol=1e-7):
    """Verify decomposition: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all i,j,k."""
    n = m = p = 3
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _construct_rank23():
    """
    Construct rank-23 decomposition for 3x3 matrix multiplication.
    """
    n = m = p = 3
    rank = 23
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Block A: 2x2 Winograd for indices {0,1}
    # Term 0
    U[0,0]=1; U[1,0]=1; V[0,0]=1; V[3,0]=1; W[0,0]=1; W[1,0]=1
    # Term 1
    U[0,1]=1; U[3,1]=1; V[0,1]=1; V[3,1]=1; W[0,1]=1; W[1,1]=1
    # Term 2
    U[0,2]=1; U[3,2]=-1; V[0,2]=1; V[3,2]=-1; W[0,2]=1; W[1,2]=-1
    # Term 3
    U[1,3]=1; U[4,3]=1; V[1,3]=1; V[4,3]=1; W[1,3]=1; W[4,3]=1
    # Term 4
    U[1,4]=1; U[4,4]=1; V[1,4]=1; V[4,4]=-1; W[1,4]=1; W[4,4]=-1
    # Term 5
    U[1,5]=1; U[4,5]=-1; V[1,5]=1; V[4,5]=-1; W[1,5]=1; W[4,5]=-1
    # Term 6
    U[0,6]=1; U[4,6]=1; V[0,6]=1; V[4,6]=1; W[0,6]=1; W[4,6]=1
    
    # Block B: 2x2 Winograd for indices {1,2}
    # Term 7
    U[4,7]=1; U[5,7]=1; V[4,7]=1; V[7,7]=1; W[4,7]=1; W[5,7]=1
    # Term 8
    U[4,8]=1; U[7,8]=1; V[4,8]=1; V[7,8]=1; W[4,8]=1; W[5,8]=1
    # Term 9
    U[4,9]=1; U[7,9]=-1; V[4,9]=1; V[7,9]=-1; W[4,9]=1; W[5,9]=-1
    # Term 10
    U[5,10]=1; U[8,10]=1; V[5,10]=1; V[8,10]=1; W[5,10]=1; W[8,10]=1
    # Term 11
    U[5,11]=1; U[8,11]=1; V[5,11]=1; V[8,11]=-1; W[5,11]=1; W[8,11]=-1
    # Term 12
    U[5,12]=1; U[8,12]=-1; V[5,12]=1; V[8,12]=-1; W[5,12]=1; W[8,12]=-1
    # Term 13
    U[4,13]=1; U[8,13]=1; V[4,13]=1; V[8,13]=1; W[4,13]=1; W[8,13]=1
    
    # Terms 14-22: Correction terms
    # The remaining 12 entries are:
    # 002, 012, 020, 021, 022, 102, 120, 200, 201, 202, 210, 220
    
    # We have 9 slots. We can combine some.
    # Let's look at the structure.
    # 002: A00 * B02
    # 102: A10 * B02  -> Combine with 002? (A00+A10) * B02? No, that gives A00*B02 + A10*B02.
    # If we use term 14: (A00+A10) * B02 -> covers 002, 102.
    #   U[0,14]=1; U[3,14]=1
    #   V[2,14]=1
    #   W[2,14]=1  (for i=0)
    #   W[2,14]=1  (for i=1)? No, W index is k*n+i.
    #   For 002: k=2, i=0 -> W[6]
    #   For 102: k=2, i=1 -> W[7]
    #   So W[6,14]=1, W[7,14]=1.
    #   This covers 002 and 102.
    
    # 020: A02 * B20
    # 220: A22 * B20 -> Combine? (A02+A22) * B20?
    #   U[2,15]=1; U[8,15]=1
    #   V[6,15]=1
    #   W[6,15]=1 (for 020: k=2, i=0)
    #   W[6,15]=1 (for 220: k=2, i=2 -> W[8])? Wait.
    #   020: k=2, i=0 -> W[6]
    #   220: k=2, i=2 -> W[8]
    #   So W[6,15]=1, W[8,15]=1.
    #   This covers 020 and 220.
    
    # 012: A01 * B12
    # 112: Covered by Block B? 
    #   Block B covers (1,1,2). Yes, term 10 covers A11*B12 (partially).
    #   Let's check Block B coverage again.
    #   Block B (1,2)x(1,2)x(1,2).
    #   (1,1,2) is in Block B.
    #   So 012 is the only one with A01*B12.
    #   Is there a partner for 012?
    #   212: A21 * B12.
    #   Combine (A01+A21) * B12?
    #   U[1,16]=1; U[7,16]=1
    #   V[5,16]=1
    #   W[5,16]=1 (for 012: k=2, i=0 -> W[6])
    #   W[5,16]=1 (for 212: k=2, i=2 -> W[8])? Wait.
    #   012: k=2, i=0 -> W[6]
    #   212: k=2, i=2 -> W[8]
    #   So W[6,16]=1, W[8,16]=1.
    #   This covers 012 and 212.
    
    # 021: A02 * B21
    # 221: A22 * B21 -> Combine? (A02+A22) * B21?
    #   U[2,17]=1; U[8,17]=1
    #   V[7,17]=1
    #   W[7,17]=1 (for 021: k=1, i=0 -> W[3])
    #   W[7,17]=1 (for 221: k=1, i=2 -> W[7])? Wait.
    #   021: k=1, i=0 -> W[3]
    #   221: k=1, i=2 -> W[7]
    #   So W[3,17]=1, W[7,17]=1.
    #   This covers 021 and 221.
    
    # 120: A12 * B20
    # 220: Already covered by term 15?
    #   Term 15 was (A02+A22)*B20. It covers 020 and 220.
    #   So 120 is A12*B20.
    #   Is there a partner?
    #   020 covered. 220 covered.
    #   Maybe 120 combines with something else?
    #   What about 200? A20*B00.
    #   120: A12*B20.
    #   200: A20*B00.
    #   These don't seem to combine easily.
    
    # Let's list remaining after 14, 15, 16, 17:
    # Covered by 14: 002, 102
    # Covered by 15: 020, 220
    # Covered by 16: 012, 212
    # Covered by 17: 021, 221
    
    # Original list: 002, 012, 020, 021, 022, 102, 120, 200, 201, 202, 210, 220
    # Removed: 002, 012, 020, 021, 102, 220
    # Also removed: 212, 221 (which were NOT in the original list? 
    #   Wait. Block B covers (1,2)x(1,2)x(1,2).
    #   (2,1,2) -> i=2, j=1, k=2.
    #   Is (2,1,2) in Block B? Yes.
    #   So 212 and 221 are covered by Block B?
    #   Block B terms:
    #   7: (1,1,1)
    #   8: (1,1,1)
    #   9: (1,1,1)
    #   10: (1,1,2) -> covers (1,1,2), (1,2,2), (2,1,2), (2,2,2)
    #   11: (1,1,2) -> covers (1,1,2), (1,2,2), (2,1,2), (2,2,2)
    #   12: (1,1,2) -> covers (1,1,2), (1,2,2), (2,1,2), (2,2,2)
    #   13: (1,1,2) -> covers (1,1,2), (1,2,2), (2,1,2), (2,2,2)
    #   
    #   So (2,1,2) [212] and (2,2,1) [221] are NOT in the set of 8 covered by Block B?
    #   Block B covers:
    #   (1,1,1)
    #   (1,1,2), (1,2,2), (2,1,2), (2,2,2) from terms 10-13.
    #   (1,2,1), (2,2,1) from term 7? No.
    #   Let's re-verify Block B coverage.
    #   Term 7: (a11+a12)*(b11+b21) -> covers (1,1,1), (1,1,2), (1,2,1), (1,2,2) ??
    #   Wait, the Winograd basis vectors are:
    #   v1 = e00+e01 (in A) -> (1,1,0,0) in 2x2.
    #   v2 = e00+e10 -> (1,0,1,0)
    #   v3 = e00-e10 -> (1,0,-1,0)
    #   v4 = e01+e11 -> (0,1,0,1)
    #   v5 = e01+e11 -> (0,1,0,1)
    #   v6 = e01-e11 -> (0,1,0,-1)
    #   v7 = e00+e11 -> (1,0,0,1)
    #   
    #   For Block B (indices 1,2):
    #   Term 7: (A11+A12)*(B11+B21).
    #   A part: A11, A12. (Rows 1,2; Col 1,2 of A).
    #   B part: B11, B21. (Rows 1,2; Col 1,2 of B).
    #   Products: A11*B11, A11*B21, A12*B11, A12*B21.
    #   Output indices (i,j,k):
    #   A11*B11 -> (1,1,1)
    #   A11*B21 -> (1,2,1)
    #   A12*B11 -> (2,1,1)
    #   A12*B21 -> (2,2,1)
    #   
    #   Term 8: (A11+A21)*(B11+B21).
    #   A11, A21.
    #   A11*B11 -> (1,1,1)
    #   A11*B21 -> (1,2,1)
    #   A21*B11 -> (2,1,1)
    #   A21*B21 -> (2,2,1)
    #   
    #   Term 9: (A11-A21)*(B11-B21).
    #   A11*B11 -> (1,1,1)
    #   -A11*B21 -> -(1,2,1)
    #   -A21*B11 -> -(2,1,1)
    #   A21*B21 -> (2,2,1)
    #   
    #   Term 10: (A12+A22)*(B12+B22).
    #   A12, A22.
    #   A12*B12 -> (1,1,2)
    #   A12*B22 -> (1,2,2)
    #   A22*B12 -> (2,1,2)
    #   A22*B22 -> (2,2,2)
    #   
    #   Term 11: (A12+A22)*(B12-B22).
    #   A12*B12 -> (1,1,2)
    #   -A12*B22 -> -(1,2,2)
    #   A22*B12 -> (2,1,2)
    #   -A22*B22 -> -(2,2,2)
    #   
    #   Term 12: (A12-A22)*(B12-B22).
    #   A12*B12 -> (1,1,2)
    #   -A12*B22 -> -(1,2,2)
    #   -A22*B12 -> -(2,1,2)
    #   A22*B22 -> (2,2,2)
    #   
    #   Term 13: (A11+A22)*(B11+B22).
    #   A11*B11 -> (1,1,1)
    #   A11*B22 -> (1,2,2)
    #   A22*B11 -> (2,1,1)
    #   A22*B22 -> (2,2,2)
    #   
    #   Sum of coefficients for each (i,j,k) in Block B:
    #   (1,1,1): 1+1+1+1 = 4? No, we need 1.
    #   Wait, the Winograd decomposition sums to 1 for each entry in the 2x2 block.
    #   Let's check (1,1,1).
    #   Term 7: +1
    #   Term 8: +1
    #   Term 9: +1
    #   Term 13: +1
    #   Sum = 4. This is wrong.
    #   
    #   The Winograd algorithm for 2x2x2 uses 7 terms to produce the 8 entries.
    #   The output is a 2x2 matrix.
    #   The terms are:
    #   m1 = (a00+a01)*(b00+b10)
    #   m2 = (a00+a10)*(b00+b10)
    #   m3 = (a00-a10)*(b00-b10)
    #   m4 = (a01+a11)*(b01+b11)
    #   m5 = (a01+a11)*(b01-b11)
    #   m6 = (a01-a11)*(b01-b11)
    #   m7 = (a00+a11)*(b00+b11)
    #   
    #   c00 = m1 - m2 + m3 + m7 ?
    #   c01 = m1 - m4 + m5 + m6 ?
    #   c10 = m2 + m3 - m7 ?
    #   c11 = m4 + m5 - m6 + m7 ?
    #   
    #   Let's check c00 (1,1,1 in our notation if 1->0, 2->1).
    #   m1: a00*b00 -> +1
    #   m2: a00*b00 -> -1
    #   m3: a00*b00 -> +1
    #   m7: a00*b00 -> +1
    #   Sum = 1 - 1 + 1 + 1 = 2. Still wrong.
    #   
    #   Actually, the coefficients in W are not just +1/-1.
    #   In the standard Winograd 2x2:
    #   c00 = m1 - m2 + m3 + m7
    #   c01 = m1 - m4 + m5 + m6
    #   c10 = m2 + m3 - m7
    #   c11 = m4 + m5 - m6 + m7
    #   
    #   So for (1,1,1) [local 0,0,0]:
    #   Term 7 (m1): W[0,0] = 1
    #   Term 8 (m2): W[0,1] = -1
    #   Term 9 (m3): W[0,2] = 1
    #   Term 13 (m7): W[0,6] = 1
    #   Sum = 2.
    #   
    #   Wait, the standard Winograd is for real numbers.
    #   c00 = (a00+a01)(b00+b10) - (a00+a10)(b00+b10) + (a00-a10)(b00-b10) + (a00+a11)(b00+b11)
    #   = a00(b00+b10-b00-b10+b00-b10+b00+b11) ...
    #   = a00(2b00 - b10 + b11) ...
    #   This is not 1.
    #   
    #   The correct combination for c00 = a00*b00:
    #   1/4 * [ (a00+a01)(b00+b10) - (a00-a01)(b00+b10) ... ]
    #   
    #   Actually, the rank of 2x2x2 tensor is 7.
    #   The decomposition is:
    #   T = 1/4 [
    #   (e0+e1)_A (e0+e1)_B (e0+e1)_C  # 000, 001, 010, 011
    #   (e0+e1)_A (e0-e1)_B (e0-e1)_C  # 000, -001, -010, 011
    #   ...
    #   ]
    #   
    #   With 1/4 factors, the sum is 1.
    #   So W coefficients should be 1/4, -1/4, etc.
    
    # Let's use 0.25 for all coefficients in the Winograd blocks.
    
    # Block A (0,1)
    # Term 0 (m1): U[0,0]=1, U[1,0]=1; V[0,0]=1, V[3,0]=1; W[0,0]=0.25, W[1,0]=0.25, W[3,0]=0.25, W[4,0]=0.25?
    #   No, W is for output.
    #   m1 covers 000, 001, 010, 011.
    #   W[0,0]=0.25 (000)
    #   W[1,0]=0.25 (001)
    #   W[3,0]=0.25 (010)
    #   W[4,0]=0.25 (011)
    
    # Term 1 (m2): (a00+a10)(b00+b10)
    #   Covers 000, 001, 100, 101.
    #   W[0,1]=-0.25
    #   W[1,1]=-0.25
    #   W[3,1]=-0.25
    #   W[4,1]=-0.25
    
    # Term 2 (m3): (a00-a10)(b00-b10)
    #   Covers 000, -001, -100, 101.
    #   W[0,2]=0.25
    #   W[1,2]=-0.25
    #   W[3,2]=-0.25
    #   W[4,2]=0.25
    
    # Term 3 (m4): (a01+a11)(b01+b11)
    #   Covers 001, 011, 101, 111.
    #   W[1,3]=0.25
    #   W[4,3]=0.25
    #   W[7,3]=0.25
    #   W[8,3]=0.25  (111 -> W[8])
    
    # Term 4 (m5): (a01+a11)(b01-b11)
    #   Covers 001, -011, 101, -111.
    #   W[1,4]=0.25
    #   W[4,4]=-0.25
    #   W[7,4]=0.25
    #   W[8,4]=-0.25
    
    # Term 5 (m6): (a01-a11)(b01-b11)
    #   Covers 001, -011, -101, 111.
    #   W[1,5]=0.25
    #   W[4,5]=-0.25
    #   W[7,5]=-0.25
    #   W[8,5]=0.25
    
    # Term 6 (m7): (a00+a11)(b00+b11)
    #   Covers 000, 011, 100, 111.
    #   W[0,6]=0.25
    #   W[4,6]=0.25
    #   W[3,6]=0.25
    #   W[8,6]=0.25
    
    # Let's check 000:
    #   T0: 0.25
    #   T1: -0.25
    #   T2: 0.25
    #   T6: 0.25
    #   Sum = 0.5. Still not 1.
    
    # The factor is 1/4?
    # If W is 1, then sum is 2.
    # If W is 1/2, then sum is 1.
    # Let's try 0.5.
    
    # Reset
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Block A
    # T0
    U[0,0]=1; U[1,0]=1; V[0,0]=1; V[3,0]=1; W[0,0]=0.5; W[1,0]=0.5; W[3,0]=0.5; W[4,0]=0.5
    # T1
    U[0,1]=1; U[3,1]=1; V[0,1]=1; V[3,1]=1; W[0,1]=-0.5; W[1,1]=-0.5; W[3,1]=-0.5; W[4,1]=-0.5
    # T2
    U[0,2]=1; U[3,2]=-1; V[0,2]=1; V[3,2]=-1; W[0,2]=0.5; W[1,2]=-0.5; W[3,2]=-0.5; W[4,2]=0.5
    # T3
    U[1,3]=1; U[4,3]=1; V[1,3]=1; V[4,3]=1; W[1,3]=0.5; W[4,3]=0.5; W[7,3]=0.5; W[8,3]=0.5
    # T4
    U[1,4]=1; U[4,4]=1; V[1,4]=1; V[4,4]=-1; W[1,4]=0.5; W[4,4]=-0.5; W[7,4]=0.5; W[8,4]=-0.5
    # T5
    U[1,5]=1; U[4,5]=-1; V[1,5]=1; V[4,5]=-1; W[1,5]=0.5; W[4,5]=-0.5; W[7,5]=-0.5; W[8,5]=0.5
    # T6
    U[0,6]=1; U[4,6]=1; V[0,6]=1; V[4,6]=1; W[0,6]=0.5; W[4,6]=0.5; W[3,6]=0.5; W[8,6]=0.5
    
    # Block B (indices 1,2)
    # T7
    U[4,7]=1; U[5,7]=1; V[4,7]=1; V[7,7]=1; W[4,7]=0.5; W[5,7]=0.5; W[7,7]=0.5; W[8,7]=0.5
    # T8
    U[4,8]=1; U[7,8]=1; V[4,8]=1; V[7,8]=1; W[4,8]=-0.5; W[5,8]=-0.5; W[7,8]=-0.5; W[8,8]=-0.5
    # T9
    U[4,9]=1; U[7,9]=-1; V[4,9]=1; V[7,9]=-1; W[4,9]=0.5; W[5,9]=-0.5; W[7,9]=-0.5; W[8,9]=0.5
    # T10
    U[5,10]=1; U[8,10]=1; V[5,10]=1; V[8,10]=1; W[5,10]=0.5; W[8,10]=0.5; W[7,10]=0.5; W[8,10]=0.5
    # T11
    U[5,11]=1; U[8,11]=1; V[5,11]=1; V[8,11]=-1; W[5,11]=0.5; W[8,11]=-0.5; W[7,11]=0.5; W[8,11]=-0.5
    # T12
    U[5,12]=1; U[8,12]=-1; V[5,12]=1; V[8,12]=-1; W[5,12]=0.5; W[8,12]=-0.5; W[7,12]=-0.5; W[8,12]=0.5
    # T13
    U[4,13]=1; U[8,13]=1; V[4,13]=1; V[8,13]=1; W[4,13]=0.5; W[8,13]=0.5; W[7,13]=0.5; W[8,13]=0.5
    
    # Verify Block A
    if _verify(U, V, W):
        return U.tolist(), V.tolist(), W.tolist()
    
    # If it fails, it's likely the corrections are missing or wrong.
    # Given the complexity and risk, and that rank 26 is a safe improvement over 27,
    # I will return None to trigger the rank 26 fallback.
    return None

def _construct_rank26():
    """
    Construct rank-26 decomposition for 3x3 matrix multiplication.
    Two 2x2 Winograd blocks + 12 identity corrections.
    """
    n = m = p = 3
    rank = 26
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Block A (0,1)
    # T0
    U[0,0]=1; U[1,0]=1; V[0,0]=1; V[3,0]=1; W[0,0]=0.5; W[1,0]=0.5; W[3,0]=0.5; W[4,0]=0.5
    # T1
    U[0,1]=1; U[3,1]=1; V[0,1]=1; V[3,1]=1; W[0,1]=-0.5; W[1,1]=-0.5; W[3,1]=-0.5; W[4,1]=-0.5
    # T2
    U[0,2]=1; U[3,2]=-1; V[0,2]=1; V[3,2]=-1; W[0,2]=0.5; W[1,2]=-0.5; W[3,2]=-0.5; W[4,2]=0.5
    # T3
    U[1,3]=1; U[4,3]=1; V[1,3]=1; V[4,3]=1; W[1,3]=0.5; W[4,3]=0.5; W[7,3]=0.5; W[8,3]=0.5
    # T4
    U[1,4]=1; U[4,4]=1; V[1,4]=1; V[4,4]=-1; W[1,4]=0.5; W[4,4]=-0.5; W[7,4]=0.5; W[8,4]=-0.5
    # T5
    U[1,5]=1; U[4,5]=-1; V[1,5]=1; V[4,5]=-1; W[1,5]=0.5; W[4,5]=-0.5; W[7,5]=-0.5; W[8,5]=0.5
    # T6
    U[0,6]=1; U[4,6]=1; V[0,6]=1; V[4,6]=1; W[0,6]=0.5; W[4,6]=0.5; W[3,6]=0.5; W[8,6]=0.5
    
    # Block B (1,2)
    # T7
    U[4,7]=1; U[5,7]=1; V[4,7]=1; V[7,7]=1; W[4,7]=0.5; W[5,7]=0.5; W[7,7]=0.5; W[8,7]=0.5
    # T8
    U[4,8]=1; U[7,8]=1; V[4,8]=1; V[7,8]=1; W[4,8]=-0.5; W[5,8]=-0.5; W[7,8]=-0.5; W[8,8]=-0.5
    # T9
    U[4,9]=1; U[7,9]=-1; V[4,9]=1; V[7,9]=-1; W[4,9]=0.5; W[5,9]=-0.5; W[7,9]=-0.5; W[8,9]=0.5
    # T10
    U[5,10]=1; U[8,10]=1; V[5,10]=1; V[8,10]=1; W[5,10]=0.5; W[8,10]=0.5; W[7,10]=0.5; W[8,10]=0.5
    # T11
    U[5,11]=1; U[8,11]=1; V[5,11]=1; V[8,11]=-1; W[5,11]=0.5; W[8,11]=-0.5; W[7,11]=0.5; W[8,11]=-0.5
    # T12
    U[5,12]=1; U[8,12]=-1; V[5,12]=1; V[8,12]=-1; W[5,12]=0.5; W[8,12]=-0.5; W[7,12]=-0.5; W[8,12]=0.5
    # T13
    U[4,13]=1; U[8,13]=1; V[4,13]=1; V[8,13]=1; W[4,13]=0.5; W[8,13]=0.5; W[7,13]=0.5; W[8,13]=0.5
    
    # Corrections (14-25)
    missing = [
        (0,0,2), (0,1,2), (0,2,0), (0,2,1), (0,2,2),
        (1,0,2), (1,2,0), (2,0,0), (2,0,1), (2,0,2), (2,1,0), (2,2,0)
    ]
    
    for idx, (i,j,k) in enumerate(missing):
        col = 14 + idx
        U[i*m+j, col] = 1
        V[j*p+k, col] = 1
        W[k*n+i, col] = 1
    
    if _verify(U, V, W):
        return U.tolist(), V.tolist(), W.tolist()
    
    return None

def _schoolbook():
    """Return the schoolbook rank-27 decomposition."""
    n = m = p = 3
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
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    
    if time.time() - _T0 < _DEADLINE - 10:
        # Try rank 23
        result = _construct_rank23()
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
        
        # Try rank 26
        result = _construct_rank26()
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook rank-27
    U, V, W = _schoolbook()
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
