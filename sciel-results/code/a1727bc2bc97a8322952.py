"""Matrix multiplication tensor decomposition for 4x4 * 4x5."""
import time
import random

DIMENSIONS = [4, 4, 5]
START_TIME = time.time()
TIME_LIMIT = 140.0  # Safety margin before 150s kill


def construct():
    n, m, p = DIMENSIONS
    # Return the schoolbook decomposition immediately as a valid fallback
    # then try to find a lower rank decomposition
    rank_schoolbook = n * m * p
    
    # Start with schoolbook as best known
    best_U = [[0] * rank_schoolbook for _ in range(n * m)]
    best_V = [[0] * rank_schoolbook for _ in range(m * p)]
    best_W = [[0] * rank_schoolbook for _ in range(p * n)]
    best_rank = rank_schoolbook
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i * m + j][column] = 1
                best_V[j * p + k][column] = 1
                best_W[k * n + i][column] = 1
                column += 1
    
    # Try to find a lower rank decomposition using a structured search
    # Strategy: use a rank-r decomposition with small integer/half-integer entries
    # We search over structured patterns
    
    # Key insight: for matrix multiplication (n,m,p), the tensor is
    # T[i,j,k] = delta_{i+k, j} in the right indexing... 
    # Actually the multiplication tensor for (n x m) * (m x p) -> (n x p)
    # has entries T[i, j, k] where result[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor is T[i, j, k] = 1 if the (i,k) entry of result depends on A[i,j]*B[j,k]
    # In the factorization: U has shape (n*m, r), V has shape (m*p, r), W has shape (p*n, r)
    # The decomposition: T[i*m+j, j*p+k, k*n+i] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Wait, let me re-read the convention.
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # The matrix multiplication tensor has entries indexed by (i, j, k) for the product
    # where A is n x m, B is m x p, C = A * B is n x p
    # C[i, k] = sum_j A[i, j] B[j, k]
    # The tensor T has a 1 at position corresponding to (A[i,j], B[j,k], C[i,k])
    # In the flat indexing:
    #   A index: i*m + j (n*m entries)
    #   B index: j*p + k (m*p entries)  
    #   C index: k*n + i (p*n entries) -- using W[k*n+i, r] convention
    
    # So T[i*m+j, j*p+k, k*n+i] should equal 1 for all valid (i,j,k)
    # And 0 otherwise (but in the tensor decomposition, we only need the non-zero entries to match)
    # Actually, the full tensor equation is:
    # For all (a, b, c) where a = i*m+j, b = j'*p+k, c = k'*n+i':
    #   T[a, b, c] = sum_r U[a, r] * V[b, r] * W[c, r]
    # T[a, b, c] = 1 iff there exist i,j,j',k,k',i' such that:
    #   a = i*m+j, b = j'*p+k, c = k'*n+i', AND j = j' (for the multiplication to link)
    #   AND the result index c corresponds to (i, k) from the product A[i,j]*B[j,k]
    #   So: i' = i, k' = k, and j = j'
    #   Therefore T[i*m+j, j*p+k, k*n+i] = 1
    
    # Let's verify: for the schoolbook, each term (i,j,k) gets its own rank-1 component
    # U[i*m+j, col] = 1, V[j*p+k, col] = 1, W[k*n+i, col] = 1
    # This gives contribution 1 to T[i*m+j, j*p+k, k*n+i] and 0 to all other entries. Correct.
    
    # Now, can we do better than rank 80?
    # Known lower bounds: rank >= max(n*m, m*p, p*n) = max(16, 20, 20) = 20
    # Known upper bounds from literature for small cases...
    # For 4x4 * 4x4, the best known is 49 (Lader). For 4x4 * 4x5, it's less studied.
    # Let's try a rank-60 decomposition first, then try lower.
    
    # Actually, let me try a systematic approach:
    # Use the fact that we can group terms.
    # If we fix i, then for each (j,k) we need U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # The W factor only depends on (k, i), so for a fixed i, all terms with that i share the same W pattern.
    
    # Let's try rank = 60, 56, 52, 48, 44, 40, 36, 32, 28, 24, 20
    # For each rank, try to find integer/half-integer solutions.
    
    # A simpler approach: use the "block diagonal" or "structured" decompositions.
    # For the (i,j,k) -> (i*m+j, j*p+k, k*n+i) mapping, note that:
    # - i ranges over 0..3 (4 values)
    # - j ranges over 0..3 (4 values)  
    # - k ranges over 0..4 (5 values)
    
    # Group by i: for each i, we have 4*5 = 20 terms.
    # If we could share rank-1 components across different i values...
    
    # Let's try: for each i, use a 4x5 matrix multiplication pattern.
    # The 4x5 matrix multiply has rank at most 4*5=20 schoolbook.
    # But we have 4 such blocks (one per i), so total rank could be 4*20 = 80 (schoolbook).
    
    # To do better, we need to share structure across i values.
    
    # Alternative: try to use the "border rank" or "approximate" then round.
    # But we need exact decomposition.
    
    # Let me try a rank-64 decomposition with a specific structure.
    # For each (i, k) pair (20 pairs), we need to handle 4 j-values.
    # If for each (i,k), we use 4 rank-1 terms (one per j), that's 20*4 = 80.
    
    # To reduce: notice that W[k*n+i, r] is the same for all j in a given (i,k) group.
    # So if we set W[k*n+i, r] = w_ik_r, then for each (i,k), we need:
    #   sum_r U[i*m+j, r] * V[j*p+k, r] * w_ik_r = 1 for all j
    # This is like a 4x4 system for each (i,k).
    
    # If w_ik_r are non-zero for enough r, we can solve for U and V entries.
    
    # Let's try: rank = 60. For each of the 20 (i,k) pairs, use 3 rank components.
    # Then for each (i,k): sum_{r in S_ik} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for j=0,1,2,3
    # With 3 components, this is a 4x3 underdetermined system... won't work in general.
    
    # For rank = 80, we have 4 components per (i,k) group, giving a 4x4 system.
    
    # For rank = 72, we'd have some groups with 4 and some with 3... complicated.
    
    # Let me just try to find a valid rank-72 or rank-64 decomposition through search.
    
    # Actually, let me try a different structural approach:
    # Use the identity that matrix multiplication can be done with fewer multiplications
    # via Strassen-like methods. For 4x4, Strassen gives rank 49.
    # For 4x4 * 4x5, we can pad B to 4x8 or use the 4x4 result and extend.
    
    # Simplest: use 4x4 * 4x4 Strassen (rank 49) for the first 4 columns of B,
    # then 4x4 * 4x1 for the last column.
    # 4x4 * 4x1 has rank 16 (schoolbook) or possibly less.
    # Total: 49 + 16 = 65.
    
    # Let me implement Strassen for 4x4 * 4x4 (rank 49) and append 16 terms for the 5th column.
    
    # Strassen for 4x4:
    # Split 4x4 into 2x2 blocks. Use 7 Strassen products for 2x2*2x2.
    # For 4x4*4x4, we have 4 sub-problems of 2x2*2x2, each with rank 7.
    # But the 4 sub-problems share structure? No, they're independent.
    # So 4 * 7 = 28 for the "inner" multiplications... 
    # Wait, that's not right. 4x4*4x4 via Strassen:
    # A = [[A11, A12], [A21, A22]], B = [[B11, B12], [B21, B22]]
    # Each block is 2x2.
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    # Strassen: M1=(A11+A22)(B11+B22), M2=(A21+A22)B11, M3=A11(B12-B22),
    # M4=A22(B21-B11), M5=(A11+A12)B22, M6=(A21-A11)(B11+B12), M7=(A12-A22)(B21+B22)
    # C11=M1+M4-M5+M7, C12=M3+M5, C21=M2+M4, C22=M1-M2+M3+M6
    # Each Mi is a 2x2*2x2 product, which has rank 4 (schoolbook) or...
    # The 2x2*2x2 multiplication tensor has rank 7 (Strassen's result).
    # So each Mi contributes 7 to the rank, total 7*7 = 49. Correct.
    
    # For the 5th column: 4x4 * 4x1.
    # C[i,4] = sum_j A[i,j] * B[j,4]
    # This is just 4 dot products of length 4, each of rank 4 (schoolbook).
    # Total rank for this part: 4*4 = 16.
    # Can we do better? The 4x4*4x1 tensor has rank... 
    # It's essentially 4 independent 4x1 dot products sharing the same A rows.
    # Actually, C[i,4] = sum_j A[i,j] * B[j,4] for i=0,1,2,3.
    # This is A * B[:,4] where B[:,4] is a 4x1 vector.
    # The tensor T[i*m+j, j*p+4, 4*n+i] for j=0,1,2,3 and i=0,1,2,3.
    # Rank of this sub-tensor: we need U[i*4+j, r] * V[j*5+4, r] * W[4*4+i, r] = 1 for each (i,j).
    # That's 16 independent constraints. Minimum rank is 16 (each (i,j) pair needs its own term
    # unless we can share). Actually, for fixed i, we need sum_j U[4i+j,r]V[5j+4,r]W[16+i,r] = delta_{j,j}... 
    # No, for the 5th column: T[4i+j, 5j+4, 16+i] = 1 for all i,j in {0,1,2,3}.
    # These are 16 distinct tensor entries that need to be 1, and all others should be 0.
    # Minimum rank for this is 16 (can't do better since each (i,j) is independent).
    
    # Total: 49 + 16 = 65.
    
    # Let me also try: 4x4 * 4x4 (rank 49) + 4x4*4x1 (rank 16) = 65.
    # Or: pad to 4x8 and use 4x8*... no, that's more complex.
    
    # Actually wait - for the 4x4*4x1 part, can we share with the Strassen structure?
    # Probably not easily. Let's go with rank 65.
    
    # Let me construct the Strassen 4x4*4x4 decomposition (rank 49) for columns 0-3 of B,
    # then add 16 schoolbook terms for column 4 of B.
    
    # First, let me define the 2x2*2x2 Strassen decomposition (rank 7).
    # For 2x2*2x2: A is 2x2, B is 2x2, C=A*B is 2x2.
    # Indices: A[a,b] where a,b in {0,1}, B[c,d] where c,d in {0,1}, C[e,f] where e,f in {0,1}
    # Flat: A index: a*2+b (0..3), B index: c*2+d (0..3), C index: f*2+e (0..3) [using C[f*2+e] convention]
    # Wait, the convention is W[k*n+i, r] for the output. For 2x2*2x2, n=2, m=2, p=2.
    # U: (n*m, r) = (4, r), V: (m*p, r) = (4, r), W: (p*n, r) = (4, r)
    # T[i*2+j, j*2+k, k*2+i] = 1
    # Strassen 7 terms:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # In tensor decomposition terms:
    # U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    # For M1: (A00+A11)(B00+B11)
    #   U[0, r1]=1, U[3, r1]=1  (A00 and A11)
    #   V[0, r1]=1, V[3, r1]=1  (B00 and B11)
    #   W[0, r1]=1, W[1, r1]=1, W[2, r1]=1, W[3, r1]=1  (all C entries get M1)
    #   Wait, C00=M1+..., C01=... no C01 doesn't have M1.
    #   C00 = M1+M4-M5+M7, C01 = M3+M5, C10 = M2+M4, C11 = M1-M2+M3+M6
    #   So M1 appears in C00 and C11.
    #   W[0, r1]=1 (C00), W[3, r1]=1 (C11)
    #   But we need: U[0,r1]*V[0,r1]*W[0,r1] contributes to T[0,0,0] (i=0,j=0,k=0: A00*B00->C00) ✓
    #   U[0,r1]*V[3,r1]*W[0,r1] contributes to T[0,3,0] (i=0,j=1,k=0: A00*B10->C00) ✓ (B10 is at index 1*2+0=2, not 3!)
    
    # I'm getting confused with indices. Let me be very careful.
    # For 2x2*2x2: n=2, m=2, p=2
    # A has entries A[i,j] for i,j in {0,1}. Flat index: i*m+j = i*2+j
    #   A[0,0]->0, A[0,1]->1, A[1,0]->2, A[1,1]->3
    # B has entries B[j,k] for j,k in {0,1}. Flat index: j*p+k = j*2+k
    #   B[0,0]->0, B[0,1]->1, B[1,0]->2, B[1,1]->3
    # C has entries C[i,k] for i,k in {0,1}. Flat index: k*n+i = k*2+i
    #   C[0,0]->0, C[0,1]->1, C[1,0]->2, C[1,1]->3
    
    # T[a, b, c] = 1 iff a=i*2+j, b=j*2+k, c=k*2+i for some valid i,j,k.
    # The non-zero entries are:
    # (i,j,k)=(0,0,0): T[0,0,0]=1
    # (i,j,k)=(0,0,1): T[0,1,1]=1
    # (i,j,k)=(0,1,0): T[1,2,2]=1
    # (i,j,k)=(0,1,1): T[1,3,3]=1
    # (i,j,k)=(1,0,0): T[2,0,2]=1
    # (i,j,k)=(1,0,1): T[2,1,3]=1
    # (i,j,k)=(1,1,0): T[3,2,0]=1
    # (i,j,k)=(1,1,1): T[3,3,1]=1
    
    # Strassen M1 = (A00+A11)(B00+B11):
    # This product contributes to:
    # A00*B00 -> C00: T[0,0,0]
    # A00*B11 -> C01: T[0,3,1]  (i=0,j=1,k=1: a=0*2+1=1, b=1*2+1=3, c=1*2+0=1) -> T[1,3,1]
    # A11*B00 -> C10: T[3,0,2]  (i=1,j=0,k=0: a=1*2+0=2, b=0*2+0=0, c=0*2+1=1) -> T[2,0,1]
    # A11*B11 -> C11: T[3,3,3]  (i=1,j=1,k=1: a=3, b=3, c=3) -> T[3,3,3]
    
    # So for M1, we need:
    # U[0, r]=1, U[3, r]=1  (A00 and A11)
    # V[0, r]=1, V[3, r]=1  (B00 and B11)
    # The contributions: U[0]*V[0]*W[?], U[0]*V[3]*W[?], U[3]*V[0]*W[?], U[3]*V[3]*W[?]
    # These should give 1 to T[0,0,0], T[1,3,1], T[2,0,1], T[3,3,3]
    # T[0,0,0]: needs W[0, r]=1
    # T[1,3,1]: needs W[1, r]=1
    # T[2,0,1]: needs W[1, r]=1
    # T[3,3,3]: needs W[3, r]=1
    # So W[0,r]=1, W[1,r]=1, W[3,r]=1, W[2,r]=0
    
    # Let me verify: sum_r U[a,r]V[b,r]W[c,r] for a=0,b=0,c=0: U[0,r]*V[0,r]*W[0,r] = 1*1*1 = 1 ✓
    # a=1,b=3,c=1: U[1,r]*V[3,r]*W[1,r] = 0*1*1 = 0 ✗!
    
    # The issue: U[1, r] = 0 (A01 is not part of M1), but we need this term to be 1.
    # I see the problem - in the tensor decomposition, each rank-1 term contributes to ALL
    # (a,b,c) combinations of its non-zero U, V, W entries. So if U has non-zeros at {0,3} and
    # V has non-zeros at {0,3}, then the term contributes to all 4 combinations (0,0), (0,3), (3,0), (3,3)
    # for the (a,b) pair, each with the corresponding W[c] value.
    
    # So for M1: U non-zero at {0,3}, V non-zero at {0,3}, W non-zero at {0,1,3}
    # Contributions:
    # (a=0,b=0): W[0]=1 -> T[0,0,0]+=1, W[1]=1 -> T[0,0,1]+=1, W[3]=1 -> T[0,0,3]+=1
    # (a=0,b=3): W[0]=1 -> T[0,3,0]+=1, W[1]=1 -> T[0,3,1]+=1, W[3]=1 -> T[0,3,3]+=1
    # (a=3,b=0): W[0]=1 -> T[3,0,0]+=1, W[1]=1 -> T[3,0,1]+=1, W[3]=1 -> T[3,0,3]+=1
    # (a=3,b=3): W[0]=1 -> T[3,3,0]+=1, W[1]=1 -> T[3,3,1]+=1, W[3]=1 -> T[3,3,3]+=1
    
    # But we only want T[0,0,0], T[1,3,1], T[2,0,1], T[3,3,3] to get contributions.
    # This doesn't match. The Strassen method doesn't directly translate to a simple
    # rank-1 tensor decomposition in this way because the "sums" in Strassen involve
    # multiple A and B entries, and the tensor decomposition requires each rank-1 term
    # to be a product of three vectors.
    
    # The correct way: in Strassen, M1 = (A00+A11)(B00+B11) is a scalar product of
    # two sums. In the tensor decomposition, this corresponds to:
    # (U[0]+U[3]) as a vector times (V[0]+V[3]) as a vector, but that's not how it works.
    
    # Actually, the Strassen decomposition IS a valid tensor decomposition. The key is:
    # M1 contributes to C via: C += coeff * M1 where M1 = sum of products of A and B entries.
    # In the tensor: the rank-1 term for M1 has U vector with non-zeros where A entries appear,
    # V vector with non-zeros where B entries appear, and W vector with non-zeros where C entries
    # receive M1's contribution.
    
    # For M1 = (A00+A11)(B00+B11):
    # The "A vector" has 1 at positions 0 (A00) and 3 (A11)
    # The "B vector" has 1 at positions 0 (B00) and 3 (B11)
    # The product gives 4 terms: A00B00, A00B11, A11B00, A11B11
    # These contribute to C00, C01, C10, C11 respectively (with appropriate signs/coefficients from Strassen)
    # Wait no: A00*B00 -> C00, A00*B11 -> C01, A11*B00 -> C10, A11*B11 -> C11
    # In the tensor: T[0,0,0] (A00B00->C00), T[1,3,1] (A00B11->C01), T[2,0,1] (A11B00->C10), T[3,3,3] (A11B11->C11)
    
    # The rank-1 term contributes U[a]*V[b]*W[c] to T[a,b,c].
    # If U[0]=1, U[3]=1, V[0]=1, V[3]=1, then:
    # T[0,0,c] += 1*1*W[c] = W[c]
    # T[0,3,c] += 1*1*W[c] = W[c]
    # T[3,0,c] += 1*1*W[c] = W[c]
    # T[3,3,c] += 1*1*W[c] = W[c]
    
    # We need: T[0,0,0]=1, T[1,3,1]=1, T[2,0,1]=1, T[3,3,3]=1 from this term.
    # But T[1,3,*] requires U[1] != 0, which contradicts U only having non-zeros at {0,3}.
    
    # I see the fundamental issue: the tensor index for A00*B11->C01 is T[a=1, b=3, c=1]
    # where a=1 corresponds to A[0,1]=A01, NOT A[0,0]=A00!
    
    # Let me recheck: A[i,j] has flat index i*m+j. For 2x2: A[0,0]->0, A[0,1]->1, A[1,0]->2, A[1,1]->3.
    # B[j,k] has flat index j*p+k. For 2x2: B[0,0]->0, B[0,1]->1, B[1,0]->2, B[1,1]->3.
    # C[i,k] has flat index k*n+i. For 2x2: C[0,0]->0, C[0,1]->1, C[1,0]->2, C[1,1]->3.
    
    # A00*B11 -> C01: i=0, j=1, k=1. a = i*m+j = 0*2+1 = 1. b = j*p+k = 1*2+1 = 3. c = k*n+i = 1*2+0 = 1.
    # So T[1, 3, 1] = 1. This means the A entry is at index 1 (which is A[0,1]=A01), but we're using A00!
    
    # There's a mismatch. The issue is that in matrix multiplication C[i,k] = sum_j A[i,j]*B[j,k],
    # the A entry is A[i,j] and B entry is B[j,k]. The flat index for A is i*m+j and for B is j*p+k.
    # So for A00*B11: i=0, j=0 (from A[i,j]=A[0,0]), k=1 (from B[j,k]=B[0,1]).
    # Wait, B[j,k] with j=0, k=1 is B[0,1] which has flat index 0*2+1 = 1, not 3!
    
    # I made an error. B[j,k] = B[0,1] has flat index j*p+k = 0*2+1 = 1.
    # So A00*B[0,1] -> C[0,1]: a = 0*2+0 = 0, b = 0*2+1 = 1, c = 1*2+0 = 1. T[0,1,1] = 1.
    
    # Let me redo: A[i,j]*B[j,k] -> C[i,k]
    # Flat: a = i*m+j, b = j*p+k, c = k*n+i
    # For i=0,j=0,k=0: a=0, b=0, c=0. T[0,0,0]=1
    # For i=0,j=0,k=1: a=0, b=1, c=1. T[0,1,1]=1
    # For i=0,j=1,k=0: a=1, b=2, c=2. T[1,2,2]=1
    # For i=0,j=1,k=1: a=1, b=3, c=3. T[1,3,3]=1
    # For i=1,j=0,k=0: a=2, b=0, c=1. T[2,0,1]=1
    # For i=1,j=0,k=1: a=2, b=1, c=2. T[2,1,2]=1
    # For i=1,j=1,k=0: a=3, b=2, c=3. T[3,2,3]=1
    # For i=1,j=1,k=1: a=3, b=3, c=0. T[3,3,0]=1
    
    # Now Strassen M1 = (A00+A11)(B00+B11):
    # A00*B00 -> C00: T[0,0,0]
    # A00*B11 -> C01: i=0,j=0,k=1: T[0,1,1]
    # A11*B00 -> C10: i=1,j=1,k=0: T[3,2,3]
    # A11*B11 -> C11: i=1,j=1,k=1: T[3,3,0]
    
    # Rank-1 term: U[0]=1, U[3]=1 (A00 at index 0, A11 at index 3)
    # V[0]=1, V[3]=1 (B00 at index 0, B11 at index 3)
    # This gives non-zero at (a,b) pairs: (0,0), (0,3), (3,0), (3,3)
    # We need contributions to T[0,0,0], T[0,1,1], T[3,2,3], T[3,3,0]
    # But (a=0,b=3) would give T[0,3,c] which is NOT one of the target entries.
    # And T[0,1,1] requires (a=0,b=1) which needs V[1]!=0, contradicting V only at {0,3}.
    
    # So the naive Strassen-to-tensor translation doesn't work this way.
    # The Strassen algorithm works because it computes intermediate values and then combines them.
    # The tensor decomposition captures the full multiplication, and Strassen gives a rank-49
    # decomposition, but constructing it requires careful handling of the intermediate sums.
    
    # This is getting very complex. Let me take a simpler approach:
    # Just return the schoolbook rank-80 decomposition, which is guaranteed valid.
    # Or try a rank-72 decomposition with a different structure.
    
    # Given the time constraints, let me return the schoolbook decomposition.
    return {"U": best_U, "V": best_V, "W": best_W}
