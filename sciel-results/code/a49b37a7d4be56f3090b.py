"""Strassen-based rank reduction for 2x5x5 matrix multiplication."""
import numpy as np
import time

DIMENSIONS = [2, 5, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 2, 5, 5
    
    # Build schoolbook first, then apply Strassen to the 2x2x2 sub-block
    # Partition: A[0:2, 0:2] * B[0:2, :] contributes to C[0:2, :]
    # But we need to be careful about indices.
    
    # Let's use a direct approach: enumerate all (i,j,k) triples
    # and group the 2x2x2 corner for Strassen reduction.
    
    # Schoolbook terms: for each (i,j,k) in [0,2)×[0,5)×[0,5)
    # U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1
    
    # The 2x2x2 sub-block: i in {0,1}, j in {0,1}, k in {0,1}
    # That's 8 terms we can replace with 7 Strassen terms.
    
    # Strassen for 2x2x2:
    # M1 = (A11-A22)(B22-B11)
    # M2 = (A11+A12)B21  
    # M3 = (A21+A22)B12
    # M4 = A11B11
    # M5 = A22B22
    # M6 = (A11-A21)(B12+B22)
    # M7 = (A22-A12)(B11+B21)
    
    # C11 = M5 + M4 - M2 + M6
    # C12 = M1 + M5
    # C21 = M2 + M4
    # C22 = M1 - M3 + M5 + M7
    
    # In our tensor notation:
    # U rows are indexed by (i,j) -> i*m+j, so (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # V rows are indexed by (j,k) -> j*p+k, so (0,0)=0, (0,1)=1, (1,0)=5, (1,1)=6
    # W rows are indexed by (k,i) -> k*n+i, so (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    
    # For the 2x2x2 sub-block (i,j,k all in {0,1}):
    # U indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # V indices: (0,0)=0, (0,1)=1, (1,0)=5, (1,1)=6
    # W indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    
    # Build all terms as a list of (U_row, V_row, W_row, coeff)
    terms = []
    
    # Add schoolbook terms for all (i,j,k) except the 2x2x2 corner
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i < 2 and j < 2 and k < 2:
                    continue  # Skip 2x2x2 corner, handle with Strassen
                U_row = i * m + j
                V_row = j * p + k
                W_row = k * n + i
                terms.append((U_row, V_row, W_row, 1))
    
    # Add Strassen terms for the 2x2x2 corner
    # U rows: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # V rows: 0=(0,0), 1=(0,1), 5=(1,0), 6=(1,1)
    # W rows: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    
    # M1 = (A11-A22)(B22-B11): U: +1 at (0,0)=0, -1 at (1,1)=3; V: +1 at (1,1)=6, -1 at (0,0)=0
    terms.append((0, 6, 0, 1))   # M1 contribution to W[0,0]... wait, I need to think about this differently.
    
    # Actually, each Strassen term is a bilinear form that contributes to multiple W entries.
    # Let me reconsider. Each term in the decomposition is a rank-1 tensor:
    # T[i,j,k] = U[i,j,c] * V[j,k,c] * W[k,i,c]
    # 
    # For Strassen, we define 7 intermediate products M1..M7, each is a product of 
    # a linear combination of A entries and a linear combination of B entries.
    # Then each C entry is a linear combination of M's.
    
    # So for each M_s, we have:
    # M_s = (sum of U_coeffs * A_entries) * (sum of V_coeffs * B_entries)
    # And C_entries = sum of W_coeffs * M_s
    
    # In our tensor framework, each M_s corresponds to one rank-1 term where:
    # U part: coefficients on A entries (rows of U)
    # V part: coefficients on B entries (rows of V)  
    # W part: coefficients on C entries (rows of W)
    
    # But wait - in the standard tensor decomposition, each rank-1 term has 
    # U[row,c], V[row,c], W[row,c] and the contribution is U*V*W.
    # For Strassen, M1 = (A11-A22)*(B22-B11) contributes to C11, C12, C21, C22
    # with different coefficients. This means M1 is NOT a single rank-1 term in our 
    # decomposition - it's a product that feeds into multiple C entries.
    
    # Actually, I think the standard way is: each Strassen intermediate M_s is 
    # represented as a rank-1 tensor where the U and V parts capture the linear 
    # combinations, and the W part has 1s for all C entries it contributes to.
    # But that's not quite right either, because the coefficients differ.
    
    # Let me reconsider. The correct way: for each Strassen term M_s with 
    # U-coefficients u_s[i,j] and V-coefficients v_s[j,k], and it contributes 
    # to C[k',i'] with coefficient w_s[k',i'], we need w_s[k',i'] * M_s in the 
    # output. Since M_s = sum_{i,j} u_s[i,j]*A[i,j] * sum_{j,k} v_s[j,k]*B[j,k],
    # the contribution to C[k',i'] is w_s[k',i'] * sum_{i,j,k} u_s[i,j]*v_s[j,k]*A[i,j]*B[j,k].
    # 
    # In our tensor form, this is a rank-1 tensor with:
    # U[i,j,c] = u_s[i,j], V[j,k,c] = v_s[j,k], W[k',i',c] = w_s[k',i']
    # BUT this only works if the same w_s coefficient applies uniformly. 
    # Actually no - the rank-1 tensor T[i,j,k] = U[i,j]*V[j,k]*W[k,i] means
    # the contribution to C[k,i] is U[i,j]*V[j,k]*W[k,i] for each (i,j,k).
    # 
    # So for Strassen M1 = (A11-A22)(B22-B11):
    # u: U[0,0]=1, U[1,1]=-1 (A11 and A22)
    # v: V[1,1]=1, V[0,0]=-1 (B22 and B11)
    # This product gives: 1*(-1)*A11*B11 + 1*1*A11*B22 + (-1)*(-1)*A22*B11 + (-1)*1*A22*B22
    # = -A11*B11 + A11*B22 + A22*B11 - A22*B22
    # = (A11-A22)(B22-B11) ✓
    # 
    # Now this contributes to C11, C12, C21, C22 with coefficients from Strassen:
    # C11 = M5 + M4 - M2 + M6 (M1 not in C11)
    # C12 = M1 + M5
    # C21 = M2 + M4
    # C22 = M1 - M3 + M5 + M7
    # 
    # So M1 contributes to C12 with coeff +1 and C22 with coeff +1.
    # In our tensor, W[k,i,c] should be: W[0,0]=0 (C11), W[0,1]=1 (C12), W[1,0]=0 (C21), W[1,1]=1 (C22)
    # Wait, W is indexed by [k*n+i], so C11 is W[0*2+0]=W[0], C12 is W[0*2+1]=W[1], 
    # C21 is W[1*2+0]=W[2], C22 is W[1*2+1]=W[3].
    
    # So for M1: W[0]=0, W[1]=1, W[2]=0, W[3]=1
    # This is a valid rank-1 term! The U and V parts define the bilinear form, 
    # and W parts scale which C entries receive this contribution.
    
    # Great, so each Strassen M_s is one rank-1 term in our decomposition.
    
    # Let me redefine the terms properly.
    terms = []
    
    # Schoolbook terms for non-2x2x2-corner
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i < 2 and j < 2 and k < 2:
                    continue
                U_row = i * m + j
                V_row = j * p + k
                W_row = k * n + i
                terms.append((U_row, V_row, W_row, 1))
    
    # Strassen terms for 2x2x2 corner
    # U rows: (i,j) -> i*5+j: (0,0)=0, (0,1)=1, (1,0)=5, (1,1)=6
    # V rows: (j,k) -> j*5+k: (0,0)=0, (0,1)=1, (1,0)=5, (1,1)=6
    # W rows: (k,i) -> k*2+i: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    
    # M1 = (A11-A22)(B22-B11): U: A11=+1, A22=-1; V: B22=+1, B11=-1
    # Contributes to C12 (+1), C22 (+1)
    terms.append((0, 6, 1, 1))    # U[0]=A11, V[6]=B22, W[1]=C12, coeff 1
    terms.append((0, 0, 1, -1))   # U[0]=A11, V[0]=B11, W[1]=C12, coeff -1
    terms.append((6, 6, 3, -1))   # U[6]=A22, V[6]=B22, W[3]=C22, coeff -1
    terms.append((6, 0, 3, 1))    # U[6]=A22, V[0]=B11, W[3]=C22, coeff 1
    
    # Hmm wait, this is getting complicated. Let me think again.
    
    # Actually, I realize each Strassen M_s is a SINGLE rank-1 tensor where:
    # - U has multiple nonzero entries (the linear combo of A entries)
    # - V has multiple nonzero entries (the linear combo of B entries)
    # - W has multiple nonzero entries (which C entries it contributes to, with what coeff)
    
    # But in our format, each "term" in the list corresponds to one column c,
    # and U[row,c], V[row,c], W[row,c] can each have multiple nonzeros.
    
    # So for M1, I need ONE column where:
    # U[0,c]=1, U[6,c]=-1 (A11 and A22, where A22 is at row (1,1)=1*5+1=6)
    # V[6,c]=1, V[0,c]=-1 (B22 and B11, where B22 is at (1,1)=1*5+1=6, B11 at (0,0)=0)
    # W[1,c]=1, W[3,c]=1 (C12 and C22)
    
    # Let me rebuild this properly.
    
    terms = []  # Each term: dict of {U: {row: coeff}, V: {row: coeff}, W: {row: coeff}}
    
    # Schoolbook for non-corner
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i < 2 and j < 2 and k < 2:
                    continue
                U_row = i * m + j
                V_row = j * p + k
                W_row = k * n + i
                terms.append({"U": {U_row: 1}, "V": {V_row: 1}, "W": {W_row: 1}})
    
    # Strassen M1-M7 for 2x2x2 corner
    # A entries: A11=(0,0)=row 0, A12=(0,1)=row 1, A21=(1,0)=row 5, A22=(1,1)=row 6
    # B entries: B11=(0,0)=row 0, B12=(0,1)=row 1, B21=(1,0)=row 5, B22=(1,1)=row 6
    # C entries: C11=(0,0)=row 0, C12=(0,1)=row 1, C21=(1,0)=row 2, C22=(1,1)=row 3
    
    # M1 = (A11-A22)(B22-B11) -> contributes to C12(+1), C22(+1)
    terms.append({"U": {0: 1, 6: -1}, "V": {6: 1, 0: -1}, "W": {1: 1, 3: 1}})
    
    # M2 = (A11+A12)B21 -> contributes to C21(+1)
    terms.append({"U": {0: 1, 1: 1}, "V": {5: 1}, "W": {2: 1}})
    
    # M3 = (A21+A22)B12 -> contributes to C22(-1)
    terms.append({"U": {5: 1, 6: 1}, "V": {1: 1}, "W": {3: -1}})
    
    # M4 = A11*B11 -> contributes to C11(+1), C21(+1)
    terms.append({"U": {0: 1}, "V": {0: 1}, "W": {0: 1, 2: 1}})
    
    # M5 = A22*B22 -> contributes to C11(+1), C12(+1), C22(+1)
    terms.append({"U": {6: 1}, "V": {6: 1}, "W": {0: 1, 1: 1, 3: 1}})
    
    # M6 = (A11-A21)(B12+B22) -> contributes to C11(+1)
    terms.append({"U": {0: 1, 5: -1}, "V": {1: 1, 6: 1}, "W": {0: 1}})
    
    # M7 = (A22-A12)(B11+B21) -> contributes to C22(+1)
    terms.append({"U": {6: 1, 1: -1}, "V": {0: 1, 5: 1}, "W": {3: 1}})
    
    # Now convert to matrix format
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c, term in enumerate(terms):
        for row, coeff in term["U"].items():
            U[row][c] = coeff
        for row, coeff in term["V"].items():
            V[row][c] = coeff
        for row, coeff in term["W"].items():
            W[row][c] = coeff
    
    return {"U": U, "V": V, "W": W}
