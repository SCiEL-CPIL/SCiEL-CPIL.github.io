import time
import numpy as np

DIMENSIONS = [2, 2, 3]

def construct():
    _t0 = time.time()
    _deadline = _t0 + 140.0
    n, m, p = DIMENSIONS

    def verify(U, V, W, rank):
        """Verify the decomposition is exact."""
        # Build tensor: T[a,b,c] where a=i*m+j, b=j*p+k, c=k*n+i
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    # T[a,b,c] should equal sum_r U[a,r]*V[b,r]*W[c,r]
                    total = 0
                    for r in range(rank):
                        total += U[a][r] * V[b][r] * W[c][r]
                    if total != 1:
                        return False
        # Also verify all other entries are 0
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    # Check if (a,b,c) corresponds to a valid multiplication entry
                    i_a = a // m
                    j_a = a % m
                    j_b = b // p
                    k_b = b % p
                    k_c = c // n
                    i_c = c % n
                    is_valid = (j_a == j_b) and (i_a == i_c)
                    total = 0
                    for r in range(rank):
                        total += U[a][r] * V[b][r] * W[c][r]
                    if is_valid:
                        if total != 1:
                            return False
                    else:
                        if total != 0:
                            return False
        return True

    def try_rank10_search():
        """Search for rank-10 decomposition using structured approach."""
        rank = 10
        # Start from known rank-11 structure, try to merge two terms
        # The 4 schoolbook terms for col 2: A00*B02, A01*B12, A10*B02, A11*B12
        # Try to absorb these into Strassen factors
        
        # Strategy: Modify the 7 Strassen terms to also handle col 2
        # Each Strassen term has a U vector (over 4 A-entries), V vector (over 6 B-entries), W vector (over 6 C-entries)
        # We need to add col-2 contributions to V and W while keeping the bilinear form correct
        
        # Let's parameterize: for each of the 7 Strassen terms r, we can add:
        #   V[r][4] += alpha_r (coefficient for B02)
        #   V[r][5] += beta_r  (coefficient for B12)
        #   W[r][4] += gamma_r (coefficient for C02)  
        #   W[r][5] += delta_r (coefficient for C12)
        
        # The col-2 contribution from term r is:
        #   U[r][0]*(alpha_r*B02 + beta_r*B12) * (gamma_r*C02 + delta_r*C12)
        # Wait, that's not right. The tensor entry for (i,j,k=2) is:
        #   sum_r U[i*m+j][r] * V[j*p+2][r] * W[2*n+i][r]
        
        # For k=2: V[j*p+2][r] = V[j*3+2][r]
        #   j=0: V[2][r] (B02)
        #   j=1: V[5][r] (B12)
        # W[2*n+i][r] = W[2*2+i][r] = W[4+i][r]
        #   i=0: W[4][r] (C02)
        #   i=1: W[5][r] (C12)
        
        # So for entry (i,j,2): sum_r U[i*m+j][r] * V[j*3+2][r] * W[4+i][r] = 1 if i,j match
        
        # Required:
        # (0,0,2): sum_r U[0][r]*V[2][r]*W[4][r] = 1  (A00*B02->C02)
        # (0,1,2): sum_r U[1][r]*V[5][r]*W[4][r] = 1  (A01*B12->C02)
        # (1,0,2): sum_r U[2][r]*V[2][r]*W[5][r] = 1  (A10*B02->C12)
        # (1,1,2): sum_r U[3][r]*V[5][r]*W[5][r] = 1  (A11*B12->C12)
        # All other (i,j,2) entries must be 0
        
        # Base Strassen U vectors (for the 2x2 part, rows 0-3 of U):
        # Term 0: U = [1, 0, 0, 1]  (A00+A11)
        # Term 1: U = [0, 0, 1, 1]  (A10+A11)
        # Term 2: U = [1, 0, 0, 0]  (A00)
        # Term 3: U = [0, 0, 0, 1]  (A11)
        # Term 4: U = [1, 1, 0, 0]  (A00+A01)
        # Term 5: U = [0, 0, 1, -1] (A10-A00) -- wait, let me recheck
        # Actually from the current solution:
        # Term 0: U[0][0]=1, U[3][0]=1 -> [1,0,0,1]
        # Term 1: U[2][1]=1, U[3][1]=1 -> [0,0,1,1]
        # Term 2: U[0][2]=1 -> [1,0,0,0]
        # Term 3: U[3][3]=1 -> [0,0,0,1]
        # Term 4: U[0][4]=1, U[1][4]=1 -> [1,1,0,0]
        # Term 5: U[2][5]=1, U[0][5]=-1 -> [-1,0,1,0]
        # Term 6: U[1][6]=1, U[3][6]=-1 -> [0,1,0,-1]
        
        U_base = [
            [1, 0, 0, 1],  # term 0
            [0, 0, 1, 1],  # term 1
            [1, 0, 0, 0],  # term 2
            [0, 0, 0, 1],  # term 3
            [1, 1, 0, 0],  # term 4
            [-1, 0, 1, 0], # term 5
            [0, 1, 0, -1], # term 6
        ]
        
        # For each term r, let v0_r = V[2][r] (B02 coeff), v1_r = V[5][r] (B12 coeff)
        # Let w0_r = W[4][r] (C02 coeff), w1_r = W[5][r] (C12 coeff)
        
        # We need:
        # sum_r U[0][r] * v0_r * w0_r = 1  (A00*B02)
        # sum_r U[1][r] * v1_r * w0_r = 1  (A01*B12)
        # sum_r U[2][r] * v0_r * w1_r = 1  (A10*B02)
        # sum_r U[3][r] * v1_r * w1_r = 1  (A11*B12)
        
        # And cross terms must be 0:
        # sum_r U[1][r] * v0_r * w0_r = 0  (A01*B02 should be 0)
        # sum_r U[0][r] * v1_r * w0_r = 0  (A00*B12 should be 0)
        # sum_r U[3][r] * v0_r * w1_r = 0  (A11*B02 should be 0)
        # sum_r U[2][r] * v1_r * w1_r = 0  (A10*B12 should be 0)
        
        # This is a system of equations. Let's try a simple ansatz:
        # Set v0_r = a_r, v1_r = b_r, w0_r = c_r, w1_r = d_r
        # Try: a_r = U[0][r], b_r = U[1][r], c_r = U[0][r], d_r = U[1][r]
        # Then:
        # A00*B02: sum U[0][r]^3 -- not generally 1
        
        # Better ansatz: Try to use only 2 terms for col-2 absorption
        # Set v0_r = v1_r = w0_r = w1_r = 0 for all r except two terms
        # For terms s and t:
        # U[0][s]*v0_s*w0_s + U[0][t]*v0_t*w0_t = 1
        # U[1][s]*v1_s*w0_s + U[1][t]*v1_t*w0_t = 1
        # U[2][s]*v0_s*w1_s + U[2][t]*v0_t*w1_t = 1
        # U[3][s]*v1_s*w1_s + U[3][t]*v1_t*w1_t = 1
        
        # Try s=2 (U=[1,0,0,0]) and t=3 (U=[0,0,0,1]):
        # 1*v0_2*w0_2 + 0 = 1 -> v0_2*w0_2 = 1
        # 0 + 0 = 1 -> IMPOSSIBLE
        
        # Try s=0 (U=[1,0,0,1]) and t=1 (U=[0,0,1,1]):
        # 1*v0_0*w0_0 + 0 = 1 -> v0_0*w0_0 = 1
        # 0 + 0 = 1 -> IMPOSSIBLE
        
        # Try s=4 (U=[1,1,0,0]) and t=5 (U=[-1,0,1,0]):
        # 1*v0_4*w0_4 + (-1)*v0_5*w0_5 = 1
        # 1*v1_4*w0_4 + 0 = 1 -> v1_4*w0_4 = 1
        # 0 + 1*v0_5*w1_5 = 1 -> v0_5*w1_5 = 1
        # 0 + 0 = 1 -> IMPOSSIBLE
        
        # Try s=2 (U=[1,0,0,0]) and t=4 (U=[1,1,0,0]):
        # 1*v0_2*w0_2 + 1*v0_4*w0_4 = 1
        # 0 + 1*v1_4*w0_4 = 1 -> v1_4*w0_4 = 1
        # 0 + 0 = 1 -> IMPOSSIBLE
        
        # Try s=0 (U=[1,0,0,1]) and t=4 (U=[1,1,0,0]):
        # 1*v0_0*w0_0 + 1*v0_4*w0_4 = 1
        # 0 + 1*v1_4*w0_4 = 1 -> v1_4*w0_4 = 1
        # 0 + 0 = 1 -> IMPOSSIBLE
        
        # It seems hard to do with just 2 terms. Let's try 3 terms.
        # Use terms 2, 3, and 4:
        # s=2 (U=[1,0,0,0]), t=3 (U=[0,0,0,1]), u=4 (U=[1,1,0,0])
        # A00*B02: v0_2*w0_2 + v0_4*w0_4 = 1
        # A01*B12: v1_4*w0_4 = 1 -> v1_4 = 1/w0_4
        # A10*B02: v0_4*w1_4 = 1 -> v0_4 = 1/w1_4
        # A11*B12: v1_3*w1_3 = 1 -> v1_3 = 1/w1_3
        
        # Cross terms:
        # A01*B02: v1_4*w0_4 * 0 + ... wait, let me redo
        # Entry (i,j,2) = sum_r U[i*m+j][r] * V[j*3+2][r] * W[4+i][r]
        # (0,0,2): U[0]*V[2]*W[4] sum = U[0][2]*v0_2*w0_2 + U[0][3]*v0_3*w0_3 + U[0][4]*v0_4*w0_4
        #   = 1*v0_2*w0_2 + 0 + 1*v0_4*w0_4 = 1
        # (0,1,2): U[1]*V[5]*W[4] = U[1][2]*v1_2*w0_2 + U[1][3]*v1_3*w0_3 + U[1][4]*v1_4*w0_4
        #   = 0 + 0 + 1*v1_4*w0_4 = 1 -> v1_4*w0_4 = 1
        # (1,0,2): U[2]*V[2]*W[5] = U[2][2]*v0_2*w1_2 + U[2][3]*v0_3*w1_3 + U[2][4]*v0_4*w1_4
        #   = 0 + 0 + 0 = 0 != 1  PROBLEM!
        
        # U[2][4] = 0, so term 4 doesn't help for (1,0,2). Need a term with U[2][r]!=0.
        # Terms with U[2][r]!=0: term 1 (U[2]=1), term 5 (U[2]=1)
        
        # Let's use terms 1, 3, 4:
        # s=1 (U=[0,0,1,1]), t=3 (U=[0,0,0,1]), u=4 (U=[1,1,0,0])
        # (0,0,2): U[0]*V[2]*W[4] = 0*v0_1*w0_1 + 0*v0_3*w0_3 + 1*v0_4*w0_4 = v0_4*w0_4 = 1
        # (0,1,2): U[1]*V[5]*W[4] = 0*v1_1*w0_1 + 0*v1_3*w0_3 + 1*v1_4*w0_4 = v1_4*w0_4 = 1
        # (1,0,2): U[2]*V[2]*W[5] = 1*v0_1*w1_1 + 0*v0_3*w1_3 + 0*v0_4*w1_4 = v0_1*w1_1 = 1
        # (1,1,2): U[3]*V[5]*W[5] = 1*v1_1*w1_1 + 1*v1_3*w1_3 + 0*v1_4*w1_4 = v1_1*w1_1 + v1_3*w1_3 = 1
        
        # Cross terms:
        # (0,0,2) done, (0,1,2) done, (1,0,2) done, (1,1,2) done
        # Need: (1,0,2) uses U[2], but also check (0,0,2) doesn't get contributions from terms 1,3
        # U[0][1]=0, U[0][3]=0, so OK
        # (1,1,2): U[3][1]=1, U[3][3]=1, so v1_1*w1_1 + v1_3*w1_3 = 1
        
        # Set: v0_4=1, w0_4=1, v1_4=1, w0_4=1 (consistent)
        # v0_1=1, w1_1=1
        # v1_1*w1_1 + v1_3*w1_3 = 1 -> 1*1 + v1_3*w1_3 = 1 -> v1_3*w1_3 = 0
        # Set v1_3=0, w1_3=0
        
        # Now check all cross terms are 0:
        # (0,0,2): only term 4 contributes: 1*1*1=1 ✓
        # (0,1,2): only term 4 contributes: 1*1*1=1 ✓  
        # (1,0,2): only term 1 contributes: 1*1*1=1 ✓
        # (1,1,2): term 1 contributes 1*1*1=1, term 3 contributes 1*0*0=0, total=1 ✓
        
        # But wait, I need to check that the OTHER entries (k=0,1) are still correct
        # The base Strassen handles k=0,1. Adding V[2],V[5],W[4],W[5] entries only affects k=2 entries.
        # For k=0: V[j*3+0][r] and W[0*2+i][r] = W[i][r]
        # For k=1: V[j*3+1][r] and W[1*2+i][r] = W[2+i][r]
        # These are separate from V[2],V[5],W[4],W[5] so no interference!
        
        # So the rank-10 solution is:
        # Base Strassen for k=0,1 (7 terms)
        # Plus: term 1 gets V[2][1]=1, W[5][1]=1
        #        term 3 gets nothing extra
        #        term 4 gets V[2][4]=1, V[5][4]=1, W[4][4]=1
        
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        # Term 0: (A00+A11)(B00+B11)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[4][0] = 1
        W[0][0] = 1; W[3][0] = 1
        
        # Term 1: (A10+A11)B00
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[1][1] = 1; W[3][1] = -1
        # Col-2 absorption:
        V[2][1] = 1  # B02
        W[5][1] = 1  # C12
        
        # Term 2: A00(B01-B11)
        U[0][2] = 1
        V[1][2] = 1; V[4][2] = -1
        W[2][2] = 1; W[3][2] = 1
        
        # Term 3: A11(B10-B00)
        U[3][3] = 1
        V[3][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[1][3] = 1
        
        # Term 4: (A00+A01)B11
        U[0][4] = 1; U[1][4] = 1
        V[4][4] = 1
        W[0][4] = -1; W[2][4] = 1
        # Col-2 absorption:
        V[2][4] = 1  # B02
        V[5][4] = 1  # B12
        W[4][4] = 1  # C02
        
        # Term 5: (A10-A00)(B00+B01)
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1
        
        # Term 6: (A01-A11)(B10+B11)
        U[1][6] = 1; U[3][6] = -1
        V[3][6] = 1; V[4][6] = 1
        W[0][6] = 1
        
        # Terms 7,8: for remaining col-2 entries?
        # We handled: A00*B02 (term4), A01*B12 (term4), A10*B02 (term1), A11*B12 (term1+term3)
        # A11*B12: U[3][1]=1, V[5][1]=1, W[5][1]=1 -> contributes 1
        # But we also need to make sure A11*B12 doesn't get extra from term 3
        # U[3][3]=1, V[5][3]=0, W[5][3]=0 -> contributes 0. Good.
        
        # Verify
        if verify(U, V, W, rank):
            return {"U": U, "V": V, "W": W}
        
        return None

    # Try rank-10
    result = try_rank10_search()
    if result is not None:
        return result
    
    # Fallback to rank-11
    rank = 11
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    U[0][0] += 1; U[3][0] += 1
    V[0][0] += 1; V[4][0] += 1
    W[0][0] += 1; W[3][0] += 1
    
    U[2][1] += 1; U[3][1] += 1
    V[0][1] += 1
    W[1][1] += 1; W[3][1] += -1
    
    U[0][2] += 1
    V[1][2] += 1; V[4][2] += -1
    W[2][2] += 1; W[3][2] += 1
    
    U[3][3] += 1
    V[3][3] += 1; V[0][3] += -1
    W[0][3] += 1; W[1][3] += 1
    
    U[0][4] += 1; U[1][4] += 1
    V[4][4] += 1
    W[0][4] += -1; W[2][4] += 1
    
    U[2][5] += 1; U[0][5] += -1
    V[0][5] += 1; V[1][5] += 1
    W[3][5] += 1
    
    U[1][6] += 1; U[3][6] += -1
    V[3][6] += 1; V[4][6] += 1
    W[0][6] += 1
    
    U[0][7] += 1
    V[2][7] += 1
    W[4][7] += 1
    
    U[1][8] += 1
    V[5][8] += 1
    W[4][8] += 1
    
    U[2][9] += 1
    V[2][9] += 1
    W[5][9] += 1
    
    U[3][10] += 1
    V[5][10] += 1
    W[5][10] += 1
    
    return {"U": U, "V": V, "W": W}
