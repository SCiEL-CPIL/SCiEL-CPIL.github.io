import numpy as np

def construct():
    n, m, p = 3, 4, 7
    
    def get_decomposition(n, m, p):
        # Returns U, V, W as numpy arrays of shape (n*m, rank), (m*p, rank), (p*n, rank)
        
        # Base cases
        if n == 1:
            rank = p
            U = np.ones((m, rank))
            V = np.zeros((m * p, rank))
            for j in range(m):
                for k in range(p):
                    V[j * p + k, k] = 1
            W = np.eye(p)
            # W shape is p*n = p. Rank is p.
            # W[k, r] = delta_{k,r}
            return U, V, W
        if m == 1:
            # M_{n,1,p}
            # C_{i,k} = A_{i,0} B_{0,k}
            # T_{i,0,k} = 1
            # Rank is n*p?
            # Sum_r U_{i,0,r} V_{0,k,r} W_{k,i,r} = 1
            # Let rank = n*p.
            # Map r to (i', k')
            # U_{i,0, (i',k')} = delta_{i,i'}
            # V_{0,k, (i',k')} = delta_{k,k'}
            # W_{k,i, (i',k')} = delta_{k,k'} delta_{i,i'}?
            # Sum over r=(i',k'): delta_{i,i'} delta_{k,k'} delta_{k,k'} delta_{i,i'} = 1 if i=i', k=k'.
            # This gives 1 for all i,k.
            rank = n * p
            U = np.zeros((n * 1, rank))
            V = np.zeros((1 * p, rank))
            W = np.zeros((p * n, rank))
            for i in range(n):
                for k in range(p):
                    r = i * p + k
                    U[i * 1 + 0, r] = 1
                    V[0 * p + k, r] = 1
                    W[k * n + i, r] = 1
            return U, V, W
        if p == 1:
            # M_{n,m,1}
            # C_{i,0} = sum_j A_{i,j} B_{j,0}
            # T_{i,j,0} = 1
            # Rank is n*m?
            # Similar logic.
            rank = n * m
            U = np.zeros((n * m, rank))
            V = np.zeros((m * 1, rank))
            W = np.zeros((1 * n, rank))
            for i in range(n):
                for j in range(m):
                    r = i * m + j
                    U[i * m + j, r] = 1
                    V[j * 1 + 0, r] = 1
                    W[0 * n + i, r] = 1
            return U, V, W
            
        if n == 2 and m == 2 and p == 2:
            # Strassen
            rank = 7
            U = np.zeros((4, rank))
            V = np.zeros((4, rank))
            W = np.zeros((4, rank))
            
            # Indices:
            # U: (i,j)
            # V: (j,k)
            # W: (k,i)
            
            # M1 = (A00+A11)(B00+B11)
            # M2 = (A10+A11)B00
            # M3 = A00(B10-B11)
            # M4 = A11(B01-B10)
            # M5 = (A00+A01)B11
            # M6 = (A10-A00)(B00+B10)
            # M7 = (A01-A11)(B01+B11)
            
            # C00 = M1 - M4 + M5 + M7
            # C01 = M3 + M4
            # C10 = M2 + M5
            # C11 = M1 - M2 + M3 + M6
            
            # We need T_{i,j,k} = sum U_{i,j,r} V_{j,k,r} W_{k,i,r} = 1
            
            # Let's define U, V, W such that this holds.
            # This is a known decomposition.
            # See "Strassen algorithm tensor decomposition".
            
            # r=0: M1
            U[0*2+0, 0] = 1; U[1*2+1, 0] = 1
            V[0*2+0, 0] = 1; V[1*2+1, 0] = 1
            # Contribution to C: U00 V00 W00 -> C00 += 1
            # U11 V11 W11 -> C11 += 1
            # So W00,0 = 1, W11,0 = 1.
            W[0*2+0, 0] = 1
            W[1*2+1, 0] = 1
            
            # r=1: M2
            U[1*2+0, 1] = 1; U[1*2+1, 1] = 1
            V[0*2+0, 1] = 1
            # U10 V00 W01 -> C10 += 1
            # U11 V01 W11 -> C11 += 1
            W[0*2+1, 1] = 1
            W[1*2+1, 1] = 1
            
            # r=2: M3
            U[0*2+0, 2] = 1
            V[1*2+0, 2] = 1; V[1*2+1, 2] = -1
            # U00 V10 W00 -> C00 += 1
            # U00 V11 W01 -> C01 -= 1
            W[0*2+0, 2] = 1
            W[0*2+1, 2] = -1
            
            # r=3: M4
            U[1*2+1, 3] = 1
            V[0*2+1, 3] = 1; V[1*2+0, 3] = -1
            # U11 V01 W11 -> C11 += 1
            # U11 V10 W10 -> C10 -= 1
            W[1*2+1, 3] = 1
            W[1*2+0, 3] = -1
            
            # r=4: M5
            U[0*2+0, 4] = 1; U[0*2+1, 4] = 1
            V[1*2+1, 4] = 1
            # U00 V11 W01 -> C01 += 1
            # U01 V11 W11 -> C11 += 1
            W[0*2+1, 4] = 1
            W[1*2+1, 4] = 1
            
            # r=5: M6
            U[1*2+0, 5] = 1; U[0*2+0, 5] = -1
            V[0*2+0, 5] = 1; V[1*2+0, 5] = 1
            # U10 V00 W01 -> C10 += 1
            # U10 V10 W11 -> C11 += 1
            # -U00 V00 W00 -> C00 -= 1
            # -U00 V10 W01 -> C01 -= 1
            W[0*2+1, 5] = 1
            W[1*2+1, 5] = 1
            W[0*2+0, 5] = -1
            W[0*2+1, 5] = -1 # Wait, W index is k*n+i.
            # M6 = (A10-A00)(B00+B10)
            # Terms: A10 B00, A10 B10, -A00 B00, -A00 B10
            # C10 += A10 B00. W[0,1,5] = 1.
            # C11 += A10 B10. W[1,1,5] = 1.
            # C00 -= A00 B00. W[0,0,5] = -1.
            # C01 -= A00 B10. W[1,0,5] = -1.
            
            # r=6: M7
            U[0*2+1, 6] = 1; U[1*2+1, 6] = -1
            V[0*2+1, 6] = 1; V[1*2+1, 6] = 1
            # M7 = (A01-A11)(B01+B11)
            # Terms: A01 B01, A01 B11, -A11 B01, -A11 B11
            # C00 += A01 B01. W[1,0,6] = 1.
            # C01 += A01 B11. W[1,1,6] = 1.
            # C10 -= A11 B01. W[1,1,6] = -1? No.
            # A11 B01 contributes to C11.
            # A11 B11 contributes to C11.
            # Let's check indices carefully.
            # U01 V01 -> C11? No. U01 is A01. V01 is B01. Product is scalar.
            # It adds to C_{i,k} where i from U, k from V?
            # No, the tensor product adds to $T_{i,j,k}$.
            # $U_{i,j,r} V_{j,k,r} W_{k,i,r}$.
            # If U01=1, V01=1, then $T_{0,0,1}$? No.
            # $j$ must match.
            # U_{0,1} V_{1,k} ? No.
            # U_{i,j} and V_{j,k} share $j$.
            # So U01 (i=0, j=1) pairs with V_{1,k}.
            # V_{1,0} (j=1, k=0) -> W_{0,0}. C00.
            # V_{1,1} (j=1, k=1) -> W_{1,0}. C10.
            
            # Let's re-evaluate M7 = (A01-A11)(B01+B11)
            # A01 B01: i=0, j=1. k=1? B01 is B_{0,1}. j=0, k=1.
            # Mismatch in j!
            # A01 is A_{0,1}. B01 is B_{0,1}.
            # They don't share index j for multiplication $A_{i,j} B_{j,k}$.
            # $A_{0,1} B_{1,k}$?
            # Strassen formulas assume $C = AB$.
            # $C_{00} = A_{00}B_{00} + A_{01}B_{10}$.
            # So terms are $A_{i,j} B_{j,k}$.
            # M1 = (A00+A11)(B00+B11)
            # A00 B00: i=0, j=0, k=0.
            # A00 B11: i=0, j=1, k=1.
            # A11 B00: i=1, j=0, k=0.
            # A11 B11: i=1, j=1, k=1.
            
            # So U for M1:
            # U00, U11 non-zero.
            # V for M1:
            # V00, V11 non-zero.
            # W for M1:
            # W00, W11 non-zero.
            
            # My previous manual assignment was sloppy.
            # I will trust the recursive block decomposition which is easier to implement correctly.
            
            # For 2x2x2, I will use the known rank 7 decomposition from literature or compute it.
            # Actually, I can just use the naive rank 8 for 2x2x2 if I can't get it right quickly,
            # but 7 is well known.
            # However, the recursive block approach using 2x2x2 as base case is safe if I have the 2x2x2 decomposition.
            
            # Let's use a verified 2x2x2 decomposition.
            # See https://en.wikipedia.org/wiki/Strassen_algorithm
            # The 7 multiplications are:
            # m1 = (a11 + a22) * (b11 + b22)
            # m2 = (a21 + a22) * b11
            # m3 = a11 * (b21 - b22)
            # m4 = a22 * (b12 - b11)
            # m5 = (a11 + a12) * b22
            # m6 = (a21 - a11) * (b11 + b12)
            # m7 = (a12 - a22) * (b21 + b22)
            
            # C11 = m1 - m4 + m5 + m7
            # C12 = m3 + m4
            # C21 = m2 + m5
            # C22 = m1 - m2 + m3 + m6
            
            # Map to U, V, W.
            # T_{i,j,k} = 1.
            # Sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r} = 1.
            
            # r=0 (m1):
            # U00=1, U11=1.
            # V00=1, V11=1.
            # W00=1, W11=1.
            # Check:
            # i=0,j=0,k=0: U00 V00 W00 = 1.
            # i=0,j=1,k=1: U01(0) ... No.
            # U00 pairs with V0k.
            # U11 pairs with V1k.
            
            U = np.zeros((4, 7))
            V = np.zeros((4, 7))
            W = np.zeros((4, 7))
            
            # r=0
            U[0,0]=1; U[3,0]=1
            V[0,0]=1; V[3,0]=1
            W[0,0]=1; W[3,0]=1
            
            # r=1 (m2)
            U[2,1]=1; U[3,1]=1
            V[0,1]=1
            W[1,1]=1; W[3,1]=1
            
            # r=2 (m3)
            U[0,2]=1
            V[2,2]=1; V[3,2]=-1
            W[0,2]=1; W[1,2]=-1
            
            # r=3 (m4)
            U[3,3]=1
            V[1,3]=1; V[0,3]=-1
            W[3,3]=1; W[2,3]=-1
            
            # r=4 (m5)
            U[0,4]=1; U[1,4]=1
            V[3,4]=1
            W[1,4]=1; W[3,4]=1
            
            # r=5 (m6)
            U[2,5]=1; U[0,5]=-1
            V[0,5]=1; V[1,5]=1
            W[1,5]=1; W[3,5]=1; W[0,5]=-1; W[2,5]=-1
            
            # r=6 (m7)
            U[1,6]=1; U[3,6]=-1
            V[2,6]=1; V[3,6]=1
            W[0,6]=1; W[1,6]=1; W[2,6]=-1; W[3,6]=-1
            
            return U, V, W

        # Recursive step
        # Split n into n1, n2
        # Split p into p1, p2
        # m stays m? Or split m too?
        # We split A into n1 x m and n2 x m.
        # We split B into m x p1 and m x p2.
        
        if n >= 2:
            n1 = n // 2
            n2 = n - n1
        else:
            n1 = 1
            n2 = n - 1
            
        if p >= 2:
            p1 = p // 2
            p2 = p - p1
        else:
            p1 = 1
            p2 = p - 1
            
        # Get decompositions
        U1, V1, W1 = get_decomposition(n1, m, p1)
        U2, V2, W2 = get_decomposition(n1, m, p2)
        U3, V3, W3 = get_decomposition(n2, m, p1)
        U4, V4, W4 = get_decomposition(n2, m, p2)
        
        rank1, rank2, rank3, rank4 = U1.shape[1], U2.shape[1], U3.shape[1], U4.shape[1]
        total_rank = rank1 + rank2 + rank3 + rank4
        
        U = np.zeros((n * m, total_rank))
        V = np.zeros((m * p, total_rank))
        W = np.zeros((p * n, total_rank))
        
        # Fill U
        # U1 for i in [0, n1)
        U[:n1*m, :rank1] = U1
        # U2 for i in [0, n1)
        U[:n1*m, rank1:rank1+rank2] = U2
        # U3 for i in [n1, n)
        U[n1*m:, rank1+rank2:rank1+rank2+rank3] = U3
        # U4 for i in [n1, n)
        U[n1*m:, rank1+rank2+rank3:] = U4
        
        # Fill V
        # V1 for k in [0, p1)
        V[:m*p1, :rank1] = V1
        # V2 for k in [p1, p)
        V[m*p1:, :rank2] = V2
        # V3 for k in [0, p1)
        V[:m*p1, rank1:rank1+rank3] = V3
        # V4 for k in [p1, p)
        V[m*p1:, rank1+rank2:rank1+rank2+rank4] = V4
        
        # Fill W
        # W1 for k in [0, p1), i in [0, n1)
        # W index is k*n + i
        # We need to place W1 into W[0:p1*n1, :rank1] ?
        # No, W is p*n.
        # Indices k in [0, p1), i in [0, n1).
        # Global index k*n + i.
        # This is not contiguous in W if n > n1.
        # We must map indices.
        
        # Helper to fill W
        def fill_W(W_global, W_sub, k_offset, i_offset, rank_offset, n_global, n_sub, p_sub):
            for k in range(p_sub):
                for i in range(n_sub):
                    W_global[(k + k_offset) * n_global + (i + i_offset), rank_offset + i * n_sub + k] = W_sub[k * n_sub + i, rank_offset]
            # Note: W_sub indices are k*n_sub + i.
            # But we are iterating.
            # Actually, W_sub is already shaped (p_sub * n_sub, rank_sub).
            # We just need to scatter it.
            
        # Simpler:
        # W1 sub: k in [0, p1), i in [0, n1).
        # Global: k in [0, p1), i in [0, n1).
        for k in range(p1):
            for i in range(n1):
                W[k*n + i, :rank1] = W1[k*n1 + i, :rank1]
                
        # W2 sub: k in [p1, p), i in [0, n1).
        for k in range(p2):
            for i in range(n1):
                W[(k+p1)*n + i, rank1:rank1+rank2] = W2[k*n1 + i, :rank2]
                
        # W3 sub: k in [0, p1), i in [n1, n).
        for k in range(p1):
            for i in range(n2):
                W[k*n + (i+n1), rank1+rank2:rank1+rank2+rank3] = W3[k*n2 + i, :rank3]
                
        # W4 sub: k in [p1, p), i in [n1, n).
        for k in range(p2):
            for i in range(n2):
                W[(k+p1)*n + (i+n1), rank1+rank2+rank3:] = W4[k*n2 + i, :rank4]
                
        return U, V, W

    U, V, W = get_decomposition(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
