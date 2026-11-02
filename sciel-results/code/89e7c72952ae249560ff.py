<<<<<<< SEARCH
def _strassen_2x2x2():
    """
    Return U, V, W for 2x2 * 2x2 -> 2x2 with rank 7 (Strassen).
    
    U shape: (4, 7), indexed by [i*2+j] where i=row of A, j=col of A
    V shape: (4, 7), indexed by [j*2+k] where j=row of B, k=col of B
    W shape: (4, 7), indexed by [k*2+i] where k=col of C, i=row of C
    
    The decomposition satisfies:
      sum_c U[i*2+j, c] * V[j*2+k, c] * W[k*2+i, c] = 1 for all (i,j,k) in {0,1}^3
    """
    # Strassen's 7 products:
    # p1 = (a00 + a10)(b00 + b10)
    # p2 = (a10 + a11)(b00 + b01)
    # p3 = (a00 + a01)(b10 + b11)
    # p4 = a00(b00 - b10)
    # p5 = a11(b01 - b00)
    # p6 = (a00 + a10)(b10 - b00)  -- wait, let me use a verified scheme
    
    # Here's a verified rank-7 scheme for 2x2*2x2->2x2:
    # Using the standard Strassen formulation:
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # p1 = (a + b)(e + f)
    # p2 = (c + d)(e + f)  
    # p3 = a(h - f)
    # p4 = d(g - e)
    # p5 = (a + e)(h + f)
    # p6 = (c + a)(f - h)
    # p7 = (b + d)(g - c)
    
    # C[0,0] = p1 + p4 - p5 + p7
    # C[0,1] = p5 - p1 + p2 + p6  -- wait, let me recheck
    # Actually: C[0,0] = p1 - p3 + p4 + p7? Let me just verify numerically.
    
    # Let me use a different, well-known explicit rank-7 decomposition.
    # I'll construct it carefully.
    
    # For 2x2*2x2->2x2, the tensor has 8 terms. We need to combine them into 7.
    # One known approach:
    
    # Term 0: covers (i,j,k) = (0,0,0) and (1,0,0)
    # U: U[0]=1 (i=0,j=0), U[2]=1 (i=1,j=0)
    # V: V[0]=1 (j=0,k=0)
    # W: W[0]=1 (k=0,i=0), W[1]=1 (k=0,i=1)
    # Check: T[0,0,0] += U[0]*V[0]*W[0] = 1*1*1 = 1
    #         T[1,0,0] += U[2]*V[0]*W[1] = 1*1*1 = 1
    #         T[0,0,1] += U[0]*V[1]*W[2] = 1*0*0 = 0
    #         T[1,0,1] += U[2]*V[1]*W[3] = 1*0*0 = 0
    #         T[0,1,0] += U[1]*V[2]*W[0] = 0*0*1 = 0
    #         T[1,1,0] += U[3]*V[2]*W[1] = 0*0*1 = 0
    #         T[0,1,1] += U[1]*V[3]*W[2] = 0*0*0 = 0
    #         T[1,1,1] += U[3]*V[3]*W[3] = 0*0*0 = 0
    # Good: covers (0,0,0) and (1,0,0).
    
    # Term 1: covers (0,1,1) and (1,1,1)
    # U: U[1]=1 (i=0,j=1), U[3]=1 (i=1,j=1)
    # V: V[3]=1 (j=1,k=1)
    # W: W[2]=1 (k=1,i=0), W[3]=1 (k=1,i=1)
    # Check: T[0,1,1] += U[1]*V[3]*W[2] = 1*1*1 = 1
    #         T[1,1,1] += U[3]*V[3]*W[3] = 1*1*1 = 1
    #         T[0,1,0] += U[1]*V[2]*W[0] = 1*0*0 = 0
    #         T[1,1,0] += U[3]*V[2]*W[1] = 1*0*0 = 0
    #         T[0,0,0] += U[0]*V[0]*W[0] = 0*0*0 = 0
    #         T[1,0,0] += U[2]*V[0]*W[1] = 0*0*0 = 0
    #         T[0,0,1] += U[0]*V[1]*W[2] = 0*0*1 = 0
    #         T[1,0,1] += U[2]*V[1]*W[3] = 0*0*1 = 0
    # Good: covers (0,1,1) and (1,1,1).
    
    # Remaining: (0,0,1), (0,1,0), (1,0,1), (1,1,0) -- 4 terms, need 5 more rank-1 terms.
    # But we only have 3 more terms (rank 7 total). So we need to combine some.
    
    # Term 2: covers (0,0,1)
    # U: U[0]=1 (i=0,j=0)
    # V: V[1]=1 (j=0,k=1)
    # W: W[2]=1 (k=1,i=0)
    # Check: T[0,0,1] += U[0]*V[1]*W[2] = 1*1*1 = 1
    #         All others: 0 (since only one nonzero in each vector)
    # Good.
    
    # Term 3: covers (0,1,0)
    # U: U[1]=1 (i=0,j=1)
    # V: V[2]=1 (j=1,k=0)
    # W: W[0]=1 (k=0,i=0)
    # Check: T[0,1,0] += U[1]*V[2]*W[0] = 1*1*1 = 1
    #         All others: 0
    # Good.
    
    # Term 4: covers (1,0,1)
    # U: U[2]=1 (i=1,j=0)
    # V: V[1]=1 (j=0,k=1)
    # W: W[3]=1 (k=1,i=1)
    # Check: T[1,0,1] += U[2]*V[1]*W[3] = 1*1*1 = 1
    #         All others: 0
    # Good.
    
    # Term 5: covers (1,1,0)
    # U: U[3]=1 (i=1,j=1)
    # V: V[2]=1 (j=1,k=0)
    # W: W[1]=1 (k=0,i=1)
    # Check: T[1,1,0] += U[3]*V[2]*W[1] = 1*1*1 = 1
    #         All others: 0
    # Good.
    
    # Wait, that's 6 terms covering all 8 entries. But rank is 7, not 6.
    # I must have made an error. Let me recheck term 0.
    
    # Term 0: U=[1,0,1,0], V=[1,0,0,0], W=[1,1,0,0]
    # T[0,0,0] += U[0]*V[0]*W[0] = 1*1*1 = 1
    # T[0,0,1] += U[0]*V[1]*W[2] = 1*0*0 = 0
    # T[0,1,0] += U[1]*V[2]*W[0] = 0*0*1 = 0
    # T[0,1,1] += U[1]*V[3]*W[2] = 0*0*0 = 0
    # T[1,0,0] += U[2]*V[0]*W[1] = 1*1*1 = 1
    # T[1,0,1] += U[2]*V[1]*W[3] = 1*0*0 = 0
    # T[1,1,0] += U[3]*V[2]*W[1] = 0*0*1 = 0
    # T[1,1,1] += U[3]*V[3]*W[3] = 0*0*0 = 0
    # OK, term 0 gives T[0,0,0]=1 and T[1,0,0]=1.
    
    # Term 1: U=[0,1,0,1], V=[0,0,0,1], W=[0,0,1,1]
    # T[0,0,0] += 0*0*0 = 0
    # T[0,0,1] += 0*0*1 = 0
    # T[0,1,0] += 1*0*0 = 0
    # T[0,1,1] += 1*1*1 = 1
    # T[1,0,0] += 0*0*0 = 0
    # T[1,0,1] += 0*0*1 = 0
    # T[1,1,0] += 1*0*0 = 0
    # T[1,1,1] += 1*1*1 = 1
    # OK, term 1 gives T[0,1,1]=1 and T[1,1,1]=1.
    
    # So far: (0,0,0), (1,0,0), (0,1,1), (1,1,1) covered.
    # Remaining: (0,0,1), (0,1,0), (1,0,1), (1,1,0).
    
    # Term 2: U=[1,0,0,0], V=[0,1,0,0], W=[0,0,1,0]
    # T[0,0,1] += 1*1*1 = 1. Others 0. Good.
    
    # Term 3: U=[0,1,0,0], V=[0,0,1,0], W=[0,1,0,0]
    # T[0,1,0] += 1*1*1 = 1. Others 0. Good.
    
    # Term 4: U=[0,0,1,0], V=[0,1,0,0], W=[0,0,0,1]
    # T[1,0,1] += 1*1*1 = 1. Others 0. Good.
    
    # Term 5: U=[0,0,0,1], V=[0,0,1,0], W=[0,1,0,0]
    # T[1,1,0] += 1*1*1 = 1. Others 0. Good.
    
    # That's 6 terms. But the rank is 7. I must be wrong about the tensor.
    
    # Actually, I think the issue is that the matrix multiplication tensor is NOT
    # the all-ones tensor. Let me reconsider.
    
    # The tensor M_{2,2,2} is defined as:
    # M = sum_{i,j,k} e_{(i,j)} ⊗ e_{(j,k)} ⊗ e_{(k,i)}
    
    # For 2x2*2x2->2x2, the indices are:
    # U index: (i,j) where i in {0,1} (row of A), j in {0,1} (col of A)
    # V index: (j,k) where j in {0,1} (row of B), k in {0,1} (col of B)
    # W index: (k,i) where k in {0,1} (col of C), i in {0,1} (row of C)
    
    # The tensor entry at (U_idx, V_idx, W_idx) is 1 if there exists (i,j,k) such that
    # U_idx = (i,j), V_idx = (j,k), W_idx = (k,i).
    
    # For a given (U_idx, V_idx, W_idx) = ((i1,j1), (j2,k2), (k3,i3)),
    # the entry is 1 iff i1=i3, j1=j2, k2=k3.
    
    # So the tensor is NOT all ones. It's a specific pattern.
    
    # For example, T[(0,0), (0,0), (0,0)] = 1 (i=0,j=0,k=0)
    # T[(0,0), (0,1), (1,0)] = 1 (i=0,j=0,k=1)
    # T[(0,0), (1,0), (0,0)] = 0 (j1=0 != j2=1)
    
    # So my decomposition above was for the all-ones tensor, not the matrix multiplication tensor.
    # I need to decompose the correct tensor.
    
    # Let me just use the schoolbook decomposition (rank 8) for 2x2*2x2 to be safe.
    # The total rank will be 96, same as full schoolbook.
    
    # Actually, let me just implement the full schoolbook decomposition for 3x4*4x8.
    # It's guaranteed correct with rank 96.
    
    U = np.zeros((4, 8))
    V = np.zeros((4, 8))
    W = np.zeros((4, 8))
    col = 0
    for i in range(2):
        for j in range(2):
            for k in range(2):
                U[i*2+j, col] = 1
                V[j*2+k, col] = 1
                W[k*2+i, col] = 1
                col += 1
    return U, V, W
=======
def _strassen_2x2x2():
    """
    Return U, V, W for 2x2 * 2x2 -> 2x2 with rank 7 (Strassen).
    
    U shape: (4, 7), indexed by [i*2+j] where i=row of A, j=col of A
    V shape: (4, 7), indexed by [j*2+k] where j=row of B, k=col of B
    W shape: (4, 7), indexed by [k*2+i] where k=col of C, i=row of C
    
    The decomposition satisfies:
      sum_c U[i*2+j, c] * V[j*2+k, c] * W[k*2+i, c] = 1 for all (i,j,k) in {0,1}^3
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # p1 = (a+d)(e+g)
    # A: a=1, d=1 -> U[0]=1, U[3]=1
    # B: e=1, g=1 -> V[0]=1, V[2]=1
    # C: C00=1, C11=1 -> W[0]=1, W[3]=1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[2, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # p2 = (c+d)e
    # A: c=1, d=1 -> U[2]=1, U[3]=1
    # B: e=1 -> V[0]=1
    # C: C10=1, C11=-1 -> W[1]=1, W[3]=-1
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    # p3 = a(f-h)
    # A: a=1 -> U[0]=1
    # B: f=1, h=-1 -> V[1]=1, V[3]=-1
    # C: C01=1, C11=1 -> W[2]=1, W[3]=1
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    # p4 = d(g-e)
    # A: d=1 -> U[3]=1
    # B: g=1, e=-1 -> V[2]=1, V[0]=-1
    # C: C00=1, C10=1 -> W[0]=1, W[1]=1
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    # p5 = (a+b)h
    # A: a=1, b=1 -> U[0]=1, U[1]=1
    # B: h=1 -> V[3]=1
    # C: C00=-1, C01=1 -> W[0]=-1, W[2]=1
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    # p6 = (c-a)(e+f)
    # A: c=1, a=-1 -> U[2]=1, U[0]=-1
    # B: e=1, f=1 -> V[0]=1, V[1]=1
    # C: C11=1 -> W[3]=1
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # p7 = (b-d)(g+h)
    # A: b=1, d=-1 -> U[1]=1, U[3]=-1
    # B: g=1, h=1 -> V[2]=1, V[3]=1
    # C: C00=1 -> W[0]=1
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W
>>>>>>> REPLACE

