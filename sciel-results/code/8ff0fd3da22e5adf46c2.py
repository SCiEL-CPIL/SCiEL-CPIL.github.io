def construct():
    n, m, p = 2, 5, 5
    rank = n * m  # 10
    
    U = np.zeros((n*m, rank), dtype=np.float64)
    V = np.zeros((m*p, rank), dtype=np.float64)
    W = np.zeros((p*n, rank), dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            r = i * m + j
            # U index: a = i*m + j
            U[i*m + j, r] = 1.0
            # V index: b = j*p + k? 
            # Wait, V depends on j and k.
            # In the sum C_ik = sum_j A_ij B_jk, the term A_ij B_jk uses index j for A and B.
            # My mapping r = i*m + j fixes i and j.
            # For a fixed r (fixed i, j), V should be 1 for all k?
            # Let's re-evaluate.
            
            # T[5i+j, 5j+k, 5k+i] = 1.
            # We want sum_r U[5i+j, r] V[5j+k, r] W[5k+i, r] = 1.
            
            # If r = 5i + j:
            # U[5i+j, 5i+j] = 1.
            # We need V[5j+k, 5i+j] = 1 for all k?
            # And W[5k+i, 5i+j] = 1 for all k?
            
            # Let's check if this works for all k.
            # For a fixed i, j, the tensor entries are at (5i+j, 5j+k, 5k+i) for k=0..4.
            # If we set V[5j+k, 5i+j] = 1 for all k=0..4, and W[5k+i, 5i+j] = 1 for all k=0..4.
            # Then for entry (5i+j, 5j+k, 5k+i):
            # Sum_r ... = U[5i+j, 5i+j] * V[5j+k, 5i+j] * W[5k+i, 5i+j] = 1 * 1 * 1 = 1.
            
            # What about other entries?
            # If a != 5i+j, U[a, r] = 0 for r=5i+j?
            # Yes, U is identity.
            
            # So for each i, j, we add a rank-1 component.
            # U has 1 at (5i+j, 5i+j).
            # V has 1s at (5j+0, 5i+j), (5j+1, 5i+j), ..., (5j+4, 5i+j).
            # W has 1s at (0*2+i, 5i+j), (1*2+i, 5i+j), ..., (4*2+i, 5i+j).
            
            # Let's verify no cross-talk.
            # Take an entry T[a, b, c] where T=1.
            # a=5i+j, b=5j+k, c=5k+i.
            # Only r=5i+j has U[a,r]!=0.
            # So only r=5i+j contributes.
            # V[b, r] = V[5j+k, 5i+j] = 1.
            # W[c, r] = W[5k+i, 5i+j] = 1.
            # Sum = 1. Correct.
            
            # Take an entry T[a, b, c] where T=0.
            # Case 1: a is not of form 5i'+j' consistent with b, c?
            # Actually, if T[a,b,c]=0, it means there is no i,j,k such that a=5i+j, b=5j+k, c=5k+i.
            # If U[a, r] != 0, then r=a=5i+j for some i,j.
            # Then we need V[b, r] W[c, r] != 0 for the sum to be non-zero.
            # V[b, 5i+j] is 1 if b = 5j+k for some k. i.e., b mod 5 == j.
            # W[c, 5i+j] is 1 if c = 5k+i for some k. i.e., c mod 2 == i.
            # If both are 1, then b=5j+k, c=5k+i.
            # This implies a=5i+j, b=5j+k, c=5k+i, which means T[a,b,c] would be 1.
            # Contradiction. So if T=0, the sum is 0.
            
            pass

    # Fill the matrices
    for i in range(n):
        for j in range(m):
            r = i * m + j
            U[i*m + j, r] = 1.0
            
            for k in range(p):
                V[j*p + k, r] = 1.0
                W[k*n + i, r] = 1.0

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
