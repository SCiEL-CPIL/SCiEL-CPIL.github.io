"""Matrix multiplication tensor decomposition for 2x3x8."""
import time
import numpy as np

DIMENSIONS = [2, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145.0

    # Build target tensor
    M = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                M[i*m+j, j*p+k, k*n+i] = 1.0

    # Strategy: split 3x8 into two 3x4 blocks
    # Block 0: rows j=0,1 (2 rows), cols k=0..3 (4 cols) -> 2x2x4 tensor
    # Block 1: rows j=1,2 (2 rows), cols k=4..7 (4 cols) -> 2x2x4 tensor
    # But j=1 is shared, so we need to handle overlap carefully
    # 
    # Alternative: split by columns only
    # Left block: j=0..2, k=0..3 -> 3x4
    # Right block: j=0..2, k=4..7 -> 3x4
    # Each 3x4 block can be split into 2x2x4 + 2x1x4
    #
    # Let's try: 2x3x8 = 2x3x4 + 2x3x4 (two column blocks)
    # For each 2x3x4: split rows into 2+1
    #   2x2x4 (Strassen) + 2x1x4 (schoolbook)
    #
    # 2x2x4 tensor: rank? We can use 7-term Strassen-like or find minimal
    # Actually for 2x2x4, the tensor is:
    # T[i,j,k] = 1 if (i,j,k) corresponds to (A[i,j]*B[j,k])
    # where A is 2x2, B is 2x4
    # This has 2*2*4 = 16 entries, but many are zero
    # Non-zero: for each j, we have A[i,j] * B[j,k]
    # So it's sum over j of (column j of A) outer (row j of B)
    # That's rank at most 2 (one for each j)
    # Wait, that's the rank of the bilinear map, not tensor rank
    #
    # Let me think differently. The tensor for 2x2x4 multiplication:
    # M224[i,j,k] = 1 if i in {0,1}, j in {0,1}, k in {0,1,2,3}
    # and the entry is 1 when this corresponds to A[i,j]*B[j,k]
    # Actually the standard matrix mult tensor has:
    # T[a,b,c] = 1 where a = i*m+j, b = j*p+k, c = k*n+i
    # For 2x2x4: n=2, m=2, p=4
    # a = i*2+j (i in 0,1; j in 0,1) -> 4 values
    # b = j*4+k (j in 0,1; k in 0..3) -> 8 values  
    # c = k*2+i (k in 0..3; i in 0,1) -> 8 values
    # T[a,b,c] = 1 when a=i*2+j, b=j*4+k, c=k*2+i
    #
    # This is a 4x8x8 tensor. Its rank is at most 4*8*8=256 but much less.
    # Actually it's a structured tensor. Let me just try to find a low-rank decomposition.
    
    # Try to find rank-r decomposition for 2x2x4
    n2, m2, p2 = 2, 2, 4
    M224 = np.zeros((n2*m2, m2*p2, p2*n2))
    for i in range(n2):
        for j in range(m2):
            for k in range(p2):
                M224[i*m2+j, j*p2+k, k*n2+i] = 1.0
    
    # Try ranks 8, 9, 10, 11 for 2x2x4
    best_rank_224 = None
    best_decomp_224 = None
    
    for target_rank in [8, 9, 10, 11, 12]:
        if time.time() > deadline - 30:
            break
        np.random.seed(42 + target_rank)
        U224 = np.random.randn(n2*m2, target_rank)
        V224 = np.random.randn(m2*p2, target_rank)
        W224 = np.random.randn(p2*n2, target_rank)
        
        for it in range(200):
            if time.time() > deadline - 20:
                break
            # ALS update for U
            for r in range(target_rank):
                O_r = np.outer(V224[:,r], W224[:,r])
                R = np.zeros((n2*m2, m2*p2, p2*n2))
                for r2 in range(target_rank):
                    if r2 != r:
                        for a in range(n2*m2):
                            R[a,:,:] += U224[a,r2] * np.outer(V224[:,r2], W224[:,r2])
                for a in range(n2*m2):
                    residual = M224[a,:,:] - R[a,:,:]
                    denom = np.sum(O_r * O_r)
                    U224[a,r] = np.sum(residual * O_r) / denom if denom != 0 else 0
            
            # ALS update for V
            for r in range(target_rank):
                P_r = np.outer(U224[:,r], W224[:,r])
                S = np.zeros((n2*m2, m2*p2, p2*n2))
                for r2 in range(target_rank):
                    if r2 != r:
                        for b in range(m2*p2):
                            S[:,b,:] += V224[b,r2] * np.outer(U224[:,r2], W224[:,r2])
                for b in range(m2*p2):
                    residual = M224[:,b,:] - S[:,b,:]
                    denom = np.sum(P_r * P_r)
                    V224[b,r] = np.sum(residual * P_r) / denom if denom != 0 else 0
            
            # ALS update for W
            for r in range(target_rank):
                P_r = np.outer(U224[:,r], V224[:,r])
                S = np.zeros((n2*m2, m2*p2, p2*n2))
                for r2 in range(target_rank):
                    if r2 != r:
                        for c in range(p2*n2):
                            S[:,:,c] += W224[c,r2] * np.outer(U224[:,r2], V224[:,r2])
                for c in range(p2*n2):
                    residual = M224[:,:,c] - S[:,:,c]
                    denom = np.sum(P_r * P_r)
                    W224[c,r] = np.sum(residual * P_r) / denom if denom != 0 else 0
        
        # Check
        M224_recon = np.zeros((n2*m2, m2*p2, p2*n2))
        for r in range(target_rank):
            M224_recon += np.einsum('a,b,c->abc', U224[:,r], V224[:,r], W224[:,r])
        
        if np.allclose(M224_recon, M224, atol=1e-5):
            best_rank_224 = target_rank
            best_decomp_224 = (U224.copy(), V224.copy(), W224.copy())
            break
    
    if best_decomp_224 is not None:
        U224, V224, W224 = best_decomp_224
        r224 = best_rank_224
        
        # Now construct full 2x3x8 decomposition
        # Split 3x8 into two 3x4 column blocks
        # Left block: j=0,1,2; k=0,1,2,3
        # Right block: j=0,1,2; k=4,5,6,7
        #
        # For each 3x4 block, split rows:
        #   Top 2 rows (j=0,1) -> 2x2x4
        #   Bottom 1 row (j=2) -> 2x1x4
        #
        # Left block (k=0..3):
        #   2x2x4 part: i=0,1; j=0,1; k=0..3
        #   2x1x4 part: i=0,1; j=2; k=0..3
        # Right block (k=4..7):
        #   2x2x4 part: i=0,1; j=0,1; k=4..7
        #   2x1x4 part: i=0,1; j=2; k=4..7
        #
        # Total rank = 2 * (r224 + 2*4) = 2 * (r224 + 8)
        
        rank_214 = 2 * 4  # 2x1x4 schoolbook
        rank_total = 2 * (r224 + rank_214)
        
        U = np.zeros((n*m, rank_total))
        V = np.zeros((m*p, rank_total))
        W = np.zeros((p*n, rank_total))
        
        for block in range(2):
            k_offset = block * 4
            base_col = block * (r224 + rank_214)
            
            # 2x2x4 part for this block
            for t in range(r224):
                col = base_col + t
                # U: i in {0,1}, j in {0,1} -> indices i*3 + j
                for li in range(2):
                    for lj in range(2):
                        a_local = li*2 + lj
                        u_global = li*3 + lj
                        U[u_global, col] = U224[a_local, t]
                
                # V: j in {0,1}, k in {0..3} -> indices j*4 + k, but k is local
                for lj in range(2):
                    for lk in range(4):
                        b_local = lj*4 + lk
                        v_global = lj*8 + (k_offset + lk)
                        V[v_global, col] = V224[b_local, t]
                
                # W: k in {0..3}, i in {0,1} -> indices k*2 + i, but k is local
                for lk in range(4):
                    for li in range(2):
                        c_local = lk*2 + li
                        w_global = (k_offset + lk)*2 + li
                        W[w_global, col] = W224[c_local, t]
            
            # 2x1x4 schoolbook part for this block
            for i in range(2):
                for lk in range(4):
                    col = base_col + r224 + i*4 + lk
                    U[i*3 + 2, col] = 1  # j=2
                    V[2*8 + (k_offset + lk), col] = 1  # j=2, k=k_offset+lk
                    W[(k_offset + lk)*2 + i, col] = 1
        
        # Verify
        M_full_recon = np.zeros((n*m, m*p, p*n))
        for r in range(rank_total):
            M_full_recon += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
        
        if np.allclose(M_full_recon, M, atol=1e-5):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: schoolbook
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
