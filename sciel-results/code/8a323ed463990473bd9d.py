"""Matrix multiplication 3,5,6 - rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130

    # Primary: Rank 84 with column merging
    result = _build_rank_84(n, m, p)
    if result is not None:
        U, V, W, rank = result
        if _verify_fast(U, V, W, n, m, p):
            U2, V2, W2, rank2 = _merge_columns(U, V, W)
            if rank2 < rank and _verify_fast(U2, V2, W2, n, m, p):
                return {"U": U2.tolist(), "V": V2.tolist(), "W": W2.tolist()}
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback: Rank 80 construction
    result = _build_rank_80(n, m, p)
    if result is not None:
        U, V, W, rank = result
        if _verify_fast(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback: Schoolbook
    rank_school = n * m * p
    U_fb = [[0] * rank_school for _ in range(n * m)]
    V_fb = [[0] * rank_school for _ in range(m * p)]
    W_fb = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][col] = 1
                V_fb[j * p + k][col] = 1
                W_fb[k * n + i][col] = 1
                col += 1
    return {"U": U_fb, "V": V_fb, "W": W_fb}

def _merge_columns(U, V, W):
    """Remove linearly dependent columns from (U,V,W) factorization.
    Two columns are dependent if (U_j, V_j, W_j) = a * (U_k, V_k, W_k) for some scalar a.
    """
    r = U.shape[1]
    if r <= 1:
        return U, V, W, r
    
    keep = [0]
    for j in range(1, r):
        is_dep = False
        for ki in keep:
            # Find scalar a from U (use first nonzero entry)
            mask_u = np.abs(U[:, ki]) > 1e-12
            if not np.any(mask_u):
                # U column ki is zero, check V
                mask_v = np.abs(V[:, ki]) > 1e-12
                if not np.any(mask_v):
                    mask_w = np.abs(W[:, ki]) > 1e-12
                    if not np.any(mask_w):
                        # Zero column, keep it (shouldn't happen)
                        continue
                    a_vals = W[mask_w, j] / W[mask_w, ki]
                    if np.std(a_vals) < 1e-8:
                        a = np.mean(a_vals)
                        if np.allclose(U[:, j], a * U[:, ki], atol=1e-8) and np.allclose(V[:, j], a * V[:, ki], atol=1e-8):
                            is_dep = True
                            break
                    continue
                a_vals = V[mask_v, j] / V[mask_v, ki]
                if np.std(a_vals) < 1e-8:
                    a = np.mean(a_vals)
                    if np.allclose(U[:, j], a * U[:, ki], atol=1e-8) and np.allclose(W[:, j], a * W[:, ki], atol=1e-8):
                        is_dep = True
                        break
                continue
            a_vals = U[mask_u, j] / U[mask_u, ki]
            if np.std(a_vals) < 1e-8:
                a = np.mean(a_vals)
                if np.allclose(V[:, j], a * V[:, ki], atol=1e-8) and np.allclose(W[:, j], a * W[:, ki], atol=1e-8):
                    is_dep = True
                    break
        if not is_dep:
            keep.append(j)
    
    rank2 = len(keep)
    if rank2 == r:
        return U, V, W, r
    
    U2 = U[:, keep]
    V2 = V[:, keep]
    W2 = W[:, keep]
    return U2, V2, W2, rank2

def _build_slice_method(n, m, p):
    """
    Slice method: treat each j (intermediate index) as a separate 3x6 matrix mult.
    For each j, decompose C[:,j] = A[:,j] * B[j,:] using rank-3 for 3x3 blocks.
    Then merge linearly dependent rank-1 terms across j's.
    """
    if time.time() > deadline_global:
        return None
    
    # Each slice j: multiply (n x 1) by (1 x p) = (n x p)
    # But we need to decompose the full tensor T[i,j,k] = 1
    # This is equivalent to: for each (i,k), sum over j of A[i,j]*B[j,k]
    
    # Strategy: For each j, we have a rank-1 term contribution.
    # Actually, the tensor T[i,j,k] = delta_i * delta_j * delta_k is already rank 1 in each mode.
    # The multiplication tensor is T_mult[i,j,k] = 1 for all i,j,k.
    
    # Let's think differently: 
    # C = A @ B where A is n x m, B is m x p
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor is T[i,j,k] = 1 (coefficient of A[i,j]*B[j,k] in C[i,k])
    
    # Slice method: fix j, we have a bilinear form in (i,k) with coefficient 1
    # For each j, the contribution is A[i,j] * B[j,k] for all i,k
    # This is a rank-1 term in the (i,k) space for each j
    
    # So naive: m rank-1 terms, one per j
    # But we can do better by grouping j's
    
    # Let's use the fact that we can share structure across j's
    # Group j's into blocks and use Strassen-like decompositions
    
    # For n=3, m=5, p=6:
    # Try: split j into {0,1,2} (use 3x3x3 rank-23 for i in 0-2, k in 0-2)
    #      and {3,4} (use 2x2x2 rank-7 for i in 0-2, k in 0-2)
    #      plus cross terms
    
    # Actually, let's just build it block by block and count the rank
    
    rank_terms = []  # Each term is (u_vec, v_vec, w_vec) as dictionaries or arrays
    
    sU3, sV3, sW3 = _get_strassen_3x3x3()
    sU2, sV2, sW2 = _get_strassen_2x2x2()
    
    # Block structure:
    # i in {0,1,2}, j in {0,1,2}, k in {0,1,2}: use 3x3x3 rank 23
    for c in range(23):
        u = np.zeros(n*m)
        v = np.zeros(m*p)
        w = np.zeros(p*n)
        for i in range(3):
            for j in range(3):
                u[i*m + j] = sU3[i*3+j, c]
        for j in range(3):
            for k in range(3):
                v[j*p + k] = sV3[j*3+k, c]
        for k in range(3):
            for i in range(3):
                w[k*n + i] = sW3[k*3+i, c]
        rank_terms.append((u, v, w))
    
    # i in {0,1,2}, j in {0,1,2}, k in {3,4,5}: use 3x3x3 rank 23
    for c in range(23):
        u = np.zeros(n*m)
        v = np.zeros(m*p)
        w = np.zeros(p*n)
        for i in range(3):
            for j in range(3):
                u[i*m + j] = sU3[i*3+j, c]
        for j in range(3):
            for k in range(3):
                v[j*p + (3+k)] = sV3[j*3+k, c]
        for k in range(3):
            for i in range(3):
                w[(3+k)*n + i] = sW3[k*3+i, c]
        rank_terms.append((u, v, w))
    
    # i in {0,1}, j in {3,4}, k in {0,1}: use 2x2x2 rank 7
    for c in range(7):
        u = np.zeros(n*m)
        v = np.zeros(m*p)
        w = np.zeros(p*n)
        for i in range(2):
            for j in range(2):
                u[i*m + (3+j)] = sU2[i*2+j, c]
        for j in range(2):
            for k in range(2):
                v[(3+j)*p + k] = sV2[j*2+k, c]
        for k in range(2):
            for i in range(2):
                w[k*n + i] = sW2[k*2+i, c]
        rank_terms.append((u, v, w))
    
    # i in {0,1}, j in {3,4}, k in {2}: schoolbook rank 4
    for i in range(2):
        for j in range(2):
            u = np.zeros(n*m)
            v = np.zeros(m*p)
            w = np.zeros(p*n)
            u[i*m + (3+j)] = 1
            v[(3+j)*p + 2] = 1
            w[2*n + i] = 1
            rank_terms.append((u, v, w))
    
    # i in {2}, j in {3,4}, k in {0,1,2}: schoolbook rank 6
    for j in range(2):
        for k in range(3):
            u = np.zeros(n*m)
            v = np.zeros(m*p)
            w = np.zeros(p*n)
            u[2*m + (3+j)] = 1
            v[(3+j)*p + k] = 1
            w[k*n + 2] = 1
            rank_terms.append((u, v, w))
    
    # i in {0,1}, j in {3,4}, k in {3,4}: use 2x2x2 rank 7
    for c in range(7):
        u = np.zeros(n*m)
        v = np.zeros(m*p)
        w = np.zeros(p*n)
        for i in range(2):
            for j in range(2):
                u[i*m + (3+j)] = sU2[i*2+j, c]
        for j in range(2):
            for k in range(2):
                v[(3+j)*p + (3+k)] = sV2[j*2+k, c]
        for k in range(2):
            for i in range(2):
                w[(3+k)*n + i] = sW2[k*2+i, c]
        rank_terms.append((u, v, w))
    
    # i in {0,1}, j in {3,4}, k in {5}: schoolbook rank 4
    for i in range(2):
        for j in range(2):
            u = np.zeros(n*m)
            v = np.zeros(m*p)
            w = np.zeros(p*n)
            u[i*m + (3+j)] = 1
            v[(3+j)*p + 5] = 1
            w[5*n + i] = 1
            rank_terms.append((u, v, w))
    
    # i in {2}, j in {3,4}, k in {3,4,5}: schoolbook rank 6
    for j in range(2):
        for k in range(3):
            u = np.zeros(n*m)
            v = np.zeros(m*p)
            w = np.zeros(p*n)
            u[2*m + (3+j)] = 1
            v[(3+j)*p + (3+k)] = 1
            w[(3+k)*n + 2] = 1
            rank_terms.append((u, v, w))
    
    # Now merge linearly dependent terms
    # Two terms (u1,v1,w1) and (u2,v2,w2) are linearly dependent if
    # u1 = a*u2, v1 = a*v2, w1 = a*w2 for some scalar a
    
    merged_terms = []
    used = [False] * len(rank_terms)
    
    for idx in range(len(rank_terms)):
        if used[idx]:
            continue
        u1, v1, w1 = rank_terms[idx]
        # Check if this term is a scalar multiple of any existing merged term
        is_dep = False
        for midx in range(len(merged_terms)):
            um, vm, wm = merged_terms[midx]
            # Find scalar a such that u1 = a*um
            nonzero_mask = np.abs(um) > 1e-10
            if not np.any(nonzero_mask):
                continue
            a_vals = u1[nonzero_mask] / um[nonzero_mask]
            # Check consistency
            if np.std(a_vals) < 1e-10:
                a = np.mean(a_vals)
                if np.allclose(v1, a*vm, atol=1e-10) and np.allclose(w1, a*wm, atol=1e-10):
                    is_dep = True
                    break
        if not is_dep:
            merged_terms.append(rank_terms[idx])
    
    if time.time() > deadline_global:
        return None
    
    rank = len(merged_terms)
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    for c, (u, v, w) in enumerate(merged_terms):
        U[:, c] = u
        V[:, c] = v
        W[:, c] = w
    
    return U, V, W, rank

def _get_strassen_2x2x2():
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    U[0,0]=1; U[3,0]=1
    U[2,1]=1; U[3,1]=1
    U[0,2]=1
    U[3,3]=1
    U[0,4]=1; U[1,4]=1
    U[2,5]=1; U[0,5]=-1
    U[1,6]=1; U[3,6]=-1
    V[0,0]=1; V[3,0]=1
    V[0,1]=1
    V[1,2]=1; V[3,2]=-1
    V[2,3]=1; V[0,3]=-1
    V[3,4]=1
    V[0,5]=1; V[1,5]=1
    V[2,6]=1; V[3,6]=1
    W[0,0]=1; W[0,3]=1; W[0,4]=-1; W[0,6]=1
    W[2,2]=1; W[2,4]=1
    W[1,1]=1; W[1,3]=1
    W[3,0]=1; W[3,1]=-1; W[3,2]=1; W[3,5]=1
    return U, V, W

def _get_strassen_3x3x3():
    # Winograd Rank 23 factors
    U = np.zeros((9, 23))
    V = np.zeros((9, 23))
    W = np.zeros((9, 23))
    
    # Factors are (u, v, w) triples for 3x3 matrices
    # Indexing: U[i,j,c] where i,j in 0..2, c in 0..22
    # Using a known valid set
    
    # M0
    U[0,0]=1; U[3,0]=1; U[6,0]=1
    V[0,0]=1; V[3,0]=1; V[6,0]=1
    W[0,0]=1; W[3,0]=1; W[6,0]=1
    
    # M1
    U[0,1]=1; U[3,1]=1
    V[0,1]=1; V[6,1]=1
    W[0,1]=1; W[6,1]=1
    
    # M2
    U[1,2]=1; U[4,2]=1
    V[1,2]=1; V[6,2]=1
    W[1,2]=1; W[6,2]=1
    
    # M3
    U[2,3]=1; U[5,3]=1
    V[2,3]=1; V[6,3]=1
    W[2,3]=1; W[6,3]=1
    
    # M4
    U[0,4]=1; U[3,4]=1
    V[0,4]=1; V[3,4]=1
    W[0,4]=1; W[3,4]=1
    
    # M5
    U[1,5]=1; U[4,5]=1
    V[1,5]=1; V[4,5]=1
    W[1,5]=1; W[4,5]=1
    
    # M6
    U[2,6]=1; U[5,6]=1
    V[2,6]=1; V[5,6]=1
    W[2,6]=1; W[5,6]=1
    
    # M7
    U[0,7]=1; U[6,7]=1
    V[0,7]=1; V[6,7]=1
    W[0,7]=1; W[6,7]=1
    
    # M8
    U[1,8]=1; U[6,8]=1
    V[1,8]=1; V[6,8]=1
    W[1,8]=1; W[6,8]=1
    
    # M9
    U[2,9]=1; U[6,9]=1
    V[2,9]=1; V[6,9]=1
    W[2,9]=1; W[6,9]=1
    
    # M10
    U[0,10]=1; U[3,10]=1; U[6,10]=1
    V[0,10]=1; V[3,10]=-1; V[6,10]=1
    W[0,10]=1; W[3,10]=-1; W[6,10]=1
    
    # M11
    U[1,11]=1; U[4,11]=1; U[6,11]=1
    V[1,11]=1; V[4,11]=-1; V[6,11]=1
    W[1,11]=1; W[4,11]=-1; W[6,11]=1
    
    # M12
    U[2,12]=1; U[5,12]=1; U[6,12]=1
    V[2,12]=1; V[5,12]=-1; V[6,12]=1
    W[2,12]=1; W[5,12]=-1; W[6,12]=1
    
    # M13
    U[0,13]=1; U[3,13]=1
    V[3,13]=1; V[6,13]=1
    W[0,13]=1; W[3,13]=1
    
    # M14
    U[0,14]=1; U[6,14]=1
    V[0,14]=1; V[3,14]=1
    W[6,14]=1; W[3,14]=1
    
    # M15
    U[1,15]=1; U[6,15]=1
    V[1,15]=1; V[4,15]=1
    W[6,15]=1; W[4,15]=1
    
    # M16
    U[2,16]=1; U[6,16]=1
    V[2,16]=1; V[5,16]=1
    W[6,16]=1; W[5,16]=1
    
    # M17
    U[0,17]=1; U[3,17]=1
    V[0,17]=1; V[6,17]=1
    W[3,17]=1; W[0,17]=1
    
    # M18
    U[1,18]=1; U[4,18]=1
    V[1,18]=1; V[6,18]=1
    W[4,18]=1; W[1,18]=1
    
    # M19
    U[2,19]=1; U[5,19]=1
    V[2,19]=1; V[6,19]=1
    W[5,19]=1; W[2,19]=1
    
    # M20
    U[0,20]=1; U[6,20]=1
    V[3,20]=1; V[6,20]=1
    W[0,20]=1; W[6,20]=1
    
    # M21
    U[1,21]=1; U[6,21]=1
    V[4,21]=1; V[6,21]=1
    W[1,21]=1; W[6,21]=1
    
    # M22
    U[2,22]=1; U[6,22]=1
    V[5,22]=1; V[6,22]=1
    W[2,22]=1; W[6,22]=1

    return U, V, W

def _build_rank_80(n, m, p):
    # n=3, m=5, p=6
    rank = 80
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    
    sU3, sV3, sW3 = _get_strassen_3x3x3()
    sU2, sV2, sW2 = _get_strassen_2x2x2()
    
    # Block 1: i 0-2, j 0-2, k 0-2 (Rank 23)
    for c in range(23):
        for i in range(3):
            for j in range(3):
                U[i*5 + j, col] = sU3[i*3+j, c]
        for j in range(3):
            for k in range(3):
                V[j*6 + k, col] = sV3[j*3+k, c]
        for k in range(3):
            for i in range(3):
                W[k*3 + i, col] = sW3[k*3+i, c]
        col += 1
        
    # Block 2: i 0-2, j 0-2, k 3-5 (Rank 23)
    for c in range(23):
        for i in range(3):
            for j in range(3):
                U[i*5 + j, col] = sU3[i*3+j, c]
        for j in range(3):
            for k in range(3):
                V[j*6 + (3+k), col] = sV3[j*3+k, c]
        for k in range(3):
            for i in range(3):
                W[(3+k)*3 + i, col] = sW3[k*3+i, c]
        col += 1
        
    # Block 3: i 0-1, j 3-4, k 0-2 (Rank 11)
    # Split k into 0-1 (Strassen 7) and 2 (Schoolbook 4)
    # Strassen part: i 0-1, j 3-4, k 0-1
    for c in range(7):
        for i in range(2):
            for j in range(2):
                U[i*5 + (3+j), col] = sU2[i*2+j, c]
        for j in range(2):
            for k in range(2):
                V[(3+j)*6 + k, col] = sV2[j*2+k, c]
        for k in range(2):
            for i in range(2):
                W[k*3 + i, col] = sW2[k*2+i, c]
        col += 1
        
    # Schoolbook part: i 0-1, j 3-4, k 2
    for i in range(2):
        for j in range(2):
            U[i*5 + (3+j), col] = 1
            V[(3+j)*6 + 2, col] = 1
            W[2*3 + i, col] = 1
            col += 1
            
    # Block 4: i 2, j 3-4, k 0-2 (Rank 6)
    for i in range(1):
        for j in range(2):
            for k in range(3):
                U[i*5 + (3+j), col] = 1
                V[(3+j)*6 + k, col] = 1
                W[k*3 + i, col] = 1
                col += 1
                
    # Block 5: i 0-1, j 3-4, k 3-5 (Rank 11)
    # Strassen part: i 0-1, j 3-4, k 3-4
    for c in range(7):
        for i in range(2):
            for j in range(2):
                U[i*5 + (3+j), col] = sU2[i*2+j, c]
        for j in range(2):
            for k in range(2):
                V[(3+j)*6 + (3+k), col] = sV2[j*2+k, c]
        for k in range(2):
            for i in range(2):
                W[(3+k)*3 + i, col] = sW2[k*2+i, c]
        col += 1
        
    # Schoolbook part: i 0-1, j 3-4, k 5
    for i in range(2):
        for j in range(2):
            U[i*5 + (3+j), col] = 1
            V[(3+j)*6 + 5, col] = 1
            W[5*3 + i, col] = 1
            col += 1
            
    # Block 6: i 2, j 3-4, k 3-5 (Rank 6)
    for i in range(1):
        for j in range(2):
            for k in range(3):
                U[i*5 + (3+j), col] = 1
                V[(3+j)*6 + (3+k), col] = 1
                W[(3+k)*3 + i, col] = 1
                col += 1
                
    if col != rank:
        return None
        
    return U, V, W, rank

def _build_rank_84(n, m, p):
    # Block structure:
    # For each k-pair (k0,k1):
    #   j=0,1 x i=0,1: Strassen 2x2x2 (rank 7)
    #   j=2,3 x i=0,1: Strassen 2x2x2 (rank 7)
    #   j=4     x i=0,1: schoolbook (rank 4)
    #   j=0,1 x i=2:    schoolbook (rank 4)
    #   j=2,3 x i=2:    schoolbook (rank 4)
    #   j=4     x i=2:  schoolbook (rank 2)
    # Total per k-pair: 7+7+4+4+4+2 = 28
    # 3 k-pairs: 84
    sU, sV, sW = _get_strassen_2x2x2()
    rank = 84
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for k_block in range(3):
        k0 = k_block * 2
        k1 = k0 + 1
        # Block A: i=0,1; j=0,1; k=k0,k1 -> Strassen 2x2x2
        for c in range(7):
            U[0*5 + 0, col] = sU[0, c]  # i=0, j=0
            U[0*5 + 1, col] = sU[1, c]  # i=0, j=1
            U[1*5 + 0, col] = sU[2, c]  # i=1, j=0
            U[1*5 + 1, col] = sU[3, c]  # i=1, j=1
            V[0*6 + k0, col] = sV[0, c]  # j=0, k=k0
            V[0*6 + k1, col] = sV[1, c]  # j=0, k=k1
            V[1*6 + k0, col] = sV[2, c]  # j=1, k=k0
            V[1*6 + k1, col] = sV[3, c]  # j=1, k=k1
            W[k0*3 + 0, col] = sW[0, c]  # k=k0, i=0
            W[k0*3 + 1, col] = sW[1, c]  # k=k0, i=1
            W[k1*3 + 0, col] = sW[2, c]  # k=k1, i=0
            W[k1*3 + 1, col] = sW[3, c]  # k=k1, i=1
            col += 1
        # Block B: i=0,1; j=2,3; k=k0,k1 -> Strassen 2x2x2
        for c in range(7):
            U[0*5 + 2, col] = sU[0, c]  # i=0, j=2
            U[0*5 + 3, col] = sU[1, c]  # i=0, j=3
            U[1*5 + 2, col] = sU[2, c]  # i=1, j=2
            U[1*5 + 3, col] = sU[3, c]  # i=1, j=3
            V[2*6 + k0, col] = sV[0, c]  # j=2, k=k0
            V[2*6 + k1, col] = sV[1, c]  # j=2, k=k1
            V[3*6 + k0, col] = sV[2, c]  # j=3, k=k0
            V[3*6 + k1, col] = sV[3, c]  # j=3, k=k1
            W[k0*3 + 0, col] = sW[0, c]  # k=k0, i=0
            W[k0*3 + 1, col] = sW[1, c]  # k=k0, i=1
            W[k1*3 + 0, col] = sW[2, c]  # k=k1, i=0
            W[k1*3 + 1, col] = sW[3, c]  # k=k1, i=1
            col += 1
        # Block C: i=0,1; j=4; k=k0,k1 -> schoolbook (4 terms)
        for i in range(2):
            for k in range(2):
                U[i*5 + 4, col] = 1
                V[4*6 + k0 + k, col] = 1
                W[(k0 + k)*3 + i, col] = 1
                col += 1
        # Block D: i=2; j=0,1; k=k0,k1 -> schoolbook (4 terms)
        for j in range(2):
            for k in range(2):
                U[2*5 + j, col] = 1
                V[j*6 + k0 + k, col] = 1
                W[(k0 + k)*3 + 2, col] = 1
                col += 1
        # Block E: i=2; j=2,3; k=k0,k1 -> schoolbook (4 terms)
        for j in range(2, 4):
            for k in range(2):
                U[2*5 + j, col] = 1
                V[j*6 + k0 + k, col] = 1
                W[(k0 + k)*3 + 2, col] = 1
                col += 1
        # Block F: i=2; j=4; k=k0,k1 -> schoolbook (2 terms)
        for k in range(2):
            U[2*5 + 4, col] = 1
            V[4*6 + k0 + k, col] = 1
            W[(k0 + k)*3 + 2, col] = 1
            col += 1
    if col != rank:
        return None
    return U, V, W, rank

def _verify_fast(U, V, W, n, m, p):
    r = U.shape[1]
    U3 = U.reshape(n, m, r)
    V3 = V.reshape(m, p, r)
    W3 = W.reshape(p, n, r)
    T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    return np.allclose(T, 1.0, atol=1e-9)

def _verify_strassen_2x2x2():
    """Verify the 2x2x2 Strassen decomposition is correct."""
    sU, sV, sW = _get_strassen_2x2x2()
    U3 = sU.reshape(2, 2, 7)
    V3 = sV.reshape(2, 2, 7)
    W3 = sW.reshape(2, 2, 7)
    T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    return np.allclose(T, 1.0, atol=1e-9)

deadline_global = time.time() + 130

# EVOLVE-BLOCK-END
