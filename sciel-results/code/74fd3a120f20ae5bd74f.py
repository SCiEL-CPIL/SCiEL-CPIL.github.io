import time
import numpy as np

DIMENSIONS = [4, 4, 7]

def construct():
    start_time = time.time()
    DEADLINE = 140  # safety margin before 150s
    
    n, m, p = DIMENSIONS
    nm = n * m  # 16
    mp = m * p  # 28
    pn = p * n  # 28
    
    # Build target tensor as sparse list of (a, b, c) where T[a,b,c] = 1
    # a = i*m+j, b = j*p+k, c = k*n+i
    nz_entries = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                nz_entries.append((a, b, c))
    
    # Schoolbook decomposition as baseline
    rank_school = n * m * p  # 112
    U_school = [[0] * rank_school for _ in range(nm)]
    V_school = [[0] * rank_school for _ in range(mp)]
    W_school = [[0] * rank_school for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school
    
    # Helper: verify a decomposition exactly
    def verify(U, V, W):
        r = len(U[0])
        # Build the decomposed tensor
        T_decomp = np.zeros((nm, mp, pn), dtype=np.int64)
        for c in range(r):
            uc = np.array(U[:, c])
            vc = np.array(V[:, c])
            wc = np.array(W[:, c])
            T_decomp += np.einsum('a,b,c->abc', uc, vc, wc)
        
        # Check all non-zero entries match
        for (a, b, c) in nz_entries:
            if T_decomp[a, b, c] != 1:
                return False
        
        # Check all zero entries are zero
        # Only need to check entries that could potentially be non-zero
        # For efficiency, check a subset or use the fact that if rank is low,
        # most entries should be zero
        # Full check:
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    if (a, b, c) not in set(nz_entries):
                        if T_decomp[a, b, c] != 0:
                            return False
        return True
    
    # Helper: check if a candidate rank-r decomposition is valid
    # using only the non-zero constraints (faster)
    def check_nz(U, V, W):
        r = len(U[0])
        for (a, b, c) in nz_entries:
            val = sum(U[a][rr] * V[b][rr] * W[c][rr] for rr in range(r))
            if val != 1:
                return False
        return True
    
    # Try to find lower rank decompositions
    # Strategy: use the structure of the tensor to initialize
    # and use alternating minimization with exact integer arithmetic
    
    # For each candidate rank, try multiple random initializations
    # and use ALS to converge
    
    # Key insight: the tensor has a lot of symmetry. 
    # For each output index c = k*n+i, the corresponding slice 
    # T[:,:,c] has non-zeros at positions (i*m+j, j*p+k) for all j.
    # This is a 4x28 matrix with 4 non-zeros.
    
    # Let's try ranks from 112 down to some minimum
    # Minimum possible rank: at least max(nm, mp, pn) = 28 by slice rank
    # But likely higher. Let's try from 112 down.
    
    # Actually, let's be smarter. Use the fact that we can group terms.
    # For a fixed (i, k), the entries are at (i*m+j, j*p+k, k*n+i) for j=0..3.
    # These 4 entries share the same i and k.
    
    # Let's try to find a decomposition where each "term" handles 
    # multiple non-zero entries.
    
    # Initialize with schoolbook and try to merge terms
    # or use a randomized search
    
    # Try a structured approach: for each (i, k) pair, we have 4 entries.
    # Can we represent these 4 entries with fewer than 4 rank-1 terms?
    
    # The 4 entries for fixed (i,k) are:
    # (i*m+0, 0*p+k, k*n+i) = (i*m, k, k*n+i)
    # (i*m+1, 1*p+k, k*n+i) = (i*m+1, p+k, k*n+i)
    # (i*m+2, 2*p+k, k*n+i) = (i*m+2, 2p+k, k*n+i)
    # (i*m+3, 3*p+k, k*n+i) = (i*m+3, 3p+k, k*n+i)
    
    # These 4 entries have the same c-index. For a rank-1 term to contribute
    # to all 4, it would need W[c] != 0 and U[i*m+j] != 0 and V[j*p+k] != 0 
    # for all j. But a single rank-1 term only contributes to entries where 
    # all three indices are non-zero in the respective vectors.
    
    # So one rank-1 term can contribute to at most one entry in each (a,b) pair 
    # for a given c. Actually no - it can contribute to multiple entries if 
    # the vectors have non-zeros at multiple positions.
    
    # Let's just do a randomized search with ALS
    
    np.random.seed(42)
    
    def try_rank_als(target_rank, n_restarts=20, max_iter=50):
        nonlocal best_U, best_V, best_W, best_rank
        
        for restart in range(n_restarts):
            if time.time() - start_time > DEADLINE - 10:
                return
            
            # Random initialization with small values
            U = np.random.randint(0, 2, size=(nm, target_rank)).astype(np.float64)
            V = np.random.randint(0, 2, size=(mp, target_rank)).astype(np.float64)
            W = np.random.randint(0, 2, size=(pn, target_rank)).astype(np.float64)
            
            # Ensure at least some non-zeros
            if np.sum(U) == 0:
                U[0, 0] = 1
            if np.sum(V) == 0:
                V[0, 0] = 1
            if np.sum(W) == 0:
                W[0, 0] = 1
            
            for it in range(max_iter):
                if time.time() - start_time > DEADLINE - 10:
                    return
                
                # Check current error
                err = 0
                for (a, b, c) in nz_entries:
                    val = sum(U[a, rr] * V[b, rr] * W[c, rr] for rr in range(target_rank))
                    if abs(val - 1) > 1e-9:
                        err += abs(val - 1)
                
                if err < 1e-9:
                    # Check zero entries
                    valid = True
                    for a in range(nm):
                        for b in range(mp):
                            for c in range(pn):
                                if (a, b, c) not in set(nz_entries):
                                    val = sum(U[a, rr] * V[b, rr] * W[c, rr] for rr in range(target_rank))
                                    if abs(val) > 1e-9:
                                        valid = False
                                        break
                            if not valid:
                                break
                        if not valid:
                            break
                    
                    if valid:
                        # Convert to exact form
                        U_list = [[int(round(U[a, r])) for r in range(target_rank)] for a in range(nm)]
                        V_list = [[int(round(V[b, r])) for r in range(target_rank)] for b in range(mp)]
                        W_list = [[int(round(W[c, r])) for r in range(target_rank)] for c in range(pn)]
                        best_U, best_V, best_W = U_list, V_list, W_list
                        best_rank = target_rank
                        return
                    break
                
                # ALS: update U
                # For each a, solve: T[a,:,:] = U[a,:] @ (V*W)
                # where (V*W)[r, b*c] = V[b,r] * W[c,r]
                M_U = np.zeros((target_rank, mp * pn))
                for r in range(target_rank):
                    M_U[r, :] = np.outer(V[:, r], W[:, r]).flatten()
                
                T_flat = np.zeros((nm, mp * pn))
                for (a, b, c) in nz_entries:
                    T_flat[a, b * pn + c] = 1.0
                
                try:
                    U_new, _, _, _ = np.linalg.lstsq(M_U.T, T_flat.T, rcond=None)
                    U = U_new.T
                except:
                    pass
                
                # Update V
                M_V = np.zeros((target_rank, nm * pn))
                for r in range(target_rank):
                    M_V[r, :] = np.outer(U[:, r], W[:, r]).flatten()
                
                T_flat_V = np.zeros((mp, nm * pn))
                for (a, b, c) in nz_entries:
                    T_flat_V[b, a * pn + c] = 1.0
                
                try:
                    V_new, _, _, _ = np.linalg.lstsq(M_V.T, T_flat_V.T, rcond=None)
                    V = V_new.T
                except:
                    pass
                
                # Update W
                M_W = np.zeros((target_rank, nm * mp))
                for r in range(target_rank):
                    M_W[r, :] = np.outer(U[:, r], V[:, r]).flatten()
                
                T_flat_W = np.zeros((pn, nm * mp))
                for (a, b, c) in nz_entries:
                    T_flat_W[c, a * mp + b] = 1.0
                
                try:
                    W_new, _, _, _ = np.linalg.lstsq(M_W.T, T_flat_W.T, rcond=None)
                    W = W_new.T
                except:
                    pass
        
        # After ALS, check if we got close and try to round
        for restart in range(min(5, n_restarts)):
            if time.time() - start_time > DEADLINE - 5:
                return
            
            U = np.random.randint(0, 2, size=(nm, target_rank)).astype(np.float64)
            V = np.random.randint(0, 2, size=(mp, target_rank)).astype(np.float64)
            W = np.random.randint(0, 2, size=(pn, target_rank)).astype(np.float64)
            
            if np.sum(U) == 0:
                U[0, 0] = 1
            if np.sum(V) == 0:
                V[0, 0] = 1
            if np.sum(W) == 0:
                W[0, 0] = 1
            
            for it in range(30):
                if time.time() - start_time > DEADLINE - 5:
                    return
                
                M_U = np.zeros((target_rank, mp * pn))
                for r in range(target_rank):
                    M_U[r, :] = np.outer(V[:, r], W[:, r]).flatten()
                T_flat = np.zeros((nm, mp * pn))
                for (a, b, c) in nz_entries:
                    T_flat[a, b * pn + c] = 1.0
                try:
                    U_new, _, _, _ = np.linalg.lstsq(M_U.T, T_flat.T, rcond=None)
                    U = U_new.T
                except:
                    pass
                
                M_V = np.zeros((target_rank, nm * pn))
                for r in range(target_rank):
                    M_V[r, :] = np.outer(U[:, r], W[:, r]).flatten()
                T_flat_V = np.zeros((mp, nm * pn))
                for (a, b, c) in nz_entries:
                    T_flat_V[b, a * pn + c] = 1.0
                try:
                    V_new, _, _, _ = np.linalg.lstsq(M_V.T, T_flat_V.T, rcond=None)
                    V = V_new.T
                except:
                    pass
                
                M_W = np.zeros((target_rank, nm * mp))
                for r in range(target_rank):
                    M_W[r, :] = np.outer(U[:, r], V[:, r]).flatten()
                T_flat_W = np.zeros((pn, nm * mp))
                for (a, b, c) in nz_entries:
                    T_flat_W[c, a * mp + b] = 1.0
                try:
                    W_new, _, _, _ = np.linalg.lstsq(M_W.T, T_flat_W.T, rcond=None)
                    W = W_new.T
                except:
                    pass
            
            # Check if close to integer solution
            U_rounded = np.round(U)
            V_rounded = np.round(V)
            W_rounded = np.round(W)
            
            if np.allclose(U, U_rounded, atol=0.01) and np.allclose(V, V_rounded, atol=0.01) and np.allclose(W, W_rounded, atol=0.01):
                U_list = [[int(U_rounded[a, r]) for r in range(target_rank)] for a in range(nm)]
                V_list = [[int(V_rounded[b, r]) for r in range(target_rank)] for b in range(mp)]
                W_list = [[int(W_rounded[c, r]) for r in range(target_rank)] for c in range(pn)]
                
                # Verify
                valid = True
                for (a, b, c) in nz_entries:
                    val = sum(U_list[a][rr] * V_list[b][rr] * W_list[c][rr] for rr in range(target_rank))
                    if val != 1:
                        valid = False
                        break
                if valid:
                    best_U, best_V, best_W = U_list, V_list, W_list
                    best_rank = target_rank
                    return
    
    # Try ranks from 112 down, but be strategic
    # The slice rank lower bound is max(nm, mp, pn) = 28
    # Let's try from 112 down to 28, but prioritize certain ranks
    
    # Actually, let's first check if the schoolbook can be improved
    # by checking for linear dependencies in the schoolbook decomposition
    
    # For now, try a few candidate ranks
    candidate_ranks = list(range(111, 27, -1))
    
    for target_rank in candidate_ranks:
        if time.time() - start_time > DEADLINE - 15:
            break
        
        try_rank_als(target_rank, n_restarts=5, max_iter=30)
        
        if best_rank < rank_school:
            break
    
    return {"U": best_U, "V": best_V, "W": best_W}
