"""Tensor decomposition search for 4x4 * 4x6 matrix multiplication."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [4, 4, 6]
n, m, p = DIMENSIONS

# Time budget - keep safety margin
DEADLINE = 120.0
START_TIME = time.time()

def time_left():
    return DEADLINE - (time.time() - START_TIME)

def verify_decomposition(U, V, W, tol=1e-10):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Check all valid (i,j,k): sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.dot(np.multiply(U[a], V[b]), W[c])
                if abs(val - 1.0) > tol:
                    return False, f"Valid ({i},{j},{k}): got {val}, expected 1.0"
    
    # Check invalid combinations give 0
    for a in range(n * m):
        for b in range(m * p):
            for c in range(p * n):
                ia = a // m; ja = a % m
                jb = b // p; kb = b % p
                kc = c // n; ic = c % n
                if ja != jb or kb != kc or ia != ic:
                    val = np.dot(np.multiply(U[a], V[b]), W[c])
                    if abs(val) > tol:
                        return False, f"Invalid ({ia},{ja},{jb},{kb},{kc},{ic}): got {val}, expected 0"
    
    return True, "OK"

def schoolbook():
    """Return the schoolbook (rank n*m*p) decomposition."""
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}

def strassen_2x2_terms():
    """Return the 7 Strassen terms for 2x2*2x2->2x2.
    Each term is (u_dict, v_dict, w_dict).
    u_dict: (i,j)->coeff, v_dict: (j,k)->coeff, w_dict: (i,k)->coeff
    """
    return [
        ({(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}, {(0,0):1,(1,0):1}),
        ({(1,0):1,(1,1):1}, {(0,0):1}, {(0,0):-1,(1,0):1}),
        ({(0,0):1}, {(0,1):1,(1,1):-1}, {(0,1):1,(1,1):1}),
        ({(1,1):1}, {(1,0):1,(0,0):-1}, {(0,0):1,(0,1):1}),
        ({(0,0):1,(0,1):1}, {(1,1):1}, {(1,1):-1}),
        ({(1,0):1,(0,0):-1}, {(0,0):1,(0,1):1}, {(1,1):1}),
        ({(0,1):1,(1,1):-1}, {(1,0):1,(1,1):1}, {(0,0):-1,(1,1):-1}),
    ]

def build_strassen_block_decomposition():
    """Build rank-84 decomposition: 12 independent 2x2*2x2 multiplications, each via Strassen rank 7."""
    st = strassen_2x2_terms()
    rank = 12 * 7  # 84
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for I in range(2):   # block row of A
        for K in range(3): # block col of B
            for J in range(2): # block col of A / block row of B
                for (u_dict, v_dict, w_dict) in st:
                    for (i_loc, j_loc), u_c in u_dict.items():
                        gi = I * 2 + i_loc
                        gj = J * 2 + j_loc
                        U[gi * m + gj, col] += u_c
                    for (j_loc, k_loc), v_c in v_dict.items():
                        gj = J * 2 + j_loc
                        gk = K * 2 + k_loc
                        V[gj * p + gk, col] += v_c
                    for (i_loc, k_loc), w_c in w_dict.items():
                        gk = K * 2 + k_loc
                        gi = I * 2 + i_loc
                        W[gk * n + gi, col] += w_c
                    col += 1
    
    return U, V, W, rank

def try_merge_pairs(U, V, W, max_attempts=2000):
    """Try to merge pairs of rank-1 terms into single terms (rank reduction by 1 per success).
    Two terms r1, r2 can be merged if their supports are compatible.
    Returns (new_U, new_V, new_W, new_rank) or None.
    """
    rank = U.shape[1]
    U = U.copy()
    V = V.copy()
    W = W.copy()
    merged = 0
    
    # For each pair of columns, check if they can be merged
    # Two rank-1 terms can be merged if the resulting 2x2x2 sub-tensor has rank 1
    # i.e., U[:,r1]*V[:,r1]*W[:,r1] + U[:,r2]*V[:,r2]*W[:,r2] is rank 1
    # This is equivalent to: the outer product structure allows combining
    # A simpler sufficient condition: if U[:,r1] is proportional to U[:,r2] 
    # AND V[:,r1] proportional to V[:,r2] AND W[:,r1] proportional to W[:,r2]
    # Then they're already a single rank-1 term (just scale).
    
    # Better approach: for each pair (r1, r2), check if the sum of the two rank-1 tensors
    # is itself rank 1. This is true iff the 3D tensor T = u1⊗v1⊗w1 + u2⊗v2⊗w2 has rank 1,
    # which means all 2x2 minors of the "unfolding" vanish, or equivalently:
    # there exist scalars a,b,c such that u2=a*u1, v2=b*v1, w2=c*w1 with abc=1
    # (after appropriate scaling) -- no, that's for the individual vectors being proportional.
    
    # Actually, the sum of two rank-1 tensors is rank 1 iff the two tensors are "proportional"
    # in a specific sense. Let me use a different approach:
    # Try to find a single rank-1 term that equals the sum of two rank-1 terms.
    # u1⊗v1⊗w1 + u2⊗v2⊗w2 = u⊗v⊗w
    # This requires the 3-way tensor to be decomposable, which is a strong condition.
    
    # Alternative: try to find a rank-1 term that, when subtracted, leaves the rest valid.
    # This is harder. Let me try a simpler structural merge:
    # If two terms have the same U vector (up to scale) and same V vector (up to scale),
    # then they can be merged into one term with combined W.
    
    for r1 in range(rank):
        if merged >= 10:  # limit merges
            break
        for r2 in range(r1 + 1, rank):
            # Check if U[:,r1] and U[:,r2] are proportional
            u1 = U[:, r1]
            u2 = U[:, r2]
            v1 = V[:, r1]
            v2 = V[:, r2]
            w1 = W[:, r1]
            w2 = W[:, r2]
            
            # Check proportionality of U vectors
            if np.linalg.norm(u1) < 1e-12 or np.linalg.norm(u2) < 1e-12:
                continue
            # u2 = alpha * u1
            alpha = np.dot(u1, u2) / np.dot(u1, u1)
            if np.linalg.norm(u2 - alpha * u1) > 1e-8 * np.linalg.norm(u1):
                continue
            
            # v2 = beta * v1
            if np.linalg.norm(v1) < 1e-12 or np.linalg.norm(v2) < 1e-12:
                continue
            beta = np.dot(v1, v2) / np.dot(v1, v1)
            if np.linalg.norm(v2 - beta * v1) > 1e-8 * np.linalg.norm(v1):
                continue
            
            # Then the sum is (alpha*beta) * (u1⊗v1⊗w1 + w2/(alpha*beta) ... )
            # Actually: u1⊗v1⊗w1 + u2⊗v2⊗w2 = u1⊗v1⊗w1 + alpha*u1⊗beta*v1⊗w2
            # = u1⊗v1⊗(w1 + alpha*beta*w2)
            # So merged term: U=u1, V=v1, W=w1+alpha*beta*w2
            
            w_merged = w1 + alpha * beta * w2
            
            # Verify: does this merged term plus all others still give correct result?
            # The old contribution was u1⊗v1⊗w1 + u2⊗v2⊗w2 = u1⊗v1⊗(w1+alpha*beta*w2)
            # So yes, it's equivalent.
            
            # Merge: replace r1 with merged, zero out r2
            W[:, r1] = w_merged
            U[:, r2] = 0
            V[:, r2] = 0
            W[:, r2] = 0
            merged += 1
            # Shift r2 to end (swap with last non-zero column)
            # For simplicity, just mark as zero and continue
            break
    
    if merged == 0:
        return None
    
    # Remove zero columns
    nonzero_cols = [c for c in range(rank) if np.linalg.norm(U[:, c]) > 1e-12]
    U = U[:, nonzero_cols]
    V = V[:, nonzero_cols]
    W = W[:, nonzero_cols]
    
    return U, V, W, U.shape[1]

def try_hooi_optimization(U, V, W, max_iter=50, tol=1e-6):
    """Try HOOI-style optimization to reduce rank.
    Actually, HOOI optimizes within a fixed rank. For rank reduction,
    we need a different approach. Instead, let's try to find a lower-rank
    approximation and then check if it's exact.
    """
    rank = U.shape[1]
    
    # Try reducing rank by 1, 2, ... using iterative truncation
    for new_rank in range(rank - 1, max(rank - 10, 0), -1):
        if time_left() < 10:
            break
        
        # Project onto rank new_rank subspace
        # SVD of U
        try:
            U_svd = np.linalg.svd(U, full_matrices=False)
            U_trunc = U_svd[0][:, :new_rank] @ np.diag(U_svd[1][:new_rank]) @ U_svd[2][:new_rank, :]
            V_svd = np.linalg.svd(V, full_matrices=False)
            V_trunc = V_svd[0][:, :new_rank] @ np.diag(V_svd[1][:new_rank]) @ V_svd[2][:new_rank, :]
            W_svd = np.linalg.svd(W, full_matrices=False)
            W_trunc = W_svd[0][:, :new_rank] @ np.diag(W_svd[1][:new_rank]) @ W_svd[2][:new_rank, :]
        except:
            continue
        
        # Check if truncated version is still exact
        valid, _ = verify_decomposition(U_trunc, V_trunc, W_trunc, tol=1e-8)
        if valid:
            return U_trunc, V_trunc, W_trunc, new_rank
    
    return None

def construct():
    """Construct a tensor decomposition for 4x4 * 4x6 matrix multiplication."""
    if time_left() < 5:
        return schoolbook()
    
    best = schoolbook()
    best_rank = n * m * p
    
    # Try Strassen-based block decomposition (rank 84)
    if time_left() > 15:
        U, V, W, rank = build_strassen_block_decomposition()
        valid, msg = verify_decomposition(U, V, W)
        if valid:
            best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            best_rank = rank
            print(f"Strassen block decomposition: rank {rank}")
        else:
            print(f"Strassen block failed: {msg}")
    
    # Try to reduce rank further via pair merging
    if time_left() > 10 and best_rank < n * m * p:
        U = np.array(best["U"])
        V = np.array(best["V"])
        W = np.array(best["W"])
        
        # Multiple passes of pair merging
        for _ in range(3):
            if time_left() < 5:
                break
            result = try_merge_pairs(U, V, W)
            if result is None:
                break
            U, V, W, new_rank = result
            if new_rank < best_rank:
                valid, _ = verify_decomposition(U, V, W)
                if valid:
                    best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    best_rank = new_rank
                    print(f"After merging: rank {new_rank}")
    
    # Try HOOI-based rank reduction
    if time_left() > 10 and best_rank < n * m * p:
        U = np.array(best["U"])
        V = np.array(best["V"])
        W = np.array(best["W"])
        result = try_hooi_optimization(U, V, W)
        if result is not None:
            U, V, W, new_rank = result
            if new_rank < best_rank:
                valid, _ = verify_decomposition(U, V, W)
                if valid:
                    best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    best_rank = new_rank
                    print(f"After HOOI: rank {new_rank}")
    
    return best

# EVOLVE-BLOCK-END
