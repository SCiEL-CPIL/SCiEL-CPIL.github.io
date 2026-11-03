import time
import numpy as np
from itertools import combinations
from scipy.optimize import minimize, lsq_linear

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START

def _strassen_2x2x2():
    """Returns U, V, W for 2x2x2 Strassen (rank 7)."""
    U = np.zeros((4, 7), dtype=np.int64)
    V = np.zeros((4, 7), dtype=np.int64)
    W = np.zeros((4, 7), dtype=np.int64)
    U[0,0]=1; U[3,0]=1; V[0,0]=1; V[3,0]=1; W[0,0]=1; W[3,0]=1; W[1,0]=-1; W[2,0]=1
    U[2,1]=1; U[3,1]=1; V[0,1]=1; W[2,1]=1; W[3,1]=-1
    U[0,2]=1; V[2,2]=1; V[3,2]=-1; W[0,2]=-1; W[1,2]=1
    U[3,3]=1; V[1,3]=1; V[0,3]=-1; W[1,3]=-1; W[3,3]=1
    U[0,4]=1; U[1,4]=1; V[3,4]=1; W[0,4]=1; W[1,4]=-1; W[3,4]=1
    U[2,5]=1; U[0,5]=-1; V[0,5]=1; V[1,5]=1; W[2,5]=-1; W[3,5]=1
    U[1,6]=1; U[3,6]=-1; V[2,6]=1; V[3,6]=1; W[0,6]=1; W[2,6]=-1
    return U, V, W


def _build_4x4x4_49():
    """4x4x4 via 49-term Strassen recursion."""
    U2, V2, W2 = _strassen_2x2x2()
    U4 = np.zeros((16, 49), dtype=np.int64)
    V4 = np.zeros((16, 49), dtype=np.int64)
    W4 = np.zeros((16, 49), dtype=np.int64)

    # 2x2 block offsets for 4x4
    # A (4x4): blocks (0,0)=top-left, (1,0)=top-right, (0,1)=bot-left, (1,1)=bot-right
    # But we're treating 4x4 as 2x2 blocks of 2x2
    # A blocks: A_00 (rows 0-1, cols 0-1), A_10 (rows 0-1, cols 2-3), A_01 (rows 2-3, cols 0-1), A_11 (rows 2-3, cols 2-3)
    # B blocks similarly, C blocks similarly
    
    # For M_{4,4,4}: A(4x4)*B(4x4)=C(4x4)
    # Using 2x2 Strassen recursion:
    # A = [[A_00, A_10], [A_01, A_11]]
    # B = [[B_00, B_01], [B_10, B_11]]  (note: B is 4x4, so B_00 is top-left 2x2, etc.)
    # Wait, for standard matrix mult: C = A*B
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # In block form with 2x2 blocks:
    # C_00 = A_00*B_00 + A_10*B_10
    # C_01 = A_00*B_01 + A_10*B_11
    # C_10 = A_01*B_00 + A_11*B_10
    # C_11 = A_01*B_01 + A_11*B_11
    # Where A_00 is top-left 2x2, A_10 is top-right 2x2, etc.
    # And B_00 is top-left 2x2, B_01 is top-right 2x2, B_10 is bottom-left 2x2, B_11 is bottom-right 2x2.
    
    # Strassen terms for each 2x2 block multiplication:
    # M0 = (A_00 + A_11)*(B_00 + B_10)
    # M1 = A_11*(B_00 - B_10)  -- wait, let me use the standard Strassen formulas.
    
    # Standard Strassen for [[P0, P1],[P2, P3]] * [[Q0, Q1],[Q2, Q3]] = [[R0, R1],[R2, R3]]:
    # M0 = (P0+P3)(Q0+Q3)
    # M1 = (P2+P3)Q0
    # M2 = P0(Q1-Q3)
    # M3 = P3(Q2-Q0)
    # M4 = (P0+P1)Q3
    # M5 = (P3+P2)(Q1+Q3)
    # M6 = (P1-P3)(Q2+Q0)
    # R0 = M0+M3-M5+M6
    # R1 = M1+M4
    # R2 = M2+M5
    # R3 = M0-M1+M2+M4
    
    # For C_00 = A_00*B_00 + A_10*B_10:
    # This is sum of two 2x2x2 multiplications.
    # We can't directly apply Strassen to a sum.
    # Instead, we compute 49 terms = 7*7 from applying Strassen to the whole 4x4x4.
    
    # Actually, the 49-term construction for 4x4x4 uses:
    # Treat 4x4 as 2x2 blocks, and apply Strassen recursively.
    # This gives 7^2 = 49 terms.
    
    # Let me define the block structure properly.
    # A (4x4) has blocks: A_top_left (rows 0-1, cols 0-1), A_top_right (rows 0-1, cols 2-3),
    #                     A_bot_left (rows 2-3, cols 0-1), A_bot_right (rows 2-3, cols 2-3)
    # B (4x4) has blocks: B_top_left (rows 0-1, cols 0-1), B_top_right (rows 0-1, cols 2-3),
    #                     B_bot_left (rows 2-3, cols 0-1), B_bot_right (rows 2-3, cols 2-3)
    # C (4x4) has blocks: C_top_left, C_top_right, C_bot_left, C_bot_right
    
    # Strassen for the 4x4x4 tensor:
    # The 7 Strassen terms for the outer 2x2 block multiplication are:
    # M0 = (A_TL + A_BR) * (B_TL + B_BL)  -- wait, this isn't quite right.
    
    # Let me use the correct Strassen formula for block multiplication.
    # For 2x2 block multiplication where each block is 2x2:
    # M0 = (A_TL + A_BR)(B_TL + B_BR)
    # M1 = (A_BL + A_BR) * B_TL
    # M2 = A_TL * (B_TR - B_BR)
    # M3 = A_BR * (B_BL - B_TL)
    # M4 = (A_TL + A_TR) * B_BR
    # M5 = (A_BR + A_BL) * (B_TR + B_BR)
    # M6 = (A_TR - A_BR) * (B_BL + B_TL)
    # C_TL = M0 + M3 - M5 + M6
    # C_TR = M1 + M4
    # C_BL = M2 + M5
    # C_BR = M0 - M1 + M2 + M4
    
    # But each M_t is itself a 2x2 * 2x2 = 2x2 product, which requires 7 scalar products.
    # So total = 7 * 7 = 49 terms.
    
    # Index convention:
    # U index: a = i*4 + j where i is row of A (0-3), j is col of A (0-3)
    # V index: b = j*4 + k where j is row of B (0-3), k is col of B (0-3)
    # W index: c = k*4 + i where k is row of C (0-3), i is col of C (0-3)
    
    # A_TL: rows 0-1, cols 0-1 -> U indices: 0*4+0=0, 0*4+1=1, 1*4+0=4, 1*4+1=5
    # A_TR: rows 0-1, cols 2-3 -> U indices: 0*4+2=2, 0*4+3=3, 1*4+2=6, 1*4+3=7
    # A_BL: rows 2-3, cols 0-1 -> U indices: 2*4+0=8, 2*4+1=9, 3*4+0=12, 3*4+1=13
    # A_BR: rows 2-3, cols 2-3 -> U indices: 2*4+2=10, 2*4+3=11, 3*4+2=14, 3*4+3=15
    
    # B_TL: rows 0-1, cols 0-1 -> V indices: 0*4+0=0, 0*4+1=1, 1*4+0=4, 1*4+1=5
    # B_TR: rows 0-1, cols 2-3 -> V indices: 0*4+2=2, 0*4+3=3, 1*4+2=6, 1*4+3=7
    # B_BL: rows 2-3, cols 0-1 -> V indices: 2*4+0=8, 2*4+1=9, 3*4+0=12, 3*4+1=13
    # B_BR: rows 2-3, cols 2-3 -> V indices: 2*4+2=10, 2*4+3=11, 3*4+2=14, 3*4+3=15
    
    # C_TL: rows 0-1, cols 0-1 -> W indices: 0*4+0=0, 0*4+1=1, 1*4+0=4, 1*4+1=5
    # C_TR: rows 0-1, cols 2-3 -> W indices: 0*4+2=2, 0*4+3=3, 1*4+2=6, 1*4+3=7
    # C_BL: rows 2-3, cols 0-1 -> W indices: 2*4+0=8, 2*4+1=9, 3*4+0=12, 3*4+1=13
    # C_BR: rows 2-3, cols 2-3 -> W indices: 2*4+2=10, 2*4+3=11, 3*4+2=14, 3*4+3=15
    
    A_TL = [0, 1, 4, 5]
    A_TR = [2, 3, 6, 7]
    A_BL = [8, 9, 12, 13]
    A_BR = [10, 11, 14, 15]
    
    B_TL = [0, 1, 4, 5]
    B_TR = [2, 3, 6, 7]
    B_BL = [8, 9, 12, 13]
    B_BR = [10, 11, 14, 15]
    
    C_TL = [0, 1, 4, 5]
    C_TR = [2, 3, 6, 7]
    C_BL = [8, 9, 12, 13]
    C_BR = [10, 11, 14, 15]
    
    # Strassen M terms and their C block contributions:
    # M0 -> C_TL (+1), C_BR (+1)
    # M1 -> C_TR (+1), C_BR (-1)
    # M2 -> C_BL (+1), C_BR (+1)
    # M3 -> C_TL (+1), C_BR (0) -- wait let me recheck
    # M4 -> C_TR (+1), C_BR (+1)
    # M5 -> C_TL (-1), C_BL (+1)
    # M6 -> C_TL (+1)
    
    # Let me use the standard Strassen formulas again:
    # M0 = (A_TL + A_BR)(B_TL + B_BR)
    # M1 = (A_BL + A_BR) B_TL
    # M2 = A_TL (B_TR - B_BR)
    # M3 = A_BR (B_BL - B_TL)
    # M4 = (A_TL + A_TR) B_BR
    # M5 = (A_BR + A_BL)(B_TR + B_BR)
    # M6 = (A_TR - A_BR)(B_BL + B_TL)
    # C_TL = M0 + M3 - M5 + M6
    # C_TR = M1 + M4
    # C_BL = M2 + M5
    # C_BR = M0 - M1 + M2 + M4
    
    # So:
    # M0: C_TL=+1, C_BR=+1
    # M1: C_TR=+1, C_BR=-1
    # M2: C_BL=+1, C_BR=+1
    # M3: C_TL=+1
    # M4: C_TR=+1, C_BR=+1
    # M5: C_TL=-1, C_BL=+1
    # M6: C_TL=+1
    
    # Define U blocks (A side) for each M term:
    # M0: A_TL + A_BR -> U indices from A_TL and A_BR, all with coefficient +1
    # M1: A_BL + A_BR -> U indices from A_BL and A_BR, all with coefficient +1
    # M2: A_TL -> U indices from A_TL, coefficient +1
    # M3: A_BR -> U indices from A_BR, coefficient +1
    # M4: A_TL + A_TR -> U indices from A_TL and A_TR, coefficient +1
    # M5: A_BR + A_BL -> U indices from A_BR and A_BL, coefficient +1
    # M6: A_TR - A_BR -> U indices from A_TR (+1) and A_BR (-1)
    
    U_blocks = {
        0: (A_TL + A_BR, [1]*8),
        1: (A_BL + A_BR, [1]*8),
        2: (A_TL, [1]*4),
        3: (A_BR, [1]*4),
        4: (A_TL + A_TR, [1]*8),
        5: (A_BR + A_BL, [1]*8),
        6: (A_TR + A_BR, [1]*4 + [-1]*4),
    }
    
    # Define V blocks (B side) for each M term:
    # M0: B_TL + B_BR -> V indices from B_TL and B_BR, coefficient +1
    # M1: B_TL -> V indices from B_TL, coefficient +1
    # M2: B_TR - B_BR -> V indices from B_TR (+1) and B_BR (-1)
    # M3: B_BL - B_TL -> V indices from B_BL (+1) and B_TL (-1)
    # M4: B_BR -> V indices from B_BR, coefficient +1
    # M5: B_TR + B_BR -> V indices from B_TR and B_BR, coefficient +1
    # M6: B_BL + B_TL -> V indices from B_BL and B_TL, coefficient +1
    
    V_blocks = {
        0: (B_TL + B_BR, [1]*8),
        1: (B_TL, [1]*4),
        2: (B_TR + B_BR, [1]*4 + [-1]*4),
        3: (B_BL + B_TL, [1]*4 + [-1]*4),
        4: (B_BR, [1]*4),
        5: (B_TR + B_BR, [1]*8),
        6: (B_BL + B_TL, [1]*8),
    }
    
    # Define W blocks (C side) for each M term and C sub-block:
    # M0: C_TL(+1), C_BR(+1)
    # M1: C_TR(+1), C_BR(-1)
    # M2: C_BL(+1), C_BR(+1)
    # M3: C_TL(+1)
    # M4: C_TR(+1), C_BR(+1)
    # M5: C_TL(-1), C_BL(+1)
    # M6: C_TL(+1)
    
    C_blocks_for_M = {
        0: [(C_TL, 1), (C_BR, 1)],
        1: [(C_TR, 1), (C_BR, -1)],
        2: [(C_BL, 1), (C_BR, 1)],
        3: [(C_TL, 1)],
        4: [(C_TR, 1), (C_BR, 1)],
        5: [(C_TL, -1), (C_BL, 1)],
        6: [(C_TL, 1)],
    }
    
    for outer_t in range(7):
        u_idx_list, u_coefs = U_blocks[outer_t]
        v_idx_list, v_coefs = V_blocks[outer_t]
        
        for inner_s in range(7):
            col = outer_t * 7 + inner_s
            
            # Inner Strassen term s uses U2[:,s], V2[:,s], W2[:,s]
            for local_i in range(4):
                val = U2[local_i, inner_s]
                if val != 0:
                    for bi, coef in zip(u_idx_list, u_coefs):
                        U4[bi, col] += val * coef
            
            for local_j in range(4):
                val = V2[local_j, inner_s]
                if val != 0:
                    for bj, coef in zip(v_idx_list, v_coefs):
                        V4[bj, col] += val * coef
            
            for local_k in range(4):
                val = W2[local_k, inner_s]
                if val != 0:
                    for c_list, c_coef in C_blocks_for_M[outer_t]:
                        for bk in c_list:
                            W4[bk, col] += val * c_coef
    
    return U4, V4, W4


def _build_4x4x8_98():
    """4x4x8 via two 49-term 4x4x4 blocks (rank 98)."""
    U4, V4, W4 = _build_4x4x4_49()
    U = np.zeros((16, 98), dtype=np.int64)
    V = np.zeros((32, 98), dtype=np.int64)
    W = np.zeros((32, 98), dtype=np.int64)
    U[:, 0:49] = U4
    V[0:16, 0:49] = V4
    W[0:16, 0:49] = W4
    U[:, 49:98] = U4
    for j in range(4):
        for kl in range(4):
            V[j*8+kl+4, 49:98] = V4[j*4+kl, 0:49]
    for kl in range(4):
        for i in range(4):
            W[(kl+4)*4+i, 49:98] = W4[kl*4+i, 0:49]
    return U, V, W


def _build_4x4x8_84():
    """
    Try to build a rank-84 decomposition for 4x4x8.
    Idea: Split the 8 columns into 4+4, but use a smarter combination.
    
    Alternative approach: Use the fact that R(4,4,8) <= R(4,4,4) + R(4,4,4) = 98,
    but maybe we can share some terms between the two halves.
    
    Let's try a different decomposition:
    Split B(4x8) into B1(4x4) and B2(4x4).
    C = A*B1 + A*B2 (both 4x4 results, stacked in C).
    
    The two products A*B1 and A*B2 are independent 4x4x4 multiplications.
    They share the same A matrix, so maybe we can combine them.
    
    C_top[i,k] = sum_j A[i,j]*B1[j,k] for i,k in 0..3
    C_bot[i,k] = sum_j A[i,j]*B2[j,k] for i,k in 0..3
    
    Using 49-term Strassen for each gives 98 total.
    But maybe we can do better by treating it as a single 4x4x8 problem.
    
    Let's try: treat B as a 4x8 matrix and use a decomposition that exploits
    the structure more directly.
    
    Another idea: Use 2x2 Strassen on the 4x4 A matrix, and handle the 8 columns
    of B more cleverly.
    
    Actually, let's try a hybrid: use 7-term Strassen on A (splitting into 2x2 blocks),
    and for each Strassen term, handle the 8 columns of B.
    
    For each of the 7 outer Strassen terms M_t:
    M_t = A_t * B_t where A_t is a linear combination of A blocks, and B_t is a linear combination of B blocks.
    
    A_t is 4x4, B_t is 4x8.
    C_t contributes to C with a linear combination.
    
    M_t = A_t * B_t where A_t is 4x4, B_t is 4x8, result is 4x8.
    We need to decompose this 4x4x8 product.
    
    R(4,4,8) <= R(4,2,8) + R(2,4,8) = ... hmm this gets complex.
    
    For now, let's just return the 98-term solution if we can't find a better one.
    """
    U, V, W = _build_4x4x8_98()
    return U, V, W


def _verify_decomposition(U, V, W, T, atol=1e-9):
    """Verify that the decomposition exactly reconstructs T."""
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    return np.allclose(T_recon, T, atol=atol)


def _rank_of_matrix(A):
    """Compute the numerical rank of matrix A."""
    try:
        s = np.linalg.svd(A, compute_uv=False)
        return np.sum(s > 1e-10 * s[0] if len(s) > 0 else 0)
    except:
        return A.shape[1]


def _try_column_pruning(U, V, W, T, deadline):
    """
    Try to prune columns that are linearly dependent.
    Use a greedy approach: for each column, check if removing it still leaves
    a valid decomposition (approximately).
    """
    U = U.astype(np.float64).copy()
    V = V.astype(np.float64).copy()
    W = W.astype(np.float64).copy()
    rank = U.shape[1]
    
    keep = [True] * rank
    changed = True
    iter_count = 0
    max_iter = 20
    
    while changed and iter_count < max_iter:
        changed = False
        iter_count += 1
        if deadline and time.time() > deadline - 15:
            break
        
        for r in range(rank):
            if not keep[r]:
                continue
            
            # Try removing column r
            others = [s for s in range(rank) if s != r and keep[s]]
            if len(others) == 0:
                continue
            
            U_test = U[:, others]
            V_test = V[:, others]
            W_test = W[:, others]
            
            if _verify_decomposition(U_test, V_test, W_test, T, atol=1e-6):
                keep[r] = False
                changed = True
    
    idx = [i for i in range(rank) if keep[i]]
    return U[:, idx], V[:, idx], W[:, idx]


def _als_optimize(U, V, W, T, n, m, p, max_iter=50, deadline=None):
    """
    ALS-style optimization to reduce rank.
    T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
    """
    dim_a, dim_b, dim_c = n*m, m*p, p*n
    U = U.astype(np.float64).copy()
    V = V.astype(np.float64).copy()
    W = W.astype(np.float64).copy()
    rank = U.shape[1]
    
    for it in range(max_iter):
        if deadline and time.time() > deadline - 10:
            break
        
        # Update U: for each a, U[a,:] should satisfy
        # T[a,:,:] = sum_r U[a,r] * outer(V[:,r], W[:,r])
        # This is a least squares problem.
        
        # For efficiency, we can update one at a time or in batch.
        
        # Build the system: T_flat[a, :] = U[a, :] @ M^T
        # where M is the matrix with columns outer(V[:,r], W[:,r])
        
        M = np.zeros((dim_b * dim_c, rank), dtype=np.float64)
        for r in range(rank):
            M[:, r] = np.outer(V[:, r], W[:, r]).flatten()
        
        T_flat = T.reshape(dim_a, dim_b * dim_c)
        
        # Solve U @ M^T ≈ T_flat
        try:
            U, residuals, _, _ = np.linalg.lstsq(M, T_flat, rcond=1e-12)
        except:
            break
        
        # Update V
        M2 = np.zeros((dim_a * dim_c, rank), dtype=np.float64)
        for r in range(rank):
            M2[:, r] = np.outer(U[:, r], W[:, r]).flatten()
        
        T_flat2 = T.reshape(dim_b, dim_a * dim_c)
        
        try:
            V, _, _, _ = np.linalg.lstsq(M2, T_flat2, rcond=1e-12)
        except:
            break
        
        # Update W
        M3 = np.zeros((dim_a * dim_b, rank), dtype=np.float64)
        for r in range(rank):
            M3[:, r] = np.outer(U[:, r], V[:, r]).flatten()
        
        T_flat3 = T.reshape(dim_c, dim_a * dim_b)
        
        try:
            W, _, _, _ = np.linalg.lstsq(M3, T_flat3, rcond=1e-12)
        except:
            break
        
        # Check convergence
        if it % 5 == 0:
            T_recon = np.einsum('ar,br,cr->abc', U, V, W)
            if np.allclose(T_recon, T, atol=1e-8):
                # Try to prune
                U_pr, V_pr, W_pr = _try_column_pruning(U, V, W, T, deadline)
                if U_pr.shape[1] < rank:
                    return U_pr, V_pr, W_pr
                rank = U_pr.shape[1]
                U, V, W = U_pr, V_pr, W_pr
    
    return U, V, W


def _random_restart_als(T, n, m, p, target_rank, n_restarts=5, deadline=None):
    """
    Try to find a decomposition of rank `target_rank` using ALS with random restarts.
    """
    dim_a, dim_b, dim_c = n*m, m*p, p*n
    best_U, best_V, best_W = None, None, None
    best_rank = target_rank + 1
    
    np.random.seed(42)
    
    for restart in range(n_restarts):
        if deadline and time.time() > deadline - 15:
            break
        
        U = np.random.randn(dim_a, target_rank)
        V = np.random.randn(dim_b, target_rank)
        W = np.random.randn(dim_c, target_rank)
        
        # Normalize
        for r in range(target_rank):
            U[:, r] /= np.linalg.norm(U[:, r]) + 1e-10
            V[:, r] /= np.linalg.norm(V[:, r]) + 1e-10
            W[:, r] /= np.linalg.norm(W[:, r]) + 1e-10
        
        U_out, V_out, W_out = _als_optimize(U, V, W, T, n, m, p, max_iter=30, deadline=deadline)
        
        if U_out.shape[1] < best_rank:
            T_recon = np.einsum('ar,br,cr->abc', U_out, V_out, W_out)
            if np.allclose(T_recon, T, atol=1e-6):
                best_U, best_V, best_W = U_out, V_out, W_out
                best_rank = U_out.shape[1]
                if best_rank <= target_rank:
                    break
    
    if best_U is not None:
        return best_U, best_V, best_W, best_rank
    return None, None, None, target_rank + 1


def _to_half_integer_list(mat):
    """Convert a matrix to a list of lists with half-integer or integer entries."""
    mat = np.asarray(mat, dtype=np.float64)
    # Try half-integers first
    mat2 = mat * 2
    rounded2 = np.round(mat2)
    if np.allclose(mat2, rounded2, atol=1e-6):
        return (rounded2 / 2).tolist()
    # Try integers
    rounded_int = np.round(mat)
    if np.allclose(mat, rounded_int, atol=1e-6):
        return rounded_int.astype(int).tolist()
    # Fall back to float list
    return mat.tolist()


def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s kill

    n, m, p = DIMENSIONS  # 4, 4, 8
    dim_a, dim_b, dim_c = n*m, m*p, p*n

    # Build the full 4x4x8 multiplication tensor
    T = np.zeros((dim_a, dim_b, dim_c), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1

    # Fallback: Schoolbook (rank 128)
    rank_schoolbook = n * m * p
    U_school = np.zeros((dim_a, rank_schoolbook), dtype=np.int64)
    V_school = np.zeros((dim_b, rank_schoolbook), dtype=np.int64)
    W_school = np.zeros((dim_c, rank_schoolbook), dtype=np.int64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m + j, col] = 1
                V_school[j*p + k, col] = 1
                W_school[k*n + i, col] = 1
                col += 1

    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_schoolbook

    # Try 98-rank constructive solution
    if time.time() < deadline - 30:
        try:
            U98, V98, W98 = _build_4x4x8_98()
            if _verify_decomposition(U98, V98, W98, T, atol=1e-9):
                best_U, best_V, best_W = U98, V98, W98
                best_rank = 98
        except Exception:
            pass

    # Try column pruning on the 98-term solution
    if best_rank == 98 and time.time() < deadline - 20:
        try:
            U_pr, V_pr, W_pr = _try_column_pruning(
                best_U.astype(np.float64),
                best_V.astype(np.float64),
                best_W.astype(np.float64),
                T,
                deadline
            )
            if U_pr.shape[1] < best_rank:
                if _verify_decomposition(U_pr, V_pr, W_pr, T, atol=1e-6):
                    best_U, best_V, best_W = U_pr, V_pr, W_pr
                    best_rank = U_pr.shape[1]
        except Exception:
            pass

    # Try ALS optimization to reduce rank
    if best_rank > 80 and time.time() < deadline - 15:
        try:
            # Try targeting rank 90, 85, 80
            for target in [90, 85, 80]:
                if time.time() > deadline - 10:
                    break
                U_al, V_al, W_al, rank_al = _random_restart_als(
                    T, n, m, p, target, n_restarts=3, deadline=deadline
                )
                if U_al is not None and rank_al < best_rank:
                    best_U, best_V, best_W = U_al, V_al, W_al
                    best_rank = rank_al
        except Exception:
            pass

    # Final verification
    T_recon = np.einsum('ar,br,cr->abc', best_U, best_V, best_W)
    if not np.allclose(T_recon, T, atol=1e-6):
        # Fall back to schoolbook
        best_U, best_V, best_W = U_school, V_school, W_school
        best_rank = rank_schoolbook

    U_out = _to_half_integer_list(best_U)
    V_out = _to_half_integer_list(best_V)
    W_out = _to_half_integer_list(best_W)

    return {"U": U_out, "V": V_out, "W": W_out}

# EVOLVE-BLOCK-END
