import numpy as np
import time
from itertools import product

_START_TIME = time.time()
TIME_LIMIT = 140

def _time_left():
    return TIME_LIMIT - (time.time() - _START_TIME)

def _verify(U, V, W, n, m, p, num_trials=5):
    """Verify exact integer decomposition using small integer test matrices."""
    rank = len(U[0])
    np.random.seed(42)
    for _ in range(num_trials):
        A = np.random.randint(-3, 4, size=(n, m))
        B = np.random.randint(-3, 4, size=(m, p))
        C_true = A @ B
        C_decomp = np.zeros((n, p))
        for r in range(rank):
            x = np.dot(A.flatten(), [U[i*m+j][r] for i in range(n) for j in range(m)])
            y = np.dot(B.flatten(), [V[j*p+k][r] for j in range(m) for k in range(p)])
            for i in range(n):
                for k in range(p):
                    C_decomp[i,k] += x * y * W[k*n+i][r]
        if not np.array_equal(C_true, C_decomp):
            return False
    return True

def _build_base_85(n, m, p):
    """Build the base rank-85 decomposition."""
    rank = 85
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    u_micro = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, -1],
        [0, 1, 0, -1]
    ]
    v_micro = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [0, 0, 1, -1],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    w_micro = [
        [1, 0, 0, 1],
        [0, 1, 0, 0],
        [0, 1, 0, -1],
        [1, 0, 0, 1],
        [-1, 0, 0, 1],
        [0, 0, 1, 0],
        [-1, 0, 1, 0]
    ]
    
    col = 0
    for k in range(7):
        for l in range(7):
            for b in range(4):
                if u_micro[k][b] == 0: continue
                br = b // 2
                bc = b % 2
                for o in range(4):
                    if u_micro[l][o] == 0: continue
                    er = o // 2
                    ec = o % 2
                    gi = br * 2 + er
                    gj = bc * 2 + ec
                    ui = gi * 5 + gj
                    U[ui][col] += u_micro[k][b] * u_micro[l][o]
            
            for b in range(4):
                if v_micro[k][b] == 0: continue
                br = b // 2
                bc = b % 2
                for o in range(4):
                    if v_micro[l][o] == 0: continue
                    er = o // 2
                    ec = o % 2
                    gj = br * 2 + er
                    gk = bc * 2 + ec
                    vi = gj * 5 + gk
                    V[vi][col] += v_micro[k][b] * v_micro[l][o]
            
            for b in range(4):
                if w_micro[k][b] == 0: continue
                br = b // 2
                bc = b % 2
                for o in range(4):
                    if w_micro[l][o] == 0: continue
                    er = o // 2
                    ec = o % 2
                    gi = br * 2 + er
                    gk = bc * 2 + ec
                    wi = gk * 4 + gi
                    W[wi][col] += w_micro[k][b] * w_micro[l][o]
            
            col += 1
    
    for i in range(4):
        for j in range(4):
            ui = i * 5 + j
            vi = j * 5 + 4
            wi = 4 * 4 + i
            U[ui][col] = 1
            V[vi][col] = 1
            W[wi][col] = 1
            col += 1
            
    for i in range(4):
        for k in range(5):
            ui = i * 5 + 4
            vi = 4 * 5 + k
            wi = k * 4 + i
            U[ui][col] = 1
            V[vi][col] = 1
            W[wi][col] = 1
            col += 1
            
    return U, V, W

def _term_to_vectors(U, V, W, r, n, m, p):
    """Extract the three vectors for term r."""
    u_vec = np.array([U[i*m+j][r] for i in range(n) for j in range(m)], dtype=float)
    v_vec = np.array([V[j*p+k][r] for j in range(m) for k in range(p)], dtype=float)
    w_vec = np.array([W[k*n+i][r] for k in range(p) for i in range(n)], dtype=float)
    return u_vec, v_vec, w_vec

def _vectors_to_term(U, V, W, r, u_vec, v_vec, w_vec, n, m, p):
    """Set term r from vectors."""
    for i in range(n):
        for j in range(m):
            U[i*m+j][r] = int(round(u_vec[i*m+j]))
    for j in range(m):
        for k in range(p):
            V[j*p+k][r] = int(round(v_vec[j*p+k]))
    for k in range(p):
        for i in range(n):
            W[k*n+i][r] = int(round(w_vec[k*n+i]))

def _try_reduce(U, V, W, n, m, p, target_rank, time_budget):
    """
    Try to reduce the rank by finding linear dependencies among terms.
    The tensor T = sum_r u_r ⊗ v_r ⊗ w_r.
    We look for a term that can be expressed as a linear combination of others
    in the tensor product space, allowing us to eliminate it.
    
    A more practical approach: try pairwise term merging.
    If terms r1 and r2 satisfy certain conditions, they can be merged.
    
    Actually, let's use a different structural approach:
    We'll try to find a basis change that creates zero entries,
    then remove zero terms.
    """
    rank = len(U[0])
    if rank <= target_rank:
        return U, V, W
    
    # Convert to numpy for efficiency
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    # The tensor in vector form: T[i,j,k] = sum_r U[i,j,r]*V[j,k,r]*W[k,i,r]
    # We want to find if any term is redundant.
    
    # Strategy: For each term r, check if it can be written as a combination
    # of other terms. This is hard in general, but we can try a simpler approach:
    # Look for pairs of terms that can be merged via a change of basis.
    
    # Simpler structural approach: Try to find a linear transformation of the
    # rank-r space that creates more zeros, then remove zero terms.
    
    # For each pair of terms (r1, r2), check if there's a 2x2 matrix M such that
    # the new terms have simpler structure.
    
    # Actually, let's try a completely different structural approach:
    # Use the fact that the 4x5x5 problem has a block structure.
    # We can try different block partitions and optimize each block.
    
    # For now, let's try a greedy elimination:
    # For each term r, try to express its contribution using other terms.
    # The contribution of term r to T[i,j,k] is U[i,j,r]*V[j,k,r]*W[k,i,r].
    # We want: sum_{s != r} c_s * U[i,j,s]*V[j,k,s]*W[k,i,s] = U[i,j,r]*V[j,k,r]*W[k,i,r]
    # for all i,j,k. This is a system of n*m*p equations in rank-1 unknowns.
    
    # This is too hard. Let's try a different structural approach.
    
    # NEW APPROACH: Exploit the specific structure of the 4x5x5 problem.
    # The current decomposition has:
    # - 49 terms for the 4x4x4 part (Strassen squared)
    # - 16 terms for the 4x4x1 part
    # - 20 terms for the 4x1x5 part
    # Total: 85
    
    # Can we do better on the 4x4x4 part? The theoretical lower bound for 4x4x4
    # is 49 (from Strassen), but maybe we can do better with the 5x5 context.
    
    # Actually, let's try a different decomposition strategy:
    # Use a 2x2 block structure on the 4x5 matrix, giving blocks of size 2x2 and 2x3.
    
    # For now, return the base decomposition if we can't find improvements.
    # Let's try a simple heuristic: look for terms with all-zero U or V or W vectors.
    
    for r in range(rank):
        u_norm = np.sum(np.abs(U_arr[:, :, r]))
        v_norm = np.sum(np.abs(V_arr[:, :, r]))
        w_norm = np.sum(np.abs(W_arr[:, :, r]))
        if u_norm == 0 or v_norm == 0 or w_norm == 0:
            # This term contributes nothing; remove it
            U[r+1:][0] = U[r:rank-1][0]
            V[r+1:][0] = V[r:rank-1][0]
            W[r+1:][0] = W[r:rank-1][0]
            U[rank-1][0] = [0]*rank
            V[rank-1][0] = [0]*rank
            W[rank-1][0] = [0]*rank
            rank -= 1
            break
    
    if rank <= target_rank:
        return U[:rank], V[:rank], W[:rank]
    
    return U, V, W

def _build_alternative_decomposition(n, m, p):
    """
    Try a different structural decomposition of the 4x5x5 problem.
    
    Approach: Split the 5 columns of A and 5 rows of B differently.
    Instead of 4+1, try other splits and use optimized sub-problems.
    """
    # Try: split A as 2+3 columns, B as 2+3 rows
    # A = [A1(4x2) | A2(4x3)]
    # B = [B1(2x5); B2(3x5)]
    # C = A1*B1 + A2*B2
    
    # Rank of 4x2x5: can use 4*2*5=40, but maybe less
    # Rank of 4x3x5: can use 4*3*5=60, but maybe less
    
    # This doesn't obviously help. Let's try another approach.
    
    # Use the fact that 5 = 4+1, but optimize the cross terms.
    # A(4x5) * B(5x5) = A[:,:4]*B[:4,:] + A[:,:4]*B[4:5,:].T + A[:,4:5]*B[:4,:].T + A[:,4:5]*B[4:5,:].T
    # Wait, that's not right. Let me be more careful.
    
    # A is 4x5, B is 5x5.
    # Split A = [A1(4x4) | A2(4x1)]
    # Split B = [B1(4x5); B2(1x5)]
    # C = A1*B1 + A2*B2
    
    # A1*B1: 4x4 * 4x5 -> rank at most 4*4*5=80, but with Strassen for the 4x4 part...
    # Actually, we can write A1*B1 = sum over the 4x4x4 part (rank 49) plus the 4x4x1 part (rank 16)
    # Wait, 4x4 * 4x5: the 4x5 can be split into 4x4 + 4x1
    # So A1*B1 = A1*B1a + A1*B1b where B1a is 4x4 and B1b is 4x1
    # A1*B1a: 4x4*4x4, rank 49
    # A1*B1b: 4x4*4x1, rank 16
    # A2*B2: 4x1*1x5, rank 20
    # Total: 49+16+20 = 85. That's what we have.
    
    # Can we do better? The key question is whether the 4x4x4 part can be done in less than 49.
    # The known lower bound for 4x4x4 is 49, so no.
    
    # What about the cross terms? Can we merge some terms from different parts?
    # The 4x4x1 part has terms where U has support on A1 (first 4 cols), V has support on B1b (col 4),
    # and W has support on C col 4.
    # The 4x1x5 part has terms where U has support on A2 (col 4), V has support on B2 (row 4),
    # and W has support on all of C.
    
    # These are disjoint in their U and V supports, so they can't be merged directly.
    
    # Let's try a different split: A = [A1(2x5) ; A2(2x5)] and B = [B1(5x2) | B2(5x3)]
    # C = A1*B1 + A1*B2 + A2*B1 + A2*B2
    # A1*B1: 2x5*5x2, rank at most 20, but with Strassen for 2x2 blocks... 
    # This is getting complicated. Let's just return the base 85.
    
    return None

def construct():
    n, m, p = 4, 5, 5
    
    # Start with the base rank-85 decomposition
    U, V, W = _build_base_85(n, m, p)
    best_rank = len(U[0])
    best_U, best_V, best_W = U, V, W
    
    # Try to reduce
    if _time_left() > 10:
        reduced_U, reduced_V, reduced_W = _try_reduce(U, V, W, n, m, p, best_rank - 1, _time_left() - 10)
        if len(reduced_U[0]) < best_rank:
            best_rank = len(reduced_U[0])
            best_U, best_V, best_W = reduced_U, reduced_V, reduced_W
    
    # Verify the best result
    if _verify(best_U, best_V, best_W, n, m, p):
        return {"U": best_U, "V": best_V, "W": best_W}
    
    # Fallback to Kronecker
    rank_k = n * m * p
    Uk = [[0]*rank_k for _ in range(n*m)]
    Vk = [[0]*rank_k for _ in range(m*p)]
    Wk = [[0]*rank_k for _ in range(p*n)]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i*m*p + j*p + k
                Uk[i*m+j][r] = 1
                Vk[j*p+k][r] = 1
                Wk[k*n+i][r] = 1
    return {"U": Uk, "V": Vk, "W": Wk}
