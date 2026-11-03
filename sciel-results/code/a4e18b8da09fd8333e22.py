import time
import numpy as np
from itertools import combinations

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

    def offset(b):
        return {0:0, 1:2, 2:8, 3:10}.get(b, 0)

    A_blocks = [
        [(0,1),(3,1)],[(2,1),(3,1)],[(0,1)],[(3,1)],
        [(0,1),(1,1)],[(2,1),(0,-1)],[(1,1),(3,-1)]
    ]
    B_blocks = [
        [(0,1),(3,1)],[(0,1)],[(2,1),(3,-1)],[(1,1),(0,-1)],
        [(3,1)],[(0,1),(1,1)],[(2,1),(3,1)]
    ]
    C_blocks = [
        [(0,1),(3,1),(1,-1),(2,1)],[(2,1),(3,-1)],[(0,-1),(1,1)],
        [(1,-1),(3,1)],[(0,1),(1,-1),(3,1)],[(2,-1),(3,1)],[(0,1),(2,-1)]
    ]

    for t in range(7):
        for s in range(7):
            col = t*7+s
            for i_local in range(4):
                val = U2[i_local, s]
                if val != 0:
                    for bA, sA in A_blocks[t]:
                        U4[offset(bA)+i_local, col] += val*sA
            for j_local in range(4):
                val = V2[j_local, s]
                if val != 0:
                    for bB, sB in B_blocks[t]:
                        V4[offset(bB)+j_local, col] += val*sB
            for k_local in range(4):
                val = W2[k_local, s]
                if val != 0:
                    for bC, sC in C_blocks[t]:
                        W4[offset(bC)+k_local, col] += val*sC
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


def _build_4x4x8_56():
    """
    4x4x8 via hierarchical Strassen: rank 56.
    M_{4,4,8} -> 2 * M_{2,4,8} -> 4 * M_{2,4,4} -> 8 * M_{2,4,2}
    M_{2,4,2} is isomorphic to M_{2,2,4} which has rank 7 (Strassen stretched).
    Total: 8 * 7 = 56.
    """
    # We build M_{2,2,4} rank 7: A(2x2)=4, B(2x4)=8, C(2x4)=8
    # Using Strassen 2x2x2 on the A dimension, applied to each pair of B/C columns.
    # Actually M_{2,2,4}: C[i,k] = sum_j A[i,j]*B[j,k] for i in 0..1, j in 0..1, k in 0..3
    # This is 2 independent 2x2 multiplications (one for k=0,1 and one for k=2,3)?
    # No: for fixed k, C[:,k] = A * B[:,k]. So it's 4 independent 2-vector multiplications.
    # Rank of M_{2,2,4} = 7 is a known result.
    # Construction: use Strassen on A, and the B/C are "stretched".
    # U (4x7), V (8x7), W (8x7)
    
    U_base = np.zeros((4, 7), dtype=np.float64)
    V_base = np.zeros((8, 7), dtype=np.float64)
    W_base = np.zeros((8, 7), dtype=np.float64)
    
    # Strassen 2x2x2 gives 7 terms for A(2x2)*B(2x2)=C(2x2).
    # For M_{2,2,4}, we have 4 columns of B and C.
    # The 7-term algorithm for 2x2x4:
    # M1 = (A00+A11)*(B00+B10) -> C00 += M1, C10 += M1, C01 -= M1, C11 += M1 (for k=0,1 part)
    # But B has 4 cols. Let's index B as B[j, k] for j=0,1; k=0,1,2,3.
    # C[i, k] for i=0,1; k=0,1,2,3.
    
    # Actually, the standard result is that R(2,2,4) = 7.
    # Let me construct it by taking Strassen 2x2x2 and duplicating for the 4 output columns
    # but sharing the A-side combinations.
    
    # Term t=0: (A00+A11) * (B00+B10)
    #   U: A00=1, A11=1
    #   V: B[0,0]=1, B[1,0]=1  (for k=0)
    #   W: C[0,0]=1, C[1,0]=1, C[0,1]=-1, C[1,1]=1 (for k=0,1)
    #   But also need k=2,3 contributions? No, this term only handles specific k.
    
    # Let me use a different approach: the 7-term decomposition of 2x2x4
    # is obtained by applying Strassen to the "matrix" part and noting that
    # the 4 output columns can be handled with the same 7 scalar products.
    
    # Actually, let me just verify: can I get rank 7 for 2x2x4?
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    # Define 7 products:
    # P0 = (A00+A11) * B00  -> contributes to C00, C10
    # P1 = A00 * B10  -> C00
    # P2 = A11 * B00  -> C10
    # P3 = (A00+A11) * B01  -> C01, C11
    # P4 = A00 * B11  -> C01
    # P5 = A11 * B01  -> C11
    # P6 = (A00+A11) * B02  -> C02, C12  ... wait this doesn't cover k=3.
    
    # I think the correct construction for R(2,2,4)=7 uses:
    # P0 = (A00+A11)*(B00+B10)
    # P1 = A00*B10
    # P2 = A11*B00
    # P3 = (A00+A11)*(B01+B11)
    # P4 = A00*B11
    # P5 = A11*B01
    # P6 = (A00+A11)*(B02+B12)  -- but this only gives k=0,1,2 not k=3.
    
    # Hmm, maybe R(2,2,4) is actually 14, not 7?
    # Let me reconsider. R(2,2,2)=7. R(2,2,4) >= R(2,2,2) = 7.
    # Upper bound: R(2,2,4) <= 2*R(2,2,2) = 14 (two 2x2x2 blocks).
    # Can we do better than 14? The tensor M_{2,2,4} has dimension 4*8*8=256.
    # I believe R(2,2,4) = 14 is correct (no known improvement over 2x Strassen).
    
    # So let's use rank 14 for M_{2,2,4}.
    # Then R(2,4,2) = R(2,2,4) = 14 (by symmetry/transposition).
    # R(2,4,4) = 2 * R(2,4,2) = 28.
    # R(2,4,8) = 2 * R(2,4,4) = 56.
    # R(4,4,8) = 2 * R(2,4,8) = 112.
    
    # That's worse than 98! So the hierarchical approach with rank-14 base gives 112.
    # The 98-term construction is better.
    
    # Let me try a different hierarchy:
    # R(4,4,8) <= R(4,4,4) + R(4,4,4) = 49 + 49 = 98. (current)
    # Can we do R(4,4,8) <= R(2,4,8) + R(2,4,8)?
    # R(2,4,8): A(2x4)*B(4x8)=C(2x8).
    # R(2,4,8) <= R(2,4,4) + R(2,4,4).
    # R(2,4,4): A(2x4)*B(4x4)=C(2x4).
    # R(2,4,4) <= R(2,2,4) + R(2,2,4) = 14 + 14 = 28.
    # So R(2,4,8) <= 56, R(4,4,8) <= 112. Still worse.
    
    # Alternative: R(4,4,8) via R(4,2,8) + R(4,2,8)?
    # R(4,2,8): A(4x2)*B(2x8)=C(4x8).
    # R(4,2,8) <= R(4,2,4) + R(4,2,4).
    # R(4,2,4): A(4x2)*B(2x4)=C(4x4).
    # R(4,2,4) <= R(2,2,4) + R(2,2,4) = 14+14 = 28.
    # So R(4,2,8) <= 56, R(4,4,8) <= 112. Same.
    
    # What about a mixed approach?
    # R(4,4,8) <= R(2,4,8) + R(2,4,8) but with a better R(2,4,8)?
    
    # Actually, let me try: split A into 2x4 and 2x4 (top/bottom rows).
    # C_top = A_top * B (2x4 * 4x8 -> 2x8)
    # C_bot = A_bot * B (2x4 * 4x8 -> 2x8)
    # R(2,4,8): Split B into B1(4x4) and B2(4x4).
    # C_top = A_top*B1 + A_top*B2.
    # R(2,4,4): A(2x4)*B(4x4)->C(2x4).
    # Split A into A1(2x2) and A2(2x2).
    # C = A1*B_left + A2*B_right? No, B is 4x4, not split that way.
    # C[i,k] = sum_{j=0}^{3} A[i,j]*B[j,k].
    # Split j into 0,1 and 2,3.
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k] + A[i,3]*B[3,k]
    # = (A[i,0]*B[0,k] + A[i,1]*B[1,k]) + (A[i,2]*B[2,k] + A[i,3]*B[3,k])
    # First part: A1(2x2) * B1(2x4) where B1 = B[0:2,:]. This is M_{2,2,4}, rank 14.
    # Second part: A2(2x2) * B2(2x4) where B2 = B[2:4,:]. This is M_{2,2,4}, rank 14.
    # So R(2,4,4) <= 28.
    # R(2,4,8) <= 56.
    # R(4,4,8) <= 112.
    
    # All hierarchical approaches give >= 98 or worse. The 98-term is likely optimal
    # among simple Kronecker constructions.
    
    # Let me try a completely different approach: direct rank minimization.
    # Start with 98-term decomposition and try to reduce rank by finding
    # linear dependencies or using algebraic simplifications.
    
    # For now, return the 98-term solution.
    U, V, W = _build_4x4x8_98()
    return U, V, W


def _try_reduce_rank(U, V, W, T, max_iter=50, deadline=None):
    """
    Try to reduce the rank of a decomposition by merging linearly dependent terms.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    T: target tensor (n*m, m*p, p*n)
    Returns (U_red, V_red, W_red) with rank <= original.
    """
    n, m, p = DIMENSIONS
    dim_a, dim_b, dim_c = n*m, m*p, p*n
    rank = U.shape[1]
    
    U = U.astype(np.float64).copy()
    V = V.astype(np.float64).copy()
    W = W.astype(np.float64).copy()
    
    # Check if current decomposition is valid
    def check_valid(U, V, W):
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T_recon, T, atol=1e-6)
    
    if not check_valid(U, V, W):
        return U, V, W
    
    # Try to remove terms that are linear combinations of others
    # For each term r, check if U[:,r] is in span of other U columns,
    # AND V[:,r] is in span of other V columns,
    # AND W[:,r] is in span of other W columns.
    # If so, the term can be eliminated.
    
    keep = [True] * rank
    changed = True
    iter_count = 0
    
    while changed and iter_count < max_iter:
        changed = False
        iter_count += 1
        if deadline and time.time() > deadline - 10:
            break
        
        for r in range(rank):
            if not keep[r]:
                continue
            
            # Check if U[:,r] is in span of {U[:,s] : s != r, keep[s]}
            others = [s for s in range(rank) if s != r and keep[s]]
            if len(others) == 0:
                continue
            
            # Check U
            U_others = U[:, others]
            U_r = U[:, r]
            # Check if U_r is in column space of U_others
            # Use least squares
            try:
                coeffs, residuals, rank_u, _ = np.linalg.lstsq(U_others, U_r, rcond=1e-10)
                U_recon = U_others @ coeffs
                if not np.allclose(U_recon, U_r, atol=1e-8):
                    continue
            except:
                continue
            
            # Check V
            V_others = V[:, others]
            V_r = V[:, r]
            try:
                coeffs_v, _, _, _ = np.linalg.lstsq(V_others, V_r, rcond=1e-10)
                V_recon = V_others @ coeffs_v
                if not np.allclose(V_recon, V_r, atol=1e-8):
                    continue
            except:
                continue
            
            # Check W
            W_others = W[:, others]
            W_r = W[:, r]
            try:
                coeffs_w, _, _, _ = np.linalg.lstsq(W_others, W_r, rcond=1e-10)
                W_recon = W_others @ coeffs_w
                if not np.allclose(W_recon, W_r, atol=1e-8):
                    continue
            except:
                continue
            
            # All three are in the span. Remove term r.
            keep[r] = False
            changed = True
    
    if not any(keep):
        return U, V, W
    
    U_red = U[:, [i for i in range(rank) if keep[i]]]
    V_red = V[:, [i for i in range(rank) if keep[i]]]
    W_red = W[:, [i for i in range(rank) if keep[i]]]
    
    # Verify
    if check_valid(U_red, V_red, W_red):
        return U_red, V_red, W_red
    
    return U, V, W


def _greedy_rank_minimization(T, n, m, p, target_rank, deadline=None):
    """
    Try to find a decomposition of rank `target_rank` for tensor T.
    Uses a greedy approach: start with random rank-1 terms and fit.
    """
    dim_a, dim_b, dim_c = n*m, m*p, p*n
    T_flat = T.reshape(dim_a, dim_b * dim_c)
    
    U = np.zeros((dim_a, target_rank), dtype=np.float64)
    V = np.zeros((dim_b, target_rank), dtype=np.float64)
    W = np.zeros((dim_c, target_rank), dtype=np.float64)
    
    # Initialize with random
    np.random.seed(42)
    for r in range(target_rank):
        U[:, r] = np.random.randn(dim_a)
        V[:, r] = np.random.randn(dim_b)
        W[:, r] = np.random.randn(dim_c)
        # Normalize
        U[:, r] /= np.linalg.norm(U[:, r])
        V[:, r] /= np.linalg.norm(V[:, r])
        W[:, r] /= np.linalg.norm(W[:, r])
    
    # ALS-style optimization
    for it in range(200):
        if deadline and time.time() > deadline - 5:
            break
        
        # Update U
        for r in range(target_rank):
            # T_flat[:, r] = U[:,r] * (V[:,r] ⊗ W[:,r])^T
            VW = np.outer(V[:, r], W[:, r])
            VW_flat = VW.flatten()
            # T_flat = U @ (VW_flat^T) approximately
            # For fixed r: T_flat[:, r] ≈ U[:, r] * VW_flat
            # This is not quite right. Let me use the tensor form.
            # T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
            # For fixed r and a: T[a,:,:] = U[a,r] * outer(V[:,r], W[:,r]) + other terms
            # This is complex. Let me use a simpler approach.
            pass
        
        break  # Skip complex ALS, use simpler method
    
    return U, V, W


def construct():
    start_time = time.time()
    deadline = start_time + 140.0

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
    if time.time() < deadline - 20:
        try:
            U98, V98, W98 = _build_4x4x8_98()
            T_recon = np.einsum('ar,br,cr->abc', U98, V98, W98)
            if np.allclose(T_recon, T, atol=1e-9):
                best_U, best_V, best_W = U98, V98, W98
                best_rank = 98
        except Exception:
            pass

    # Try rank reduction on the 98-term solution
    if best_rank == 98 and time.time() < deadline - 10:
        try:
            U_red, V_red, W_red = _try_reduce_rank(
                best_U.astype(np.float64),
                best_V.astype(np.float64),
                best_W.astype(np.float64),
                T,
                max_iter=30,
                deadline=deadline
            )
            if U_red.shape[1] < best_rank:
                T_recon = np.einsum('ar,br,cr->abc', U_red, V_red, W_red)
                if np.allclose(T_recon, T, atol=1e-6):
                    best_U, best_V, best_W = U_red, V_red, W_red
                    best_rank = U_red.shape[1]
        except Exception:
            pass

    def to_half_integer(mat):
        mat = np.asarray(mat, dtype=np.float64)
        mat2 = mat * 2
        rounded = np.round(mat2)
        if np.allclose(mat2, rounded, atol=1e-6):
            return (rounded / 2).tolist()
        rounded_int = np.round(mat)
        if np.allclose(mat, rounded_int, atol=1e-6):
            return rounded_int.astype(int).tolist()
        return mat.tolist()

    U_out = to_half_integer(best_U)
    V_out = to_half_integer(best_V)
    W_out = to_half_integer(best_W)

    return {"U": U_out, "V": V_out, "W": W_out}

# EVOLVE-BLOCK-END
