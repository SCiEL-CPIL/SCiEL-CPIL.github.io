import time
import random
import itertools
import numpy as np
from scipy.linalg import qr, solve

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 120.0  # seconds
    n, m, p = DIMENSIONS  # 3, 4, 6
    
    def make_schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W, rank
    
    def make_strassen_2x2():
        """
        Return U, V, W for 2x2x2 Strassen with shapes (4,7), (4,7), (4,7).
        U indexed as U[i*2+j], V as V[j*2+k], W as W[k*2+i] for i,j,k in {0,1}.
        Must satisfy: for all (i,j,k) in {0,1}^3, sum_r U[i*2+j,r]*V[j*2+k,r]*W[k*2+i,r] = 1
        """
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1 = (A00+A11)*(B00+B11)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # M2 = (A10+A11)*B00
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = -1
        
        # M3 = A00*(B01-B11)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1; W[3, 2] = 1
        
        # M4 = A11*(B10-B00)
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1
        
        # M5 = (A00+A01)*B11
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1
        
        # M6 = (A10-A00)*(B00+B01)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        
        # M7 = (A01-A11)*(B10+B11)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1
        
        return U, V, W
    
    def verify_decomposition(U, V, W, rank, n, m, p):
        """Check: for all (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1"""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += W[k * n + i, r] * U[i * m + j, r] * V[j * p + k, r]
                    if abs(s - 1.0) > 1e-8:
                        return False
        return True
    
    def verify_decomposition_fast(U, V, W, rank, n, m, p):
        """Vectorized verification using tensor contraction."""
        # Reshape to 3D
        U3 = U.reshape(n, m, rank)
        V3 = V.reshape(m, p, rank)
        W3 = W.reshape(p, n, rank)
        
        # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # This is a 3-tensor. We need T[i,j,k] = 1 for all i,j,k.
        # Efficient: for each r, form outer products and sum.
        T = np.zeros((n, m, p))
        for r in range(rank):
            # U[i,j,r], V[j,k,r], W[k,i,r]
            # T[i,j,k] += U[i,j,r] * V[j,k,r] * W[k,i,r]
            # This is like a tensor product with specific index patterns
            # Let's do it with einsum for clarity, or manual loops for small sizes
            for i in range(n):
                for j in range(m):
                    u_val = U[i * m + j, r]
                    if u_val == 0:
                        continue
                    for k in range(p):
                        T[i, j, k] += u_val * V[j * p + k, r] * W[k * n + i, r]
        
        return np.allclose(T, 1.0, atol=1e-8)
    
    def make_blocked_strassen():
        """
        Build a blocked Strassen decomposition for 3x4x6.
        Split: A rows [0,1] and [2], A cols [0,1] and [2,3], B cols [0,1],[2,3],[4,5]
        6 blocks of type (2,2,2) using Strassen (rank 7 each)
        6 blocks of type (1,2,2) using schoolbook (rank 4 each)
        Total rank: 6*7 + 6*4 = 66
        """
        n, m, p = 3, 4, 6
        total_rank = 6 * 7 + 6 * 4  # 66
        U = np.zeros((n * m, total_rank), dtype=float)
        V = np.zeros((m * p, total_rank), dtype=float)
        W = np.zeros((p * n, total_rank), dtype=float)
        
        # Get Strassen 2x2x2 decomposition
        sU, sV, sW = make_strassen_2x2()
        
        col = 0
        a_row_blocks = [(0, 2), (2, 3)]  # (start, end)
        a_col_blocks = [(0, 2), (2, 4)]  # (start, end)
        b_col_blocks = [(0, 2), (2, 4), (4, 6)]  # (start, end)
        
        for (ar_s, ar_e) in a_row_blocks:
            ar_size = ar_e - ar_s  # 2 or 1
            for (ac_s, ac_e) in a_col_blocks:
                ac_size = ac_e - ac_s  # 2
                b_row_s = ac_s  # B rows match A cols
                b_row_e = ac_e
                for (bc_s, bc_e) in b_col_blocks:
                    bc_size = bc_e - bc_s  # 2
                    
                    if ar_size == 2:
                        # Use Strassen for (2,2,2) block
                        for r in range(7):
                            for ii in range(2):
                                for jj in range(2):
                                    U[(ar_s + ii) * m + (ac_s + jj), col] = sU[ii * 2 + jj, r]
                            for jj in range(2):
                                for kk in range(2):
                                    V[(b_row_s + jj) * p + (bc_s + kk), col] = sV[jj * 2 + kk, r]
                            for kk in range(2):
                                for ii in range(2):
                                    W[(bc_s + kk) * n + (ar_s + ii), col] = sW[kk * 2 + ii, r]
                            col += 1
                    else:
                        # ar_size == 1: use schoolbook for (1,2,2) block, rank 4
                        i0 = ar_s
                        for jj in range(2):
                            for kk in range(2):
                                U[i0 * m + (ac_s + jj), col] = 1.0
                                V[(b_row_s + jj) * p + (bc_s + kk), col] = 1.0
                                W[(bc_s + kk) * n + i0, col] = 1.0
                                col += 1
        
        assert col == total_rank, f"Column count {col} != expected {total_rank}"
        
        if not verify_decomposition(U, V, W, total_rank, n, m, p):
            return None
        
        return U, V, W, total_rank

    def make_blocked_strassen_v2():
        """
        Alternative blocking:
        A rows: [0,1,2] -> keep as 3? No, Strassen needs powers of 2 or specific sizes.
        Let's try:
        A rows: [0,1] (2), [2] (1)
        A cols: [0,1,2,3] -> [0,1] (2), [2,3] (2)
        B cols: [0,1,2,3,4,5] -> [0,1] (2), [2,3] (2), [4,5] (2)
        This is the same as above.
        
        What if we group differently?
        A rows: [0,1,2]
        Can we use a 3x?x? decomposition?
        Known: 3x3x3 has rank 23 (Laderman).
        3x2x2?
        
        Let's try a different split for the 3x4x6 case.
        Maybe use 2x2x2 Strassen for the 2x2x2 parts and handle the rest carefully.
        
        Current: 66.
        Schoolbook: 72.
        
        Can we do better?
        If we use Laderman for 3x3x3, rank 23.
        3x4x6 = (3x3x3) + (3x3x3) + (3x3x? no, 4=3+1, 6=3+3).
        3x4x6 = 3x(3+1)x(3+3) = 3x3x3 + 3x1x3 + 3x3x3 + 3x1x3.
        Rank: 23 (Laderman) + 3 (schoolbook 3x1x3) + 23 (Laderman) + 3 (schoolbook 3x1x3) = 52.
        This is much better!
        
        Let's implement Laderman 3x3x3.
        """
        pass
    
    def make_laderman_3x3():
        """
        Laderman's 23-product algorithm for 3x3 matrix multiplication.
        Reference: Laderman, 1976.
        This is complex to implement from scratch correctly.
        Let's try to find or derive it.
        
        Alternatively, use a known decomposition.
        Since implementing Laderman is error-prone, let's stick to the blocked Strassen for now
        but see if we can optimize the 1x2x2 blocks or other combinations.
        
        Actually, 3x3x3 rank is 23.
        3x1x3 is just 3 multiplications (diagonal of a 3x3? No, A is 3x1, B is 1x3, C is 3x3.
        C[i,k] = A[i,0] * B[0,k].
        This is an outer product.
        Rank of outer product tensor?
        C[i,k] = sum_r U[i,r] V[k,r] W[i,k,r]? No.
        The tensor T[i,j,k] where A is 3x1, B is 1x3.
        j is fixed at 0.
        T[i,0,k] = 1.
        Rank is 1? No.
        C = A B. A is 3x1, B is 1x3.
        This is rank 1 in the matrix sense, but in tensor rank?
        The bilinear map is (a, b) -> a b^T.
        This is rank 1 in the tensor decomposition sense?
        C[i,k] = a_i b_k.
        T[i,j,k] = 1 if j=0, else 0.
        T[i,0,k] = 1.
        Can we write T[i,0,k] = sum_r U[i,0,r] V[0,k,r] W[k,0? no, W[k*n+i, r].
        Indices: U[i*m+j, r] with m=1, so U[i, r].
        V[j*p+k, r] with j=0, so V[k, r].
        W[k*n+i, r] with n=3, so W[3k+i, r].
        
        We need sum_r U[i,r] V[k,r] W[3k+i, r] = 1 for all i,k.
        If we set rank 1: U[i,0]=1, V[k,0]=1, W[3k+i,0]=1?
        Then sum = 1*1*1 = 1.
        So rank is 1?
        Wait, W has shape (p*n, rank) = (3*3, 1) = (9,1).
        W[3k+i, 0] = 1 for all k,i.
        U has shape (n*m, rank) = (3*1, 1) = (3,1).
        U[i, 0] = 1.
        V has shape (m*p, rank) = (1*3, 1) = (3,1).
        V[k, 0] = 1 (since j=0, index is 0*3+k = k).
        
        So yes, 3x1x3 has rank 1.
        
        So 3x4x6 = 3x3x3 + 3x1x3 + 3x3x3 + 3x1x3.
        Rank = 23 + 1 + 23 + 1 = 48.
        
        Let's verify the decomposition structure.
        A is 3x4. Split cols into [0,1,2] and [3].
        B is 4x6. Split rows into [0,1,2] and [3].
        B cols into [0,1,2] and [3,4,5].
        
        C = A[:,:3] * B[:3, :3] + A[:,:3] * B[:3, 3:] + A[:,3:] * B[3:, :3] + A[:,3:] * B[3:, 3:]
        C = C11 + C12 + C21 + C22
        
        C11: 3x3 * 3x3 -> 3x3. Rank 23 (Laderman).
        C12: 3x3 * 3x3 -> 3x3. Rank 23 (Laderman).
        C21: 3x1 * 1x3 -> 3x3. Rank 1.
        C22: 3x1 * 1x3 -> 3x3. Rank 1.
        
        Total Rank = 23 + 23 + 1 + 1 = 48.
        
        We need Laderman's 3x3x3 decomposition.
        """
        # Laderman 3x3x3 decomposition
        # U (9, 23), V (9, 23), W (9, 23)
        # Indices: U[i*3+j, r], V[j*3+k, r], W[k*3+i, r]
        U = np.zeros((9, 23))
        V = np.zeros((9, 23))
        W = np.zeros((9, 23))
        
        # This is tedious to hardcode. Let's use a known set.
        # From Wikipedia or other sources, Laderman's 23 products.
        # I will use a verified set if I can recall or construct it.
        # Since I can't look it up, I will implement a generic search or use a known simple one.
        # Actually, 3x3x3 rank 23 is standard.
        # Let's try to construct it or use a fallback.
        
        # Fallback: use Strassen on 3x3x3?
        # 3x3x3 can be decomposed into 2x2x2 (Strassen) + remainder.
        # 3 = 2 + 1.
        # 3x3x3 = (2+1)x(2+1)x(2+1)
        # = 2x2x2 + 2x2x1 + 2x1x2 + 2x1x1 + 1x2x2 + 1x2x1 + 1x1x2 + 1x1x1
        # Ranks:
        # 2x2x2: 7
        # 2x2x1: 4 (schoolbook)
        # 2x1x2: 4
        # 2x1x1: 2
        # 1x2x2: 4
        # 1x2x1: 2
        # 1x1x2: 2
        # 1x1x1: 1
        # Total: 7 + 4 + 4 + 2 + 4 + 2 + 2 + 1 = 26.
        # This is worse than 23.
        
        # Let's stick to the 66 rank solution for now, as implementing Laderman correctly is risky.
        # But 48 is a significant improvement.
        # I will attempt to implement Laderman using a known pattern.
        
        # Laderman's algorithm:
        # M1 = (a11+a22+a33)(b11+b22+b33)
        # ... this is getting too complex to derive from memory.
        
        # Let's return None to indicate failure and fall back to blocked Strassen.
        return None

    # Start with schoolbook
    best_U, best_V, best_W, best_rank = make_schoolbook()
    
    if time.time() - start_time > deadline - 10:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Try blocked Strassen
    result = make_blocked_strassen()
    if result is not None:
        bU, bV, bW, brank = result
        if brank < best_rank:
            best_U, best_V, best_W, best_rank = bU, bV, bW, brank
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
