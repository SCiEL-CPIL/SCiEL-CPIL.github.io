import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = 145.0

    n, m, p = 2, 3, 6

    def schoolbook():
        rank = n * m * p  # 36
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    def verify(U, V, W, tol=1e-9):
        rank = len(U[0])
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)

        U_3d = U_arr.reshape(n, m, rank)
        V_3d = V_arr.reshape(m, p, rank)
        W_3d = W_arr.reshape(p, n, rank)

        T_target = np.ones((n, m, p))
        T_calc = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)

        return np.allclose(T_target, T_calc, atol=tol)

    def strassen_2x2():
        """Return rank-7 Strassen factors for 2x2 * 2x2.
        U: (4, 7), V: (4, 7), W: (4, 7)
        Indexing: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
        """
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))

        # M1 = (A00+A11)(B00+B11)
        # A00*A00 + A11*B11 + A00*B11 + A11*B00
        # U contributions: A00: +1, A11: +1
        # V contributions: B00: +1, B11: +1
        # W contributions: C00: +1, C11: -1
        U[0, 0] += 1.0  # A00
        U[3, 0] += 1.0  # A11
        V[0, 0] += 1.0  # B00
        V[3, 0] += 1.0  # B11
        W[0, 0] += 1.0  # C00
        W[3, 0] -= 1.0  # C11

        # M2 = (A10+A11)B00
        # A10*B00 + A11*B00
        # U: A10: +1, A11: +1
        # V: B00: +1
        # W: C10: +1
        U[2, 1] += 1.0  # A10
        U[3, 1] += 1.0  # A11
        V[0, 1] += 1.0  # B00
        W[2, 1] += 1.0  # C10

        # M3 = A00(B01-B11)
        # A00*B01 - A00*B11
        # U: A00: +1
        # V: B01: +1, B11: -1
        # W: C01: +1
        U[0, 2] += 1.0  # A00
        V[1, 2] += 1.0  # B01
        V[3, 2] -= 1.0  # B11
        W[1, 2] += 1.0  # C01

        # M4 = A11(B10-B00)
        # A11*B10 - A11*B00
        # U: A11: +1
        # V: B10: +1, B00: -1
        # W: C11: +1
        U[3, 3] += 1.0  # A11
        V[2, 3] += 1.0  # B10
        V[0, 3] -= 1.0  # B00
        W[3, 3] += 1.0  # C11

        # M5 = (A00+A10)B10
        # A00*B10 + A10*B10
        # U: A00: +1, A10: +1
        # V: B10: +1
        # W: C00: +1
        U[0, 4] += 1.0  # A00
        U[2, 4] += 1.0  # A10
        V[2, 4] += 1.0  # B10
        W[0, 4] += 1.0  # C00

        # M6 = (A01-A11)(B10+B11)
        # A01*B10 + A01*B11 - A11*B10 - A11*B11
        # U: A01: +1, A11: -1
        # V: B10: +1, B11: +1
        # W: C01: +1, C11: +1
        U[1, 5] += 1.0  # A01
        U[3, 5] -= 1.0  # A11
        V[2, 5] += 1.0  # B10
        V[3, 5] += 1.0  # B11
        W[1, 5] += 1.0  # C01
        W[3, 5] += 1.0  # C11

        # M7 = (A10-A00)(B00+B01)
        # A10*B00 + A10*B01 - A00*B00 - A00*B01
        # U: A10: +1, A00: -1
        # V: B00: +1, B01: +1
        # W: C10: +1, C11: -1
        U[2, 6] += 1.0  # A10
        U[0, 6] -= 1.0  # A00
        V[0, 6] += 1.0  # B00
        V[1, 6] += 1.0  # B01
        W[2, 6] += 1.0  # C10
        W[3, 6] -= 1.0  # C11

        return U, V, W

    def schoolbook_block(n_b, m_b, p_b):
        """Return schoolbook factors for n_b x m_b * m_b x p_b.
        U: (n_b*m_b, rank), V: (m_b*p_b, rank), W: (p_b*n_b, rank)
        """
        rank = n_b * m_b * p_b
        U = np.zeros((n_b * m_b, rank))
        V = np.zeros((m_b * p_b, rank))
        W = np.zeros((p_b * n_b, rank))
        col = 0
        for i in range(n_b):
            for j in range(m_b):
                for k in range(p_b):
                    U[i * m_b + j, col] = 1.0
                    V[j * p_b + k, col] = 1.0
                    W[k * n_b + i, col] = 1.0
                    col += 1
        return U, V, W

    def strassen_decomposition():
        """Construct the full 2x3 * 3x6 decomposition using Strassen blocks.
        
        A (2x3) = [A11 A12; A21 A22] where A11,A21 are 1x2, A12,A22 are 1x1
        B (3x6) = [B11 B12; B21 B22] where B11,B21 are 2x3, B12,B22 are 1x3
        
        C (2x6) = [A11*B11 + A12*B21,  A11*B12 + A12*B22;
                   A21*B11 + A22*B21,  A21*B12 + A22*B22]
        
        Top-left: A11(1x2) * B11(2x3) -> 1x3  [schoolbook rank 6]
        Top-right: A12(1x1) * B21(1x3) -> 1x3  [schoolbook rank 3]
        Bottom-left: A21(1x2) * B11(2x3) -> 1x3  [schoolbook rank 6]
        Bottom-right: A22(1x1) * B21(1x3) -> 1x3  [schoolbook rank 3]
        
        Wait, that's just schoolbook. Let me reconsider the block structure.
        
        Better: A (2x3), B (3x6).
        Split A into rows: A_top (1x3), A_bot (1x3)
        Split B into columns: B_left (3x3), B_right (3x3)
        
        C_top_left = A_top * B_left  (1x3 * 3x3 = 1x3)  rank 9
        C_top_right = A_top * B_right (1x3 * 3x3 = 1x3)  rank 9
        C_bot_left = A_bot * B_left  (1x3 * 3x3 = 1x3)  rank 9
        C_bot_right = A_bot * B_right (1x3 * 3x3 = 1x3)  rank 9
        Total: 36 (schoolbook)
        
        Alternative: Use Strassen on overlapping 2x2 sub-blocks.
        
        Let me try: A = [a b c; d e f] (2x3), B = [g h i; j k l; m n o] (3x6)
        
        Actually, let me try a different split.
        A (2x3) split as: [A_2x2 | A_2x1]
        B (3x6) split as: [B_2x6; B_1x6]
        
        C = [A_2x2 * B_2x6 + A_2x1 * B_1x6]
        
        A_2x2 * B_2x6: Use Strassen on the 2x2 part.
        A_2x2 = [a b; d e], B_2x6 = [g h i j k l; m n o p q r]
        
        Actually, for 2x2 * 2x6, I can split B into two 2x3 blocks:
        A_2x2 * B_2x3_left + A_2x2 * B_2x3_right
        
        For 2x2 * 2x3: split A as [a b; d e], B as [g h i; m n o]
        C = [ag+b m, ah+b n, ai+b o; dg+em, dh+en, di+eo]
        
        This is 2x2 * 2x3. Rank? At most 12 (schoolbook).
        Can we do better? 
        
        Let me just try the direct approach: use Strassen for 2x2*2x2 parts.
        
        A (2x3) = [a b c; d e f]
        B (3x6) = [g h i j k l; m n o p q r; s t u v w x]
        
        C[0,0] = ag + bm + cs
        C[0,1] = ah + bn + ct
        C[0,2] = ai + bo + cu
        C[0,3] = aj + bp + cv
        C[0,4] = ak + bq + cw
        C[0,5] = al + br + cx
        C[1,0] = dg + em + fs
        C[1,1] = dh + en + ft
        C[1,2] = di + eo + fu
        C[1,3] = dj + ep + fv
        C[1,4] = dk + eq + fw
        C[1,5] = dl + er + fx
        
        Let me try Strassen on the 2x2 sub-block using columns 0,1 of B:
        A_sub = [a b; d e], B_sub = [g h; m n]
        C_sub = [ag+b m, ah+b n; dg+em, dh+en]
        
        Using Strassen rank 7 for this 2x2*2x2:
        M1 = (a+e)(g+n) -> contributes to C[0,0] and C[1,1]
        M2 = (d+e)g -> contributes to C[1,0]
        M3 = a(h-n) -> contributes to C[0,1]
        M4 = e(m-g) -> contributes to C[1,1]
        M5 = (a+d)m -> contributes to C[0,0]
        M6 = (b-e)(m+n) -> contributes to C[0,1] and C[1,1]
        M7 = (d-a)(g+h) -> contributes to C[1,0] and C[1,1]
        
        So for the 2x2*2x2 part (columns 0,1), we use 7 terms.
        For the remaining terms involving c, f and columns 2-5, we use schoolbook.
        
        C[0,0] = ag + bm + cs: Strassen gives ag+b m, need cs
        C[0,1] = ah + bn + ct: Strassen gives ah+b n, need ct
        C[0,2] = ai + bo + cu: all schoolbook
        C[0,3] = aj + bp + cv: all schoolbook
        C[0,4] = ak + bq + cw: all schoolbook
        C[0,5] = al + br + cx: all schoolbook
        C[1,0] = dg + em + fs: Strassen gives dg+em, need fs
        C[1,1] = dh + en + ft: Strassen gives dh+en, need ft
        C[1,2] = di + eo + fu: all schoolbook
        C[1,3] = dj + ep + fv: all schoolbook
        C[1,4] = dk + eq + fw: all schoolbook
        C[1,5] = dl + er + fx: all schoolbook
        
        So: 7 (Strassen) + 2 (cs, ct) + 2 (fs, ft) + 4*2 (columns 2-5 for both rows)
        = 7 + 2 + 2 + 8 = 19
        
        Hmm, that's better than 36 but not as good as I hoped.
        
        Let me try a different approach. What if I use Strassen more aggressively?
        
        Actually, let me reconsider. The key insight from the cell finding is:
        "Block decomposition with Strassen sub-blocks reduces rank from 36 to 33"
        
        So the known improvement is 33. Let me think about how to get there.
        
        2x3 * 3x6: If I split into a 2x2*2x2 Strassen block plus the rest...
        
        Let me try: 
        A = [a b c; d e f]
        B = [g h i; j k l; m n o]  (3x3, ignoring the 6 columns for a moment)
        
        Actually, let me just implement the direct construction carefully.
        
        For 2x3 * 3x6, let me try:
        - Use Strassen (rank 7) for the 2x2 * 2x2 sub-problem (rows 0-1 of A, cols 0-1 of B, but B is 3x6 so we need to be careful)
        
        Let me just build it term by term and count.
        """
        # Let me try a cleaner approach:
        # Split B (3x6) into B_top (2x6) and B_bot (1x6)
        # A (2x3) = [A_2x2 | A_2x1] where A_2x2 is cols 0-1, A_2x1 is col 2
        #
        # C = A_2x2 * B_top + A_2x1 * B_bot
        #
        # A_2x2 * B_top: 2x2 * 2x6
        #   Split B_top into B_top_left (2x3) and B_top_right (2x3)
        #   A_2x2 * B_top_left: 2x2 * 2x3
        #   A_2x2 * B_top_right: 2x2 * 2x3
        #
        # For 2x2 * 2x3:
        #   A = [a b; d e], B = [g h i; m n o]
        #   C = [ag+bm, ah+bn, ai+bo; dg+em, dh+en, di+eo]
        #   Schoolbook rank: 12
        #   Can we do better? 
        #   Split into 2x2*2x2 (first 2 cols) + 2x1*2x1 (last col)
        #   2x2*2x2: Strassen rank 7
        #   2x1*2x1: [a; d] * [i; o] = [ai, di] -> rank 2
        #   Total: 7 + 2 = 9
        #
        # So A_2x2 * B_top: 9 + 9 = 18
        # A_2x1 * B_bot: [c; f] * (1x6) = [c*b0, c*b1, ..., c*b5; f*b0, ..., f*b5]
        #   This is 2x1 * 1x6: rank 12 (each output needs 1 term, 12 outputs)
        #   Wait: C[i,k] = A[i,2] * B[2,k]. Each (i,k) pair is one product.
        #   So rank = 2 * 6 = 12.
        #
        # Total: 18 + 12 = 30. Better than 36!
        #
        # Can we do better on the 2x1 * 1x6 part?
        # [c; f] * [b0 b1 b2 b3 b4 b5]
        # C[0,k] = c * bk, C[1,k] = f * bk
        # This is an outer product of [c; f] and [b0...b5].
        # Rank = 1? No, because c and f are different variables.
        # Actually, each C[i,k] = A[i,2] * B[2,k] is a single product, so we need 12 terms.
        # Unless we can share... but we can't since they're all different.
        #
        # Wait, actually the rank of the 2x1*1x6 tensor is 12 (each entry is a distinct product).
        #
        # Total: 18 + 12 = 30.
        #
        # Hmm, but can we do better on 2x2*2x3?
        # A = [a b; d e], B = [g h i; m n o]
        # C[0,0] = ag+bm, C[0,1] = ah+bn, C[0,2] = ai+bo
        # C[1,0] = dg+em, C[1,1] = dh+en, C[1,2] = di+eo
        #
        # Using Strassen on first 2 cols (2x2*2x2): rank 7
        # Last col: [a; d] * [i; o]: C[0,2]=ai, C[1,2]=do: rank 2
        # Total: 9
        #
        # Alternatively, can we do 2x2*2x3 in fewer than 9?
        # The tensor is 2x2x3 = 12 entries. Lower bound is max(4,6,4)=6.
        # Strassen gives 9 for 2x2*2x3. Is there a rank-8 decomposition?
        # Not that I know of. Let's go with 9.
        #
        # So total rank = 9 + 9 + 12 = 30.
        #
        # Actually wait, I need to double-check the 2x1*1x6 part.
        # A_2x1 = [c; f] (2x1), B_bot = [b0 b1 b2 b3 b4 b5] (1x6)
        # C = A_2x1 * B_bot (2x6)
        # C[0,k] = c * bk, C[1,k] = f * bk
        # Each C[i,k] is a single monomial, so we need 12 rank-1 tensors.
        # Rank = 12.
        #
        # Total: 9 + 9 + 12 = 30.
        
        # Let me build this:
        rank = 30
        U = np.zeros((n * m, rank))  # (6, 30)
        V = np.zeros((m * p, rank))  # (18, 30)
        W = np.zeros((p * n, rank))  # (12, 30)
        
        # Indexing:
        # U[i*m+j, r] where i in [0,2), j in [0,3)
        # V[j*p+k, r] where j in [0,3), k in [0,6)
        # W[k*n+i, r] where k in [0,6), i in [0,2)
        
        col = 0
        
        # Part 1: A_2x2 * B_top_left (2x2 * 2x3)
        # A_2x2 = [a b; d e] -> A[0,0]=a, A[0,1]=b, A[1,0]=d, A[1,1]=e
        # B_top_left = [g h i; m n o] -> B[0,0]=g, B[0,1]=h, B[0,2]=i, B[1,0]=m, B[1,1]=n, B[1,2]=o
        # C_top_left = [ag+bm, ah+bn, ai+bo; dg+em, dh+en, di+eo]
        # Output indices: C[0,0], C[0,1], C[0,2], C[1,0], C[1,1], C[1,2]
        
        # Strassen for 2x2*2x2 part (cols 0,1):
        # M1 = (a+e)(g+n): U: a(+1), e(+1); V: g(+1), n(+1); W: C[0,0](+1), C[1,1](-1)
        U[0, col] += 1.0  # a = A[0,0]
        U[1, col] += 1.0  # e = A[1,1]
        V[0, col] += 1.0  # g = B[0,0]
        V[1, col] += 1.0  # n = B[1,1]
        W[0, col] += 1.0  # C[0,0]: k=0,i=0
        W[1, col] -= 1.0  # C[1,1]: k=1,i=1
        col += 1
        
        # M2 = (d+e)g: U: d(+1), e(+1); V: g(+1); W: C[1,0](+1)
        U[2, col] += 1.0  # d = A[1,0]
        U[1, col] += 1.0  # e = A[1,1]
        V[0, col] += 1.0  # g = B[0,0]
        W[0, col] += 1.0  # C[1,0]: k=0,i=1
        col += 1
        
        # M3 = a(h-n): U: a(+1); V: h(+1), n(-1); W: C[0,1](+1)
        U[0, col] += 1.0  # a
        V[1, col] += 1.0  # h = B[0,1]
        V[1, col] -= 1.0  # n = B[1,1]
        W[1, col] += 1.0  # C[0,1]: k=1,i=0
        col += 1
        
        # M4 = e(m-g): U: e(+1); V: m(+1), g(-1); W: C[1,1](+1)
        U[1, col] += 1.0  # e
        V[2, col] += 1.0  # m = B[1,0]
        V[0, col] -= 1.0  # g
        W[1, col] += 1.0  # C[1,1]: k=1,i=1
        col += 1
        
        # M5 = (a+d)m: U: a(+1), d(+1); V: m(+1); W: C[0,0](+1)
        U[0, col] += 1.0  # a
        U[2, col] += 1.0  # d
        V[2, col] += 1.0  # m
        W[0, col] += 1.0  # C[0,0]
        col += 1
        
        # M6 = (b-e)(m+n): U: b(+1), e(-1); V: m(+1), n(+1); W: C[0,1](+1), C[1,1](+1)
        U[3, col] += 1.0  # b = A[0,1]
        U[1, col] -= 1.0  # e
        V[2, col] += 1.0  # m
        V[1, col] += 1.0  # n
        W[1, col] += 1.0  # C[0,1]: k=1,i=0
        W[1, col] += 1.0  # C[1,1]: k=1,i=1 -- wait, this should be different
        # Actually W index: W[k*n+i, r]. C[0,1] -> k=1,i=0 -> W[1*2+0]=W[2]
        # C[1,1] -> k=1,i=1 -> W[1*2+1]=W[3]
        # Let me redo the W indexing.
        
        # I made an error. Let me restart with careful indexing.
        # W[k*n+i, r] where k is the column of C (0-5), i is the row of C (0-1)
        # C[0,0] -> W[0*2+0] = W[0]
        # C[0,1] -> W[1*2+0] = W[2]
        # C[0,2] -> W[2*2+0] = W[4]
        # C[1,0] -> W[0*2+1] = W[1]
        # C[1,1] -> W[1*2+1] = W[3]
        # C[1,2] -> W[2*2+1] = W[5]
        # C[0,3] -> W[3*2+0] = W[6]
        # C[0,4] -> W[4*2+0] = W[8]
        # C[0,5] -> W[5*2+0] = W[10]
        # C[1,3] -> W[3*2+1] = W[7]
        # C[1,4] -> W[4*2+1] = W[9]
        # C[1,5] -> W[5*2+1] = W[11]
        
        # Let me rebuild from scratch.
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        
        # U index: U[i*m+j, r]
        # A[0,0]=a -> U[0], A[0,1]=b -> U[1], A[0,2]=c -> U[2]
        # A[1,0]=d -> U[3], A[1,1]=e -> U[4], A[1,2]=f -> U[5]
        
        # V index: V[j*p+k, r]
        # B[0,0]=g -> V[0], B[0,1]=h -> V[1], B[0,2]=i -> V[2]
        # B[0,3]=j -> V[3], B[0,4]=k -> V[4], B[0,5]=l -> V[5]
        # B[1,0]=m -> V[6], B[1,1]=n -> V[7], B[1,2]=o -> V[8]
        # B[1,3]=p -> V[9], B[1,4]=q -> V[10], B[1,5]=r -> V[11]
        # B[2,0]=s -> V[12], B[2,1]=t -> V[13], B[2,2]=u -> V[14]
        # B[2,3]=v -> V[15], B[2,4]=w -> V[16], B[2,5]=x -> V[17]
        
        # W index: W[k*n+i, r]
        # C[0,0] -> W[0], C[1,0] -> W[1]
        # C[0,1] -> W[2], C[1,1] -> W[3]
        # C[0,2] -> W[4], C[1,2] -> W[5]
        # C[0,3] -> W[6], C[1,3] -> W[7]
        # C[0,4] -> W[8], C[1,4] -> W[9]
        # C[0,5] -> W[10], C[1,5] -> W[11]
        
        # Part 1: Strassen for A_2x2 * B_top_left (2x2 * 2x3, cols 0-2 of B)
        # A_2x2 = [a b; d e] -> U indices 0,1,3,4
        # B_top_left = [g h i; m n o] -> V indices 0,1,2,6,7,8
        # C = [ag+bm, ah+bn, ai+bo; dg+em, dh+en, di+eo]
        # W indices: 0,2,4 (row 0), 1,3,5 (row 1)
        
        # M1 = (a+e)(g+n): contributes ag+an+eg+en
        #   We want: ag (C[0,0]), en (C[1,1])
        #   Extra: an (C[0,1]), eg (C[1,0])
        #   So: C[0,0]+=ag, C[1,1]+=en, C[0,1]+=an, C[1,0]+=eg
        U[0, col] += 1.0  # a
        U[4, col] += 1.0  # e
        V[0, col] += 1.0  # g
        V[7, col] += 1.0  # n
        W[0, col] += 1.0  # C[0,0]
        W[3, col] += 1.0  # C[1,1]
        W[2, col] += 1.0  # C[0,1]
        W[1, col] += 1.0  # C[1,0]
        col += 1
        
        # M2 = (d+e)g: contributes dg+eg
        #   We want: dg (C[1,0])
        #   Extra: eg (C[1,0]) -- wait, eg is already in C[1,0]
        #   C[1,0] = dg + em. So dg is part of C[1,0].
        #   But eg is NOT in C[1,0]. C[1,0] = dg + em.
        #   Hmm, I'm confusing myself. Let me be more careful.
        #
        # C[0,0] = ag + bm
        # C[0,1] = ah + bn  
        # C[0,2] = ai + bo
        # C[1,0] = dg + em
        # C[1,1] = dh + en
        # C[1,2] = di + eo
        #
        # Strassen M1 = (a+e)(g+n) = ag + an + eg + en
        #   ag -> C[0,0] (+1)
        #   an -> C[0,1] (+1)  [but C[0,1] = ah+bn, so an is extra]
        #   eg -> C[1,0] (+1)  [but C[1,0] = dg+em, so eg is extra]
        #   en -> C[1,1] (+1)
        #
        # This doesn't work directly. Strassen works for 2x2*2x2 where:
        # C[0,0] = M1 - M4 + M5
        # C[0,1] = M3 + M6
        # C[1,0] = M2 + M5
        # C[1,1] = M1 - M2 + M4
        #
        # Let me use the standard Strassen formulas properly.
        # For A = [a b; d e], B = [g h; m n]:
        # M1 = (a+e)(g+n)
        # M2 = (d+e)g
        # M3 = a(h-n)
        # M4 = e(m-g)
        # M5 = (a+d)m
        # M6 = (b-e)(m+n)
        # M7 = (d-a)(g+h)
        #
        # C[0,0] = M1 - M4 + M5
        # C[0,1] = M3 + M6
        # C[1,0] = M2 + M5
        # C[1,1] = M1 - M2 + M4
        
        # OK so let me just use the standard Strassen decomposition for the 2x2*2x2 part.
        # For the 2x2*2x3, I'll use Strassen for the first 2 columns and schoolbook for the 3rd.
        
        # Reset
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        
        # Strassen for A_2x2 * B_2x2 (first 2 columns of B_top)
        # A_2x2 = [a b; d e], B_2x2 = [g h; m n]
        # U: a=U[0], b=U[1], d=U[3], e=U[4]
        # V: g=V[0], h=V[1], m=V[6], n=V[7]
        # W: C[0,0]=W[0], C[0,1]=W[2], C[1,0]=W[1], C[1,1]=W[3]
        
        # M1 = (a+e)(g+n)
        U[0, col] += 1.0; U[4, col] += 1.0
        V[0, col] += 1.0; V[7, col] += 1.0
        W[0, col] += 1.0; W[3, col] += 1.0  # C[0,0] += M1, C[1,1] += M1
        col += 1
        
        # M2 = (d+e)g
        U[3, col] += 1.0; U[4, col] += 1.0
        V[0, col] += 1.0
        W[1, col] += 1.0  # C[1,0] += M2
        col += 1
        
        # M3 = a(h-n)
        U[0, col] += 1.0
        V[1, col] += 1.0; V[7, col] -= 1.0
        W[2, col] += 1.0  # C[0,1] += M3
        col += 1
        
        # M4 = e(m-g)
        U[4, col] += 1.0
        V[6, col] += 1.0; V[0, col] -= 1.0
        W[0, col] -= 1.0; W[3, col] += 1.0  # C[0,0] -= M4, C[1,1] += M4
        col += 1
        
        # M5 = (a+d)m
        U[0, col] += 1.0; U[3, col] += 1.0
        V[6, col] += 1.0
        W[0, col] += 1.0; W[1, col] += 1.0  # C[0,0] += M5, C[1,0] += M5
        col += 1
        
        # M6 = (b-e)(m+n)
        U[1, col] += 1.0; U[4, col] -= 1.0
        V[6, col] += 1.0; V[7, col] += 1.0
        W[2, col] += 1.0; W[3, col] += 1.0  # C[0,1] += M6, C[1,1] += M6
        col += 1
        
        # M7 = (d-a)(g+h)
        U[3, col] += 1.0; U[0, col] -= 1.0
        V[0, col] += 1.0; V[1, col] += 1.0
        W[1, col] += 1.0; W[3, col] -= 1.0  # C[1,0] += M7, C[1,1] -= M7
        col += 1
        
        # Now the 3rd column of B_top: B[0,2]=i=V[2], B[1,2]=o=V[8]
        # C[0,2] = ai + bo
        # C[1,2] = di + eo
        U[0, col] += 1.0; V[2, col] += 1.0; W[4, col] += 1.0  # a*i -> C[0,2]
        col += 1
        U[1, col] += 1.0; V[8, col] += 1.0; W[4, col] += 1.0  # b*o -> C[0,2]
        col += 1
        U[3, col] += 1.0; V[2, col] += 1.0; W[5, col] += 1.0  # d*i -> C[1,2]
        col += 1
        U[4, col] += 1.0; V[8, col] += 1.0; W[5, col] += 1.0  # e*o -> C[1,2]
        col += 1
        
        # Part 2: A_2x2 * B_top_right (2x2 * 2x3, cols 3-5 of B)
        # B_top_right = [j k l; p q r] -> V indices 3,4,5,9,10,11
        # C[0,3] = aj + bp, C[0,4] = ak + bq, C[0,5] = al + br
        # C[1,3] = dj + ep, C[1,4] = dk + eq, C[1,5] = dl + er
        # W: C[0,3]=W[6], C[0,4]=W[8], C[0,5]=W[10], C[1,3]=W[7], C[1,4]=W[9], C[1,5]=W[11]
        
        # Strassen for first 2 cols (j,k and p,q):
        # M1 = (a+e)(j+q)
        U[0, col] += 1.0; U[4, col] += 1.0
        V[3, col] += 1.0; V[10, col] += 1.0
        W[6, col] += 1.0; W[9, col] += 1.0  # C[0,3], C[1,4]
        col += 1
        
        # M2 = (d+e)j
        U[3, col] += 1.0; U[4, col] += 1.0
        V[3, col] += 1.0
        W[7, col] += 1.0  # C[1,3]
        col += 1
        
        # M3 = a(k-q)
        U[0, col] += 1.0
        V[4, col] += 1.0; V[10, col] -= 1.0
        W[8, col] += 1.0  # C[0,4]
        col += 1
        
        # M4 = e(p-j)
        U[4, col] += 1.0
        V[9, col] += 1.0; V[3, col] -= 1.0
        W[6, col] -= 1.0; W[9, col] += 1.0  # C[0,3] -= M4, C[1,4] += M4
        col += 1
        
        # M5 = (a+d)p
        U[0, col] += 1.0; U[3, col] += 1.0
        V[9, col] += 1.0
        W[6, col] += 1.0; W[7, col] += 1.0  # C[0,3] += M5, C[1,3] += M5
        col += 1
        
        # M6 = (b-e)(p+q)
        U[1, col] += 1.0; U[4, col] -= 1.0
        V[9, col] += 1.0; V[10, col] += 1.0
        W[8, col] += 1.0; W[9, col] += 1.0  # C[0,4] += M6, C[1,4] += M6
        col += 1
        
        # M7 = (d-a)(j+k)
        U[3, col] += 1.0; U[0, col] -= 1.0
        V[3, col] += 1.0; V[4, col] += 1.0
        W[7, col] += 1.0; W[9, col] -= 1.0  # C[1,3] += M7, C[1,4] -= M7
        col += 1
        
        # 3rd col: l=V[5], r=V[11]
        # C[0,5] = al + br, C[1,5] = dl + er
        U[0, col] += 1.0; V[5, col] += 1.0; W[10, col] += 1.0  # a*l
        col += 1
        U[1, col] += 1.0; V[11, col] += 1.0; W[10, col] += 1.0  # b*r
        col += 1
        U[3, col] += 1.0; V[5, col] += 1.0; W[11, col] += 1.0  # d*l
        col += 1
        U[4, col] += 1.0; V[11, col] += 1.0; W[11, col] += 1.0  # e*r
        col += 1
        
        # Part 3: A_2x1 * B_bot (2x1 * 1x6)
        # A_2x1 = [c; f] -> U[2]=c, U[5]=f
        # B_bot = [s t u v w x] -> V[12]=s, V[13]=t, V[14]=u, V[15]=v, V[16]=w, V[17]=x
        # C[0,k] = c*B[2,k], C[1,k] = f*B[2,k]
        # W: C[0,0]=W[0], C[0,1]=W[2], C[0,2]=W[4], C[0,3]=W[6], C[0,4]=W[8], C[0,5]=W[10]
        #     C[1,0]=W[1], C[1,1]=W[3], C[1,2]=W[5], C[1,3]=W[7], C[1,4]=W[9], C[1,5]=W[11]
        
        for k in range(6):
            v_idx = 12 + k  # B[2,k]
            w_idx_top = k * 2 + 0  # C[0,k]
            w_idx_bot = k * 2 + 1  # C[1,k]
            U[2, col] += 1.0; V[v_idx, col] += 1.0; W[w_idx_top, col] += 1.0  # c * B[2,k]
            col += 1
            U[5, col] += 1.0; V[v_idx, col] += 1.0; W[w_idx_bot, col] += 1.0  # f * B[2,k]
            col += 1
        
        # Total rank used:
        # Part 1: 7 (Strassen) + 4 (3rd col) = 11
        # Part 2: 7 (Strassen) + 4 (3rd col) = 11
        # Part 3: 12
        # Total: 34
        
        # Wait, that's 34, not 30. Let me recount.
        # Part 1: M1-M7 = 7 terms, then 4 terms for 3rd col = 11
        # Part 2: M1-M7 = 7 terms, then 4 terms for 3rd col = 11
        # Part 3: 12 terms
        # Total: 34
        
        # Hmm, 34 is better than 36 but not as good as I calculated earlier.
        # Earlier I said 9+9+12=30, but that was wrong because 2x2*2x3 with Strassen
        # uses 7+4=11, not 9. The 9 was 7 (Strassen for 2x2*2x2) + 2 (for 2x1*2x1).
        # But 2x2*2x3 is not 2x2*2x2 + 2x1*2x1. It's 2x2 * (2x2 + 2x1).
        # The 2x2*2x1 part has rank 4 (each of 2 rows times each of 1 col... no).
        # 2x2 * 2x1: A=[a b; d e], B=[i; o]
        # C[0] = ai+bo, C[1] = di+eo: 4 terms, rank 4.
        # So 2x2*2x3 = 2x2*2x2 (rank 7) + 2x2*2x1 (rank 4) = rank 11.
        # That matches.
        
        # So total rank = 11 + 11 + 12 = 34.
        
        # Let me verify and return.
        U_flat = U[:, :col]
        V_flat = V[:, :col]
        W_flat = W[:, :col]
        
        if col != rank:
            # Trim to actual rank used
            pass
        
        result = {"U": U_flat.tolist(), "V": V_flat.tolist(), "W": W_flat.tolist()}
        
        # Verify
        if verify(U_flat.tolist(), V_flat.tolist(), W_flat.tolist()):
            return result
        
        # If verification fails, fall back to schoolbook
        return schoolbook()

    # Try the Strassen-based decomposition
    result = strassen_decomposition()
    
    # Verify the result
    rank = len(result['U'][0])
    if not verify(result['U'], result['V'], result['W']):
        # Fallback to schoolbook
        result = schoolbook()
    
    # Check time
    if time.time() > deadline:
        return result
    
    return result

# Quick self-test
if __name__ == "__main__":
    result = construct()
    rank = len(result['U'][0])
    print(f"Found decomposition with rank {rank}")
    
    n, m, p = 2, 3, 6
    T = np.ones((n, m, p))
    
    U_arr = np.array(result['U'])
    V_arr = np.array(result['V'])
    W_arr = np.array(result['W'])
    
    U_3d = U_arr.reshape(n, m, rank)
    V_3d = V_arr.reshape(m, p, rank)
    W_3d = W_arr.reshape(p, n, rank)
    
    T_calc = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
    print(f"Max error: {np.max(np.abs(T - T_calc))}")

