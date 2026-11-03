import time
import numpy as np

DIMENSIONS = [2, 4, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n

    def verify(U, V, W):
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U_np[i * m + j] * V_np[j * p + k] * W_np[k * n + i])
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank), dtype=int)
        V = np.zeros((mp, rank), dtype=int)
        W = np.zeros((pn, rank), dtype=int)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    def strassen_2x2x2_decomp():
        """
        Returns U, V, W for 2x2x2 multiplication using Strassen's algorithm (rank 7).
        Indices:
        U: 2x2 (A)
        V: 2x2 (B)
        W: 2x2 (C)
        """
        U = np.zeros((4, 7), dtype=int)
        V = np.zeros((4, 7), dtype=int)
        W = np.zeros((4, 7), dtype=int)

        # M1 = (A00+A11)*(B00+B11)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1  # C00 += M1, C11 += M1

        # M2 = (A10+A11)*B00
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = -1  # C10 += M2, C11 -= M2

        # M3 = A00*(B01-B11)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1; W[3, 2] = 1  # C01 += M3, C11 += M3

        # M4 = A11*(B10-B00)
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1  # C00 += M4, C10 += M4

        # M5 = (A00+A01)*B11
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1  # C00 -= M5, C01 += M5

        # M6 = (A10-A00)*(B00+B01)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1  # C11 += M6

        # M7 = (A01-A11)*(B10+B11)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1  # C00 += M7

        return U, V, W

    def try_strassen_blocks():
        """
        Construct decomposition by partitioning:
        A is 2x4. B is 4x7.
        Split B into two 2x7 matrices: B_left (cols 0,1) and B_right (cols 2,3,4,5,6)?
        No, Strassen requires square sub-blocks or specific structures.
        Let's partition the 4x7 product into:
        1. A[:, 0:2] * B[0:2, :] -> 2x2 * 2x7.
           This can be seen as two 2x2x2 blocks (for B cols 0:2 and 2:4) plus a 2x2x3 remainder.
           Actually, let's just partition the 7 columns of B into groups of 2 and 1.
           B = [B0 (2x2), B1 (2x2), B2 (2x2), B3 (2x1)]? No, B is 4x7.
           Let's split the 4 rows of B into two 2-row blocks? No, A is 2x4, so A's columns correspond to B's rows.
           A = [A0 (2x2), A1 (2x2)].
           B = [[B00 (2x?); B10 (2x?)] ...]
           
           Let's use the standard block decomposition:
           A = [A0 | A1] where A0, A1 are 2x2.
           B = [B0; B1] where B0, B1 are 2x7.
           C = A0*B0 + A1*B1.
           
           Now, for A0*B0 (2x2 * 2x7):
           Split B0 into [B00 (2x2) | B01 (2x2) | B02 (2x2) | B03 (2x1)].
           A0*B0 = A0*B00 + A0*B01 + A0*B02 + A0*B03.
           Each 2x2 * 2x2 term can use Strassen (rank 7).
           The 2x2 * 2x1 term can use schoolbook (rank 2*2*1 = 4? No, 2*1*2=4? No, 2x2 * 2x1 is 2x1. Rank is 2*2*1=4? No, 2*1*2=4 is wrong. 2x2 * 2x1 involves 2*2*1 = 4 multiplications? Yes.
           
           Wait, let's just sum ranks:
           A0*B00: 7
           A0*B01: 7
           A0*B02: 7
           A0*B03: 4 (schoolbook 2x2 * 2x1)
           
           A1*B1: Same structure.
           A1*B10: 7
           A1*B11: 7
           A1*B12: 7
           A1*B13: 4
           
           Total rank = 3*7 + 4 + 3*7 + 4 = 21 + 4 + 21 + 4 = 50.
           
           Can we do better?
           The current program uses schoolbook rank 56.
           50 is an improvement.
           
           Let's implement this specific construction.
        """
        rank = 50
        U = np.zeros((nm, rank), dtype=int)
        V = np.zeros((mp, rank), dtype=int)
        W = np.zeros((pn, rank), dtype=int)
        
        # We need to map the local indices of the sub-blocks to the global indices.
        # Global A: 2x4. Indices (i, j) where i in {0,1}, j in {0,1,2,3}.
        # Global B: 4x7. Indices (j, k) where j in {0,1,2,3}, k in {0..6}.
        # Global C: 7x2. Indices (k, i) where k in {0..6}, i in {0,1}.
        
        # Block 1: A0 * B0
        # A0 corresponds to A[:, 0:2] -> j in {0,1}
        # B0 corresponds to B[0:2, :] -> j in {0,1}
        # Wait, A is 2x4, B is 4x7.
        # A * B = sum_j A[:, j] * B[j, :]
        # A0 is A[:, 0:2]. B0 is B[0:2, :].
        # So A0 * B0 contributes to C.
        
        # Let's define a helper to add a 2x2x2 Strassen block.
        def add_strassen_block(A_j_offset, B_j_offset, B_k_offset, rank_offset):
            """
            Adds a Strassen block for A[:, A_j_offset:A_j_offset+2] * B[B_j_offset:B_j_offset+2, B_k_offset:B_k_offset+2]
            A_j_offset: column offset in A (0 or 2)
            B_j_offset: row offset in B (0 or 2)
            B_k_offset: column offset in B (0, 2, 4, 6)
            """
            U_block, V_block, W_block = strassen_2x2x2_decomp()
            
            # Map U (2x2) to global U (2x4)
            # U_block indices: (i, j_local) where i in {0,1}, j_local in {0,1}
            # Global U indices: (i, j_global) where j_global = A_j_offset + j_local
            for i in range(2):
                for jl in range(2):
                    jg = A_j_offset + jl
                    U[i * 4 + jg, rank_offset:] += U_block[i * 2 + jl]
            
            # Map V (2x2) to global V (4x7)
            # V_block indices: (j_local, k_local) where j_local in {0,1}, k_local in {0,1}
            # Global V indices: (j_global, k_global) where j_global = B_j_offset + j_local, k_global = B_k_offset + k_local
            for jl in range(2):
                for kl in range(2):
                    jg = B_j_offset + jl
                    kg = B_k_offset + kl
                    V[jg * 7 + kg, rank_offset:] += V_block[jl * 2 + kl]
            
            # Map W (2x2) to global W (7x2)
            # W_block indices: (k_local, i) where k_local in {0,1}, i in {0,1}
            # Global W indices: (k_global, i) where k_global = B_k_offset + k_local
            for kl in range(2):
                for i in range(2):
                    kg = B_k_offset + kl
                    W[kg * 2 + i, rank_offset:] += W_block[kl * 2 + i]
        
        def add_schoolbook_block(A_j_offset, A_j_size, B_j_offset, B_j_size, B_k_offset, B_k_size, rank_offset):
            """
            Adds a schoolbook block for A[:, A_j_offset:A_j_offset+A_j_size] * B[B_j_offset:B_j_offset+B_j_size, B_k_offset:B_k_offset+B_k_size]
            Note: A_j_size must equal B_j_size for this simple implementation.
            """
            for i in range(2):
                for jl in range(A_j_size):
                    for kl in range(B_k_size):
                        jg = A_j_offset + jl
                        kg = B_k_offset + kl
                        U[i * 4 + jg, rank_offset] = 1
                        V[jg * 7 + kg, rank_offset] = 1
                        W[kg * 2 + i, rank_offset] = 1
                        rank_offset += 1
            return rank_offset

        rank_offset = 0
        
        # Block A0 * B0
        # A0: j in {0,1}. B0: j in {0,1}, k in {0..6}
        # Split B0 into k-blocks of size 2,2,2,1
        # k=0: Strassen
        rank_offset += 7
        add_strassen_block(0, 0, 0, rank_offset - 7)
        # k=2: Strassen
        rank_offset += 7
        add_strassen_block(0, 0, 2, rank_offset - 7)
        # k=4: Strassen
        rank_offset += 7
        add_strassen_block(0, 0, 4, rank_offset - 7)
        # k=6: Schoolbook (2x2 * 2x1)
        rank_offset = add_schoolbook_block(0, 2, 0, 2, 6, 1, rank_offset)
        
        # Block A1 * B1
        # A1: j in {2,3}. B1: j in {2,3}, k in {0..6}
        # Split B1 into k-blocks of size 2,2,2,1
        # k=0: Strassen
        rank_offset += 7
        add_strassen_block(2, 2, 0, rank_offset - 7)
        # k=2: Strassen
        rank_offset += 7
        add_strassen_block(2, 2, 2, rank_offset - 7)
        # k=4: Strassen
        rank_offset += 7
        add_strassen_block(2, 2, 4, rank_offset - 7)
        # k=6: Schoolbook (2x2 * 2x1)
        rank_offset = add_schoolbook_block(2, 2, 2, 2, 6, 1, rank_offset)
        
        # Verify rank matches
        assert rank_offset == rank, f"Rank mismatch: {rank_offset} vs {rank}"
        
        if verify(U, V, W):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        else:
            return None

    # Start with schoolbook
    best = schoolbook()
    best_rank = n * m * p

    # Try the structured Strassen block construction
    if time.time() < deadline - 10.0:
        result = try_strassen_blocks()
        if result is not None:
            best = result
            best_rank = 50

    return best
