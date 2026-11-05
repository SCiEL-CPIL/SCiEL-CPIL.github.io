import numpy as np
import time

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 5, 5, 5
    start_time = time.time()
    deadline = start_time + 140

    # Partition 5 into 2, 2, 1
    parts = [2, 2, 1]
    offsets = [0, 2, 4]

    U_cols = []
    V_cols = []
    W_cols = []

    def add_col(u_vec, v_vec, w_vec):
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)

    def add_strassen_2x2x2(off_i, off_j, off_k):
        """Strassen for 2x2x2 block at global offsets."""
        # M1 = (A00+A11)(B00+B11)
        u = [0]*25; v = [0]*25; w = [0]*25
        u[(off_i+0)*5+(off_j+0)] = 1
        u[(off_i+1)*5+(off_j+1)] = 1
        v[(off_j+0)*5+(off_k+0)] = 1
        v[(off_j+1)*5+(off_k+1)] = 1
        w[(off_k+0)*5+(off_i+0)] = 1
        w[(off_k+1)*5+(off_i+1)] = 1
        add_col(u, v, w)

        # M2 = (A10+A11)B00
        u = [0]*25; v = [0]*25; w = [0]*25
        u[(off_i+1)*5+(off_j+0)] = 1
        u[(off_i+1)*5+(off_j+1)] = 1
        v[(off_j+0)*5+(off_k+0)] = 1
        w[(off_k+0)*5+(off_i+1)] = 1
        w[(off_k+1)*5+(off_i+1)] = -1
        add_col(u, v, w)

        # M3 = A00(B01-B11)
        u = [0]*25; v = [0]*25; w = [0]*25
        u[(off_i+0)*5+(off_j+0)] = 1
        v[(off_j+0)*5+(off_k+1)] = 1
        v[(off_j+1)*5+(off_k+1)] = -1
        w[(off_k+1)*5+(off_i+0)] = 1
        w[(off_k+1)*5+(off_i+1)] = 1
        add_col(u, v, w)

        # M4 = A11(B10-B00)
        u = [0]*25; v = [0]*25; w = [0]*25
        u[(off_i+1)*5+(off_j+1)] = 1
        v[(off_j+1)*5+(off_k+0)] = 1
        v[(off_j+0)*5+(off_k+0)] = -1
        w[(off_k+0)*5+(off_i+0)] = 1
        w[(off_k+0)*5+(off_i+1)] = 1
        add_col(u, v, w)

        # M5 = (A00+A01)B11
        u = [0]*25; v = [0]*25; w = [0]*25
        u[(off_i+0)*5+(off_j+0)] = 1
        u[(off_i+0)*5+(off_j+1)] = 1
        v[(off_j+1)*5+(off_k+1)] = 1
        w[(off_k+0)*5+(off_i+0)] = -1
        w[(off_k+1)*5+(off_i+0)] = 1
        add_col(u, v, w)

        # M6 = (A10-A00)(B00+B01)
        u = [0]*25; v = [0]*25; w = [0]*25
        u[(off_i+1)*5+(off_j+0)] = 1
        u[(off_i+0)*5+(off_j+0)] = -1
        v[(off_j+0)*5+(off_k+0)] = 1
        v[(off_j+0)*5+(off_k+1)] = 1
        w[(off_k+1)*5+(off_i+1)] = 1
        add_col(u, v, w)

        # M7 = (A01-A11)(B10+B11)
        u = [0]*25; v = [0]*25; w = [0]*25
        u[(off_i+0)*5+(off_j+1)] = 1
        u[(off_i+1)*5+(off_j+1)] = -1
        v[(off_j+1)*5+(off_k+0)] = 1
        v[(off_j+1)*5+(off_k+1)] = 1
        w[(off_k+0)*5+(off_i+0)] = 1
        add_col(u, v, w)

    def add_strassen_1x2x2(off_i, off_j, off_k):
        """Strassen-like for 1x2x2 block: A is 1x2, B is 2x2, C is 1x2.
        C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
        C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
        Use 3 multiplications via Strassen trick:
        M1 = A[0,0]*(B[0,0]+B[1,0]) -> contributes to C[0,0]
        M2 = A[0,1]*(B[0,1]+B[1,1]) -> contributes to C[0,1]  
        M3 = (A[0,0]+A[0,1])*(B[0,1]+B[1,0]) - ... 
        Actually for 1x2x2, schoolbook is already 4 terms.
        Can we do better? The tensor is essentially a 2x2 outer product structure.
        Minimum rank for 1x2x2 is 2 (since it's separable in a sense).
        Actually: C[0,k] = sum_j A[0,j]*B[j,k]
        This is: C = A * B where A is 1x2, B is 2x2.
        The tensor has rank min(1*2, 2*2, 1*2) ... no.
        Rank of 1x2x2 tensor: it's the matrix product tensor restricted.
        Actually rank(1,2,2) = 2 since we can write:
        C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
        C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
        Term 1: A[0,0] * (B[0,0], B[0,1]) -> W: (1,0)
        Term 2: A[0,1] * (B[1,0], B[1,1]) -> W: (1,0)
        That's rank 2! Let me verify:
        U1 = [1,0], V1 = [1,1,0,0] (B[0,0],B[0,1],B[1,0],B[1,1] -> [1,1,0,0]), W1 = [1,0]
        U2 = [0,1], V2 = [0,0,1,1], W2 = [1,0]
        Check: C[0,0] = 1*B[0,0] + 0*B[1,0] = B[0,0]... no that's wrong.
        
        Let me be more careful. For 1x2x2:
        A has indices (0,0),(0,1) -> 2 entries
        B has indices (0,0),(0,1),(1,0),(1,1) -> 4 entries  
        C has indices (0,0),(0,1) -> 2 entries
        
        C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
        C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
        
        Rank-1 term 1: U=[1,0], V=[1,1,0,0], W=[1,1]
          Contributes: 1*B[0,0] to C[0,0], 1*B[0,1] to C[0,1] -> gives A[0,0]*B[0,*]
        Rank-1 term 2: U=[0,1], V=[0,0,1,1], W=[1,1]
          Contributes: 1*B[1,0] to C[0,0], 1*B[1,1] to C[0,1] -> gives A[0,1]*B[1,*]
        
        Total: C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] ✓
                C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1] ✓
        
        So rank 2!
        """
        # Term 1: A[0,0] * row 0 of B
        u = [0]*25; v = [0]*25; w = [0]*25
        u[(off_i+0)*5+(off_j+0)] = 1
        v[(off_j+0)*5+(off_k+0)] = 1
        v[(off_j+0)*5+(off_k+1)] = 1
        w[(off_k+0)*5+(off_i+0)] = 1
        w[(off_k+1)*5+(off_i+0)] = 1
        add_col(u, v, w)

        # Term 2: A[0,1] * row 1 of B
        u = [0]*25; v = [0]*25; w = [0]*25
        u[(off_i+0)*5+(off_j+1)] = 1
        v[(off_j+1)*5+(off_k+0)] = 1
        v[(off_j+1)*5+(off_k+1)] = 1
        w[(off_k+0)*5+(off_i+0)] = 1
        w[(off_k+1)*5+(off_i+0)] = 1
        add_col(u, v, w)

    def add_strassen_2x1x2(off_i, off_j, off_k):
        """For 2x1x2 block: A is 2x1, B is 1x2, C is 2x2.
        C[i,0] = A[i,0]*B[0,0]
        C[i,1] = A[i,0]*B[0,1]
        Rank 2:
        Term 1: col 0 of A * B[0,0] -> W col 0
        Term 2: col 0 of A * B[0,1] -> W col 1
        """
        # Term 1: A[:,0] * B[0,0]
        u = [0]*25; v = [0]*25; w = [0]*25
        u[(off_i+0)*5+(off_j+0)] = 1
        u[(off_i+1)*5+(off_j+0)] = 1
        v[(off_j+0)*5+(off_k+0)] = 1
        w[(off_k+0)*5+(off_i+0)] = 1
        w[(off_k+0)*5+(off_i+1)] = 1
        add_col(u, v, w)

        # Term 2: A[:,0] * B[0,1]
        u = [0]*25; v = [0]*25; w = [0]*25
        u[(off_i+0)*5+(off_j+0)] = 1
        u[(off_i+1)*5+(off_j+0)] = 1
        v[(off_j+0)*5+(off_k+1)] = 1
        w[(off_k+1)*5+(off_i+0)] = 1
        w[(off_k+1)*5+(off_i+1)] = 1
        add_col(u, v, w)

    def add_strassen_2x2x1(off_i, off_j, off_k):
        """For 2x2x1 block: A is 2x2, B is 2x1, C is 2x1.
        C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
        C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
        Rank 2:
        Term 1: A[:,0] * B[0,0]
        Term 2: A[:,1] * B[1,0]
        """
        # Term 1: col 0 of A * B[0,0]
        u = [0]*25; v = [0]*25; w = [0]*25
        u[(off_i+0)*5+(off_j+0)] = 1
        u[(off_i+1)*5+(off_j+0)] = 1
        v[(off_j+0)*5+(off_k+0)] = 1
        w[(off_k+0)*5+(off_i+0)] = 1
        w[(off_k+0)*5+(off_i+1)] = 1
        add_col(u, v, w)

        # Term 2: col 1 of A * B[1,0]
        u = [0]*25; v = [0]*25; w = [0]*25
        u[(off_i+0)*5+(off_j+1)] = 1
        u[(off_i+1)*5+(off_j+1)] = 1
        v[(off_j+1)*5+(off_k+0)] = 1
        w[(off_k+0)*5+(off_i+0)] = 1
        w[(off_k+0)*5+(off_i+1)] = 1
        add_col(u, v, w)

    def add_strassen_1x1x2(off_i, off_j, off_k):
        """For 1x1x2 block: A is 1x1, B is 1x2, C is 1x2.
        C[0,0] = A[0,0]*B[0,0]
        C[0,1] = A[0,0]*B[0,1]
        Rank 2 (or 1? No, need 2 since B has 2 independent entries)
        Actually rank 1: U=[1], V=[1,1], W=[1,1]
        Check: C[0,0] = 1*B[0,0]*1 = B[0,0]*A[0,0] ✓ (U picks A[0,0])
        C[0,1] = 1*B[0,1]*1 = B[0,1]*A[0,0] ✓
        Yes! Rank 1.
        """
        u = [0]*25; v = [0]*25; w = [0]*25
        u[(off_i+0)*5+(off_j+0)] = 1
        v[(off_j+0)*5+(off_k+0)] = 1
        v[(off_j+0)*5+(off_k+1)] = 1
        w[(off_k+0)*5+(off_i+0)] = 1
        w[(off_k+1)*5+(off_i+0)] = 1
        add_col(u, v, w)

    def add_strassen_1x2x1(off_i, off_j, off_k):
        """For 1x2x1 block: A is 1x2, B is 2x1, C is 1x1.
        C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
        Rank 2:
        Term 1: A[0,0]*B[0,0]
        Term 2: A[0,1]*B[1,0]
        """
        # Term 1
        u = [0]*25; v = [0]*25; w = [0]*25
        u[(off_i+0)*5+(off_j+0)] = 1
        v[(off_j+0)*5+(off_k+0)] = 1
        w[(off_k+0)*5+(off_i+0)] = 1
        add_col(u, v, w)

        # Term 2
        u = [0]*25; v = [0]*25; w = [0]*25
        u[(off_i+0)*5+(off_j+1)] = 1
        v[(off_j+1)*5+(off_k+0)] = 1
        w[(off_k+0)*5+(off_i+0)] = 1
        add_col(u, v, w)

    def add_strassen_2x1x1(off_i, off_j, off_k):
        """For 2x1x1 block: A is 2x1, B is 1x1, C is 2x1.
        C[0,0] = A[0,0]*B[0,0]
        C[1,0] = A[1,0]*B[0,0]
        Rank 1: U=[1,1], V=[1], W=[1,1]
        """
        u = [0]*25; v = [0]*25; w = [0]*25
        u[(off_i+0)*5+(off_j+0)] = 1
        u[(off_i+1)*5+(off_j+0)] = 1
        v[(off_j+0)*5+(off_k+0)] = 1
        w[(off_k+0)*5+(off_i+0)] = 1
        w[(off_k+0)*5+(off_i+1)] = 1
        add_col(u, v, w)

    def add_schoolbook(off_i, off_j, off_k, sa, sb, sc):
        """Schoolbook for sa x sb x sc block."""
        for u in range(sa):
            for v in range(sb):
                for w in range(sc):
                    u_vec = [0]*25; v_vec = [0]*25; w_vec = [0]*25
                    gi = off_i + u
                    gj = off_j + v
                    gk = off_k + w
                    u_vec[gi*5 + gj] = 1
                    v_vec[gj*5 + gk] = 1
                    w_vec[gk*5 + gi] = 1
                    add_col(u_vec, v_vec, w_vec)

    # Build the decomposition
    U_cols.clear()
    V_cols.clear()
    W_cols.clear()

    for a in range(3):
        for b in range(3):
            for c in range(3):
                if time.time() > deadline:
                    break
                    
                sa = parts[a]
                sb = parts[b]
                sc = parts[c]
                oa = offsets[a]
                ob = offsets[b]
                oc = offsets[c]
                
                if sa == 2 and sb == 2 and sc == 2:
                    add_strassen_2x2x2(oa, ob, oc)
                elif sa == 1 and sb == 2 and sc == 2:
                    add_strassen_1x2x2(oa, ob, oc)
                elif sa == 2 and sb == 1 and sc == 2:
                    add_strassen_2x1x2(oa, ob, oc)
                elif sa == 2 and sb == 2 and sc == 1:
                    add_strassen_2x2x1(oa, ob, oc)
                elif sa == 1 and sb == 1 and sc == 2:
                    add_strassen_1x1x2(oa, ob, oc)
                elif sa == 1 and sb == 2 and sc == 1:
                    add_strassen_1x2x1(oa, ob, oc)
                elif sa == 2 and sb == 1 and sc == 1:
                    add_strassen_2x1x1(oa, ob, oc)
                elif sa == 1 and sb == 1 and sc == 1:
                    # Rank 1
                    u_vec = [0]*25; v_vec = [0]*25; w_vec = [0]*25
                    u_vec[oa*5 + ob] = 1
                    v_vec[ob*5 + oc] = 1
                    w_vec[oc*5 + oa] = 1
                    add_col(u_vec, v_vec, w_vec)
                else:
                    add_schoolbook(oa, ob, oc, sa, sb, sc)
                
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break

    rank = len(U_cols)
    
    U = [[0]*rank for _ in range(25)]
    V = [[0]*rank for _ in range(25)]
    W = [[0]*rank for _ in range(25)]
    
    for col_idx in range(rank):
        u_col = U_cols[col_idx]
        v_col = V_cols[col_idx]
        w_col = W_cols[col_idx]
        for row_idx in range(25):
            U[row_idx][col_idx] = u_col[row_idx]
            V[row_idx][col_idx] = v_col[row_idx]
            W[row_idx][col_idx] = w_col[row_idx]
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
