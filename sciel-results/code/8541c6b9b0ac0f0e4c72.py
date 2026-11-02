import time
import sys

DIMENSIONS = [2, 4, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    n, m, p = DIMENSIONS
    # n=2, m=4, p=7
    
    # We will construct a decomposition of rank 52
    # Using Strassen for 2x2x2 blocks and Schoolbook for 2x2x3 blocks
    # Split A (2x4) into A1 (2x2) and A2 (2x2)
    # Split B (4x7) into B1 (2x7) and B2 (2x7)
    # B1 split into B1a (2x2), B1b (2x2), B1c (2x3)
    # B2 split into B2a (2x2), B2b (2x2), B2c (2x3)
    
    # Total rank:
    # A1*B1: 7 (Strassen) + 7 (Strassen) + 12 (Schoolbook 2x2x3) = 26
    # A2*B2: 7 (Strassen) + 7 (Strassen) + 12 (Schoolbook 2x2x3) = 26
    # Total = 52
    
    rank = 52
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add Strassen 2x2x2
    def add_strassen_222(u_offset, v_offset, w_offset, col_start):
        # u_offset: (i_offset, j_offset) in A
        # v_offset: (j_offset, k_offset) in B
        # w_offset: (k_offset, i_offset) in C
        # Local indices 0,1 for i,j,k
        # A indices: (i_offset + li, j_offset + lj)
        # B indices: (j_offset + lj, k_offset + lk)
        # C indices: (k_offset + lk, i_offset + li)
        
        i_off, j_off = u_offset
        j_off_v, k_off = v_offset
        # Note: j_off in U and j_off_v in V must match for the block to be consistent
        # In our case, for A1*B1a, j_off=0, j_off_v=0.
        
        # Strassen terms
        # M1 = (a+d)(e+h)
        # M2 = (c+d)e
        # M3 = a(f-h)
        # M4 = d(g-c)
        # M5 = (a+b)h
        # M6 = (c-d)(g+h)
        # M7 = (b-d)(e+f)
        
        terms = [
            # (u_vec, v_vec, w_vec)
            # u_vec is size 4 (2x2 A), v_vec size 4 (2x2 B), w_vec size 4 (2x2 C)
            # C index mapping: k*2+i.
            # C00 (0,0) -> 0
            # C10 (0,1) -> 1 (Wait, C is 2x2. i=1, k=0 -> 0*2+1=1)
            # C01 (1,0) -> 2 (i=0, k=1 -> 1*2+0=2)
            # C11 (1,1) -> 3 (i=1, k=1 -> 1*2+1=3)
            
            # M1: (1,0,0,1) * (1,0,0,1) -> C00, C11
            ([1,0,0,1], [1,0,0,1], [1,0,0,1]),
            # M2: (0,0,1,1) * (1,0,0,0) -> C10, C11
            ([0,0,1,1], [1,0,0,0], [0,1,0,1]),
            # M3: (1,0,0,0) * (0,1,0,-1) -> C01
            ([1,0,0,0], [0,1,0,-1], [0,0,1,0]),
            # M4: (0,0,0,1) * (0,0,1,-1) -> C00, C10
            ([0,0,0,1], [0,0,1,-1], [1,1,0,0]),
            # M5: (1,1,0,0) * (0,0,0,1) -> C00, C01
            ([1,1,0,0], [0,0,0,1], [1,0,1,0]),
            # M6: (0,0,1,-1) * (0,0,1,1) -> C11
            ([0,0,1,-1], [0,0,1,1], [0,0,0,1]),
            # M7: (0,1,0,-1) * (1,1,0,0) -> C00, C11
            ([0,1,0,-1], [1,1,0,0], [1,0,0,1]),
        ]
        
        for u, v, w in terms:
            # Map to global indices
            for li in range(2):
                for lj in range(2):
                    if u[li*2+lj] != 0:
                        U[(i_off + li) * m + (j_off + lj)][col_start] += u[li*2+lj]
            
            for lj in range(2):
                for lk in range(2):
                    if v[lj*2+lk] != 0:
                        V[(j_off_v + lj) * p + (k_off + lk)][col_start] += v[lj*2+lk]
            
            for lk in range(2):
                for li in range(2):
                    if w[lk*2+li] != 0:
                        W[(k_off + lk) * n + (i_off + li)][col_start] += w[lk*2+li]
            
            col_start += 1
        return col_start

    # Helper to add Schoolbook 2x2xk
    def add_schoolbook_22k(u_offset, v_offset, w_offset, col_start, k_size):
        i_off, j_off = u_offset
        j_off_v, k_off = v_offset
        
        # k_size is 3
        for li in range(2):
            for lj in range(2):
                for lk in range(k_size):
                    # u: A[li, lj]
                    # v: B[lj, lk]
                    # w: C[li, lk]
                    
                    U[(i_off + li) * m + (j_off + lj)][col_start] = 1
                    V[(j_off_v + lj) * p + (k_off + lk)][col_start] = 1
                    W[(k_off + lk) * n + (i_off + li)][col_start] = 1
                    col_start += 1
        return col_start

    # Block 1: A1 * B1
    # A1: i in 0,1; j in 0,1
    # B1: j in 0,1; k in 0,6
    # B1a: k in 0,1
    col = add_strassen_222((0,0), (0,0), (0,0), col) # A1(0,0..1) * B1a(0,0..1)
    # B1b: k in 2,3
    col = add_strassen_222((0,0), (0,2), (0,2), col) # A1(0,0..1) * B1b(0,2..3)
    # B1c: k in 4,5,6
    col = add_schoolbook_22k((0,0), (0,4), (0,4), col, 3) # A1(0,0..1) * B1c(0,4..6)
    
    # Block 2: A2 * B2
    # A2: i in 0,1; j in 2,3
    # B2: j in 2,3; k in 0,6
    # B2a: k in 0,1
    col = add_strassen_222((0,2), (2,0), (0,0), col) # A2(0,2..3) * B2a(2,0..1)
    # B2b: k in 2,3
    col = add_strassen_222((0,2), (2,2), (0,2), col) # A2(0,2..3) * B2b(2,2..3)
    # B2c: k in 4,5,6
    col = add_schoolbook_22k((0,2), (2,4), (0,4), col, 3) # A2(0,2..3) * B2c(2,4..6)
    
    # Verify col == rank
    if col != rank:
        print(f"Warning: rank mismatch. Expected {rank}, got {col}", file=sys.stderr)
        
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    construct()
