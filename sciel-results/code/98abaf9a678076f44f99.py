import time
import numpy as np

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110

    # Rank 54 construction
    rank = 54
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_222(i_offset, j_offset, k_offset):
        """
        Adds a 2x2x2 Strassen block.
        i_offset: row offset in A (0 or 1 for 2x2? No, A is 2x5. 
                  i_offset is the index in the 2x2 block's local rows?
                  Actually, A is 2x2. Rows 0,1.
                  j_offset: col offset in A (0..3 for 2x5? No, 0 and 2? No.)
                  Wait, A1 is 2x2. j is 0,1.
                  B1 is 2x6. k is 0..5.
                  We split k into 0,1; 2,3; 4,5.
        """
        nonlocal col
        # Strassen products for 2x2x2
        # M1 = (a+d)(e+h)
        # M2 = (c+d)e
        # M3 = a(f-h)
        # M4 = d(g-e)
        # M5 = (a+b)h
        # M6 = (c-a)(e+f)
        # M7 = (b-d)(g+h)
        
        # Indices in local 2x2 A: (0,0), (0,1), (1,0), (1,1)
        # Indices in local 2x2 B: (0,0), (0,1), (1,0), (1,1)
        # Indices in local 2x2 C: (0,0), (0,1), (1,0), (1,1)
        
        # We map local (li, lj) to global A index (i_offset+li, j_offset+lj)
        # We map local (lj, lk) to global B index (j_offset+lj, k_offset+lk)
        # We map local (li, lk) to global C index (k_offset+lk, i_offset+li)
        
        def set_u(li, lj, val):
            gi = i_offset + li
            gj = j_offset + lj
            U[gi * m + gj][col] += val
            
        def set_v(lj, lk, val):
            gj = j_offset + lj
            gk = k_offset + lk
            V[gj * p + gk][col] += val
            
        def set_w(li, lk, val):
            gi = i_offset + li
            gk = k_offset + lk
            W[gk * n + gi][col] += val
            
        # M1: (a00+a11)(b00+b11)
        set_u(0,0,1); set_u(1,1,1)
        set_v(0,0,1); set_v(1,1,1)
        set_w(0,0,1); set_w(1,1,1)
        col += 1
        
        # M2: (a10+a11)b00
        set_u(1,0,1); set_u(1,1,1)
        set_v(0,0,1)
        set_w(1,0,1)
        col += 1
        
        # M3: a00(b01-b11)
        set_u(0,0,1)
        set_v(0,1,1); set_v(1,1,-1)
        set_w(0,1,1)
        col += 1
        
        # M4: a11(b10-b00)
        set_u(1,1,1)
        set_v(1,0,1); set_v(0,0,-1)
        set_w(0,0,-1); set_w(0,1,1) # Wait. C00 has -M4, C01 has M4.
        # C00 = M1 - M4 + M5 + M7
        # C01 = M3 + M4
        # So W(0,0) gets -1, W(0,1) gets 1.
        # Correct.
        col += 1
        
        # M5: (a00+a01)b11
        set_u(0,0,1); set_u(0,1,1)
        set_v(1,1,1)
        set_w(0,0,1); set_w(1,0,1) # C00 has M5, C10 has M5.
        # C00 = ... + M5 ...
        # C10 = M2 + M5
        # Correct.
        col += 1
        
        # M6: (a10-a00)(b00+b01)
        set_u(1,0,1); set_u(0,0,-1)
        set_v(0,0,1); set_v(0,1,1)
        set_w(1,1,1) # C11 has M6.
        # C11 = M1 - M2 + M3 + M6
        # Correct.
        col += 1
        
        # M7: (a01-a11)(b10+b11)
        set_u(0,1,1); set_u(1,1,-1)
        set_v(1,0,1); set_v(1,1,1)
        set_w(0,0,1) # C00 has M7.
        # C00 = M1 - M4 + M5 + M7
        # Correct.
        col += 1

    def add_block_2x3x2(i_offset, j_offset, k_offset):
        """
        Adds a 2x3x2 block (Rank 11).
        A is 2x3. B is 3x2.
        j_offset is the start of the 3 columns in A.
        k_offset is the start of the 2 columns in B (output).
        """
        nonlocal col
        
        # Part 1: 2x1x2 (Rank 4)
        # A1 is col j_offset. B1 is row j_offset.
        # A1 = [a0, a1]^T. B1 = [b0, b1].
        # C = [a0b0, a0b1; a1b0, a1b1]
        
        # r1: a0*b0 -> C00
        U[i_offset * m + (j_offset)][col] = 1
        V[(j_offset) * p + (k_offset)][col] = 1
        W[(k_offset) * n + i_offset][col] = 1
        col += 1
        
        # r2: a0*b1 -> C01
        U[i_offset * m + (j_offset)][col] = 1
        V[(j_offset) * p + (k_offset+1)][col] = 1
        W[(k_offset+1) * n + i_offset][col] = 1
        col += 1
        
        # r3: a1*b0 -> C10
        U[(i_offset+1) * m + (j_offset)][col] = 1
        V[(j_offset) * p + (k_offset)][col] = 1
        W[(k_offset) * n + (i_offset+1)][col] = 1
        col += 1
        
        # r4: a1*b1 -> C11
        U[(i_offset+1) * m + (j_offset)][col] = 1
        V[(j_offset) * p + (k_offset+1)][col] = 1
        W[(k_offset+1) * n + (i_offset+1)][col] = 1
        col += 1
        
        # Part 2: 2x2x2 (Rank 7)
        # A2 is cols j_offset+1, j_offset+2.
        # B2 is rows j_offset+1, j_offset+2.
        # We can call add_strassen_222 with local mapping.
        # But add_strassen_222 assumes 0..1 indices.
        # We need to map local 0,1 to global j_offset+1, j_offset+2.
        # And local i 0,1 to global i_offset, i_offset+1.
        # And local k 0,1 to global k_offset, k_offset+1.
        
        # We can reuse the logic by creating a wrapper or just inlining.
        # Let's just call a helper that takes offsets.
        
        def set_u_222(li, lj, val):
            gi = i_offset + li
            gj = (j_offset + 1) + lj
            U[gi * m + gj][col] += val
            
        def set_v_222(lj, lk, val):
            gj = (j_offset + 1) + lj
            gk = k_offset + lk
            V[gj * p + gk][col] += val
            
        def set_w_222(li, lk, val):
            gi = i_offset + li
            gk = k_offset + lk
            W[gk * n + gi][col] += val
            
        # M1
        set_u_222(0,0,1); set_u_222(1,1,1)
        set_v_222(0,0,1); set_v_222(1,1,1)
        set_w_222(0,0,1); set_w_222(1,1,1)
        col += 1
        
        # M2
        set_u_222(1,0,1); set_u_222(1,1,1)
        set_v_222(0,0,1)
        set_w_222(1,0,1)
        col += 1
        
        # M3
        set_u_222(0,0,1)
        set_v_222(0,1,1); set_v_222(1,1,-1)
        set_w_222(0,1,1)
        col += 1
        
        # M4
        set_u_222(1,1,1)
        set_v_222(1,0,1); set_v_222(0,0,-1)
        set_w_222(0,0,-1); set_w_222(0,1,1)
        col += 1
        
        # M5
        set_u_222(0,0,1); set_u_222(0,1,1)
        set_v_222(1,1,1)
        set_w_222(0,0,1); set_w_222(1,0,1)
        col += 1
        
        # M6
        set_u_222(1,0,1); set_u_222(0,0,-1)
        set_v_222(0,0,1); set_v_222(0,1,1)
        set_w_222(1,1,1)
        col += 1
        
        # M7
        set_u_222(0,1,1); set_u_222(1,1,-1)
        set_v_222(1,0,1); set_v_222(1,1,1)
        set_w_222(0,0,1)
        col += 1

    # A is 2x5. B is 5x6.
    # Split A into A1 (2x2, cols 0,1) and A2 (2x3, cols 2,3,4).
    # Split B into B1 (2x6, rows 0,1) and B2 (3x6, rows 2,3,4).
    
    # Part 1: A1 * B1 (2x2 * 2x6)
    # Split B1 into 3 blocks of 2x2.
    # k offsets: 0, 2, 4.
    for k_off in [0, 2, 4]:
        add_strassen_222(i_offset=0, j_offset=0, k_offset=k_off)
        
    # Part 2: A2 * B2 (2x3 * 3x6)
    # Split B2 into 3 blocks of 3x2.
    # k offsets: 0, 2, 4.
    for k_off in [0, 2, 4]:
        add_block_2x3x2(i_offset=0, j_offset=2, k_offset=k_off)
        
    # Verify col == rank
    if col != rank:
        # Should not happen
        pass

    # Verification
    def verify_exact(U, V, W):
        r = len(U[0]) if U else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(r):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    if not verify_exact(U, V, W):
        # Fallback to schoolbook if verification fails
        U = [[0] * 60 for _ in range(n * m)]
        V = [[0] * 60 for _ in range(m * p)]
        W = [[0] * 60 for _ in range(p * n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][c] = 1
                    V[j * p + k][c] = 1
                    W[k * n + i][c] = 1
                    c += 1
                    rank = 60
        return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    result = construct()
    n, m, p = DIMENSIONS
    U, V, W = result['U'], result['V'], result['W']
    rank = len(U[0])
    max_err = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                max_err = max(max_err, abs(s - 1))
    print(f"Rank: {rank}, Max Error: {max_err}")
