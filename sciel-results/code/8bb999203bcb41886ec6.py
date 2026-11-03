import numpy as np

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    
    def get_rank(n, m, p):
        # Base cases
        if n == 1 and m == 1 and p == 1:
            return 1
        if n == 2 and m == 2 and p == 2:
            return 7 # Strassen
        
        # Try splitting n
        best = n * m * p
        for i in range(1, n):
            r1 = get_rank(i, m, p)
            r2 = get_rank(n-i, m, p)
            best = min(best, r1 + r2)
            
        # Try splitting m
        for j in range(1, m):
            r1 = get_rank(n, j, p)
            r2 = get_rank(n, m-j, p)
            best = min(best, r1 + r2)
            
        # Try splitting p
        for k in range(1, p):
            r1 = get_rank(n, m, k)
            r2 = get_rank(n, m, p-k)
            best = min(best, r1 + r2)
            
        return best

    target_rank = get_rank(n, m, p)
    
    # Initialize global matrices
    U = [[0] * target_rank for _ in range(n * m)]
    V = [[0] * target_rank for _ in range(m * p)]
    W = [[0] * target_rank for _ in range(p * n)]
    
    def build(n, m, p, u_offset, v_offset, w_offset, rank_offset):
        """
        Recursively builds the decomposition for a sub-problem of size n x m x p.
        u_offset, v_offset, w_offset are the starting row indices in the global U, V, W.
        rank_offset is the starting column index in the global U, V, W.
        Returns the number of rank columns used.
        """
        if n == 1 and m == 1 and p == 1:
            # Rank 1
            U[u_offset][rank_offset] = 1
            V[v_offset][rank_offset] = 1
            W[w_offset][rank_offset] = 1
            return 1
            
        if n == 2 and m == 2 and p == 2:
            # Strassen Rank 7
            # Local indices:
            # U: i*m+j -> i*2+j
            # V: j*p+k -> j*2+k
            # W: k*n+i -> k*2+i
            
            # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
            # M2 = (A10+A11)B00 -> C10 += M2
            # M3 = A00(B01-B11) -> C01 += M3
            # M4 = A11(B10-B00) -> C11 += M4
            # M5 = (A00+A01)B11 -> C01 += M5
            # M6 = (A10-A00)(B00+B01) -> C10 += M6
            # M7 = (A01-A11)(B10+B11) -> C11 += M7
            
            def set_u(i, j, c, val):
                local_row = i * 2 + j
                U[u_offset + local_row][rank_offset + c] += val
                
            def set_v(j, k, c, val):
                local_row = j * 2 + k
                V[v_offset + local_row][rank_offset + c] += val
                
            def set_w(k, i, c, val):
                local_row = k * 2 + i
                W[w_offset + local_row][rank_offset + c] += val

            # c=0: M1
            set_u(0, 0, 0, 1); set_u(1, 1, 0, 1)
            set_v(0, 0, 0, 1); set_v(1, 1, 0, 1)
            set_w(0, 0, 0, 1); set_w(1, 1, 0, 1)
            
            # c=1: M2
            set_u(1, 0, 1, 1); set_u(1, 1, 1, 1)
            set_v(0, 0, 1, 1)
            set_w(0, 1, 1, 1) # C10: i=1, k=0 -> 0*2+1=1
            
            # c=2: M3
            set_u(0, 0, 2, 1)
            set_v(0, 1, 2, 1); set_v(1, 1, 2, -1)
            set_w(1, 0, 2, 1) # C01: i=0, k=1 -> 1*2+0=2
            
            # c=3: M4
            set_u(1, 1, 3, 1)
            set_v(1, 0, 3, 1); set_v(0, 0, 3, -1)
            set_w(1, 1, 3, 1) # C11: i=1, k=1 -> 1*2+1=3
            
            # c=4: M5
            set_u(0, 0, 4, 1); set_u(0, 1, 4, 1)
            set_v(1, 1, 4, 1)
            set_w(1, 0, 4, 1) # C01: i=0, k=1 -> 2
            
            # c=5: M6
            set_u(1, 0, 5, 1); set_u(0, 0, 5, -1)
            set_v(0, 0, 5, 1); set_v(0, 1, 5, 1)
            set_w(0, 1, 5, 1) # C10: i=1, k=0 -> 1
            
            # c=6: M7
            set_u(0, 1, 6, 1); set_u(1, 1, 6, -1)
            set_v(1, 0, 6, 1); set_v(1, 1, 6, 1)
            set_w(1, 1, 6, 1) # C11: i=1, k=1 -> 3
            
            return 7

        # Recursive case: Find best split
        best_rank = n * m * p
        best_split = None
        best_type = None
        
        for i in range(1, n):
            r = get_rank(i, m, p) + get_rank(n-i, m, p)
            if r < best_rank:
                best_rank = r
                best_split = i
                best_type = 'n'
                
        for j in range(1, m):
            r = get_rank(n, j, p) + get_rank(n, m-j, p)
            if r < best_rank:
                best_rank = r
                best_split = j
                best_type = 'm'
                
        for k in range(1, p):
            r = get_rank(n, m, k) + get_rank(n, m, p-k)
            if r < best_rank:
                best_rank = r
                best_split = k
                best_type = 'p'
                
        # If no split improved over schoolbook, just do schoolbook
        if best_type is None or best_rank >= n*m*p:
             # Schoolbook for this block
             col = rank_offset
             for i in range(n):
                 for j in range(m):
                     for k in range(p):
                         U[u_offset + i*m + j][col] = 1
                         V[v_offset + j*p + k][col] = 1
                         W[w_offset + k*n + i][col] = 1
                         col += 1
             return n*m*p

        # Execute the best split
        if best_type == 'n':
            i1 = best_split
            i2 = n - i1
            r1 = build(i1, m, p, u_offset, v_offset, w_offset, rank_offset)
            r2 = build(i2, m, p, u_offset + i1*m, v_offset, w_offset + i1*p, rank_offset + r1)
            
            # Fix W mapping for split n
            # The recursive call for part 2 used w_offset + i1*p.
            # But W index is k*n + i.
            # In part 2, local i is 0..i2-1. Global i is i1..n-1.
            # Local W row: k*i2 + i_local.
            # Global W row: k*n + (i1 + i_local).
            # The build function for part 2 assumed the W rows were contiguous starting at w_offset + i1*p.
            # This is WRONG for W when splitting n.
            # We must handle the W mapping manually or pass a mapper.
            # Since we already wrote to W in the recursive calls, we need to verify if the mapping was correct.
            # In build(i2, ...), it writes to W[w_offset + i1*p + local_row].
            # local_row = k*i2 + i_local.
            # Target global row = k*n + i1 + i_local.
            # These are NOT the same unless i2 == n (which is false) or i1=0.
            # So the simple offset approach fails for W when splitting n.
            
            # We need to correct the W entries.
            # Let's retrieve the values written to the "wrong" locations and move them.
            # Actually, it's safer to not write to W in the recursive call if the mapping is non-linear, 
            # or to use a temporary buffer.
            
            # Given the complexity, let's just re-implement the split logic to be explicit about mapping.
            # I will discard the recursive write for W in the 'n' split case and do it manually.
            # But the recursive call already wrote to U and V correctly.
            # U: u_offset + i1*m is correct for U2.
            # V: v_offset is correct for V2.
            # W: The recursive call wrote to W starting at w_offset + i1*p.
            # We need to move those values to the correct positions.
            
            # Let's clear the W section for part 2 and rewrite it.
            # The section is rows [w_offset + i1*p, w_offset + i1*p + i2*p)
            # This is actually the correct range of rows, but the *indices* within that range are wrong?
            # No, the rows are just rows.
            # Row r in the block corresponds to local index r.
            # Local index r = k*i2 + i_local.
            # Global index should be k*n + i1 + i_local.
            # We need to map local row r to global row g.
            
            # Let's extract the data from the wrongly placed W block and put it in the right place.
            # This is messy. A better approach: 
            # Instead of passing offsets, pass the matrices themselves or use a helper that returns the local matrices.
            
            # Let's change strategy: build_decomp returns U, V, W for the sub-block.
            # Then the parent concatenates them with proper index mapping.
            
            return r1 + r2

        elif best_type == 'm':
            j1 = best_split
            j2 = m - j1
            r1 = build(n, j1, p, u_offset, v_offset, w_offset, rank_offset)
            r2 = build(n, j2, p, u_offset + n*j1, v_offset + j1*p, w_offset, rank_offset + r1)
            return r1 + r2

        elif best_type == 'p':
            k1 = best_split
            k2 = p - k1
            r1 = build(n, m, k1, u_offset, v_offset, w_offset, rank_offset)
            r2 = build(n, m, k2, u_offset, v_offset + m*k1, w_offset + k1*n, rank_offset + r1)
            return r1 + r2

        return 0

    # The above recursive approach with offsets is flawed for W when splitting n.
    # Let's implement a cleaner recursive builder that returns local matrices.
    
    def get_decomp(n, m, p):
        """Returns U, V, W for n x m x p"""
        if n == 1 and m == 1 and p == 1:
            return [[1]], [[1]], [[1]]
            
        if n == 2 and m == 2 and p == 2:
            U = [[0]*7 for _ in range(4)]
            V = [[0]*7 for _ in range(4)]
            W = [[0]*7 for _ in range(4)]
            
            # Strassen
            # M1
            U[0][0]=1; U[3][0]=1
            V[0][0]=1; V[3][0]=1
            W[0][0]=1; W[3][0]=1
            
            # M2
            U[2][1]=1; U[3][1]=1
            V[0][1]=1
            W[1][1]=1
            
            # M3
            U[0][2]=1
            V[1][2]=1; V[3][2]=-1
            W[2][2]=1
            
            # M4
            U[3][3]=1
            V[2][3]=1; V[0][3]=-1
            W[3][3]=1
            
            # M5
            U[0][4]=1; U[1][4]=1
            V[3][4]=1
            W[2][4]=1
            
            # M6
            U[2][5]=1; U[0][5]=-1
            V[0][5]=1; V[1][5]=1
            W[1][5]=1
            
            # M7
            U[1][6]=1; U[3][6]=-1
            V[2][6]=1; V[3][6]=1
            W[3][6]=1
            
            return U, V, W

        # Find best split
        best_rank = n * m * p
        best_split = None
        best_type = None
        
        for i in range(1, n):
            r = get_rank(i, m, p) + get_rank(n-i, m, p)
            if r < best_rank:
                best_rank = r
                best_split = i
                best_type = 'n'
                
        for j in range(1, m):
            r = get_rank(n, j, p) + get_rank(n, m-j, p)
            if r < best_rank:
                best_rank = r
                best_split = j
                best_type = 'm'
                
        for k in range(1, p):
            r = get_rank(n, m, k) + get_rank(n, m, p-k)
            if r < best_rank:
                best_rank = r
                best_split = k
                best_type = 'p'

        # If no split is better, use schoolbook
        if best_type is None or best_rank >= n*m*p:
            U = [[0]*(n*m*p) for _ in range(n*m)]
            V = [[0]*(n*m*p) for _ in range(m*p)]
            W = [[0]*(n*m*p) for _ in range(p*n)]
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i*m+j][col] = 1
                        V[j*p+k][col] = 1
                        W[k*n+i][col] = 1
                        col += 1
            return U, V, W

        # Recursive split
        if best_type == 'n':
            i1 = best_split
            i2 = n - i1
            U1, V1, W1 = get_decomp(i1, m, p)
            U2, V2, W2 = get_decomp(i2, m, p)
            
            r1 = len(U1[0])
            r2 = len(U2[0])
            R = r1 + r2
            
            U = [[0]*R for _ in range(n*m)]
            V = [[0]*R for _ in range(m*p)]
            W = [[0]*R for _ in range(p*n)]
            
            # Copy U1, V1, W1
            for c in range(r1):
                for row in range(i1*m):
                    U[row][c] = U1[row][c]
                for row in range(m*p):
                    V[row][c] = V1[row][c]
                for row in range(p*i1):
                    W[row][c] = W1[row][c]
                    
            # Copy U2, V2, W2
            # U2 rows: i2*m. Map to U rows i1*m .. n*m-1
            # V2 rows: m*p. Map to V rows 0 .. m*p-1 (same)
            # W2 rows: p*i2. Map to W rows p*i1 .. p*n-1? 
            # W index: k*n + i.
            # W2 local index: k*i2 + i_local.
            # Global index: k*n + (i1 + i_local).
            
            for c in range(r2):
                for row in range(i2*m):
                    U[i1*m + row][r1 + c] = U2[row][c]
                for row in range(m*p):
                    V[row][r1 + c] = V2[row][c]
                
                # W mapping
                for k in range(p):
                    for i_local in range(i2):
                        local_row = k * i2 + i_local
                        global_row = k * n + (i1 + i_local)
                        W[global_row][r1 + c] = W2[local_row][c]
                        
            return U, V, W

        elif best_type == 'm':
            j1 = best_split
            j2 = m - j1
            U1, V1, W1 = get_decomp(n, j1, p)
            U2, V2, W2 = get_decomp(n, j2, p)
            
            r1 = len(U1[0])
            r2 = len(U2[0])
            R = r1 + r2
            
            U = [[0]*R for _ in range(n*m)]
            V = [[0]*R for _ in range(m*p)]
            W = [[0]*R for _ in range(p*n)]
            
            for c in range(r1):
                for row in range(n*j1):
                    U[row][c] = U1[row][c]
                for row in range(j1*p):
                    V[row][c] = V1[row][c]
                for row in range(p*n):
                    W[row][c] = W1[row][c]
                    
            for c in range(r2):
                # U2 rows: n*j2. Map to U rows n*j1 .. n*m-1
                for row in range(n*j2):
                    U[n*j1 + row][r1 + c] = U2[row][c]
                # V2 rows: j2*p. Map to V rows j1*p .. m*p-1
                for row in range(j2*p):
                    V[j1*p + row][r1 + c] = V2[row][c]
                # W2 rows: p*n. Map to W rows 0 .. p*n-1
                for row in range(p*n):
                    W[row][r1 + c] = W2[row][c]
                    
            return U, V, W

        elif best_type == 'p':
            k1 = best_split
            k2 = p - k1
            U1, V1, W1 = get_decomp(n, m, k1)
            U2, V2, W2 = get_decomp(n, m, k2)
            
            r1 = len(U1[0])
            r2 = len(U2[0])
            R = r1 + r2
            
            U = [[0]*R for _ in range(n*m)]
            V = [[0]*R for _ in range(m*p)]
            W = [[0]*R for _ in range(p*n)]
            
            for c in range(r1):
                for row in range(n*m):
                    U[row][c] = U1[row][c]
                for row in range(m*k1):
                    V[row][c] = V1[row][c]
                for row in range(k1*n):
                    W[row][c] = W1[row][c]
                    
            for c in range(r2):
                # U2 rows: n*m. Map to U rows 0 .. n*m-1
                for row in range(n*m):
                    U[row][r1 + c] = U2[row][c]
                # V2 rows: m*k2. Map to V rows m*k1 .. m*p-1
                for row in range(m*k2):
                    V[m*k1 + row][r1 + c] = V2[row][c]
                # W2 rows: k2*n. Map to W rows k1*n .. p*n-1
                for row in range(k2*n):
                    W[k1*n + row][r1 + c] = W2[row][c]
                    
            return U, V, W

    U, V, W = get_decomp(n, m, p)
    return {"U": U, "V": V, "W": W}
