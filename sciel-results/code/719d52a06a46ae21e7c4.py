import numpy as np

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    
    def get_rank(n, m, p):
        if n == 1 and m == 1 and p == 1:
            return 1
        if n == 2 and m == 2 and p == 2:
            return 7
        
        best = n * m * p
        for i in range(1, n):
            r = get_rank(i, m, p) + get_rank(n-i, m, p)
            if r < best:
                best = r
        for j in range(1, m):
            r = get_rank(n, j, p) + get_rank(n, m-j, p)
            if r < best:
                best = r
        for k in range(1, p):
            r = get_rank(n, m, k) + get_rank(n, m, p-k)
            if r < best:
                best = r
        return best

    def build(n, m, p, U, V, W, offset_u, offset_v, offset_w):
        rank = get_rank(n, m, p)
        
        if n == 1 and m == 1 and p == 1:
            U[offset_u][offset_w] = 1
            V[offset_v][offset_w] = 1
            W[offset_w][offset_u] = 1 # Note: W index is k*n+i. Here k=0, i=0. So W[0][0]=1.
            # Wait, W is (p*n, rank). Row index is k*n+i.
            # U is (n*m, rank). Row index is i*m+j.
            # V is (m*p, rank). Row index is j*p+k.
            # In the 1x1x1 case: i=0, j=0, k=0.
            # U[0][c]=1, V[0][c]=1, W[0][c]=1.
            # My code above: U[offset_u][offset_w] is wrong. offset_w is the column index.
            # Let's fix the indexing in the helper.
            # U[row, col], V[row, col], W[row, col].
            # The recursive call should pass the column offset.
            # Let's restructure build to return the rank used and handle offsets internally or via parameters.
            
            # Actually, it's safer to pass the base indices for U, V, W rows and the column offset.
            # U row base: offset_u_row
            # V row base: offset_v_row
            # W row base: offset_w_row
            # Col offset: offset_c
            
            return 1

    # The previous attempt at recursion was messy. Let's do a clean recursive builder.
    # We need to be careful with the W mapping.
    # W row index = k * n + i.
    # If we split n, i is split into i1, i2.
    # W1 rows are k * n1 + i1.
    # W2 rows are k * n2 + i2.
    # We need to map these to the global W rows k * n + i.
    
    def construct_rec(n, m, p, U, V, W, u_row_off, v_row_off, w_row_off, c_off):
        """
        Fills U, V, W for a subproblem of size n x m x p.
        U rows are u_row_off .. u_row_off + n*m - 1.
        V rows are v_row_off .. v_row_off + m*p - 1.
        W rows are w_row_off .. w_row_off + p*n - 1.
        Columns are c_off .. c_off + rank - 1.
        Returns the rank used.
        """
        
        # Base case 1: 1x1x1
        if n == 1 and m == 1 and p == 1:
            U[u_row_off][c_off] = 1
            V[v_row_off][c_off] = 1
            W[w_row_off][c_off] = 1
            return 1

        # Base case 2: 2x2x2 (Strassen)
        if n == 2 and m == 2 and p == 2:
            # Strassen decomposition for 2x2x2
            # U is 4x7, V is 4x7, W is 4x7
            # Indices:
            # U: i*m+j -> 0:00, 1:01, 2:10, 3:11
            # V: j*p+k -> 0:00, 1:01, 2:10, 3:11
            # W: k*n+i -> 0:00, 1:10, 2:01, 3:11  (k is outer, i is inner)
            # Wait, W index is k*n+i.
            # k=0, i=0 -> 0 (C00)
            # k=0, i=1 -> 1 (C10)
            # k=1, i=0 -> 2 (C01)
            # k=1, i=1 -> 3 (C11)
            
            # Strassen formulas:
            # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
            # M2 = (A10+A11)B00 -> C10 += M2
            # M3 = A00(B01-B11) -> C01 += M3
            # M4 = A11(B10-B00) -> C11 += M4
            # M5 = (A00+A01)B11 -> C01 += M5
            # M6 = (A10-A00)(B00+B01) -> C10 += M6
            # M7 = (A01-A11)(B10+B11) -> C11 += M7
            
            # Columns c_off to c_off+6
            
            # M1
            U[u_row_off + 0][c_off + 0] = 1 # A00
            U[u_row_off + 3][c_off + 0] = 1 # A11
            V[v_row_off + 0][c_off + 0] = 1 # B00
            V[v_row_off + 3][c_off + 0] = 1 # B11
            W[w_row_off + 0][c_off + 0] = 1 # C00
            W[w_row_off + 3][c_off + 0] = 1 # C11
            
            # M2
            U[u_row_off + 2][c_off + 1] = 1 # A10
            U[u_row_off + 3][c_off + 1] = 1 # A11
            V[v_row_off + 0][c_off + 1] = 1 # B00
            W[w_row_off + 1][c_off + 1] = 1 # C10
            
            # M3
            U[u_row_off + 0][c_off + 2] = 1 # A00
            V[v_row_off + 1][c_off + 2] = 1 # B01
            V[v_row_off + 3][c_off + 2] = -1 # B11
            W[w_row_off + 2][c_off + 2] = 1 # C01
            
            # M4
            U[u_row_off + 3][c_off + 3] = 1 # A11
            V[v_row_off + 2][c_off + 3] = 1 # B10
            V[v_row_off + 0][c_off + 3] = -1 # B00
            W[w_row_off + 3][c_off + 3] = 1 # C11
            
            # M5
            U[u_row_off + 0][c_off + 4] = 1 # A00
            U[u_row_off + 1][c_off + 4] = 1 # A01
            V[v_row_off + 3][c_off + 4] = 1 # B11
            W[w_row_off + 2][c_off + 4] = 1 # C01
            
            # M6
            U[u_row_off + 2][c_off + 5] = 1 # A10
            U[u_row_off + 0][c_off + 5] = -1 # A00
            V[v_row_off + 0][c_off + 5] = 1 # B00
            V[v_row_off + 1][c_off + 5] = 1 # B01
            W[w_row_off + 1][c_off + 5] = 1 # C10
            
            # M7
            U[u_row_off + 1][c_off + 6] = 1 # A01
            U[u_row_off + 3][c_off + 6] = -1 # A11
            V[v_row_off + 2][c_off + 6] = 1 # B10
            V[v_row_off + 3][c_off + 6] = 1 # B11
            W[w_row_off + 3][c_off + 6] = 1 # C11
            
            return 7

        # Recursive case
        # Find best split
        best_rank = n * m * p
        best_split = None
        best_type = None
        
        for i in range(1, n):
            r1 = get_rank(i, m, p)
            r2 = get_rank(n-i, m, p)
            r = r1 + r2
            if r < best_rank:
                best_rank = r
                best_split = i
                best_type = 'n'
                
        for j in range(1, m):
            r1 = get_rank(n, j, p)
            r2 = get_rank(n, m-j, p)
            r = r1 + r2
            if r < best_rank:
                best_rank = r
                best_split = j
                best_type = 'm'
                
        for k in range(1, p):
            r1 = get_rank(n, m, k)
            r2 = get_rank(n, m, p-k)
            r = r1 + r2
            if r < best_rank:
                best_rank = r
                best_split = k
                best_type = 'p'
                
        # If no split is better than schoolbook, just do schoolbook
        if best_type is None or best_rank >= n*m*p:
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_idx = u_row_off + i*m + j
                        v_idx = v_row_off + j*p + k
                        w_idx = w_row_off + k*n + i
                        U[u_idx][c_off] = 1
                        V[v_idx][c_off] = 1
                        W[w_idx][c_off] = 1
                        c_off += 1
            return n*m*p

        # Apply split
        if best_type == 'n':
            i1 = best_split
            i2 = n - i1
            
            # Part 1: i1 x m x p
            # U rows: u_row_off .. u_row_off + i1*m - 1
            # V rows: v_row_off .. v_row_off + m*p - 1 (same V)
            # W rows: w_row_off .. w_row_off + p*i1 - 1 (Wait, W row is k*n+i. For subproblem n=i1, W row is k*i1+i'.
            # We need to map subproblem W rows to global W rows.
            # Global W row for (k, i') is k*n + i'.
            # Subproblem W row for (k, i') is k*i1 + i'.
            # This mapping is not a simple offset if we just copy the matrix.
            # However, we are filling U, V, W directly.
            # We need to call construct_rec for the subproblem, but it fills a contiguous block of W.
            # The subproblem W is size p*i1. The global W is size p*n.
            # The rows in global W corresponding to the first part (i < i1) are NOT contiguous in the global W array if n > i1.
            # Global W indices:
            # k=0: 0, 1, ..., n-1
            # k=1: n, n+1, ..., 2n-1
            # ...
            # The rows for i < i1 are 0..i1-1, n..n+i1-1, etc.
            # This is a strided access.
            
            # My construct_rec assumes contiguous W rows.
            # This is a problem for splitting n.
            # Splitting m or p keeps U or V contiguous?
            # Splitting m: U is n*m. Rows for first m1 are 0..n*m1-1. Contiguous.
            #             V is m*p. Rows for first m1 are 0..m1*p-1. Contiguous.
            #             W is p*n. Rows are k*n+i. No change in n. Contiguous block of columns? No, W is same size.
            # Splitting p: U is n*m. No change.
            #             V is m*p. Rows for first p1 are 0..m*p1-1. Contiguous.
            #             W is p*n. Rows for first p1 are 0..p1*n-1. Contiguous.
            
            # So splitting n is problematic for the "contiguous W" assumption.
            # We can either:
            # 1. Avoid splitting n.
            # 2. Modify construct_rec to handle strided W or return a local W and map it.
            
            # Let's check if we can avoid splitting n.
            # For 3x4x6:
            # Split m=4 into 2+2.
            # Subproblem 1: 3x2x6.
            # Subproblem 2: 3x2x6.
            # For 3x2x6:
            # Split p=6 into 2+2+2.
            # Subsub 1: 3x2x2.
            # Subsub 2: 3x2x2.
            # Subsub 3: 3x2x2.
            # For 3x2x2:
            # Split n=3 into 1+2.
            # Subsubsub 1: 1x2x2. Rank 4.
            # Subsubsub 2: 2x2x2. Rank 7.
            # Total for 3x2x2: 11.
            # Total for 3x2x6: 33.
            # Total for 3x4x6: 66.
            
            # This path requires splitting n.
            # So I must handle the W mapping for n-splits.
            
            # Solution: construct_rec returns the rank used.
            # For n-split, we cannot just pass a contiguous W slice.
            # We will build a temporary W for the subproblem and then copy it to the correct global locations.
            
            # Actually, it's easier to just implement the specific splits for this case or a generic one that handles the mapping.
            # Let's modify construct_rec to accept a list of W row indices to write to, or just handle the mapping in the caller.
            
            # Revised Plan:
            # construct_rec(n, m, p, U, V, W, u_off, v_off, w_row_indices, c_off)
            # w_row_indices is a list of length p*n specifying which global W rows to use.
            # For a subproblem of size n', p', the w_row_indices should be length p'*n'.
            
            # This is getting complex. Let's stick to the specific decomposition for 3x4x6 that we derived.
            # Rank 66.
            
            pass # Placeholder

    # Given the complexity, I will hardcode the construction for the 3x4x6 case using the derived splits.
    # 1. Split m=4 into 2+2.
    #    Part A: 3x2x6. U rows 0..5, V rows 0..11, W rows 0..17.
    #    Part B: 3x2x6. U rows 6..11, V rows 12..23, W rows 0..17.
    #    Wait, V rows for Part B are j=2,3. j*p+k = 2*6+k = 12..23.
    #    W rows are the same for both parts because n and p are same.
    #    But the columns (rank) are different.
    
    # Let's use a helper that builds into a local structure and then merges.
    # Or just build the 3x2x6 decomposition and call it twice.
    
    def build_3x2x6(U, V, W, u_off, v_off, w_off, c_off):
        # Splits p=6 into 2+2+2.
        # Sub 1: 3x2x2. p=0,1.
        # Sub 2: 3x2x2. p=2,3.
        # Sub 3: 3x2x2. p=4,5.
        
        def build_3x2x2(U, V, W, u_off, v_off, w_off, c_off, p_base):
            # Splits n=3 into 1+2.
            # Sub 1: 1x2x2. n=0.
            # Sub 2: 2x2x2. n=1,2.
            
            # 1x2x2
            # U rows: 0..1 (i=0, j=0,1)
            # V rows: 0..3 (j=0,1; k=0,1)
            # W rows: 0..1 (k=0,1; i=0) -> Global W rows: k*3+0 = 0, 3.
            
            # 2x2x2
            # U rows: 2..5 (i=1,2; j=0,1)
            # V rows: 0..3
            # W rows: 2..5 (k=0,1; i=1,2) -> Global W rows: k*3+1, k*3+2 = 1, 2, 4, 5.
            
            # This mapping is annoying.
            # Let's just use the schoolbook for 3x2x2? Rank 12.
            # 3x2x6 rank 36. 3x4x6 rank 72. No gain.
            
            # We MUST use Strassen for 2x2x2 to get the gain.
            # So we need to handle the W mapping for n-split.
            
            # Let's write a generic recursive function that handles the W mapping correctly.
            
            pass

    # I will implement a clean recursive function that returns the rank and handles the W mapping by passing a list of global row indices.
    
    def rec_build(n, m, p, U, V, W, u_off, v_off, w_indices, c_off):
        """
        w_indices: list of length p*n. w_indices[k*n+i] is the global row index in W for this element.
        """
        
        if n == 1 and m == 1 and p == 1:
            U[u_off][c_off] = 1
            V[v_off][c_off] = 1
            W[w_indices[0]][c_off] = 1
            return 1

        if n == 2 and m == 2 and p == 2:
            # Strassen
            # U rows: u_off .. u_off+3
            # V rows: v_off .. v_off+3
            # W rows: w_indices[0..3]
            
            # M1
            U[u_off+0][c_off+0] = 1
            U[u_off+3][c_off+0] = 1
            V[v_off+0][c_off+0] = 1
            V[v_off+3][c_off+0] = 1
            W[w_indices[0]][c_off+0] = 1 # k=0,i=0
            W[w_indices[3]][c_off+0] = 1 # k=1,i=1
            
            # M2
            U[u_off+2][c_off+1] = 1
            U[u_off+3][c_off+1] = 1
            V[v_off+0][c_off+1] = 1
            W[w_indices[1]][c_off+1] = 1 # k=0,i=1
            
            # M3
            U[u_off+0][c_off+2] = 1
            V[v_off+1][c_off+2] = 1
            V[v_off+3][c_off+2] = -1
            W[w_indices[2]][c_off+2] = 1 # k=1,i=0
            
            # M4
            U[u_off+3][c_off+3] = 1
            V[v_off+2][c_off+3] = 1
            V[v_off+0][c_off+3] = -1
            W[w_indices[3]][c_off+3] = 1 # k=1,i=1
            
            # M5
            U[u_off+0][c_off+4] = 1
            U[u_off+1][c_off+4] = 1
            V[v_off+3][c_off+4] = 1
            W[w_indices[2]][c_off+4] = 1 # k=1,i=0
            
            # M6
            U[u_off+2][c_off+5] = 1
            U[u_off+0][c_off+5] = -1
            V[v_off+0][c_off+5] = 1
            V[v_off+1][c_off+5] = 1
            W[w_indices[1]][c_off+5] = 1 # k=0,i=1
            
            # M7
            U[u_off+1][c_off+6] = 1
            U[u_off+3][c_off+6] = -1
            V[v_off+2][c_off+6] = 1
            V[v_off+3][c_off+6] = 1
            W[w_indices[3]][c_off+6] = 1 # k=1,i=1
            
            return 7

        # Find best split
        best_rank = n*m*p
        best_split = None
        best_type = None
        
        for i in range(1, n):
            r1 = get_rank(i, m, p)
            r2 = get_rank(n-i, m, p)
            r = r1 + r2
            if r < best_rank:
                best_rank = r
                best_split = i
                best_type = 'n'
        for j in range(1, m):
            r1 = get_rank(n, j, p)
            r2 = get_rank(n, m-j, p)
            r = r1 + r2
            if r < best_rank:
                best_rank = r
                best_split = j
                best_type = 'm'
        for k in range(1, p):
            r1 = get_rank(n, m, k)
            r2 = get_rank(n, m, p-k)
            r = r1 + r2
            if r < best_rank:
                best_rank = r
                best_split = k
                best_type = 'p'

        if best_type is None:
            # Schoolbook
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[u_off + i*m + j][c_off] = 1
                        V[v_off + j*p + k][c_off] = 1
                        W[w_indices[k*n+i]][c_off] = 1
                        c_off += 1
            return n*m*p

        # Apply split
        if best_type == 'n':
            i1 = best_split
            i2 = n - i1
            # Part 1: i1 x m x p
            w_indices_1 = []
            for k in range(p):
                for i in range(i1):
                    w_indices_1.append(w_indices[k*n+i])
            
            r1 = rec_build(i1, m, p, U, V, W, u_off, v_off, w_indices_1, c_off)
            
            # Part 2: i2 x m x p
            w_indices_2 = []
            for k in range(p):
                for i in range(i2):
                    w_indices_2.append(w_indices[k*n + i1 + i])
            
            # U offset for part 2: u_off + i1*m
            # V offset: same
            # W indices: w_indices_2
            r2 = rec_build(i2, m, p, U, V, W, u_off + i1*m, v_off, w_indices_2, c_off + r1)
            
            return r1 + r2

        elif best_type == 'm':
            j1 = best_split
            j2 = m - j1
            # Part 1: n x j1 x p
            w_indices_1 = w_indices[:] # W is same size
            
            r1 = rec_build(n, j1, p, U, V, W, u_off, v_off, w_indices_1, c_off)
            
            # Part 2: n x j2 x p
            # U offset: u_off + n*j1
            # V offset: v_off + j1*p
            # W indices: same
            r2 = rec_build(n, j2, p, U, V, W, u_off + n*j1, v_off + j1*p, w_indices, c_off + r1)
            
            return r1 + r2

        elif best_type == 'p':
            k1 = best_split
            k2 = p - k1
            # Part 1: n x m x k1
            w_indices_1 = []
            for k in range(k1):
                for i in range(n):
                    w_indices_1.append(w_indices[k*n+i])
            
            r1 = rec_build(n, m, k1, U, V, W, u_off, v_off, w_indices_1, c_off)
            
            # Part 2: n x m x k2
            w_indices_2 = []
            for k in range(k2):
                for i in range(n):
                    w_indices_2.append(w_indices[(k1+k)*n+i])
            
            # U offset: same
            # V offset: v_off + m*k1
            r2 = rec_build(n, m, k2, U, V, W, u_off, v_off + m*k1, w_indices_2, c_off + r1)
            
            return r1 + r2

    total_rank = get_rank(n, m, p)
    U = [[0]*total_rank for _ in range(n*m)]
    V = [[0]*total_rank for _ in range(m*p)]
    W = [[0]*total_rank for _ in range(p*n)]
    
    # Initial W indices: identity mapping
    w_indices = list(range(p*n))
    
    used_rank = rec_build(n, m, p, U, V, W, 0, 0, w_indices, 0)
    
    # Trim if necessary (should not be necessary if get_rank is accurate)
    if used_rank < total_rank:
        U = [row[:used_rank] for row in U]
        V = [row[:used_rank] for row in V]
        W = [row[:used_rank] for row in W]
        
    return {"U": U, "V": V, "W": W}
