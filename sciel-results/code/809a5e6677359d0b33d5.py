import time
import numpy as np

def _verify(U, V, W, n=3, m=4, p=4):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Build the tensor from U, V, W
    # T[i, j, k] = sum_r U[i, r] * V[j, r] * W[k, r]
    # i in [0, n*m), j in [0, m*p), k in [0, p*n)
    # i = row_a * m + col_a, j = row_b * p + col_b, k = col_c * n + row_c
    # T should equal 1 if col_a == row_b and row_a == row_c, else 0
    
    # Efficient check using einsum
    # T[i,j,k] = sum_r U[i,r] V[j,r] W[k,r]
    T = np.einsum('ir,jr,kr->ijk', U, V, W)
    
    # Build target tensor
    target = np.zeros((n*m, m*p, p*n))
    for row_a in range(n):
        for col_a in range(m):
            for col_c in range(p):
                if col_a == col_c:  # This is wrong, need to check properly
                    pass
    # Actually: T[i,j,k] = 1 iff (i%m == j//p) and (i//m == k%n)
    # i = row_a * m + col_a, j = row_b * p + col_b, k = col_c * n + row_c
    # Condition: col_a == row_b AND row_a == row_c
    for row_a in range(n):
        for row_b in range(m):
            for col_c in range(p):
                i = row_a * m + row_b
                j = row_b * p + col_c
                k = col_c * n + row_a
                target[i, j, k] = 1.0
    
    max_err = np.max(np.abs(T - target))
    return max_err < 1e-10

def _try_numerical_search(deadline):
    """Try to find a lower-rank decomposition using numerical optimization."""
    n, m, p = 3, 4, 4
    
    # Start from rank 43 and try to reduce
    # Use the structure: 4 Strassen blocks (28) + 16 trivial (16) = 44
    # Try to merge or find cancellations
    
    # Approach: Start with random rank-r initialization and minimize
    # ||U V^T W^T - target_tensor||^2
    # But this is non-convex. Let's try a different approach.
    
    # Actually, let's try to see if we can do 43 by finding a linear dependency
    # among the 44 terms that allows elimination.
    
    # Build the 44-term decomposition as a tensor T_44
    U44 = np.zeros((12, 44))
    V44 = np.zeros((16, 44))
    W44 = np.zeros((12, 44))
    
    col = 0
    def add_strassen(a_r, a_c, b_r, b_c, c_r, c_c):
        global col
        # m1 = (a00+a11)(b00+b11)
        U44[a_r*m+a_c, col] += 1
        U44[(a_r+1)*m+(a_c+1), col] += 1
        V44[b_r*p+b_c, col] += 1
        V44[(b_r+1)*p+(b_c+1), col] += 1
        W44[c_c*n+c_r, col] += 1
        W44[(c_c+1)*n+(c_r+1), col] += 1
        col += 1
        # m2 = (a10+a11)b00
        U44[(a_r+1)*m+a_c, col] += 1
        U44[(a_r+1)*m+(a_c+1), col] += 1
        V44[b_r*p+b_c, col] += 1
        W44[c_c*n+(a_r+1), col] += 1
        W44[(c_c+1)*n+(a_r+1), col] -= 1
        col += 1
        # m3 = a00(b01-b11)
        U44[a_r*m+a_c, col] += 1
        V44[b_r*p+(b_c+1), col] += 1
        V44[(b_r+1)*p+(b_c+1), col] -= 1
        W44[(c_c+1)*n+a_r, col] += 1
        W44[(c_c+1)*n+(a_r+1), col] += 1
        col += 1
        # m4 = a11(b10-b00)
        U44[(a_r+1)*m+(a_c+1), col] += 1
        V44[(b_r+1)*p+b_c, col] += 1
        V44[b_r*p+b_c, col] -= 1
        W44[c_c*n+a_r, col] += 1
        W44[c_c*n+(a_r+1), col] += 1
        col += 1
        # m5 = (a00+a01)b11
        U44[a_r*m+a_c, col] += 1
        U44[a_r*m+(a_c+1), col] += 1
        V44[(b_r+1)*p+(b_c+1), col] += 1
        W44[c_c*n+a_r, col] -= 1
        W44[(c_c+1)*n+a_r, col] += 1
        col += 1
        # m6 = (a10-a00)(b00+b01)
        U44[(a_r+1)*m+a_c, col] += 1
        U44[a_r*m+a_c, col] -= 1
        V44[b_r*p+b_c, col] += 1
        V44[b_r*p+(b_c+1), col] += 1
        W44[(c_c+1)*n+(a_r+1), col] += 1
        col += 1
        # m7 = (a01-a11)(b10+b11)
        U44[a_r*m+(a_c+1), col] += 1
        U44[(a_r+1)*m+(a_c+1), col] -= 1
        V44[(b_r+1)*p+b_c, col] += 1
        V44[(b_r+1)*p+(b_c+1), col] += 1
        W44[c_c*n+a_r, col] += 1
        col += 1
    
    def add_trivial_1x4x2(a_r, b_c):
        global col
        for j in range(4):
            U44[a_r*m+j, col] += 1
            V44[j*p+b_c, col] += 1
            W44[b_c*n+a_r, col] += 1
            col += 1
            U44[a_r*m+j, col] += 1
            V44[j*p+(b_c+1), col] += 1
            W44[(b_c+1)*n+a_r, col] += 1
            col += 1
    
    add_strassen(0, 0, 0, 0, 0, 0)
    add_strassen(0, 2, 2, 0, 0, 0)
    add_trivial_1x4x2(2, 0)
    add_strassen(0, 0, 0, 2, 0, 2)
    add_strassen(0, 2, 2, 2, 0, 2)
    add_trivial_1x4x2(2, 2)
    
    # Check if any column is a linear combination of others
    # For rank reduction, we need to find if the 44 rank-1 tensors are linearly dependent
    # in a way that preserves the sum
    # This is hard to check directly. Let's try a different approach.
    
    # Try random perturbations and see if we can project to lower rank
    # while maintaining correctness
    
    # Alternative: Try to find a rank-43 decomposition by solving
    # sum_{r=0}^{42} u_r v_r w_r = target_tensor
    # This is a system of polynomial equations.
    
    # Let's try a simpler approach: check if the 44 terms have any redundancy
    # by checking if removing one term and adjusting others still works
    
    # For each term r, try to express it as a combination of others
    # This requires solving a linear system for each r
    
    for target_rank in [43, 42]:
        if time.time() > deadline:
            break
        # Try to find a rank-target_rank decomposition
        # by starting from rank 44 and trying to merge terms
        
        # Simple heuristic: try removing one term at a time and see if
        # the remaining terms can be adjusted
        for drop_idx in range(44):
            if time.time() > deadline:
                break
            # Create a new decomposition without term drop_idx
            U_new = np.delete(U44, drop_idx, axis=1)
            V_new = np.delete(V44, drop_idx, axis=1)
            W_new = np.delete(W44, drop_idx, axis=1)
            
            # Check if the sum of remaining terms still equals target
            T_new = np.einsum('ir,jr,kr->ijk', U_new, V_new, W_new)
            target = np.zeros((12, 16, 12))
            for row_a in range(3):
                for row_b in range(4):
                    for col_c in range(4):
                        i = row_a * 4 + row_b
                        j = row_b * 4 + col_c
                        k = col_c * 3 + row_a
                        target[i, j, k] = 1.0
            
            if np.max(np.abs(T_new - target)) < 1e-10:
                # Found a rank 43 decomposition!
                return U_new, V_new, W_new
    
    return None

def construct():
    n, m, p = 3, 4, 4
    start_time = time.time()
    deadline = start_time + 120  # Safety margin
    
    # Build the rank-44 decomposition
    rank = 44
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    
    def add_strassen(a_r, a_c, b_r, b_c, c_r, c_c, start_col):
        c = start_col
        # m1
        U[a_r*m + a_c][c] += 1
        U[(a_r+1)*m + (a_c+1)][c] += 1
        V[b_r*p + b_c][c] += 1
        V[(b_r+1)*p + (b_c+1)][c] += 1
        W[c_c*n + c_r][c] += 1
        W[(c_c+1)*n + (c_r+1)][c] += 1
        c += 1
        # m2
        U[(a_r+1)*m + a_c][c] += 1
        U[(a_r+1)*m + (a_c+1)][c] += 1
        V[b_r*p + b_c][c] += 1
        W[c_c*n + (a_r+1)][c] += 1
        W[(c_c+1)*n + (a_r+1)][c] -= 1
        c += 1
        # m3
        U[a_r*m + a_c][c] += 1
        V[b_r*p + (b_c+1)][c] += 1
        V[(b_r+1)*p + (b_c+1)][c] -= 1
        W[(c_c+1)*n + a_r][c] += 1
        W[(c_c+1)*n + (a_r+1)][c] += 1
        c += 1
        # m4
        U[(a_r+1)*m + (a_c+1)][c] += 1
        V[(b_r+1)*p + b_c][c] += 1
        V[b_r*p + b_c][c] -= 1
        W[c_c*n + a_r][c] += 1
        W[c_c*n + (a_r+1)][c] += 1
        c += 1
        # m5
        U[a_r*m + a_c][c] += 1
        U[a_r*m + (a_c+1)][c] += 1
        V[(b_r+1)*p + (b_c+1)][c] += 1
        W[c_c*n + a_r][c] -= 1
        W[(c_c+1)*n + a_r][c] += 1
        c += 1
        # m6
        U[(a_r+1)*m + a_c][c] += 1
        U[a_r*m + a_c][c] -= 1
        V[b_r*p + b_c][c] += 1
        V[b_r*p + (b_c+1)][c] += 1
        W[(c_c+1)*n + (a_r+1)][c] += 1
        c += 1
        # m7
        U[a_r*m + (a_c+1)][c] += 1
        U[(a_r+1)*m + (a_c+1)][c] -= 1
        V[(b_r+1)*p + b_c][c] += 1
        V[(b_r+1)*p + (b_c+1)][c] += 1
        W[c_c*n + a_r][c] += 1
        c += 1
        return c
    
    def add_trivial_1x4x2(a_r, b_c, start_col):
        c = start_col
        for j in range(4):
            U[a_r*m + j][c] += 1
            V[j*p + b_c][c] += 1
            W[b_c*n + a_r][c] += 1
            c += 1
            U[a_r*m + j][c] += 1
            V[j*p + (b_c+1)][c] += 1
            W[(b_c+1)*n + a_r][c] += 1
            c += 1
        return c
    
    # Build the 44-rank decomposition
    col = add_strassen(0, 0, 0, 0, 0, 0, col)
    col = add_strassen(0, 2, 2, 0, 0, 0, col)
    col = add_trivial_1x4x2(2, 0, col)
    col = add_strassen(0, 0, 0, 2, 0, 2, col)
    col = add_strassen(0, 2, 2, 2, 0, 2, col)
    col = add_trivial_1x4x2(2, 2, col)
    
    # Verify correctness
    if not _verify(U, V, W):
        # Fallback to trivial rank-48
        rank = 48
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Try to find a lower-rank decomposition if time permits
    if time.time() < deadline - 10:
        result = _try_numerical_search(deadline - 5)
        if result is not None:
            U_new, V_new, W_new = result
            if _verify(U_new, V_new, W_new):
                return {"U": U_new.tolist(), "V": V_new.tolist(), "W": W_new.tolist()}
    
    return {"U": U, "V": V, "W": W}
