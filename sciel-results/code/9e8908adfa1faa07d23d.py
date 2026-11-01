import time

DIMENSIONS = [2, 4, 5]

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    rank = 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def u_row(i, j): return i * 4 + j
    def v_row(j, k): return j * 5 + k
    def w_row(k, i): return k * 2 + i
    
    def add_strassen(ai0, aj0, bj0, bk0, ci0, ck0, start_col):
        """Add Strassen rank-7 for 2x2x2 block."""
        A00 = (ai0, aj0)
        A01 = (ai0, aj0+1)
        A10 = (ai0+1, aj0)
        A11 = (ai0+1, aj0+1)
        B00 = (bj0, bk0)
        B01 = (bj0, bk0+1)
        B10 = (bj0+1, bk0)
        B11 = (bj0+1, bk0+1)
        C00 = (ci0, ck0)
        C01 = (ci0, ck0+1)
        C10 = (ci0+1, ck0)
        C11 = (ci0+1, ck0+1)
        
        # M1 = (A00+A11)*(B00+B11)
        c = start_col
        U[u_row(*A00)][c] += 1
        U[u_row(*A11)][c] += 1
        V[v_row(*B00)][c] += 1
        V[v_row(*B11)][c] += 1
        W[w_row(*C00)][c] += 1
        W[w_row(*C11)][c] += 1
        
        # M2 = (A10+A11)*B00
        c = start_col + 1
        U[u_row(*A10)][c] += 1
        U[u_row(*A11)][c] += 1
        V[v_row(*B00)][c] += 1
        W[w_row(*C10)][c] += 1
        W[w_row(*C11)][c] -= 1
        
        # M3 = A00*(B01-B11)
        c = start_col + 2
        U[u_row(*A00)][c] += 1
        V[v_row(*B01)][c] += 1
        V[v_row(*B11)][c] -= 1
        W[w_row(*C01)][c] += 1
        W[w_row(*C11)][c] += 1
        
        # M4 = A11*(B10-B00)
        c = start_col + 3
        U[u_row(*A11)][c] += 1
        V[v_row(*B10)][c] += 1
        V[v_row(*B00)][c] -= 1
        W[w_row(*C00)][c] += 1
        W[w_row(*C10)][c] += 1
        
        # M5 = (A00+A01)*B11
        c = start_col + 4
        U[u_row(*A00)][c] += 1
        U[u_row(*A01)][c] += 1
        V[v_row(*B11)][c] += 1
        W[w_row(*C00)][c] -= 1
        W[w_row(*C01)][c] += 1
        
        # M6 = (A10-A00)*(B00+B01)
        c = start_col + 5
        U[u_row(*A10)][c] += 1
        U[u_row(*A00)][c] -= 1
        V[v_row(*B00)][c] += 1
        V[v_row(*B01)][c] += 1
        W[w_row(*C11)][c] += 1
        
        # M7 = (A01-A11)*(B10+B11)
        c = start_col + 6
        U[u_row(*A01)][c] += 1
        U[u_row(*A11)][c] -= 1
        V[v_row(*B10)][c] += 1
        V[v_row(*B11)][c] += 1
        W[w_row(*C00)][c] += 1
        
        return start_col + 7
    
    def add_2x2x1(ai0, aj0, bj0, bk, ci0, ck, start_col):
        """Add schoolbook rank-4 for 2x2x1 block."""
        A00 = (ai0, aj0)
        A01 = (ai0, aj0+1)
        A10 = (ai0+1, aj0)
        A11 = (ai0+1, aj0+1)
        B0 = (bj0, bk)
        B1 = (bj0+1, bk)
        C0 = (ci0, ck)
        C1 = (ci0+1, ck)
        
        c = start_col
        U[u_row(*A00)][c] += 1
        V[v_row(*B0)][c] += 1
        W[w_row(*C0)][c] += 1
        
        c = start_col + 1
        U[u_row(*A01)][c] += 1
        V[v_row(*B1)][c] += 1
        W[w_row(*C0)][c] += 1
        
        c = start_col + 2
        U[u_row(*A10)][c] += 1
        V[v_row(*B0)][c] += 1
        W[w_row(*C1)][c] += 1
        
        c = start_col + 3
        U[u_row(*A11)][c] += 1
        V[v_row(*B1)][c] += 1
        W[w_row(*C1)][c] += 1
        
        return start_col + 4
    
    # Part 1: A[:, 0:2] * B[0:2, :] -> C[:, :]
    col = add_strassen(0, 0, 0, 0, 0, 0, col)   # 7 cols
    col = add_strassen(0, 0, 0, 2, 0, 2, col)   # 7 cols
    col = add_2x2x1(0, 0, 0, 4, 0, 4, col)      # 4 cols
    
    # Part 2: A[:, 2:4] * B[2:4, :] -> C[:, :]
    col = add_strassen(0, 2, 2, 0, 0, 0, col)   # 7 cols
    col = add_strassen(0, 2, 2, 2, 0, 2, col)   # 7 cols
    col = add_2x2x1(0, 2, 2, 4, 0, 4, col)      # 4 cols
    
    assert col == rank, f"Expected {rank}, got {col}"
    
    if time.time() - start_time > 140:
        pass
    
    return {"U": U, "V": V, "W": W}
