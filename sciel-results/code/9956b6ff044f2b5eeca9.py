"""Matrix multiplication 3x4x6 tensor decomposition - rank minimization."""
import time
import random

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    import numpy as np
    
    n, m, p = 3, 4, 6
    deadline = time.time() + 135  # Safety margin before 150s kill
    
    def Aidx(i, j): return i * m + j
    def Bidx(j, k): return j * p + k
    def Cidx(k, i): return k * n + i  # W[k*n+i] trace convention
    
    def verify(U, V, W, r):
        """Check exact decomposition: sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] == 1 for all (i,j,k)."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U[Aidx(i,j), :] * V[Bidx(j,k), :] * W[Cidx(k,i), :])
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    def baseline_66():
        """Block decomposition: rank 66 baseline."""
        RANK = 66
        U = [[0.0]*RANK for _ in range(n*m)]
        V = [[0.0]*RANK for _ in range(m*p)]
        W = [[0.0]*RANK for _ in range(p*n)]
        col = 0
        
        def add_strassen(a_r, a_c, b_r, b_c, c_r, c_c):
            nonlocal col
            A00 = Aidx(a_r, a_c); A01 = Aidx(a_r, a_c+1)
            A10 = Aidx(a_r+1, a_c); A11 = Aidx(a_r+1, a_c+1)
            B00 = Bidx(b_r, b_c); B01 = Bidx(b_r, b_c+1)
            B10 = Bidx(b_r+1, b_c); B11 = Bidx(b_r+1, b_c+1)
            C00 = Cidx(c_r, c_c); C01 = Cidx(c_r, c_c+1)
            C10 = Cidx(c_r+1, c_c); C11 = Cidx(c_r+1, c_c+1)
            # M1: (A00+A11)(B00+B11) -> C00+C11
            U[A00][col]+=1; U[A11][col]+=1; V[B00][col]+=1; V[B11][col]+=1; W[C00][col]+=1; W[C11][col]+=1; col+=1
            # M2: (A10+A11)B00 -> C10-C11
            U[A10][col]+=1; U[A11][col]+=1; V[B00][col]+=1; W[C10][col]+=1; W[C11][col]-=1; col+=1
            # M3: A00(B01-B11) -> C01+C11
            U[A00][col]+=1; V[B01][col]+=1; V[B11][col]-=1; W[C01][col]+=1; W[C11][col]+=1; col+=1
            # M4: A11(B10-B00) -> C00+C10
            U[A11][col]+=1; V[B10][col]+=1; V[B00][col]-=1; W[C00][col]+=1; W[C10][col]+=1; col+=1
            # M5: (A00+A01)B11 -> -C00+C01
            U[A00][col]+=1; U[A01][col]+=1; V[B11][col]+=1; W[C00][col]-=1; W[C01][col]+=1; col+=1
            # M6: (A10-A00)(B00+B01) -> C11
            U[A10][col]+=1; U[A00][col]-=1; V[B00][col]+=1; V[B01][col]+=1; W[C11][col]+=1; col+=1
            # M7: (A01-A11)(B10+B11) -> C00
            U[A01][col]+=1; U[A11][col]-=1; V[B10][col]+=1; V[B11][col]+=1; W[C00][col]+=1; col+=1
        
        def add_1x2_2x2(a_r, a_c, b_r, b_c, c_r, c_c):
            nonlocal col
            A0 = Aidx(a_r, a_c); A1 = Aidx(a_r, a_c+1)
            B00 = Bidx(b_r, b_c); B01 = Bidx(b_r, b_c+1)
            B10 = Bidx(b_r+1, b_c); B11 = Bidx(b_r+1, b_c+1)
            C0 = Cidx(c_r, c_c); C1 = Cidx(c_r, c_c+1)
            U[A0][col]+=1; V[B00][col]+=1; W[C0][col]+=1; col+=1
            U[A1][col]+=1; V[B10][col]+=1; W[C0][col]+=1; col+=1
            U[A0][col]+=1; V[B01][col]+=1; W[C1][col]+=1; col+=1
            U[A1][col]+=1; V[B11][col]+=1; W[C1][col]+=1; col+=1
        
        # i=0: rows {0}, i=1: rows {1,2}
        # j: cols {0,1}, {2,3}
        # k: cols {0,1}, {2,3}, {4,5}
        
        # i=0 (size 1) x j blocks (size 2) x k blocks (size 2): schoolbook 1x2x2, rank 4 each
        add_1x2_2x2(0,0,0,0,0,0)
        add_1x2_2x2(0,2,0,2,0,2)
        add_1x2_2x2(0,0,0,4,0,4)
        add_1x2_2x2(0,2,0,4,0,4)
        
        # i=1 (size 2) x j blocks (size 2) x k blocks (size 2): Strassen 2x2x2, rank 7 each
        add_strassen(1,0,0,0,1,0)
        add_strassen(1,2,0,2,1,2)
        add_strassen(1,0,0,4,1,4)
        add_strassen(1,2,0,4,1,4)
        add_strassen(1,0,2,0,2,0)
        add_strassen(1,2,2,2,2,2)
        add_strassen(1,0,2,4,2,4)
        add_strassen(1,2,2,4,2,4)
        
        assert col == 66, f"Expected 66, got {col}"
        return U, V, W
    
    def try_drop(U, V, W, r, l):
        """Check if removing term l still gives exact decomposition."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for ll in range(r):
                        if ll == l:
                            continue
                        s += U[Aidx(i,j)][ll] * V[Bidx(j,k)][ll] * W[Cidx(k,i)][ll]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    def local_search(U, V, W, r, deadline):
        """Iteratively try to drop redundant terms."""
        best_r = r
        best_U, best_V, best_W = U, V, W
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            for l in range(best_r):
                if try_drop(best_U, best_V, best_W, best_r, l):
                    newU = [row[:l] + row[l+1:] for row in best_U]
                    newV = [row[:l] + row[l+1:] for row in best_V]
                    newW = [row[:l] + row[l+1:] for row in best_W]
                    best_r -= 1
                    best_U, best_V, best_W = newU, newV, newW
                    improved = True
                    break
        return best_r, best_U, best_V, best_W
    
    # Start with baseline
    U, V, W = baseline_66()
    r = 66
    
    # Local search to drop redundant terms
    r, U, V, W = local_search(U, V, W, r, deadline)
    
    # Verify final result
    if not verify(U, V, W, r):
        # Fallback: return baseline
        U, V, W = baseline_66()
        r = 66
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
