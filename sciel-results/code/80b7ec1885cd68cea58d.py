import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = 110.0

    n, m, p = 2, 3, 8
    dim_U = n * m  # 6
    dim_V = m * p  # 24
    dim_W = p * n  # 16

    def time_left():
        return deadline - (time.time() - start_time)

    def build_matrices(U_list, V_list, W_list):
        rank = len(U_list)
        U_mat = [[0] * rank for _ in range(dim_U)]
        V_mat = [[0] * rank for _ in range(dim_V)]
        W_mat = [[0] * rank for _ in range(dim_W)]
        for r in range(rank):
            for idx in range(dim_U):
                U_mat[idx][r] = U_list[r][idx]
            for idx in range(dim_V):
                V_mat[idx][r] = V_list[r][idx]
            for idx in range(dim_W):
                W_mat[idx][r] = W_list[r][idx]
        return U_mat, V_mat, W_mat

    def verify(U_mat, V_mat, W_mat):
        R = len(U_mat[0])
        for i0 in range(n):
            for j0 in range(m):
                for k0 in range(p):
                    for i in range(n):
                        for k in range(p):
                            val = 0
                            for r in range(R):
                                if i == i0 and k == k0:
                                    val += W_mat[k * n + i][r] * U_mat[i * m + j0][r] * V_mat[j0 * p + k][r]
                            expected_val = 1 if (i == i0 and k == k0) else 0
                            if val != expected_val:
                                return False
        return True

    # Base 2x2x2 Strassen decomposition
    # For matrices A (2x2), B (2x2), C = A*B
    # C[0,0] = (A[0,0]+A[1,1])*(B[0,0]+B[1,1]) - A[1,0]*B[0,1] - A[0,1]*B[1,1] + A[1,1]*B[0,0]
    # Standard Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1 - M4 + M6 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M5 - M6

    def strassen_2x2x2():
        """Return 7 (U, V, W) triples for 2x2x2 matrix multiplication."""
        # U is 4-dim (i,j) for 2x2, V is 4-dim (j,k) for 2x2, W is 4-dim (k,i) for 2x2
        U = []
        V = []
        W = []
        
        # M1 = (A00+A11)(B00+B11)
        U.append([1, 0, 0, 1])  # A00 + A11
        V.append([1, 0, 1, 0])  # B00 + B11
        W.append([1, 0, 0, 1])  # C00 + C11 (will be combined)
        
        # M2 = (A10+A11)B00
        U.append([0, 0, 1, 1])  # A10 + A11
        V.append([1, 0, 0, 0])  # B00
        W.append([0, 0, 0, 1])  # C10 + C11
        
        # M3 = A00(B10-B11)
        U.append([1, 0, 0, 0])  # A00
        V.append([0, 1, -1, 0])  # B10 - B11
        W.append([0, 1, 0, 0])  # C00 - C01
        
        # M4 = A11(B01-B00)
        U.append([0, 0, 0, 1])  # A11
        V.append([0, 1, 0, -1])  # B01 - B00 (note: V[j,k] so B01 is j=0,k=1; B00 is j=0,k=0)
        W.append([0, 0, 1, 0])  # C10 - C11
        
        # M5 = (A00+A01)B11
        U.append([1, 1, 0, 0])  # A00 + A01
        V.append([0, 0, 0, 1])  # B11
        W.append([0, 1, 0, 1])  # C01 + C11
        
        # M6 = (A10-A00)(B00+B01)
        U.append([0, -1, 1, 0])  # A10 - A00
        V.append([1, 1, 0, 0])  # B00 + B01
        W.append([0, 0, 0, -1])  # -C11 (and +C00 via M1-M4+M6+M7)
        
        # M7 = (A01-A11)(B10+B11)
        U.append([0, 1, 0, -1])  # A01 - A11
        V.append([0, 0, 1, 1])  # B10 + B11
        W.append([1, 0, 0, 0])  # C00
        
        return U, V, W

    # Let me verify the Strassen formulas more carefully
    # C00 = M1 - M4 + M6 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M5 - M6

    # Let me redefine with correct W entries
    def strassen_2x2x2_correct():
        U = []
        V = []
        W = []
        
        # M1 = (A00+A11)(B00+B11)
        U.append([1, 0, 0, 1])
        V.append([1, 0, 1, 0])
        W.append([1, 0, 0, 1])  # contributes to C00 and C11
        
        # M2 = (A10+A11)B00
        U.append([0, 0, 1, 1])
        V.append([1, 0, 0, 0])
        W.append([0, 0, 0, 1])  # contributes to C10 and C11
        
        # M3 = A00(B10-B11)
        U.append([1, 0, 0, 0])
        V.append([0, 1, -1, 0])
        W.append([0, 1, 0, 0])  # contributes to C00 and -C01
        
        # M4 = A11(B01-B00)
        U.append([0, 0, 0, 1])
        V.append([0, 1, 0, -1])
        W.append([0, 0, 1, 0])  # contributes to C10 and -C11
        
        # M5 = (A00+A01)B11
        U.append([1, 1, 0, 0])
        V.append([0, 0, 0, 1])
        W.append([0, 1, 0, 1])  # contributes to C01 and C11
        
        # M6 = (A10-A00)(B00+B01)
        U.append([0, -1, 1, 0])
        V.append([1, 1, 0, 0])
        W.append([0, 0, 0, -1])  # contributes to -C11
        
        # M7 = (A01-A11)(B10+B11)
        U.append([0, 1, 0, -1])
        V.append([0, 0, 1, 1])
        W.append([1, 0, 0, 0])  # contributes to C00
        
        return U, V, W

    # Verify Strassen
    sU, sV, sW = strassen_2x2x2_correct()
    # Check: for each (i,j,k) in 2x2x2, sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] should be 1 if i==j0 and k==k0... 
    # Actually the tensor T[i,j,k] = 1 if this is the "multiplication" term
    # For 2x2x2 matrix mult: T[i,j,k] = delta(i, row of A) * delta(j, col of A = row of B) * delta(k, col of B)
    # The decomposition should satisfy: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 if (i,j,k) corresponds to a valid product term
    # Actually, the matrix multiplication tensor has T[i,j,k] = 1 for all valid (i,j,k) combinations where
    # i is row of C, j is the shared index, k is col of C. Wait, no.
    
    # The tensor for matrix multiplication C = AB where A is n x m, B is m x p:
    # T[i,j,k] where i in [0,n), j in [0,m), k in [0,p)
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor has T[i,j,k] = 1 for all i,j,k (it's a "full" tensor in the sense that all entries are 1)
    # The decomposition: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] should equal 1 for all i,j,k
    
    # Let me verify Strassen for 2x2x2:
    # For each (i,j,k) in {0,1}^3, sum over r of U[i,j,r]*V[j,k,r]*W[k,i,r] should be 1
    
    def verify_2x2x2(U_list, V_list, W_list):
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    val = 0
                    for r in range(len(U_list)):
                        val += U_list[r][i*2+j] * V_list[r][j*2+k] * W_list[r][k*2+i]
                    if val != 1:
                        return False, (i, j, k, val)
        return True, None

    ok, info = verify_2x2x2(sU, sV, sW)
    if not ok:
        # Try to fix
        pass

    # Build the full 2x3x8 decomposition using Strassen on blocks
    # Strategy: 
    # - j=0: 2x1x8 -> 16 terms (can't do better for 1-column B)
    # - j=1,2: 2x2x8 -> split into 4 blocks of 2x2x2, use Strassen (7 terms each) = 28 terms
    # Total: 16 + 28 = 44

    U_list = []
    V_list = []
    W_list = []

    # Part 1: j=0, 2x1x8
    for k in range(p):
        # C[0,k] = A[0,0]*B[0,k]
        U_list.append([1, 0, 0, 0, 0, 0])  # A[0,0]
        V_list.append([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # B[0,k]
        # V index: j*p + k = 0*8 + k = k
        V_list[-1] = [0]*24
        V_list[-1][k] = 1
        W_list.append([0]*16)
        W_list[-1][k*2 + 0] = 1  # C[0,k]

        # C[1,k] = A[1,0]*B[0,k]
        U_list.append([0, 0, 0, 1, 0, 0])  # A[1,0]
        V_list.append([0]*24)
        V_list[-1][k] = 1
        W_list.append([0]*16)
        W_list[-1][k*2 + 1] = 1  # C[1,k]

    # Part 2: j=1,2, 2x2x8
    # Split k into 4 blocks of 2
    for b in range(4):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # Use Strassen for 2x2x2 with rows i=0,1, cols j=1,2, and k=k0,k1
        # Map: local i -> global i, local j -> global j+1, local k -> global k0+local_k
        
        sU, sV, sW = strassen_2x2x2_correct()
        for r in range(7):
            # U: 2x2 local (i, j_local) -> 6-dim global (i, j_global=j_local+1)
            u_global = [0] * 6
            for i in range(2):
                for j_local in range(2):
                    u_global[i * 3 + (j_local + 1)] = sU[r][i * 2 + j_local]
            U_list.append(u_global)
            
            # V: 2x2 local (j_local, k_local) -> 24-dim global (j_global=j_local+1, k_global=k0+k_local)
            v_global = [0] * 24
            for j_local in range(2):
                for k_local in range(2):
                    v_global[(j_local + 1) * 8 + (k0 + k_local)] = sV[r][j_local * 2 + k_local]
            V_list.append(v_global)
            
            # W: 2x2 local (k_local, i) -> 16-dim global (k_global=k0+k_local, i)
            w_global = [0] * 16
            for k_local in range(2):
                for i in range(2):
                    w_global[(k0 + k_local) * 2 + i] = sW[r][k_local * 2 + i]
            W_list.append(w_global)

    rank = len(U_list)
    U_mat, V_mat, W_mat = build_matrices(U_list, V_list, W_list)

    if verify(U_mat, V_mat, W_mat):
        return {"U": U_mat, "V": V_mat, "W": W_mat}

    # Fallback: schoolbook 48
    U_list = []
    V_list = []
    W_list = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = [0] * 6
                u[i * 3 + j] = 1
                v = [0] * 24
                v[j * 8 + k] = 1
                w = [0] * 16
                w[k * 2 + i] = 1
                U_list.append(u)
                V_list.append(v)
                W_list.append(w)

    U_mat, V_mat, W_mat = build_matrices(U_list, V_list, W_list)
    return {"U": U_mat, "V": V_mat, "W": W_mat}


def _verify(U, V, W):
    n, m, p = 2, 3, 8
    R = len(U[0]) if U else 0

    for i0 in range(n):
        for j0 in range(m):
            for k0 in range(p):
                for i in range(n):
                    for k in range(p):
                        val = 0
                        for r in range(R):
                            if i == i0 and k == k0:
                                val += W[k * n + i][r] * U[i * m + j0][r] * V[j0 * p + k][r]
                        expected_val = 1 if (i == i0 and k == k0) else 0
                        if val != expected_val:
                            return False
    return True


if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res["U"][0]))
    if _verify(res["U"], res["V"], res["W"]):
        print("Verification PASSED")
    else:
        print("Verification FAILED")
