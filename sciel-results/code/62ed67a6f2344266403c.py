import time
import numpy as np
from itertools import product

def _strassen_28():
    """Return (U, V, W) as lists of lists, rank 28, 4 independent Strassen blocks."""
    n, m, p = 2, 4, 4
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    def add_block(col_offset, X_rows, Y_rows, Z_rows):
        U[X_rows[0]][col_offset + 0] += 1
        U[X_rows[3]][col_offset + 0] += 1
        V[Y_rows[0]][col_offset + 0] += 1
        V[Y_rows[3]][col_offset + 0] += 1
        W[Z_rows[0]][col_offset + 0] += 1
        W[Z_rows[3]][col_offset + 0] += 1

        U[X_rows[2]][col_offset + 1] += 1
        U[X_rows[3]][col_offset + 1] += 1
        V[Y_rows[0]][col_offset + 1] += 1
        W[Z_rows[2]][col_offset + 1] += 1
        W[Z_rows[3]][col_offset + 1] -= 1

        U[X_rows[0]][col_offset + 2] += 1
        V[Y_rows[1]][col_offset + 2] += 1
        V[Y_rows[3]][col_offset + 2] -= 1
        W[Z_rows[1]][col_offset + 2] += 1
        W[Z_rows[3]][col_offset + 2] += 1

        U[X_rows[3]][col_offset + 3] += 1
        V[Y_rows[2]][col_offset + 3] += 1
        V[Y_rows[0]][col_offset + 3] -= 1
        W[Z_rows[0]][col_offset + 3] += 1
        W[Z_rows[2]][col_offset + 3] += 1

        U[X_rows[0]][col_offset + 4] += 1
        U[X_rows[1]][col_offset + 4] += 1
        V[Y_rows[3]][col_offset + 4] += 1
        W[Z_rows[0]][col_offset + 4] -= 1
        W[Z_rows[1]][col_offset + 4] += 1

        U[X_rows[2]][col_offset + 5] += 1
        U[X_rows[0]][col_offset + 5] -= 1
        V[Y_rows[0]][col_offset + 5] += 1
        V[Y_rows[1]][col_offset + 5] += 1
        W[Z_rows[3]][col_offset + 5] += 1

        U[X_rows[1]][col_offset + 6] += 1
        U[X_rows[3]][col_offset + 6] -= 1
        V[Y_rows[2]][col_offset + 6] += 1
        V[Y_rows[3]][col_offset + 6] += 1
        W[Z_rows[0]][col_offset + 6] += 1

    X1 = [0, 1, 4, 5];  Y1 = [0, 1, 4, 5];   Z1 = [0, 2, 1, 3]
    X2 = [2, 3, 6, 7];  Y2 = [8, 9, 12, 13]; Z2 = [0, 2, 1, 3]
    X3 = [0, 1, 4, 5];  Y3 = [2, 3, 6, 7];   Z3 = [4, 6, 5, 7]
    X4 = [2, 3, 6, 7];  Y4 = [10, 11, 14, 15]; Z4 = [4, 6, 5, 7]

    add_block(0,  X1, Y1, Z1)
    add_block(7,  X2, Y2, Z2)
    add_block(14, X3, Y3, Z3)
    add_block(21, X4, Y4, Z4)
    return U, V, W


def _verify_decomp(U, V, W, n=2, m=4, p=4):
    """Check sum_k U[i,k]*V[j,k] == delta_{i//m, j//m} * (j%m) for all i in [0,nm), j in [0,mp)."""
    rank = len(U[0])
    # Build the target tensor: T[i, j] = 1 if (i//m == j//m) and (i%m == j%m), else 0
    # Actually the multiplication tensor: (A*B)[a,b] = sum_j A[a,j]*B[j,b]
    # So T[a, j, b] = delta_{a,j} ... no.
    # The tensor is T[i, j, k] where i indexes A (n*m), j indexes B (m*p), k indexes C (p*n).
    # T[i, j, k] = 1 if the multiplication A*B produces C[k] from A[i] and B[j].
    # A is n x m, B is m x p, C is p x n (trace convention W[k*n+i, r] means C row k, col i).
    # C[k, i] = sum_j A[i, j] * B[j, k]
    # So the tensor T[i, j, k] = 1 iff the term A[i,j]*B[j,k] contributes to C[k,i].
    # i is in [0, n*m): i = a*m + j where a is row of A, j is col of A (= row of B)
    # j is in [0, m*p): j = b*m + k where b is row of B, k is col of B (= row of C)
    # k is in [0, p*n): k = c*n + a where c is row of C, a is col of C (= row of A)
    # Contribution: A[a, j]*B[b, k] contributes to C[c, a] iff j == b and c == k.
    # So T[i, j, k] = 1 iff (i%m == j//m) and (j%m == k//n) and (k%n == i//m)
    
    def target(i, j, k):
        if i % m != j // m: return 0
        if j % m != k // n: return 0
        if k % n != i // m: return 0
        return 1
    
    # Verify: sum_r U[i,r]*V[j,r] should equal... wait, the decomposition is:
    # T[i, j, k] = sum_r U[i,r] * V[j,r] * W[k,r]
    # Let's check this.
    for i in range(n * m):
        for j in range(m * p):
            for k in range(p * n):
                s = 0
                for r in range(rank):
                    s += U[i][r] * V[j][r] * W[k][r]
                if s != target(i, j, k):
                    return False
    return True


def _try_rank_search(deadline):
    """Try to find a decomposition with rank < 28 using small integer entries."""
    n, m, p = 2, 4, 4
    nm, mp, pn = n * m, m * p, p * n  # 8, 16, 8
    
    def target_tensor():
        T = np.zeros((nm, mp, pn), dtype=int)
        for i in range(nm):
            for j in range(mp):
                if i % m != j // m: continue
                for k in range(pn):
                    if j % m != k // n: continue
                    if k % n != i // m: continue
                    T[i, j, k] = 1
        return T
    
    T = target_tensor()
    
    # Try rank 24-27 with entries in {-1, 0, 1}
    # This is a hard problem. Let's try a structured approach:
    # The tensor M_{2,4,4} can be seen as 4 copies of M_{1,4,4} interleaved,
    # or as M_{2,2,4} x M_{1,2,2} type structure.
    # 
    # Key insight: M_{2,4,4} = M_{2,2,4} ⊗ M_{1,2,2} in some sense.
    # Actually, let's think of it as: A is 2x4 = [A1 | A2] where A1, A2 are 2x2.
    # B is 4x4 = [[B11, B12], [B21, B22]] where each is 2x2.
    # C = A*B = [A1*B11 + A2*B21, A1*B12 + A2*B22] (each 2x2)
    # 
    # This is exactly what the current code does: 4 Strassen blocks.
    # 
    # Can we share between the 4 blocks? The blocks are:
    # Block 1: A1 * B11 -> C11 (top-left of C)
    # Block 2: A2 * B21 -> C12 (bottom-left... wait let me recheck)
    # 
    # Actually with trace convention W[k*n+i], C is stored row-major with n columns.
    # C has p=4 rows and n=2 cols.
    # C[k, i] = sum_j A[i, j] * B[j, k]
    # 
    # Let me re-derive. A is 2x4, so A[i, j] for i in {0,1}, j in {0,1,2,3}.
    # B is 4x4, B[j, k] for j in {0,1,2,3}, k in {0,1,2,3}.
    # C is 4x2, C[k, i] for k in {0,1,2,3}, i in {0,1}.
    # 
    # C[k, 0] = A[0,0]*B[0,k] + A[0,1]*B[1,k] + A[0,2]*B[2,k] + A[0,3]*B[3,k]
    # C[k, 1] = A[1,0]*B[0,k] + A[1,1]*B[1,k] + A[1,2]*B[2,k] + A[1,3]*B[3,k]
    # 
    # So C is the product of a 2x4 matrix and 4x4 matrix, giving 4x2.
    # This is equivalent to: for each row i of A (i=0,1), compute A[i,:] * B.
    # 
    # The tensor rank of M_{n,m,p} for n=2, m=4, p=4.
    # Known: M_{2,2,2} has rank 7 (Strassen).
    # M_{2,4,4} can be decomposed as 4 * M_{2,2,2} giving rank 28.
    # 
    # Lower bound: The tensor M_{2,4,4} has dimension 8*16*8 = 1024.
    # A rank-r decomposition has 3*r parameters (U, V, W each have r columns).
    # So r >= 1024/3 ≈ 341? No, that's not right because of sparsity.
    # 
    # Actually, the standard lower bound is via the slice rank or other methods.
    # For M_{n,m,p}, a lower bound is max(n*m, m*p, p*n) = max(8,16,8) = 16.
    # A better bound: the tensor is "non-decomposable" in a strong sense.
    # 
    # Let me try a computational search for rank < 28 with small entries.
    
    # Given the time constraint, let's do a quick randomized search.
    # We'll try to find U, V, W with rank r where entries are in {-2,-1,0,1,2}.
    
    best_rank = 28
    best_decomp = None
    
    for r in range(24, 28):
        if time.time() > deadline - 10:
            break
        # Randomized search
        for trial in range(500):
            if time.time() > deadline - 10:
                break
            # Initialize with small random integers
            U_cand = np.random.randint(-2, 3, (nm, r))
            V_cand = np.random.randint(-2, 3, (mp, r))
            W_cand = np.random.randint(-2, 3, (pn, r))
            
            # Check
            ok = True
            for i in range(nm):
                for j in range(mp):
                    for k in range(pn):
                        s = np.dot(U_cand[i], V_cand[j] * W_cand[k])
                        # This is not right. The decomposition is:
                        # T[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r]
                        # So for fixed i,j, sum_r U[i,r]*V[j,r] should give a "partial"
                        # Actually no. Let me re-read the problem.
                        # 
                        # The tensor T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
                        # This is a 3-way tensor decomposition (CP decomposition).
                        # 
                        # For matrix multiplication: T[i,j,k] = 1 if the indices match.
                        # 
                        # Let me verify with the Strassen decomposition.
                        # For Strassen 2x2x2: 
                        # U, V, W are 4x7 matrices.
                        # T[i,j,k] = sum_{r=0}^{6} U[i,r]*V[j,r]*W[k,r]
                        # 
                        # Let me check a specific case.
                        # i=0, j=0, k=0: should be 1 (A[0,0]*B[0,0] -> C[0,0])
                        # U[0] = [1, 0, 1, 0, 1, -1, 0] (from the code, col 0,2,4 have U[0]=1)
                        # Wait, let me trace through add_block for col_offset=0:
                        # U[0][0] += 1, U[0][2] += 1, U[0][4] += 1, U[0][5] -= 1
                        # So U[0] = [1, 0, 1, 0, 1, -1, 0]
                        # V[0][0] += 1, V[0][1] += 1, V[0][3] -= 1, V[0][5] += 1
                        # So V[0] = [1, 1, 0, -1, 0, 1, 0]
                        # W[0][0] += 1, W[0][3] += 1, W[0][4] -= 1, W[0][6] += 1
                        # So W[0] = [1, 0, 0, 1, -1, 0, 1]
                        # 
                        # T[0,0,0] = sum_r U[0,r]*V[0,r]*W[0,r]
                        # = 1*1*1 + 0*1*0 + 1*0*0 + 0*(-1)*1 + 1*0*(-1) + (-1)*1*0 + 0*0*1
                        # = 1 + 0 + 0 + 0 + 0 + 0 + 0 = 1. Correct!
                        # 
                        # OK so the check is: for all i,j,k, sum_r U[i,r]*V[j,r]*W[k,r] == T[i,j,k]
                        # 
                        # This is expensive to check for all i,j,k. Let's be smarter.
                        # For fixed i,j, define P[i,j,r] = U[i,r]*V[j,r].
                        # Then T[i,j,k] = sum_r P[i,j,r]*W[k,r].
                        # So for fixed i,j, the vector (T[i,j,0], ..., T[i,j,pn-1]) 
                        # = P[i,j,:] @ W.T
                        # 
                        # We can check this more efficiently.
                        pass
            
            # Efficient check
            ok = True
            for i in range(nm):
                for j in range(mp):
                    P = U_cand[i] * V_cand[j]  # shape (r,)
                    for k in range(pn):
                        s = np.dot(P, W_cand[k])
                        expected = 0
                        if i % m == j // m and j % m == k // n and k % n == i // m:
                            expected = 1
                        if s != expected:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            
            if ok:
                best_rank = r
                best_decomp = (U_cand.tolist(), V_cand.tolist(), W_cand.tolist())
                return best_rank, best_decomp
    
    return best_rank, best_decomp


def construct():
    t0 = time.time()
    deadline = t0 + 115.0  # safety margin under 150s kill
    
    # Try to find a better decomposition
    try:
        rank_found, decomp_found = _try_rank_search(deadline)
        if decomp_found is not None:
            U, V, W = decomp_found
            # Verify
            if _verify_decomp(U, V, W):
                return {"U": U, "V": V, "W": W}
    except Exception:
        pass
    
    # Fallback to Strassen rank 28
    U, V, W = _strassen_28()
    return {"U": U, "V": V, "W": W}
