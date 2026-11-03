import time
import numpy as np
from fractions import Fraction

N, M, P = 3, 4, 4
NM, MP, PN = N * M, M * P, P * N

def _verify(U, V, W, tol=1e-10):
    """
    Verify the tensor decomposition using floating point arithmetic.
    T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] should be 1.
    """
    rank = len(U[0])
    U_arr = np.array(U, dtype=float).reshape(N, M, rank)
    V_arr = np.array(V, dtype=float).reshape(M, P, rank)
    W_arr = np.array(W, dtype=float).reshape(P, N, rank)
    
    # Compute the tensor T
    # T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
    T = np.einsum('ijc,jkc,kic->ijk', U_arr, V_arr, W_arr)
    
    # The target tensor for matrix multiplication is 1 at (i,j,k) 
    # corresponding to the scalar product term? 
    # Wait, the tensor for matrix multiplication (A * B)_{i,k} = sum_j A_{i,j} B_{j,k}
    # is T_{i,j,k} = delta_{i,j,k} in the sense that the coefficient of A_{i,j} B_{j,k} in (AB)_{i,k} is 1.
    # So T[i,j,k] should be 1 for all i,j,k.
    
    max_err = np.max(np.abs(T - 1.0))
    return max_err < tol, max_err

def _strassen_44():
    """
    Constructs a rank-44 decomposition for 3x4 * 4x4 matrix multiplication.
    This uses Strassen's algorithm for the top 2x2 block and standard multiplication for the rest.
    """
    rank = 44
    U = [[0] * rank for _ in range(NM)]
    V = [[0] * rank for _ in range(MP)]
    W = [[0] * rank for _ in range(PN)]
    
    # Strassen steps for 2x2 block multiplication
    # Each step is (U_coeffs, V_coeffs, W_coeffs)
    # Indices are local (0,1)
    strassen_steps = [
        # M1 = (X00 + X11)(Y00 + Y11) -> Z00 += 1, Z11 += 1
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # M2 = (X10 + X11)Y00 -> Z10 += 1, Z11 -= 1
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        # M3 = X00(Y01 - Y11) -> Z01 += 1, Z11 += 1
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        # M4 = X11(Y10 - Y00) -> Z00 += 1, Z10 += 1
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        # M5 = (X00 + X01)Y11 -> Z01 += 1, Z00 -= 1
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,1): 1, (0,0): -1}),
        # M6 = (X10 - X00)(Y00 + Y01) -> Z11 += 1
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # M7 = (X01 - X11)(Y10 + Y11) -> Z00 += 1
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1})
    ]
    
    # We need to compute C[0:2, 0:4] = A[0:2, 0:4] * B[0:4, 0:4]
    # We split A into A_left (2x2) and A_right (2x2)
    # We split B into B_top (4x2) and B_bottom (4x2)
    # Actually, it's easier to view it as 4 blocks of 2x2 multiplications.
    # C[0:2, 0:2] = A[0:2, 0:2]B[0:2, 0:2] + A[0:2, 2:4]B[2:4, 0:2]
    # C[0:2, 2:4] = A[0:2, 0:2]B[0:2, 2:4] + A[0:2, 2:4]B[2:4, 2:4]
    
    # Block definitions: (x_i, x_j, y_j, y_k, z_i, z_k)
    # x: row offset in A, y: col offset in A / row offset in B, z: col offset in B / col offset in C
    # A is 3x4, B is 4x4, C is 3x4
    # Strassen applies to 2x2 blocks.
    
    blocks = [
        # C[0:2, 0:2] part 1: A[0:2, 0:2] * B[0:2, 0:2]
        (0, 0, 0, 0, 0, 0),
        # C[0:2, 0:2] part 2: A[0:2, 2:4] * B[2:4, 0:2]
        (0, 2, 2, 0, 0, 0),
        # C[0:2, 2:4] part 1: A[0:2, 0:2] * B[0:2, 2:4]
        (0, 0, 0, 2, 0, 2),
        # C[0:2, 2:4] part 2: A[0:2, 2:4] * B[2:4, 2:4]
        (0, 2, 2, 2, 0, 2)
    ]
    
    col = 0
    for (xi, xj, yj, yk, zi, zk) in blocks:
        for u_c, v_c, w_c in strassen_steps:
            for (a, b), val in u_c.items():
                U[(a + xi) * M + (b + xj)][col] += val
            for (b, c), val in v_c.items():
                V[(b + yj) * P + (c + yk)][col] += val
            for (a, c), val in w_c.items():
                W[(c + zk) * N + (a + zi)][col] += val
            col += 1
    
    # Naive part for the bottom row of C (i=2)
    # C[2, k] = sum_j A[2, j] * B[j, k]
    for j in range(M):
        for k in range(P):
            U[2 * M + j][col] = 1
            V[j * P + k][col] = 1
            W[k * N + 2][col] = 1
            col += 1
    
    assert col == rank, f"Rank mismatch: {col} vs {rank}"
    return {'U': U, 'V': V, 'W': W}

def _try_reduce_by_merge(U, V, W, deadline):
    """
    Attempts to reduce rank by merging proportional columns.
    """
    rank = len(U[0])
    best_U, best_V, best_W = U[:], V[:], W[:]
    best_rank = rank
    
    # Use fractions for exact comparison
    U_frac = [[Fraction(x) for x in row] for row in U]
    V_frac = [[Fraction(x) for x in row] for row in V]
    W_frac = [[Fraction(x) for x in row] for row in W]
    
    deadline = deadline - 2.0 # Safety margin
    
    for attempt in range(50):
        if time.time() > deadline:
            break
        
        found_merge = False
        for c1 in range(best_rank - 1):
            if time.time() > deadline:
                break
            for c2 in range(c1 + 1, best_rank):
                u1 = [U_frac[i][c1] for i in range(NM)]
                u2 = [U_frac[i][c2] for i in range(NM)]
                v1 = [V_frac[i][c1] for i in range(MP)]
                v2 = [V_frac[i][c2] for i in range(MP)]
                w1 = [W_frac[i][c1] for i in range(PN)]
                w2 = [W_frac[i][c2] for i in range(PN)]
                
                # Check if u2 = a * u1
                a = None
                valid_a = True
                for i in range(NM):
                    if u1[i] != 0:
                        if u2[i] != 0:
                            if a is None:
                                a = u2[i] / u1[i]
                            elif u2[i] / u1[i] != a:
                                valid_a = False
                                break
                        else:
                            valid_a = False
                            break
                    else:
                        if u2[i] != 0:
                            valid_a = False
                            break
                if not valid_a or a is None or a == 0:
                    continue
                
                # Check if v2 = b * v1
                b = None
                valid_b = True
                for i in range(MP):
                    if v1[i] != 0:
                        if v2[i] != 0:
                            if b is None:
                                b = v2[i] / v1[i]
                            elif v2[i] / v1[i] != b:
                                valid_b = False
                                break
                        else:
                            valid_b = False
                            break
                    else:
                        if v2[i] != 0:
                            valid_b = False
                            break
                if not valid_b or b is None or b == 0:
                    continue
                
                # Check if w2 = c * w1
                c = None
                valid_c = True
                for i in range(PN):
                    if w1[i] != 0:
                        if w2[i] != 0:
                            if c is None:
                                c = w2[i] / w1[i]
                            elif w2[i] / w1[i] != c:
                                valid_c = False
                                break
                        else:
                            valid_c = False
                            break
                    else:
                        if w2[i] != 0:
                            valid_c = False
                            break
                if not valid_c or c is None or c == 0:
                    continue
                
                # Check if a * b * c == 1
                if a * b * c != 1:
                    continue
                
                # Merge c2 into c1
                for i in range(NM):
                    U_frac[i][c1] = U_frac[i][c1] + U_frac[i][c2]
                for i in range(MP):
                    V_frac[i][c1] = V_frac[i][c1] + V_frac[i][c2]
                for i in range(PN):
                    W_frac[i][c1] = W_frac[i][c1] + W_frac[i][c2]
                
                # Remove column c2
                U_frac = [row[:c2] + row[c2+1:] for row in U_frac]
                V_frac = [row[:c2] + row[c2+1:] for row in V_frac]
                W_frac = [row[:c2] + row[c2+1:] for row in W_frac]
                
                best_rank -= 1
                best_U = [[float(x) for x in row] for row in U_frac]
                best_V = [[float(x) for x in row] for row in V_frac]
                best_W = [[float(x) for x in row] for row in W_frac]
                
                found_merge = True
                break
            if found_merge or time.time() > deadline:
                break
        
        if not found_merge:
            break
            
    return {'U': best_U, 'V': best_V, 'W': best_W}, best_rank

def construct():
    start = time.time()
    deadline = start + 140.0 # 10s safety margin
    
    # 1. Generate the base rank-44 decomposition
    result = _strassen_44()
    
    # 2. Verify it
    ok, err = _verify(result['U'], result['V'], result['W'])
    if not ok:
        # Fallback to schoolbook if something went wrong (should not happen)
        rank = NM * MP * PN
        U = [[0] * rank for _ in range(NM)]
        V = [[0] * rank for _ in range(MP)]
        W = [[0] * rank for _ in range(PN)]
        c = 0
        for i in range(N):
            for j in range(M):
                for k in range(P):
                    U[i * M + j][c] = 1
                    V[j * P + k][c] = 1
                    W[k * N + i][c] = 1
                    c += 1
        return {'U': U, 'V': V, 'W': W}
    
    best_result = result
    best_rank = len(result['U'][0])
    
    # 3. Try to reduce rank
    if time.time() < deadline - 5.0:
        merged_result, merged_rank = _try_reduce_by_merge(result['U'], result['V'], result['W'], deadline)
        if merged_rank < best_rank:
            ok, err = _verify(merged_result['U'], merged_result['V'], merged_result['W'])
            if ok:
                best_result = merged_result
                best_rank = merged_rank
    
    return best_result

if __name__ == "__main__":
    # Simple self-test
    res = construct()
    print(f"Rank: {len(res['U'][0])}")
    ok, err = _verify(res['U'], res['V'], res['W'])
    print(f"Valid: {ok}, Max Error: {err}")
