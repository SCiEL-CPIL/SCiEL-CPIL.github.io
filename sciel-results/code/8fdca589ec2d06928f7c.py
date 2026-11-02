import numpy as np
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = 2, 4, 7
    
    # Try to find a better decomposition using 2x2x4 blocks
    # 2x2x4 can potentially be done in rank 8 (vs schoolbook 8 or Strassen 7+4=11)
    # Actually schoolbook 2x2x4 = 8, so that's the same.
    # Let's try a different approach: 2x2x7 in rank 13 (vs schoolbook 14)
    # using a custom decomposition
    
    def get_rank_50():
        rank = 50
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col = 0
        def add_strassen_222(i0, j0, k0):
            nonlocal col
            if col >= rank: return False
            a = i0 * m + j0
            b = i0 * m + (j0 + 1)
            c = (i0 + 1) * m + j0
            d = (i0 + 1) * m + (j0 + 1)
            e = j0 * p + k0
            f = j0 * p + (k0 + 1)
            g = (j0 + 1) * p + k0
            h = (j0 + 1) * p + (k0 + 1)
            c00 = k0 * n + i0
            c01 = (k0 + 1) * n + i0
            c10 = k0 * n + (i0 + 1)
            c11 = (k0 + 1) * n + (i0 + 1)
            
            U[a][col] += 1; U[d][col] += 1; V[e][col] += 1; V[h][col] += 1; W[c00][col] += 1; W[c11][col] += 1; col += 1
            U[c][col] += 1; U[d][col] += 1; V[e][col] += 1; W[c10][col] += 1; W[c11][col] -= 1; col += 1
            U[a][col] += 1; V[f][col] += 1; V[h][col] -= 1; W[c01][col] += 1; W[c11][col] += 1; col += 1
            U[d][col] += 1; V[g][col] += 1; V[e][col] -= 1; W[c00][col] += 1; W[c10][col] += 1; col += 1
            U[a][col] += 1; U[b][col] += 1; V[h][col] += 1; W[c00][col] -= 1; W[c01][col] += 1; col += 1
            U[c][col] += 1; U[a][col] -= 1; V[e][col] += 1; V[f][col] += 1; W[c11][col] += 1; col += 1
            U[b][col] += 1; U[d][col] -= 1; V[g][col] += 1; V[h][col] += 1; W[c00][col] += 1; col += 1
            return True

        def add_schoolbook_221(i0, j0, k0):
            nonlocal col
            if col >= rank: return False
            a = i0 * m + j0
            b = i0 * m + (j0 + 1)
            c = (i0 + 1) * m + j0
            d = (i0 + 1) * m + (j0 + 1)
            e = j0 * p + k0
            g = (j0 + 1) * p + k0
            c00 = k0 * n + i0
            c10 = k0 * n + (i0 + 1)
            
            U[a][col] += 1; V[e][col] += 1; W[c00][col] += 1; col += 1
            U[b][col] += 1; V[g][col] += 1; W[c00][col] += 1; col += 1
            U[c][col] += 1; V[e][col] += 1; W[c10][col] += 1; col += 1
            U[d][col] += 1; V[g][col] += 1; W[c10][col] += 1; col += 1
            return True

        for k0 in [0, 2, 4]:
            add_strassen_222(0, 0, k0)
        add_schoolbook_221(0, 0, 6)
        for k0 in [0, 2, 4]:
            add_strassen_222(0, 2, k0)
        add_schoolbook_221(0, 2, 6)
        
        return {"U": U, "V": V, "W": W}

    best_solution = get_rank_50()
    best_rank = 50
    
    def verify_solution(U, V, W):
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a_idx = i * m + j
                    b_idx = j * p + k
                    c_idx = k * n + i
                    reconstructed = np.sum(U_arr[a_idx] * V_arr[b_idx] * W_arr[c_idx])
                    if abs(reconstructed - 1) > 1e-9:
                        return False
        return True
    
    # Try a 2x2x3 block in rank 5 (saving 1 per block vs schoolbook 6)
    # 2x2x3: C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] for i=0,1; k=0,1,2
    # Schoolbook: 6 terms. Try rank 5.
    # Using a trick: let X = A[i,0]+A[i,1], then C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    # = (X - A[i,1])*B[0,k] + A[i,1]*B[1,k] = X*B[0,k] + A[i,1]*(B[1,k]-B[0,k])
    # That's still 2 terms per (i,k), so 6 total. Not helpful.
    # Try: C[0,0]+C[0,1]+C[0,2] = A[0,0]*(B[0,0]+B[0,1]+B[0,2]) + A[0,1]*(B[1,0]+B[1,1]+B[1,2])
    # Define S0 = B[0,0]+B[0,1]+B[0,2], S1 = B[1,0]+B[1,1]+B[1,2]
    # T0 = A[0,0]*S0 + A[0,1]*S1 gives sum of row 0 of C
    # But we need individual C[0,k] values, not just the sum.
    # So this doesn't directly help unless we can recover individual values.
    
    # Alternative: try rank-5 decomposition of 2x2x3 using linear algebra
    # The tensor T[i,j,k] has entries: T[0,0,k]=1, T[1,0,k]=1, T[0,1,k]=1, T[1,1,k]=1 for k=0,1,2
    # Wait no, T[i,j,k] = 1 for all i,j,k in the 2x2x3 case? No.
    # T[i,j,k] = 1 if the (i,j) entry of A and (j,k) entry of B contribute to C[i,k].
    # For matrix mult, T[i,j,k] = 1 for all valid i,j,k. So 2x2x3 tensor has 12 ones.
    # Rank of this tensor: we need min r such that T[i,j,k] = sum_{r=1}^r U[i,r]*V[j,r]*W[k,r]
    
    # Let's try to find a rank-5 decomposition of 2x2x3 by solving a system
    # Actually, let's just try the current approach and verify it, then try removing terms
    
    # Verify the rank-50 solution
    if not verify_solution(best_solution["U"], best_solution["V"], best_solution["W"]):
        return get_rank_50()
    
    # Try removing each term
    U_arr = np.array(best_solution["U"], dtype=float)
    V_arr = np.array(best_solution["V"], dtype=float)
    W_arr = np.array(best_solution["W"], dtype=float)
    
    # Precompute contributions for efficiency
    # For each term r, compute its contribution to each (i,j,k)
    # contribution[r, i, j, k] = U[i*4+j, r] * V[j*7+k, r] * W[k*2+i, r]
    
    # Try removing each term
    for term_idx in range(50):
        if time.time() > deadline - 10:
            break
        
        U_test = U_arr.copy()
        V_test = V_arr.copy()
        W_test = W_arr.copy()
        U_test[:, term_idx] = 0
        V_test[:, term_idx] = 0
        W_test[:, term_idx] = 0
        
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a_idx = i * m + j
                    b_idx = j * p + k
                    c_idx = k * n + i
                    reconstructed = np.sum(U_test[a_idx] * V_test[b_idx] * W_test[c_idx])
                    if abs(reconstructed - 1) > 1e-9:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            best_solution["U"] = [[0] * 49 for _ in range(n * m)]
            best_solution["V"] = [[0] * 49 for _ in range(m * p)]
            best_solution["W"] = [[0] * 49 for _ in range(p * n)]
            for r in range(50):
                if r == term_idx:
                    continue
                new_idx = r if r < term_idx else r - 1
                for i in range(n * m):
                    best_solution["U"][i][new_idx] = U_arr[i][r]
                for j in range(m * p):
                    best_solution["V"][j][new_idx] = V_arr[j][r]
                for k in range(p * n):
                    best_solution["W"][k][new_idx] = W_arr[k][r]
            best_rank = 49
            break
    
    # Final verification
    U_arr = np.array(best_solution["U"], dtype=float)
    V_arr = np.array(best_solution["V"], dtype=float)
    W_arr = np.array(best_solution["W"], dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                reconstructed = np.sum(U_arr[a_idx] * V_arr[b_idx] * W_arr[c_idx])
                if abs(reconstructed - 1) > 1e-9:
                    return get_rank_50()
    
    return best_solution
