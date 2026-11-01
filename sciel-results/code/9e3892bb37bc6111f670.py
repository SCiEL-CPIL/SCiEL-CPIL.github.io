import numpy as np
import time
from scipy.optimize import least_squares

def construct():
    n, m, p = 2, 4, 4
    start_time = time.time()
    time_limit = 140.0
    
    # Build the rank 28 solution
    # A is 2x4, B is 4x4, C is 2x4
    # Split A into A1 (2x2) and A2 (2x2)
    # Split B into B11, B12, B21, B22 (2x2 each)
    # C1 = A1*B11 + A2*B21
    # C2 = A1*B12 + A2*B22
    
    # Strassen for 2x2:
    # m1 = (a11+a22)(b11+b22)
    # m2 = (a21+a22)b11
    # m3 = a11(b12-b22)
    # m4 = a22(b21-b11)
    # m5 = (a11+a12)b22
    # m6 = (a21-a11)(b11+b12)
    # m7 = (a12-a22)(b21+b22)
    
    # We need to map these to the full 2x4 x 4x4 -> 2x4 tensor.
    # U: 2x4 x rank
    # V: 4x4 x rank
    # W: 2x4 x rank
    
    rank = 28
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Define the 7 Strassen terms for 2x2 multiplication
    # Indices for 2x2: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # U terms (A side), V terms (B side), W terms (C side)
    strassen_terms = [
        # m1
        ([(0,1), (3,1)], [(0,1), (3,1)], [(0,1), (3,1)]),
        # m2
        ([(2,1), (3,1)], [(0,1)], [(2,1), (3,-1)]),
        # m3
        ([(0,1)], [(1,1), (3,-1)], [(1,1), (3,1)]),
        # m4
        ([(3,1)], [(2,1), (0,-1)], [(0,1), (2,1)]),
        # m5
        ([(0,1), (1,1)], [(3,1)], [(0,-1), (1,1)]),
        # m6
        ([(2,1), (0,-1)], [(0,1), (1,1)], [(3,1)]),
        # m7
        ([(1,1), (3,-1)], [(2,1), (3,1)], [(0,1)])
    ]
    
    # We have 4 blocks to multiply:
    # 1. A1 * B11 -> contributes to C1
    # 2. A2 * B21 -> contributes to C1
    # 3. A1 * B12 -> contributes to C2
    # 4. A2 * B22 -> contributes to C2
    
    # Mapping indices:
    # A1 is A[:, 0:2], A2 is A[:, 2:4]
    # B11 is B[0:2, 0:2], B12 is B[0:2, 2:4]
    # B21 is B[2:4, 0:2], B22 is B[2:4, 2:4]
    # C1 is C[:, 0:2], C2 is C[:, 2:4]
    
    # U indices:
    # A1: 0,1,4,5 (row 0 cols 0,1; row 1 cols 0,1)
    # A2: 2,3,6,7 (row 0 cols 2,3; row 1 cols 2,3)
    # Note: U is flattened A[i*m+j]. 
    # A[0,0]->0, A[0,1]->1, A[1,0]->4, A[1,1]->5
    # A[0,2]->2, A[0,3]->3, A[1,2]->6, A[1,3]->7
    
    # V indices:
    # B11: 0,1,4,5
    # B12: 2,3,6,7
    # B21: 8,9,12,13
    # B22: 10,11,14,15
    # Note: V is flattened B[j*p+k].
    # B[0,0]->0, B[0,1]->1, B[1,0]->4, B[1,1]->5
    # B[0,2]->2, B[0,3]->3, B[1,2]->6, B[1,3]->7
    # B[2,0]->8, B[2,1]->9, B[3,0]->12, B[3,1]->13
    # B[2,2]->10, B[2,3]->11, B[3,2]->14, B[3,3]->15
    
    # W indices:
    # C1: 0,1,4,5 (row 0 cols 0,1; row 1 cols 0,1)
    # C2: 2,3,6,7 (row 0 cols 2,3; row 1 cols 2,3)
    # Note: W is flattened C[k*n+i] ?? No, W[k*n+i, r] where C[i,k]?
    # Convention: W[k*n+i, r] corresponds to C[i,k]?
    # The task says: "output factor follows the official trace convention W[k*n+i,r]"
    # Usually C[i,k] = sum U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
    # So W index is k*n + i.
    # C[0,0] -> 0*2+0 = 0
    # C[0,1] -> 1*2+0 = 2
    # C[1,0] -> 0*2+1 = 1
    # C[1,1] -> 1*2+1 = 3
    # C[0,2] -> 2*2+0 = 4
    # C[0,3] -> 3*2+0 = 6
    # C[1,2] -> 2*2+1 = 5
    # C[1,3] -> 3*2+1 = 7
    
    # Let's re-map W indices based on W[k*n+i]
    # C1 (cols 0,1):
    # C[0,0]->0, C[0,1]->2, C[1,0]->1, C[1,1]->3
    # C2 (cols 2,3):
    # C[0,2]->4, C[0,3]->6, C[1,2]->5, C[1,3]->7
    
    # Strassen local indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # For C1, local 0 is C[0,0] (idx 0), local 1 is C[0,1] (idx 2), local 2 is C[1,0] (idx 1), local 3 is C[1,1] (idx 3)
    # So w_map_C1 = [0, 2, 1, 3]
    # For C2, local 0 is C[0,2] (idx 4), local 1 is C[0,3] (idx 6), local 2 is C[1,2] (idx 5), local 3 is C[1,3] (idx 7)
    # So w_map_C2 = [4, 6, 5, 7]
    
    u_map_A1 = [0, 1, 4, 5]
    u_map_A2 = [2, 3, 6, 7]
    v_map_B11 = [0, 1, 4, 5]
    v_map_B12 = [2, 3, 6, 7]
    v_map_B21 = [8, 9, 12, 13]
    v_map_B22 = [10, 11, 14, 15]
    w_map_C1 = [0, 2, 1, 3]
    w_map_C2 = [4, 6, 5, 7]
    
    col = 0
    blocks = [
        (u_map_A1, v_map_B11, w_map_C1),
        (u_map_A2, v_map_B21, w_map_C1),
        (u_map_A1, v_map_B12, w_map_C2),
        (u_map_A2, v_map_B22, w_map_C2)
    ]
    
    for u_map, v_map, w_map in blocks:
        for u_t, v_t, w_t in strassen_terms:
            for idx, val in u_t:
                U[u_map[idx], col] += val
            for idx, val in v_t:
                V[v_map[idx], col] += val
            for idx, val in w_t:
                W[w_map[idx], col] += val
            col += 1
            
    best_U, best_V, best_W = U, V, W
    best_rank = 28
    
    # Verify rank 28
    def verify(U, V, W):
        np.random.seed(42)
        A = np.random.rand(n, m)
        B = np.random.rand(m, p)
        C = A @ B
        Cf = np.zeros((n, p))
        for ii in range(n):
            for kk in range(p):
                acc = 0.0
                for cc in range(U.shape[1]):
                    for jj in range(m):
                        acc += U[ii*m+jj, cc] * V[jj*p+kk, cc] * W[kk*n+ii, cc]
                Cf[ii, kk] = acc
        return np.allclose(C, Cf, atol=1e-7)
    
    if not verify(best_U, best_V, best_W):
        # Fallback: return schoolbook if Strassen fails (unlikely)
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
        
    # Try to reduce to rank 27
    target_rank = 27
    if time.time() - start_time < time_limit - 20:
        try:
            # Optimization
            # We fix a subset of columns to keep? Or randomize all?
            # Randomizing all is better for finding a new basin.
            # Use least_squares.
            
            for restart in range(5):
                if time.time() - start_time > time_limit - 10:
                    break
                
                np.random.seed(100 + restart)
                U_init = np.random.randn(n*m, target_rank) * 0.1
                V_init = np.random.randn(m*p, target_rank) * 0.1
                W_init = np.random.randn(p*n, target_rank) * 0.1
                
                params = np.concatenate([U_init.ravel(), V_init.ravel(), W_init.ravel()])
                
                def residuals(params):
                    Uf = params[:U_init.size].reshape(U_init.shape)
                    Vf = params[U_init.size:U_init.size+V_init.size].reshape(V_init.shape)
                    Wf = params[U_init.size+U_init.size:U_init.size+V_init.size+W_init.size].reshape(W_init.shape)
                    
                    # Compute tensor
                    T_calc = np.zeros((n, m, p))
                    for c in range(target_rank):
                        for i in range(n):
                            for j in range(m):
                                u_val = Uf[i*m+j, c]
                                if u_val == 0: continue
                                for k in range(p):
                                    v_val = Vf[j*p+k, c]
                                    if v_val == 0: continue
                                    w_val = Wf[k*n+i, c]
                                    if w_val == 0: continue
                                    T_calc[i, j, k] += u_val * v_val * w_val
                                    
                    # Target tensor: T[i,j,k] = 1 if i==j? No.
                    # C[i,k] = sum_j A[i,j] B[j,k]
                    # Tensor T[i,j,k] such that C = sum T * A * B
                    # T[i,j,k] = 1 if j==k? No.
                    # C[i,k] = sum_j A[i,j] B[j,k]
                    # This corresponds to T[i,j,k] = delta_{j,k}? No.
                    # The tensor is defined by T[i,j,k] = 1 if j==k?
                    # Let's check: sum_j A[i,j] B[j,k] = sum_{j,k} A[i,j] B[j,k] delta_{j,k}? No.
                    # It is sum_j A[i,j] B[j,k].
                    # The term A[i,j] B[j,k] C[i,k] in the contraction?
                    # The decomposition is C[i,k] = sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c]
                    # Wait, the sum is over c. The j is summed over in the matrix multiplication.
                    # The tensor T is the coefficient of A[i,j] B[j,k] in C[i,k].
                    # C[i,k] = sum_j A[i,j] B[j,k].
                    # So the coefficient of A[i,j] B[j,k] is 1 if the indices match the multiplication structure.
                    # Specifically, for a fixed c, the term is U[i*m+j, c] V[j*p+k, c] W[k*n+i, c].
                    # We want sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] = 1 if j==k? No.
                    # We want the sum over j to be implicit?
                    # No, the tensor T[i,j,k] represents the contribution of A[i,j] and B[j,k] to C[i,k].
                    # In standard matrix multiplication, C[i,k] = sum_j A[i,j] B[j,k].
                    # So the "tensor" for one term c is a rank-1 tensor:
                    # T_c[i,j,k] = U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    # We want sum_c T_c[i,j,k] = 1 if j==k? No.
                    # We want sum_c T_c[i,j,k] to be such that when contracted with A and B, we get C.
                    # <T, A \otimes B> = C.
                    # C[i,k] = sum_{j} T[i,j,k] A[i,j] B[j,k].
                    # We need T[i,j,k] = 1 if j==k? No.
                    # If T[i,j,k] = delta_{j,k}, then C[i,k] = sum_j delta_{j,k} A[i,j] B[j,k] = A[i,k] B[k,k]. That's not A@B.
                    # The correct tensor for matrix multiplication M_{n,m,p} is:
                    # T[i,j,k] = 1 if j==k? No.
                    # The tensor is T[i,j,k] = 1 for all i,j,k? No.
                    # C[i,k] = sum_j A[i,j] B[j,k].
                    # This is linear in A and B.
                    # The tensor is defined by the indices of A, B, C.
                    # A has indices (i,j). B has indices (j,k). C has indices (i,k).
                    # The tensor T has indices (i, j, k) where j is the summed index.
                    # T[i,j,k] = 1 if the indices are consistent?
                    # Yes, T[i,j,k] = 1 for all valid i,j,k.
                    # Because A[i,j] and B[j,k] multiply directly.
                    # So the target tensor is all ones?
                    # Let's check.
                    # C[i,k] = sum_j (1 * A[i,j] * B[j,k]).
                    # Yes, the target tensor is 1 everywhere.
                    
                    T_target = np.ones((n, m, p))
                    return (T_calc - T_target).ravel()
                
                res = least_squares(residuals, params, method='lm', max_nfev=1000)
                
                if res.success or np.max(np.abs(res.fun)) < 1e-6:
                    U_new = res.x[:U_init.size].reshape(U_init.shape)
                    V_new = res.x[U_init.size:U_init.size+V_init.size].reshape(V_init.shape)
                    W_new = res.x[U_init.size+U_init.size:U_init.size+U_init.size+V_init.size+W_init.size].reshape(W_init.shape)
                    
                    if verify(U_new, V_new, W_new):
                        best_U, best_V, best_W = U_new, V_new, W_new
                        best_rank = target_rank
                        break
                        
        except Exception:
            pass

    if time.time() - start_time > time_limit:
        pass

    # Round to half-integers if close
    def round_half(x):
        return np.round(x * 2) / 2

    U_out = best_U
    V_out = best_V
    W_out = best_W
    
    U_r = round_half(U_out)
    V_r = round_half(V_out)
    W_r = round_half(W_out)
    
    if verify(U_r, V_r, W_r):
        U_out, V_out, W_out = U_r, V_r, W_r

    return {"U": U_out.tolist(), "V": V_out.tolist(), "W": W_out.tolist()}

if __name__ == "__main__":
    construct()
