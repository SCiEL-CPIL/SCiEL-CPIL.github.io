import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 4, 4]

def construct():
    start_time = time.time()
    deadline = 140  # 150s limit, keep 10s safety margin
    
    n, m, p = DIMENSIONS
    # n=2, m=4, p=4
    # U shape: (n*m, rank) = (8, rank)
    # V shape: (m*p, rank) = (16, rank)
    # W shape: (p*n, rank) = (8, rank)
    
    # The tensor T for matrix multiplication C = A * B
    # C_ik = sum_j A_ij B_jk
    # In tensor decomposition: T[i, j, k] = sum_r U[i, r] V[j, r] W[k, r]
    # Here indices are flattened:
    # i in [0, n*m), j in [0, m*p), k in [0, p*n)
    # Actually, the standard definition for the multiplication tensor M_{n,m,p} is:
    # T_{i, j, k} = 1 if the term A_{i, j} B_{j, k} contributes to C_{i, k}.
    # In the flattened representation used by the evaluator (implied by U, V, W shapes):
    # U corresponds to A indices (i, j_A) -> row i*m + j_A
    # V corresponds to B indices (j_B, k) -> row j_B*p + k
    # W corresponds to C indices (i, k) -> row i*p + k ? 
    # Wait, the problem states:
    # U shape (n*m, rank)
    # V shape (m*p, rank)
    # W shape (p*n, rank)
    # And "output factor follows the official trace convention W[k*n+i,r]"
    # This implies W index is k*n + i.
    # So W row index w = k*n + i.
    #
    # The condition for exact decomposition is:
    # For all valid (i, j, k) where A_ij and B_jk are multiplied:
    # The term A_{i, j} B_{j, k} appears in C_{i, k}.
    #
    # Let's define the target tensor T.
    # T is a 3-tensor of shape (n*m, m*p, p*n).
    # T[row_U, row_V, row_W] = 1 if the bilinear form U[row_U] V[row_V] W[row_W] 
    # corresponds to a valid multiplication term.
    #
    # A_{i, j} is at U row: i*m + j. (i in 0..n-1, j in 0..m-1)
    # B_{j, k} is at V row: j*p + k. (j in 0..m-1, k in 0..p-1)
    # C_{i, k} is at W row: k*n + i. (i in 0..n-1, k in 0..p-1)
    #
    # The multiplication C = A B implies:
    # C_{i, k} = sum_{j=0}^{m-1} A_{i, j} B_{j, k}
    #
    # In the tensor decomposition sum_r U[i,m+j, r] V[j,p+k, r] W[k,n+i, r]:
    # We need the sum over r to equal 1 if the indices (i, j, k) form a valid term in the product, 
    # and 0 otherwise? 
    # No, the tensor T is defined such that T_{u, v, w} = 1 if the term A_{i,j} B_{j,k} -> C_{i,k} is present.
    # Specifically, for each j, the term A_{i,j} B_{j,k} contributes to C_{i,k}.
    # So for a fixed j, the "slice" of the tensor along the V dimension index (j*p + k) and U dimension (i*m + j) 
    # must match the W dimension (k*n + i).
    #
    # Actually, the standard definition is:
    # T_{u, v, w} = 1 if u corresponds to A_{i, j}, v corresponds to B_{j, k}, and w corresponds to C_{i, k}.
    # Note that j must be the same in A and B.
    #
    # Let's construct the target tensor T.
    size_u = n * m
    size_v = m * p
    size_w = p * n
    T = np.zeros((size_u, size_v, size_w))
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0
                
    # We want U, V, W such that:
    # np.einsum('ur, vr, wr -> uvw', U, V, W) == T
    
    # Strategy 1: Construct Rank 28 via Block Strassen
    # The 2x4x4 multiplication can be viewed as two 2x2x2 multiplications 
    # if we split the 4 columns of A and 4 rows of B? 
    # A is 2x4. B is 4x4.
    # Let A = [A1, A2] where A1, A2 are 2x2.
    # Let B = [[B11, B12], [B21, B22]] where Bij are 2x2.
    # C = A B = [A1 B11 + A2 B21, A1 B12 + A2 B22].
    # C is 2x4. Let C = [C1, C2] where C1, C2 are 2x2.
    # C1 = A1 B11 + A2 B21
    # C2 = A1 B12 + A A2 B22
    
    # This is a sum of two 2x2x2 multiplications.
    # Rank of sum of two independent 2x2x2 is 7 + 7 = 14? 
    # Wait, A1, A2 are shared. B11, B21 are for C1. B12, B22 are for C2.
    # The tensors are not independent because A is shared.
    # However, we can use Strassen for each 2x2 block if we treat them appropriately.
    
    # Let's use a known Rank 28 construction.
    # Rank of 2x4x4 is known to be 25.
    # Rank 28 is achievable by "Block Strassen" or similar.
    # Actually, a simpler approach:
    # Use Rank 7 Strassen for 2x2x2.
    # We have 2x4x4.
    # Split B into 4 columns? No.
    
    # Let's try to construct Rank 28 manually using the structure.
    # If we just use schoolbook, rank is 32.
    # Can we get 28?
    # 2x4x4 can be seen as 2x2x4 + 2x2x4? No.
    
    # Let's use the fact that R(2,4,4) <= R(2,2,4) + R(2,2,4)?
    # R(2,2,4) = 10? (Laderman for 2x2x4? No, Laderman is 2x3x3=14).
    # R(2,2,4) is 10? 
    # If R(2,2,4) = 10, then R(2,4,4) <= 20? No, that's too low.
    # R(2,4,4) is known to be 25.
    # R(2,2,4) is 10.
    # If we split A into two 2x2 matrices A1, A2.
    # C = A1 B_top + A2 B_bottom.
    # B_top is 2x4, B_bottom is 2x4.
    # C1 = A1 B_top (2x4 result)
    # C2 = A2 B_bottom (2x4 result)
    # C = [C1 + C2] ? No, C is 2x4.
    # C_ik = sum_{j=0}^3 A_ij B_jk.
    # Let j=0,1 be top, j=2,3 be bottom.
    # C_ik = A_i0 B_0k + A_i1 B_1k + A_i2 B_2k + A_i3 B_3k.
    # = (A_i0 B_0k + A_i1 B_1k) + (A_i2 B_2k + A_i3 B_3k).
    # First part is 2x2x4 multiplication (A_top 2x2, B_top 2x4, C_part1 2x4).
    # Second part is 2x2x4 multiplication (A_bot 2x2, B_bot 2x4, C_part2 2x4).
    # Since A_top and A_bot are disjoint rows of A? No, same rows i, different columns j.
    # So A_top = A[:, 0:2], A_bot = A[:, 2:4].
    # These are independent variables.
    # B_top = B[0:2, :], B_bot = B[2:4, :].
    # These are independent variables.
    # C_part1 and C_part2 are added to form C.
    # Since the operations are linear and the variables are disjoint (except C is shared sum),
    # we can just sum the ranks.
    # Rank(2,2,4) is 10.
    # So Rank(2,4,4) <= 10 + 10 = 20?
    # Wait, is R(2,2,4) = 10?
    # Yes, R(2,2,4) = 10.
    # So R(2,4,4) <= 20.
    # But known bound is 25? 
    # Let's check.
    # R(2,4,4) lower bound is 25.
    # Why is my decomposition 20?
    # C = C1 + C2.
    # C1 = A_top B_top.
    # C2 = A_bot B_bot.
    # The tensor is T = T1 + T2.
    # T1 involves A_top, B_top, C.
    # T2 involves A_bot, B_bot, C.
    # The C factor is the SAME for both.
    # In tensor decomposition, if we share the C factor, can we just add ranks?
    # Sum of tensors T1 + T2.
    # T1 = sum_{r=1}^{10} u1_r v1_r w_r
    # T2 = sum_{s=1}^{10} u2_s v2_s w_s
    # T = sum u1 v1 w + sum u2 v2 w.
    # This is a valid decomposition of rank 20 if the w_r are the same?
    # No, the w_r for T1 and T2 might be different.
    # If we use the same w basis, we need to express both in the same basis.
    # But we can just take the union of the rank-1 terms.
    # T = sum_{r=1}^{10} (u1_r, v1_r, w1_r) + sum_{s=1}^{10} (u2_s, v2_s, w2_s).
    # This is a decomposition of rank 20.
    # So R(2,4,4) <= 20.
    # But literature says R(2,4,4) = 25?
    # Let me check R(2,2,4).
    # R(2,2,4) = 10.
    # If R(2,4,4) <= 20, then 25 is wrong or my logic is flawed.
    # Flaw: C is the output.
    # C_ik = C1_ik + C2_ik.
    # In the tensor T, the third index is C_ik.
    # For T1, the term is A_top_i,j B_top_j,k -> C_ik.
    # For T2, the term is A_bot_i,j' B_bot_j',k -> C_ik.
    # The C_ik is the same index in the tensor.
    # So the decomposition is indeed a sum of rank-1 tensors.
    # T = sum_{r} u1_r v1_r w1_r + sum_{s} u2_s v2_s w2_s.
    # This is a valid decomposition of rank 20.
    # So R(2,4,4) <= 20.
    # Wait, is R(2,2,4) really 10?
    # Yes, R(2,2,4) = 10.
    # So R(2,4,4) <= 20.
    # Actually, R(2,4,4) is known to be 20?
    # Let's assume we can get Rank 20.
    
    # Let's construct Rank 10 for 2x2x4.
    # A is 2x2, B is 2x4, C is 2x4.
    # C = A B.
    # C1 = A B1 (2x2), C2 = A B2 (2x2).
    # This is two 2x2x2 multiplications.
    # Rank 7 + 7 = 14?
    # But A is shared.
    # C1 = A B1, C2 = A B2.
    # We can use Strassen for 2x2x2.
    # M1..M7 for B1.
    # M1..M7 for B2.
    # Total 14.
    # Can we do better?
    # R(2,2,4) = 10 is a known result.
    # Construction:
    # Use 10 bilinear forms.
    
    # Given the time constraint and complexity of finding the exact Rank 10/20 coefficients,
    # and the fact that the current program scored 1.0 (Rank 32),
    # I will implement a **verified Rank 28** construction using a simpler block method
    # and then **Local Search** to reduce it.
    
    # Actually, let's just use **Rank 32** as the safe fallback and try to find **Rank 28** 
    # using a specific known decomposition for 2x4x4 if I can recall it, 
    # or use **Random Search** to find a valid decomposition of Rank 28.
    
    # Better Strategy:
    # 1. Start with Rank 32 Schoolbook.
    # 2. Use Local Search to try to reduce rank by setting some columns to zero 
    #    and adjusting others? No, that's not how rank reduction works.
    
    # Let's use **Symbolic Verification** to ensure correctness.
    
    def verify(U, V, W):
        # U: (8, R), V: (16, R), W: (8, R)
        # Compute T_calc = np.einsum('ur, vr, wr -> uvw', U, V, W)
        # Check if T_calc == T
        # For speed, we can check a subset or use matrix multiplication.
        # T_calc[u, v, w] = sum_r U[u, r] V[v, r] W[w, r]
        # This is a 3D tensor.
        # We can flatten T into a matrix and check?
        # T is sparse.
        # Let's just check the non-zero entries of T.
        # There are n*m*p = 32 non-zero entries.
        # For each non-zero entry (u, v, w):
        # sum_r U[u, r] V[v, r] W[w, r] should be 1.
        # For all other entries, it should be 0.
        # Checking all 8*16*8 = 1024 entries is fast.
        
        T_calc = np.einsum('ur, vr, wr -> uvw', U, V, W)
        return np.allclose(T_calc, T, atol=1e-6)

    # Fallback: Rank 32
    U32 = np.zeros((8, 32))
    V32 = np.zeros((16, 32))
    W32 = np.zeros((8, 32))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U32[u_idx, col] = 1
                V32[v_idx, col] = 1
                W32[w_idx, col] = 1
                col += 1
                
    best_U, best_V, best_W = U32, V32, W32
    best_rank = 32
    
    # Attempt Rank 28 via Local Search on a Random Initialization
    # We initialize U, V, W with random values and minimize the error.
    # Error = sum (T_calc - T)^2
    
    def error_func(params):
        # params: flattened U, V, W
        # R = 28
        R = 28
        size_u, size_v, size_w = 8, 16, 8
        U_flat, V_flat, W_flat = params[:size_u*R], params[size_u*R:size_u*R+size_v*R], params[size_u*R+size_v*R:]
        U = U_flat.reshape(size_u, R)
        V = V_flat.reshape(size_v, R)
        W = W_flat.reshape(size_w, R)
        T_calc = np.einsum('ur, vr, wr -> uvw', U, V, W)
        err = np.sum((T_calc - T)**2)
        return err
    
    # Try a few random restarts
    np.random.seed(42)
    found = False
    for attempt in range(10):
        if time.time() > deadline - 20: break
        
        R = 28
        size_u, size_v, size_w = 8, 16, 8
        U_init = np.random.randn(size_u, R)
        V_init = np.random.randn(size_v, R)
        W_init = np.random.randn(size_w, R)
        
        params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        res = minimize(error_func, params_init, method='Nelder-Mead', options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8})
        
        if res.success and res.fun < 1e-4:
            # Verify
            U_sol = res.x[:size_u*R].reshape(size_u, R)
            V_sol = res.x[size_u*R:size_u*R+size_v*R].reshape(size_v, R)
            W_sol = res.x[size_u*R+size_v*R:].reshape(size_w, R)
            
            if verify(U_sol, V_sol, W_sol):
                best_U, best_V, best_W = U_sol, V_sol, W_sol
                best_rank = R
                found = True
                break
                
    # If Rank 28 not found, try Rank 30? 
    # Or just return Rank 32.
    # Given the difficulty, Rank 32 is safe.
    # But we can try to improve Rank 32 to Rank 28 using a smarter initialization.
    
    # Alternative: Use the Rank 10 decomposition for 2x2x4 to build Rank 20 for 2x4x4.
    # If I can hardcode Rank 10 for 2x2x4, I can build Rank 20.
    # Rank 10 for 2x2x4:
    # A (2x2), B (2x4), C (2x4).
    # This is equivalent to multiplying A by B1 (2x2) and A by B2 (2x2).
    # Using Strassen (Rank 7) for each would be 14.
    # Rank 10 is better.
    # I don't have the Rank 10 coefficients memorized.
    
    # Let's stick to the Random Search for Rank 28. 
    # If it fails, return Rank 32.
    
    # Ensure integer/half-integer constraint?
    # The problem says "integer/half-integer entries".
    # My random search finds real numbers.
    # I need to round to nearest half-integer and re-verify.
    
    def round_to_half_integer(X):
        return np.round(X * 2) / 2.0

    if found:
        U_sol = round_to_half_integer(best_U)
        V_sol = round_to_half_integer(best_V)
        W_sol = round_to_half_integer(best_W)
        if verify(U_sol, V_sol, W_sol):
            best_U, best_V, best_W = U_sol, V_sol, W_sol
        else:
            # Rounding broke it, revert to unrounded if it was valid? 
            # The problem requires integer/half-integer.
            # If rounding breaks it, we must return a valid integer/half-integer solution.
            # Rank 32 is valid and integer.
            best_U, best_V, best_W = U32, V32, W32
            best_rank = 32

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

# EVOLVE-BLOCK-END
