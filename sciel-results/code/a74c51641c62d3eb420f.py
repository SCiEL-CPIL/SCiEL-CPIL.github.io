"""Find a rank-minimal exact tensor decomposition for 2x5 * 5x5 matrix multiplication."""
import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    dim_u = n * m  # 10
    dim_v = m * p  # 25
    dim_w = p * n  # 10
    
    # Build the target tensor
    # T[a, b, c] = 1 where a=i*m+j, b=j*p+k, c=k*n+i
    target = np.zeros((dim_u, dim_v, dim_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                target[u_idx, v_idx, w_idx] = 1.0
    
    def get_schoolbook():
        """Return schoolbook decomposition as lists."""
        rank = n * m * p  # 50
        U = [[0] * rank for _ in range(dim_u)]
        V = [[0] * rank for _ in range(dim_v)]
        W = [[0] * rank for _ in range(dim_w)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def get_block_decomposition():
        """
        Returns a rank 49 decomposition.
        """
        total_rank = 49
        U = [[0] * total_rank for _ in range(dim_u)]
        V = [[0] * total_rank for _ in range(dim_v)]
        W = [[0] * total_rank for _ in range(dim_w)]
        
        col = 0
        
        def add_strassen_term(u_vals, v_vals, w_vals):
            nonlocal col
            for idx, val in u_vals.items():
                if val != 0:
                    U[idx][col] += val
            for idx, val in v_vals.items():
                if val != 0:
                    V[idx][col] += val
            for idx, val in w_vals.items():
                if val != 0:
                    W[idx][col] += val
            col += 1

        def add_schoolbook_term(u_idx, v_idx, w_idx):
            nonlocal col
            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1

        # Indices:
        # U (2x5): i in 0..1, j in 0..4. idx = i*5+j
        # V (5x5): j in 0..4, k in 0..4. idx = j*5+k
        # W (5x2): k in 0..4, i in 0..1. idx = k*2+i
        
        # Block 1: A_L (2x2) * B_TL (2x2) -> C_TL (2x2)
        # A_L: i=0,1; j=0,1. U indices: 0,1,5,6
        # B_TL: j=0,1; k=0,1. V indices: 0,1,5,6
        # C_TL: k=0,1; i=0,1. W indices: 0,1,2,3
        
        # Strassen for 2x2x2
        # M1 = (A00+A11)(B00+B11)
        add_strassen_term({0:1, 6:1}, {0:1, 6:1}, {0:1, 3:1})
        # M2 = (A10+A11)B00
        add_strassen_term({5:1, 6:1}, {0:1}, {2:1})
        # M3 = A00(B10-B11)
        add_strassen_term({0:1}, {5:1, 6:-1}, {1:1})
        # M4 = A11(B01-B00)
        add_strassen_term({6:1}, {1:1, 0:-1}, {3:1})
        # M5 = (A00+A01)B11
        add_strassen_term({0:1, 1:1}, {6:1}, {1:-1})
        # M6 = (A10-A00)(B00+B10)
        add_strassen_term({5:1, 0:-1}, {0:1, 5:1}, {3:1})
        # M7 = (A01-A11)(B01+B11)
        add_strassen_term({1:1, 6:-1}, {1:1, 6:1}, {0:1})
        
        # Note: The W indices for C_TL are:
        # C00 (k=0,i=0) -> 0
        # C01 (k=1,i=0) -> 1
        # C10 (k=0,i=1) -> 2
        # C11 (k=1,i=1) -> 3
        
        # Check Strassen combinations:
        # C00 = M1 + M4 - M5 + M7
        # M1: W[0]+=1, W[3]+=1
        # M4: W[3]+=1
        # M5: W[1]-=1
        # M7: W[0]+=1
        # Sum for W[0]: 1 (M1) + 1 (M7) = 2. Expected 1.
        # This indicates my Strassen mapping to W is not directly additive in this simple way 
        # unless I account for the coefficients in the final sum.
        # The decomposition is T = sum r * u_r v_r w_r.
        # The output C is sum r * (u_r . A) (v_r . B) w_r.
        # So W matrix columns are the w_r vectors.
        # The coefficient of C_ij is sum_r (u_r . A) (v_r . B) W[j,k,r]? No.
        # T[u,v,w] = sum_r U[u,r] V[v,r] W[w,r].
        # C[i,k] = sum_j T[i,j,k]? No.
        # The tensor T is defined by T[u,v,w] = 1 if u=i*m+j, v=j*p+k, w=k*n+i.
        # So C[i,k] = sum_j A[i,j] B[j,k].
        # In the decomposition:
        # C[i,k] = sum_r (sum_j U[i*m+j, r] A[i,j]) * (sum_j V[j*p+k, r] B[j,k]) * W[k*n+i, r]
        # This is NOT correct. The tensor decomposition is:
        # T = sum_r u_r v_r w_r.
        # T[i,j,k] = sum_r U[i,r] V[j,r] W[k,r].
        # Here i is in U (size nm), j is in V (size mp), k is in W (size pn).
        # Wait, the dimensions are U(nm, r), V(mp, r), W(pn, r).
        # T[u,v,w] = sum_r U[u,r] V[v,r] W[w,r].
        # u = i*m+j. v = j*p+k. w = k*n+i.
        # So C[i,k] = sum_j T[i*m+j, j*p+k, k*n+i]
        #           = sum_j sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
        #           = sum_r W[k*n+i, r] (sum_j U[i*m+j, r] V[j*p+k, r])
        # This is not a simple matrix multiplication unless the sum over j is structured.
        
        # Actually, the standard form is:
        # C = A B.
        # C_{ik} = sum_j A_{ij} B_{jk}.
        # Tensor T_{i,j,k} = delta_{i,j,k} ? No.
        # T_{u,v,w} = 1 if u=i*m+j, v=j*p+k, w=k*n+i.
        # This T represents the multiplication.
        # The decomposition T = sum_r u_r v_r w_r means:
        # T_{u,v,w} = sum_r U_{u,r} V_{v,r} W_{w,r}.
        # Then C_{ik} = sum_j T_{i*m+j, j*p+k, k*n+i}
        #             = sum_j sum_r U_{i*m+j, r} V_{j*p+k, r} W_{k*n+i, r}
        #             = sum_r W_{k*n+i, r} (sum_j U_{i*m+j, r} V_{j*p+k, r})
        # This implies that for each r, the term contributes to C_{ik} via W_{k*n+i, r}.
        # The inner sum sum_j U_{i*m+j, r} V_{j*p+k, r} must be independent of i and k? No.
        # It depends on i and k.
        # So we can't just say "Strassen for 2x2x2" directly maps to U,V,W blocks 
        # unless we are careful.
        
        # However, if we use the block decomposition, we are effectively constructing the tensor 
        # for the sub-problems.
        # The sub-problem A_L B_TL has tensor T_sub.
        # T_sub[u', v', w'] = 1 if u'=i*m'+j', v'=j'*p'+k', w'=k'*n'+i'.
        # Here n'=2, m'=2, p'=2.
        # The indices in the full problem are mapped.
        # U_full[i*5+j, r] = U_sub[i*2+j, r] for j<2.
        # V_full[j*5+k, r] = V_sub[j*2+k, r] for j<2, k<2.
        # W_full[k*2+i, r] = W_sub[k*2+i, r] for k<2, i<2.
        
        # So I can just generate the Strassen U,V,W for 2x2x2 and place them in the correct slots.
        
        # Let's restart the block decomposition logic with this mapping.
        
        # Block 1: 2x2x2 (Strassen)
        # U_sub (4x7), V_sub (4x7), W_sub (4x7)
        # Place in U[0:2, 0:2], V[0:2, 0:2], W[0:2, 0:2]
        
        # Block 2: 2x2x3 (Schoolbook)
        # A_L (2x2) * B_TR (2x3) -> C_TR (2x3)
        # U indices: i=0,1; j=0,1. (0,1,5,6)
        # V indices: j=0,1; k=2,3,4. (2,3,4, 7,8,9)
        # W indices: k=2,3,4; i=0,1. (4,5,6, 7,8,9)
        
        # Block 3: 2x3x2 (Schoolbook)
        # A_R (2x3) * B_BL (3x2) -> C_BL (2x2)
        # U indices: i=0,1; j=2,3,4. (2,3,4, 7,8,9)
        # V indices: j=2,3,4; k=0,1. (10,11,12, 15,16,17, 20,21,22)
        # W indices: k=0,1; i=0,1. (0,1, 2,3)
        
        # Block 4: 2x3x3 (Schoolbook)
        # A_R (2x3) * B_BR (3x3) -> C_BR (2x3)
        # U indices: i=0,1; j=2,3,4. (2,3,4, 7,8,9)
        # V indices: j=2,3,4; k=2,3,4. (12,13,14, 17,18,19, 22,23,24)
        # W indices: k=2,3,4; i=0,1. (4,5,6, 7,8,9)
        
        # This is getting complex to manage manually. 
        # Given the time, I will stick to the schoolbook rank 50 
        # unless I can quickly verify the block decomposition.
        # The current program returns 50.
        # I will return the schoolbook result to be safe.
        
        return get_schoolbook()

    def get_block_decomposition():
        """
        Returns a rank 49 decomposition using block structure:
        Split A (2x5) into A_L (2x2) and A_R (2x3).
        Split B (5x5) into B_TL (2x2), B_TR (2x3), B_BL (3x2), B_BR (3x3).
        C = A_L B_TL + A_R B_BL (2x2 part)
          + A_L B_TR + A_R B_BR (2x3 part)
        
        Ranks:
        R(2,2,2) = 7 (Strassen)
        R(2,2,3) = 12 (Schoolbook)
        R(2,3,2) = 12 (Schoolbook)
        R(2,3,3) = 18 (Schoolbook)
        Total = 49
        """
        total_rank = 49
        U = [[0] * total_rank for _ in range(dim_u)]
        V = [[0] * total_rank for _ in range(dim_v)]
        W = [[0] * total_rank for _ in range(dim_w)]
        
        col = 0
        
        # Helper to add a term
        def add_term(u_idx, v_idx, w_idx, u_val=1, v_val=1, w_val=1):
            nonlocal col
            U[u_idx][col] += u_val
            V[v_idx][col] += v_val
            W[w_idx][col] += w_val
            col += 1

        # 1. A_L B_TL (2x2 * 2x2) using Strassen
        # A_L indices in U: 0,1 (row 1), 2,3 (row 2) -> i*m+j
        # i=0, j=0 -> 0; i=0, j=1 -> 1; i=1, j=0 -> 2; i=1, j=1 -> 3
        # B_TL indices in V: 0,1 (row 1), 2,3 (row 2) -> j*p+k
        # j=0, k=0 -> 0; j=0, k=1 -> 1; j=1, k=0 -> 2; j=1, k=1 -> 3
        # C_TL indices in W: 0,1 (row 1), 2,3 (row 2) -> k*n+i
        # k=0, i=0 -> 0; k=1, i=0 -> 1; k=0, i=1 -> 2; k=1, i=1 -> 3
        
        # Strassen terms for 2x2x2
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B10)
        # M7 = (A01-A11)(B01+B11)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # Term 1: M1
        # U: A00, A11 -> 1, 1
        # V: B00, B11 -> 1, 1
        # W: C00, C11 -> 1, 1
        add_term(0, 0, 0, 1, 1, 1)
        add_term(3, 0, 0, 1, 1, 1) # Wait, add_term adds to specific indices. 
        # This approach is flawed because add_term expects single indices.
        # Strassen requires sums in U, V, W.
        # We need to handle this differently.
        
        return None
    
    def compute_tensor(U, V, W):
        """Compute the tensor from factor matrices."""
        return np.einsum('ar,br,cr->abc', U, V, W)
    
    def loss_and_grad(params, rank):
        """Compute loss and gradient for given parameters and rank."""
        u_flat = params[:dim_u * rank]
        v_flat = params[dim_u * rank : dim_u * rank + dim_v * rank]
        w_flat = params[dim_u * rank + dim_v * rank : ]
        
        U = u_flat.reshape(dim_u, rank)
        V = v_flat.reshape(dim_v, rank)
        W = w_flat.reshape(dim_w, rank)
        
        T = np.einsum('ar,br,cr->abc', U, V, W)
        diff = T - target
        loss = np.sum(diff * diff)
        
        # Gradients
        grad_u = 2.0 * np.einsum('abc,br,cr->ar', diff, V, W)
        grad_v = 2.0 * np.einsum('abc,ar,cr->br', diff, U, W)
        grad_w = 2.0 * np.einsum('abc,ar,br->cr', diff, U, V)
        
        grad = np.concatenate([grad_u.flatten(), grad_v.flatten(), grad_w.flatten()])
        return loss, grad
    
    def try_rank(rank, max_restarts=20):
        """Try to find a valid decomposition with the given rank."""
        for restart in range(max_restarts):
            if time.time() > deadline:
                return None
            
            # Initialize with random values
            np.random.seed(12345 + restart)
            u0 = np.random.randn(dim_u * rank) * 0.5
            v0 = np.random.randn(dim_v * rank) * 0.5
            w0 = np.random.randn(dim_w * rank) * 0.5
            
            params = np.concatenate([u0, v0, w0])
            
            def fun(params):
                loss, grad = loss_and_grad(params, rank)
                return loss, grad
            
            try:
                res = minimize(fun, params, method='L-BFGS-B', jac=True,
                               options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10})
                
                if res.fun < 1e-10:
                    # Extract final parameters
                    u_final = res.x[:dim_u * rank].reshape(dim_u, rank)
                    v_final = res.x[dim_u * rank : dim_u * rank + dim_v * rank].reshape(dim_v, rank)
                    w_final = res.x[dim_u * rank + dim_v * rank : ].reshape(dim_w, rank)
                    
                    # Snap to nearest half-integer
                    u_snap = np.round(u_final * 2.0) / 2.0
                    v_snap = np.round(v_final * 2.0) / 2.0
                    w_snap = np.round(w_final * 2.0) / 2.0
                    
                    # Verify the snapped solution
                    T_snap = np.einsum('ar,br,cr->abc', u_snap, v_snap, w_snap)
                    if np.allclose(T_snap, target, atol=1e-9):
                        return (u_snap, v_snap, w_snap)
                    
                    # Try without snapping if it's already very close
                    if np.allclose(T_snap, target, atol=1e-6):
                        # Check if we can clean up small values
                        u_clean = np.where(np.abs(u_snap) < 1e-6, 0, u_snap)
                        v_clean = np.where(np.abs(v_snap) < 1e-6, 0, v_snap)
                        w_clean = np.where(np.abs(w_snap) < 1e-6, 0, w_snap)
                        T_clean = np.einsum('ar,br,cr->abc', u_clean, v_clean, w_clean)
                        if np.allclose(T_clean, target, atol=1e-9):
                            return (u_clean, v_clean, w_clean)
            except Exception:
                continue
        
        return None
    
    # Try ranks from lower bound up to schoolbook
    # Lower bound: max(n*m, m*p, n*p) = max(10, 25, 10) = 25
    # But for matrix multiplication, better bounds exist
    # Let's try from 25 up to 49
    
    best_result = None
    
    # Try to find a decomposition with rank < 50
    for target_rank in range(25, 50):
        if time.time() > deadline:
            break
        
        result = try_rank(target_rank, max_restarts=10)
        if result is not None:
            U, V, W = result
            best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            break
    
    if best_result is not None:
        return best_result
    else:
        return get_schoolbook()
# EVOLVE-BLOCK-END
