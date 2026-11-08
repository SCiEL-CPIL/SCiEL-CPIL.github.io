"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    # Schoolbook rank
    schoolbook_rank = n * m * p
    best_rank = schoolbook_rank
    best_U = None
    best_V = None
    best_W = None
    
    def initialize_schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W
    
    U_init, V_init, W_init = initialize_schoolbook()
    best_U, best_V, best_W = U_init, V_init, W_init
    
    # Try to find a lower rank decomposition using randomized rank-revealing factorization
    # We are looking for a rank r decomposition where r < n*m*p
    
    # The tensor T_{i,j,k} = delta_{i,j,k} for matrix multiplication A (n x m) * B (m x p) -> C (p x n)
    # Actually, the standard tensor for matrix multiplication M_{n,m,p} has entries:
    # T_{(i,j), (j,k), (k,i)} = 1 for all valid i,j,k
    # Here indices are:
    # U index: i*m + j (size n*m)
    # V index: j*p + k (size m*p)
    # W index: k*n + i (size p*n)
    
    # We want to find rank r such that there exist U (n*m x r), V (m*p x r), W (p*n x r)
    # such that for all i, j, k:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 if (i,j,k) is a valid triple (which it always is in the support)
    # and 0 otherwise? No, the tensor is sparse.
    
    # The tensor T has entries T_{a,b,c} where a = i*m+j, b = j*p+k, c = k*n+i.
    # Note that for a fixed triple (i,j,k), the indices are coupled.
    # Specifically, if we fix a and b, does that determine c?
    # a = i*m + j => i = a // m, j = a % m
    # b = j*p + k => j = b // p, k = b % p
    # For consistency, we must have j from a == j from b.
    # If consistent, then k is determined, and i is determined. Then c = k*n + i is determined.
    # So the tensor is actually a set of 1s at specific positions.
    # Number of non-zero entries is n*m*p.
    
    # We can view this as a matrix M of size (n*m) x (m*p) where M[a, b] = 1 if j_a == j_b, else 0?
    # No, that's not right. The third index W couples i and k.
    
    # Let's try a different approach: Randomized SVD / Tensor Decomposition.
    # We can form the matricizations of the tensor.
    # T_{i,j,k} = 1.
    # Mat-1: (i,j) vs k. Shape (n*m, p).
    #   Entry [(i,j), k] = 1.
    #   This matrix has rank min(n*m, p).
    # Mat-2: (i,k) vs j. Shape (n*p, m).
    #   Entry [(i,k), j] = 1.
    #   Rank min(n*p, m).
    # Mat-3: (j,k) vs i. Shape (m*p, n).
    #   Entry [(j,k), i] = 1.
    #   Rank min(m*p, n).
    
    # The rank of the tensor is at least the max of the ranks of its matricizations.
    # rank(T) >= max(min(n*m, p), min(n*p, m), min(m*p, n))
    # For n=2, m=3, p=8:
    # min(6, 8) = 6
    # min(16, 3) = 3
    # min(24, 2) = 2
    # Lower bound: 6.
    
    # Can we achieve rank 6?
    # If rank is 6, then the mat-1 matricization (6x8) must have rank 6.
    # The mat-1 matrix is:
    # Rows indexed by (i,j), cols by k.
    # Row (i,j) has 1s in all k columns? No.
    # T_{i,j,k} = 1 for all i,j,k.
    # So Mat-1 is a 6x8 matrix of all 1s.
    # Rank of a matrix of all 1s is 1.
    # Wait, my tensor definition is wrong for the matricization logic.
    
    # Let's re-read the tensor definition for matrix multiplication.
    # C_{i,k} = sum_j A_{i,j} B_{j,k}
    # The tensor T has indices (i,j), (j,k), (i,k) ?? No.
    # The bilinear form is sum_{i,j,k} A_{i,j} B_{j,k} W_{i,k,j} C_{i,k} ??
    # The standard decomposition is:
    # C_{i,k} = sum_c (sum_j U_{i,j,c} A_{i,j}) * (sum_j V_{j,k,c} B_{j,k}) * W_{i,k,c}
    # Wait, the standard form is:
    # C_{i,k} = sum_c (U_c^T A) (V_c^T B) W_c
    # Where U_c is n x m, V_c is m x p, W_c is p x n.
    # In the flattened interface:
    # U has shape (n*m, rank). Column c is U_c flattened.
    # V has shape (m*p, rank). Column c is V_c flattened.
    # W has shape (p*n, rank). Column c is W_c flattened.
    
    # The condition for exact decomposition is:
    # For all i, k:
    # sum_c (sum_j U[i*m+j, c]) * (sum_j' V[j'*p+k, c]) * W[k*n+i, c] ... No.
    
    # The bilinear map is:
    # <U_c, A> * <V_c, B> * W_c
    # where <U_c, A> = sum_{i,j} U_{i,j,c} A_{i,j}
    # <V_c, B> = sum_{j,k} V_{j,k,c} B_{j,k}
    
    # The resulting matrix is sum_c W_c * (<U_c, A> <V_c, B>)
    # We want this to equal A B.
    # (AB)_{i,k} = sum_j A_{i,j} B_{j,k}
    
    # So we require:
    # sum_c W_{i,k,c} (sum_{i',j} U_{i',j,c} A_{i',j}) (sum_{j',k'} V_{j',k',c} B_{j',k'}) = sum_j A_{i,j} B_{j,k}
    
    # This must hold for all A, B.
    # This implies that the tensor T_{i,j, j',k', i,k} ... is complicated.
    
    # Let's use the property that the rank of M_{n,m,p} is known for small cases.
    # M_{2,3,8}
    # Lower bound is max(n*m, m*p, p*n) / something? No.
    # Lower bound is max(n, m, p) * min(n, m, p) ? No.
    # A simple lower bound is max(n*m, m*p, p*n) is not true.
    # The lower bound is max(n, m, p) is not true.
    # The lower bound is n*m*p / max(n,m,p)^2 ? No.
    
    # Let's check known ranks.
    # M_{2,2,2} = 7
    # M_{2,3,3} = ?
    # M_{3,3,3} = 23
    
    # For M_{2,3,8}, schoolbook is 48.
    # We can try to use the Winograd algorithm or similar.
    # However, finding the exact minimal rank is hard.
    
    # Strategy:
    # 1. Start with schoolbook.
    # 2. Try to reduce rank by solving a linear system for a smaller rank.
    # 3. Use random initialization for U, V, W of rank r, and solve for W given U, V?
    #    Or solve for one factor?
    
    # Let's try a simple rank reduction.
    # Can we merge some terms?
    # If n=2, m=3, p=8.
    # The structure is small in n and m.
    
    # Let's try rank = n*m*p - 1, etc.
    # Actually, let's try to find a decomposition of rank r by random sampling and least squares.
    # Fix U, V randomly of rank r.
    # Then the equation is linear in W?
    # sum_c W_{i,k,c} (sum_{i',j} U_{i',j,c} A_{i',j}) (sum_{j',k'} V_{j',k',c} B_{j',k'}) = (AB)_{i,k}
    
    # Let x_c = (sum_{i',j} U_{i',j,c} A_{i',j}) (sum_{j',k'} V_{j',k',c} B_{j',k'})
    # Then sum_c W_{i,k,c} x_c = (AB)_{i,k}
    
    # This must hold for all A, B.
    # This is a tensor equation.
    # We can sample random A, B to form a linear system for W.
    
    def try_rank(r):
        if r >= best_rank:
            return None
        
        # Randomly initialize U and V
        # We need to ensure U and V are "generic" enough.
        # Use small integers or random floats?
        # The problem allows integer/half-integer entries.
        # Let's use random integers in {-1, 0, 1} or similar.
        
        U = np.random.randint(-1, 2, size=(n*m, r))
        V = np.random.randint(-1, 2, size=(m*p, r))
        
        # We need to solve for W.
        # The condition is that the tensor defined by U, V, W equals the matmul tensor.
        # We can construct the linear system by sampling random A and B.
        # Let's sample S pairs of (A, B).
        # For each pair, we compute the target C = A B.
        # We compute the "features" for each c:
        # f_c = (<U_c, A>, <V_c, B>)
        # Then we need W such that sum_c W_{i,k,c} f_{c,1} f_{c,2} = C_{i,k}
        
        # This is a linear system in the entries of W.
        # Number of unknowns: r * p * n
        # Number of equations: S * p * n
        
        # Let's try S = 10 or 20.
        S = 20
        A_samples = []
        B_samples = []
        C_samples = []
        F_samples = [] # Shape (S, r, 2) ? Or (S, r) for each factor?
        
        for _ in range(S):
            A = np.random.randint(-1, 2, size=(n, m))
            B = np.random.randint(-1, 2, size=(m, p))
            C = A @ B
            A_samples.append(A)
            B_samples.append(B)
            C_samples.append(C)
            
            # Compute features
            # u_proj = A_flat @ U  (shape r)
            # v_proj = B_flat @ V  (shape r)
            A_flat = A.flatten()
            B_flat = B.flatten()
            u_proj = A_flat @ U
            v_proj = B_flat @ V
            F_samples.append((u_proj, v_proj))
        
        # Now build the linear system for W.
        # W is flattened to (p*n*r).
        # For each sample s, and each (i,k), we have:
        # sum_c W_{i,k,c} * u_proj[s,c] * v_proj[s,c] = C_samples[s][i,k]
        
        # Let's stack all these equations.
        # Total equations: S * p * n
        # Unknowns: p * n * r
        
        # We can build a big matrix M and vector b.
        # M is (S*p*n) x (p*n*r)
        
        # To save memory, we can solve sequentially or use least squares.
        # Let's build the matrix.
        
        num_eqs = S * p * n
        num_unk = p * n * r
        
        M = np.zeros((num_eqs, num_unk))
        b = np.zeros(num_eqs)
        
        idx = 0
        for s in range(S):
            u_proj, v_proj = F_samples[s]
            C = C_samples[s]
            for i in range(n):
                for k in range(p):
                    # W index for (i,k) is k*n + i
                    # The term is sum_c W[k*n+i, c] * u_proj[c] * v_proj[c]
                    # In the flattened W vector, the offset for (i,k) is (k*n+i)*r
                    # The entries are W_flat[(k*n+i)*r : (k*n+i)*r + r]
                    
                    # The coefficients for these r unknowns are u_proj[c] * v_proj[c]
                    coeffs = u_proj * v_proj
                    M[idx, (k*n+i)*r : (k*n+i)*r + r] = coeffs
                    b[idx] = C[i,k]
                    idx += 1
        
        # Solve least squares
        try:
            W_flat, residuals, rank_m, sv = np.linalg.lstsq(M, b, rcond=None)
            W = W_flat.reshape(p*n, r)
            
            # Verify the solution
            # Check if the decomposition is exact (within tolerance)
            # We should check on new random samples to avoid overfitting.
            err = 0.0
            for _ in range(10):
                A = np.random.randint(-1, 2, size=(n, m))
                B = np.random.randint(-1, 2, size=(m, m)) # Typo, should be (m,p)
                B = np.random.randint(-1, 2, size=(m, p))
                C = A @ B
                
                A_flat = A.flatten()
                B_flat = B.flatten()
                u_proj = A_flat @ U
                v_proj = B_flat @ V
                
                # Reconstruct C
                # C_recon[i,k] = sum_c W[k*n+i, c] * u_proj[c] * v_proj[c]
                C_recon = np.zeros((n, p))
                for i in range(n):
                    for k in range(p):
                        C_recon[i,k] = np.sum(W[k*n+i, :] * u_proj * v_proj)
                
                err = max(err, np.max(np.abs(C - C_recon)))
            
            if err < 1e-9:
                return U, V, W
        except:
            pass
        
        return None
    
    # Try ranks from schoolbook-1 downwards
    # But this is slow. Let's try a few specific ranks.
    # Known lower bound is 6.
    # Let's try ranks close to the lower bound? No, that's hard.
    # Let's try ranks slightly below schoolbook.
    
    # Actually, the random method might not find low ranks easily.
    # Let's stick to schoolbook if we can't find better quickly.
    
    # Try a few random attempts for rank schoolbook - 1, schoolbook - 2, etc.
    # But with time limit.
    
    attempts = 0
    max_attempts = 100
    
    while time.time() < deadline - 5.0 and attempts < max_attempts:
        r = best_rank - 1
        if r < 6: # Don't go below lower bound
            break
        res = try_rank(r)
        if res is not None:
            U, V, W = res
            best_rank = r
            best_U, best_V, best_W = U, V, W
            # Try to go lower
        else:
            # If we failed, maybe try again with different random seeds?
            # Or just accept schoolbook if we can't improve.
            # Let's try a few more random inits for the same rank?
            # For now, just break or try next rank?
            # Let's try the same rank a few times.
            attempts += 1
            if attempts % 10 == 0:
                # Try a lower rank?
                # No, let's just stop if we fail a few times.
                # Actually, let's try rank - 2, - 3...
                pass
        
        # Check time
        if time.time() > deadline - 10.0:
            break
    
    # Convert to lists for return
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
