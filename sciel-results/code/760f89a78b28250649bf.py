"""Systematic rank-1 peeling for 2x3x5 matrix multiplication tensor."""
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank = n*m*p = 30
    # We know that for matrix multiplication, the rank is at most n*m*p
    # But we can do better. The known bounds:
    # For 2x3x5, let's try to find a decomposition with rank < 30
    
    # The tensor T has components T[i,j,k] = 1 if the (i,j) entry of A and (j,k) entry of B
    # contribute to (i,k) entry of C. Actually, the matrix multiplication tensor is:
    # T_{i,j,k} where A is n x m, B is m x p, C is n x p
    # C_{i,k} = sum_j A_{i,j} B_{j,k}
    # So the tensor is sum over j of (e_i in R^n) (e_j in R^m) (e_k in R^p)
    # Wait, that's not quite right. Let me think again.
    
    # The matrix multiplication tensor for A (n x m) * B (m x p) = C (n x p)
    # is: T = sum_{j=1}^{m} (sum_{i=1}^{n} sum_{k=1}^{p} e_i ⊗ e_j ⊗ e_k)
    # No, actually: C_{i,k} = sum_j A_{i,j} B_{j,k}
    # The tensor is: T = sum_j (A_j) ⊗ (B_j) ⊗ (C_j) where...
    # Actually, the standard formulation:
    # T = sum_{i,j,k} T_{i,j,k} e_i ⊗ e_j ⊗ e_k
    # where T_{i,j,k} = 1 if the product A_{i,j} * B_{j,k} contributes to C_{i,k}
    # But that's only for the diagonal case. 
    
    # Let me reconsider. The matrix multiplication tensor is:
    # T = sum_{j=1}^m (sum_{i=1}^n sum_{k=1}^p e_i ⊗ e_j ⊗ e_k)
    # No. Let's use the correct formulation.
    # 
    # For multiplying A (n x m) by B (m x p) to get C (n x p):
    # C_{i,k} = sum_{j=1}^m A_{i,j} B_{j,k}
    # 
    # The tensor T in R^{n*m} ⊗ R^{m*p} ⊗ R^{n*p} (using the convention from the task)
    # has the property that:
    # <T, u ⊗ v ⊗ w> = sum_{i,j,k} u_{i,j} v_{j,k} w_{k,i}  ... no wait
    # 
    # Actually with the convention W[k*n+i, r], the output is indexed by (k, i) where
    # k is the row of C (0..p-1) and i is the column of C (0..n-1)? 
    # Wait, C is n x p, so C has n rows and p columns.
    # 
    # Let me re-read: "The output factor follows the official trace convention W[k*n+i,r]"
    # So W has shape (p*n, rank). The index is k*n + i where k ranges over 0..p-1 and i over 0..n-1.
    # This means the output matrix C is indexed as C[k, i] where k is the row (0..p-1) and i is the column (0..n-1).
    # But C should be n x p (n rows, p columns). So maybe C[i, k] = C_{row=i, col=k}?
    # Then W[k*n+i] would be C[i, k] transposed? Or maybe the convention is that C is stored as p x n?
    
    # Let me just follow the schoolbook decomposition from the seed program:
    # U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1
    # So U is indexed by (i, j) where i is row of A (0..n-1), j is column of A / row of B (0..m-1)
    # V is indexed by (j, k) where j is row of B (0..m-1), k is column of B (0..p-1)
    # W is indexed by (k, i) where k is column of B / ... and i is row of A
    # 
    # The contraction is: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # For the schoolbook, this gives 1 when we fix i, j, k, and 0 otherwise.
    # The full sum over c gives: sum_{i,j,k} A[i,j] * B[j,k] * W[k*n+i]
    # We want this to equal C[i,k] = sum_j A[i,j] B[j,k]
    # So we need sum_j A[i,j] B[j,k] W[k*n+i] = C[i,k] for all i, k
    # This means W[k*n+i] should act as a delta: we need the sum over j to give C[i,k]
    # In the schoolbook, for each (i,j,k) triple, there's a separate rank-1 term.
    # So the sum is sum_{i',j',k'} A[i',j'] B[j',k'] * (delta_{i',i} delta_{j',j} delta_{k',k} for the specific term)
    # Actually in the schoolbook, each term (i,j,k) contributes A[i,j]*B[j,k]*W[k*n+i] where W[k*n+i]=1 for that term.
    # So the total is sum_{i,j,k} A[i,j] B[j,k] * 1 (for each i,j,k) but only when matching...
    # 
    # I think the correct interpretation is:
    # The tensor T has components T[(i,j), (j',k), (k',i')] where:
    # T[(i,j),(j',k),(k',i')] = 1 if i=i' and j=j' and k=k', else 0
    # Wait no. Let me just trust the seed program's schoolbook.
    
    # For the schoolbook: rank = n*m*p = 2*3*5 = 30
    # The tensor is: T = sum_{i,j,k} e_{(i,j)} ⊗ e_{(j,k)} ⊗ e_{(k,i)}
    # where e_{(i,j)} is the unit vector in R^{n*m} at position i*m+j
    # e_{(j,k)} is the unit vector in R^{m*p} at position j*p+k  
    # e_{(k,i)} is the unit vector in R^{p*n} at position k*n+i
    
    # This is a sum of n*m*p rank-1 tensors.
    
    # Now, can we do better? The rank of the matrix multiplication tensor for n x m by m x p
    # is known to be at most n*m*p, but can be lower.
    # 
    # For small cases:
    # 2x2x2: rank 7 (Strassen)
    # 2x3x3, 3x3x3: known results
    # 2x3x5: not sure of the exact minimal rank
    #
    # Let's try to find a decomposition with rank < 30 using rank-1 peeling.
    
    # Build the tensor as a 3D array
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Try rank-1 peeling
    # Start with residual = T
    residual = T.copy()
    U_list = []
    V_list = []
    W_list = []
    
    rank = 0
    max_rank = 30  # schoolbook rank
    
    while rank < max_rank:
        # Find the best rank-1 approximation to residual
        # A rank-1 tensor is u ⊗ v ⊗ w
        # We want to find u, v, w such that <residual, u⊗v◗> is maximized
        # or equivalently, find u, v, w that best approximate residual
        #
        # One approach: fix two factors, solve for the third
        # Or use power iteration / alternating least squares
        
        # Let's use a simple approach: random initialization + ALS
        best_score = -1
        best_u = None
        best_v = None
        best_w = None
        
        # Try multiple random initializations
        np.random.seed(rank)  # deterministic
        for trial in range(50):
            u = np.random.randn(n*m)
            v = np.random.randn(m*p)
            w = np.random.randn(p*n)
            
            # ALS iterations
            for _ in range(20):
                # Update u: residual contracted with v and w
                # residual is (n*m, m*p, p*n)
                # u_new[i] = sum_{j,k} residual[i,j,k] * v[j] * w[k]
                u = np.einsum('ijk,j,k->i', residual, v, w)
                # Update v
                v = np.einsum('ijk,i,k->j', residual, u, w)
                # Update w
                w = np.einsum('ijk,i,j->k', residual, u, v)
            
            # Compute score
            score = np.einsum('ijk,i,j,k', residual, u, v, w)
            if score > best_score:
                best_score = score
                best_u = u.copy()
                best_v = v.copy()
                best_w = w.copy()
        
        if best_score <= 1e-10:
            break
        
        # Normalize so that the contraction is 1
        # We want u ⊗ v ⊗ w such that <residual, u⊗v⊗w> = 1
        # Scale: let s = <residual, u⊗v⊗w> = best_score
        # Then (u/sqrt(s)) ⊗ (v/sqrt(s)) ⊗ (w) would give... no
        # Actually, we want to subtract a rank-1 tensor from residual
        # Let u, v, w be the factors. The rank-1 tensor is u ⊗ v ⊗ w
        # We want to choose scaling such that the subtraction is valid
        # 
        # For exact decomposition, we need the residual to remain a "valid" tensor
        # Let's just subtract u ⊗ v ⊗ w scaled appropriately
        # 
        # Actually, for the peeling to work exactly, we need the factors to have
        # rational (or half-integer) entries. Let's try to find such factors.
        
        # For now, let's just do floating point and see if we can get rank < 30
        # Then we'll try to rationalize
        
        # Subtract the rank-1 tensor
        residual -= np.einsum('i,j,k->ijk', best_u, best_v, best_w)
        
        U_list.append(best_u)
        V_list.append(best_v)
        W_list.append(best_w)
        rank += 1
        
        # Check if residual is close to zero
        if np.linalg.norm(residual) < 1e-10:
            break
    
    # If we didn't finish, fall back to schoolbook
    if rank >= max_rank and np.linalg.norm(residual) > 1e-10:
        # Use schoolbook
        U = [[0] * (n*m*p) for _ in range(n*m)]
        V = [[0] * (n*m*p) for _ in range(m*p)]
        W = [[0] * (n*m*p) for _ in range(p*n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][column] = 1
                    V[j*p+k][column] = 1
                    W[k*n+i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    # Build the factor matrices
    U_mat = np.array(U_list).T  # shape (n*m, rank)
    V_mat = np.array(V_list).T  # shape (m*p, rank)
    W_mat = np.array(W_list).T  # shape (p*n, rank)
    
    # Convert to list of lists with fractions for exactness
    # Try to find rational approximations
    U_frac = []
    for i in range(n*m):
        row = []
        for c in range(rank):
            val = U_mat[i, c]
            # Try to find a simple fraction
            frac = Fraction(val).limit_denominator(100)
            row.append(float(frac))
        U_frac.append(row)
    
    V_frac = []
    for i in range(m*p):
        row = []
        for c in range(rank):
            val = V_mat[i, c]
            frac = Fraction(val).limit_denominator(100)
            row.append(float(frac))
        V_frac.append(row)
    
    W_frac = []
    for i in range(p*n):
        row = []
        for c in range(rank):
            val = W_mat[i, c]
            frac = Fraction(val).limit_denominator(100)
            row.append(float(frac))
        W_frac.append(row)
    
    # Verify the decomposition
    # Check that sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = T[i*m+j, j*p+k, k*n+i]
    # for all i, j, k
    U_arr = np.array(U_frac)
    V_arr = np.array(V_frac)
    W_arr = np.array(W_frac)
    
    max_err = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_arr[i*m+j] * V_arr[j*p+k] * W_arr[k*n+i])
                expected = 1.0
                err = abs(val - expected)
                max_err = max(max_err, err)
    
    # Also check that non-diagonal entries are zero
    for i1 in range(n):
        for j1 in range(m):
            for k1 in range(p):
                for i2 in range(n):
                    for j2 in range(m):
                        for k2 in range(p):
                            if i1 != i2 or j1 != j2 or k1 != k2:
                                val = np.sum(U_arr[i1*m+j1] * V_arr[j2*p+k2] * W_arr[k2*n+i2])
                                err = abs(val)
                                max_err = max(max_err, err)
    
    if max_err > 1e-6:
        # Fall back to schoolbook
        U = [[0] * (n*m*p) for _ in range(n*m)]
        V = [[0] * (n*m*p) for _ in range(m*p)]
        W = [[0] * (n*m*p) for _ in range(p*n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][column] = 1
                    V[j*p+k][column] = 1
                    W[k*n+i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    return {"U": U_frac, "V": V_frac, "W": W_frac}
