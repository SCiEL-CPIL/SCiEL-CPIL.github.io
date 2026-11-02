"""Matrix multiplication 2x4 x 4x4 tensor decomposition - rank 27."""

import numpy as np
import time

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START

def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=4
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s
    
    def build_strassen_28():
        rank = 28
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        col = 0
        for b in range(2):  # which 2x2 block of B columns
            for a_blk in range(2):  # which 2x2 block of A columns
                for s in range(7):
                    c = col
                    j0 = 2 * a_blk
                    j1 = 2 * a_blk + 1
                    k0 = 2 * b
                    k1 = 2 * b + 1
                    
                    # U side: A[i, j] for i in {0,1}, j in {j0, j1}
                    U_contrib = {
                        0: [(0, j0, 1), (1, j1, 1)],
                        1: [(1, j0, 1), (1, j1, 1)],
                        2: [(0, j0, 1)],
                        3: [(1, j1, 1)],
                        4: [(0, j0, 1), (0, j1, 1)],
                        5: [(1, j0, 1), (0, j0, -1)],
                        6: [(0, j1, 1), (1, j1, -1)],
                    }
                    for (i, j, coeff) in U_contrib[s]:
                        U[i * 4 + j, c] += coeff
                    
                    # V side: B[j, k] for j in {j0, j1}, k in {k0, k1}
                    V_contrib = {
                        0: [(j0, k0, 1), (j1, k1, 1)],
                        1: [(j0, k0, 1)],
                        2: [(j0, k1, 1), (j1, k1, -1)],
                        3: [(j1, k0, 1), (j0, k0, -1)],
                        4: [(j1, k1, 1)],
                        5: [(j0, k0, 1), (j0, k1, 1)],
                        6: [(j1, k0, 1), (j1, k1, 1)],
                    }
                    for (j, k, coeff) in V_contrib[s]:
                        V[j * 4 + k, c] += coeff
                    
                    # W side: C[i, k] for i in {0,1}, k in {k0, k1}
                    W_contrib = {
                        0: {(0, k0): 1, (1, k1): 1},
                        1: {(1, k0): 1, (1, k1): -1},
                        2: {(0, k1): 1, (1, k1): 1},
                        3: {(0, k0): 1, (1, k0): 1},
                        4: {(0, k0): -1, (0, k1): 1},
                        5: {(1, k1): 1},
                        6: {(0, k0): 1},
                    }
                    for (i_loc, k_loc), coeff in W_contrib[s].items():
                        W[k_loc * 2 + i_loc, c] += coeff
                    
                    col += 1
        
        return U, V, W

    U28, V28, W28 = build_strassen_28()
    best_result = {"U": U28.tolist(), "V": V28.tolist(), "W": W28.tolist()}
    best_rank = 28

    # Try to find rank 27
    rank = 27
    for remove_col in range(28):
        if time.time() > deadline - 10:
            break
        
        keep = [c for c in range(28) if c != remove_col]
        U0 = U28[:, keep].copy()
        V0 = V28[:, keep].copy()
        W0 = W28[:, keep].copy()
        
        for trial in range(5):
            if time.time() > deadline - 10:
                break
            
            np.random.seed(remove_col * 100 + trial)
            
            U = U0.copy() + np.random.randn(n*m, rank) * 0.1
            V = V0.copy() + np.random.randn(m*p, rank) * 0.1
            W = W0.copy() + np.random.randn(p*n, rank) * 0.1
            
            # Gradient Descent on Frobenius norm of error
            # Error tensor E[i,j,k,r] = sum_r U[i,j,r] V[j,k,r] W[k,i,r] - delta
            # We minimize ||E||_F^2
            # Gradient w.r.t U[i,j,r] is sum_{k} V[j,k,r] W[k,i,r] * E[i,j,k,r] ? No.
            # E[i,j,k,r] is not the error. The error is for each (i,j,k) the scalar 
            # S[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r].
            # We want S[i,j,k] = 1 if i==i', j==j', k==k' (diagonal of tensor)
            # Actually, the tensor M is defined such that M[i,j,k] = 1 if i==i', j==j', k==k'?
            # No, the multiplication tensor T has T[i,j,k] = 1 if i=i', j=j', k=k' is wrong.
            # T is the tensor of the bilinear form.
            # T_{i,j,k} is the coefficient of A_{i,j} B_{j,k} in C_{i,k}.
            # So T[i,j,k] is 1 if the indices match the "diagonal" of the multiplication?
            # C[i,k] = sum_j A[i,j] B[j,k].
            # So the term A[i,j] B[j,k] contributes to C[i,k].
            # So the tensor entry T[i,j,k] corresponds to the output C[i,k]?
            # No, the tensor is in U (A), V (B), W (C) spaces.
            # U has indices (i,j). V has indices (j,k). W has indices (k,i).
            # The contraction is sum_r U[i,j,r] V[j,k,r] W[k,i,r].
            # This should equal 1 if (i,j,k) is a valid term?
            # The valid terms are all (i,j,k) such that they form a path i->j->k?
            # In matrix multiplication, every (i,j,k) is a valid term for C[i,k].
            # So T[i,j,k] should be 1 for all i,j,k?
            # No, that would mean C[i,k] = sum_{i',j',k'} ...
            # The tensor M_{n,m,p} has entries M[i,j,k] = 1 if i=i', j=j', k=k' is not right.
            # M[i,j,k] is the coefficient of A_{i,j} B_{j,k} in the expression for C_{i,k}.
            # Since C[i,k] = sum_j A[i,j] B[j,k], the coefficient of A[i,j] B[j,k] in C[i,k] is 1.
            # So M[i,j,k] = 1 for all i,j,k?
            # Wait, if M[i,j,k]=1 for all i,j,k, then the rank is n*m*p?
            # No, the rank is the minimum number of rank-1 tensors summing to M.
            # M[i,j,k] = 1 if the indices are consistent?
            # Let's check the validation in the code.
            # s = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
            # This computes sum_r U[i,j,r] V[j,k,r] W[k,i,r].
            # The code checks if this is 1.0 for all i,j,k?
            # "for i in range(n): for j in range(m): for k in range(p): ... if abs(s - 1.0) > 1e-8"
            # Yes, it checks for ALL i,j,k.
            # This implies that the tensor M has 1s at ALL positions (i,j,k).
            # Is that correct?
            # C[i,k] = sum_j A[i,j] B[j,k].
            # The term A[i,j] B[j,k] appears in C[i,k].
            # So the coefficient is 1.
            # So yes, M[i,j,k] = 1 for all i,j,k.
            # This means the tensor is the all-ones tensor?
            # No, the tensor is defined on the indices of A, B, C.
            # The "all ones" tensor has rank 1.
            # Why is the rank 28?
            # Ah, the tensor is NOT all ones.
            # The tensor M_{2,4,4} is the matrix multiplication tensor.
            # M[i,j,k] is 1 if i,j,k are such that A[i,j] and B[j,k] multiply to C[i,k].
            # This is true for ALL i,j,k.
            # So why is the rank 28?
            # Because the tensor is not just the values, but the structure of the basis?
            # No, the tensor is a linear map from U*V to W?
            # Or a trilinear form?
            # If M[i,j,k]=1 for all i,j,k, then M = u \otimes v \otimes w where u,v,w are all-ones vectors.
            # Rank 1.
            # There must be a misunderstanding of the tensor definition.
            
            # Let's re-read the validation.
            # It checks s = 1.0 for all i,j,k.
            # And checks s = 0.0 for "off-diagonals"?
            # "if i1 == i2 and j1 == j2 and k1 == k2: continue"
            # This implies that for (a,b,c) not matching (i,j,k) in some way, it should be 0.
            # But a,b,c are just indices for U, V, W.
            # U index a corresponds to (i,j).
            # V index b corresponds to (j,k).
            # W index c corresponds to (k,i).
            # The condition "i1 == i2 and j1 == j2 and k1 == k2" is always true if we iterate over the same i,j,k?
            # No, the loops are:
            # for a in range(n*m): i1 = a//m, j1 = a%m
            # for b in range(m*p): j2 = b//p, k1 = b%p
            # for c in range(p*n): i2 = c//n, k2 = c%n
            # The condition is i1==i2 and j1==j2 and k1==k2.
            # This means the indices must form a consistent cycle i->j->k->i.
            # If they don't, the term should be 0.
            # So the tensor M[i,j,k] is 1 if i,j,k form a valid path, else 0?
            # But in matrix multiplication, every (i,j,k) is a valid path for C[i,k].
            # So M[i,j,k] should be 1 for all i,j,k.
            # Why does the code check for 0 otherwise?
            # Maybe the tensor is defined differently?
            # Or maybe I am misinterpreting the indices.
            # If M[i,j,k]=1 for all i,j,k, then the rank is 1.
            # This is clearly wrong for matrix multiplication.
            
            # The matrix multiplication tensor is NOT the all-ones tensor.
            # It is the tensor that represents the bilinear map.
            # The entries of the tensor are the coefficients in the expansion.
            # C[i,k] = sum_j A[i,j] B[j,k].
            # So the coefficient of A[i,j] B[j,k] in C[i,k] is 1.
            # This means that the tensor T has T[i,j,k] = 1 if the output index is (i,k) and the input indices are (i,j) and (j,k).
            # So T is a tensor in U_A \otimes U_B \otimes U_C.
            # T_{(i,j), (j,k), (i,k)} = 1.
            # Note that the indices are linked.
            # If we flatten the indices, T[a,b,c] = 1 if a=(i,j), b=(j,k), c=(i,k) are consistent.
            # Consistency: j from a == j from b, i from a == i from c, k from b == k from c.
            # If not consistent, T[a,b,c] = 0.
            # So the tensor is sparse.
            # The number of 1s is n*m*p = 2*4*4 = 32.
            # The total size is 8*16*8 = 1024.
            # So it is not all ones.
            
            # Okay, so the gradient descent should minimize the error for the 32 non-zero entries and keep the others 0.
            # But the alternating least squares in the current code only updates based on the non-zero entries?
            # "if i == i2 and j == j2 and k1 == k2: rhs[...] = 1.0"
            # Yes, it only sets rhs to 1 for consistent indices.
            # And implicitly 0 for others (since rhs is initialized to 0).
            # So the ALS is correct.
            
            # I will replace the ALS with a simple gradient step or keep ALS but improve initialization.
            # The current ALS is slow.
            # I will use a simplified ALS with fewer iterations but better initialization.
            
            for step in range(20):
                if time.time() > deadline - 10:
                    break
                
                # Update U
                for a in range(n*m):
                    i = a // m
                    j = a % m
                    M = np.zeros((m*p*p*n, rank))
                    rhs = np.zeros(m*p*p*n)
                    for b_idx in range(m*p):
                        j2 = b_idx // p
                        k1 = b_idx % p
                        for c_idx in range(p*n):
                            i2 = c_idx // n
                            k2 = c_idx % n
                            M[b_idx*p*n + c_idx, :] = V[b_idx, :] * W[c_idx, :]
                            if i == i2 and j == j2 and k1 == k2:
                                rhs[b_idx*p*n + c_idx] = 1.0
                    try:
                        U[a, :] = np.linalg.lstsq(M, rhs, rcond=None)[0]
                    except:
                        pass
                
                # Update V
                for b_idx in range(m*p):
                    j2 = b_idx // p
                    k1 = b_idx % p
                    M = np.zeros((n*m*p*n, rank))
                    rhs = np.zeros(n*m*p*n)
                    for a in range(n*m):
                        i1 = a // m
                        j1 = a % m
                        for c_idx in range(p*n):
                            i2 = c_idx // n
                            k2 = c_idx % n
                            M[a*p*n + c_idx, :] = U[a, :] * W[c_idx, :]
                            if i1 == i2 and j1 == j2 and k1 == k2:
                                rhs[a*p*n + c_idx] = 1.0
                    try:
                        V[b_idx, :] = np.linalg.lstsq(M, rhs, rcond=None)[0]
                    except:
                        pass
                
                # Update W
                for c_idx in range(p*n):
                    i2 = c_idx // n
                    k2 = c_idx % n
                    M = np.zeros((n*m*m*p, rank))
                    rhs = np.zeros(n*m*m*p)
                    for a in range(n*m):
                        i1 = a // m
                        j1 = a % m
                        for b_idx in range(m*p):
                            j2 = b_idx // p
                            k1 = b_idx % p
                            M[a*m*p + b_idx, :] = U[a, :] * V[b_idx, :]
                            if i1 == i2 and j1 == j2 and k1 == k2:
                                rhs[a*m*p + b_idx] = 1.0
                    try:
                        W[c_idx, :] = np.linalg.lstsq(M, rhs, rcond=None)[0]
                    except:
                        pass
            
            # Validate
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                        if abs(s - 1.0) > 1e-6:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            if valid:
                best_rank = rank
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                break
        
        if best_rank == rank:
            break

    return best_result

# EVOLVE-BLOCK-END
