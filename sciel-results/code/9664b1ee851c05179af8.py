import scipy
import numpy as np
import time
import itertools

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    # Create the target tensor T
    # T[i, j, k] = 1 if ... wait, the tensor for matrix multiplication is:
    # T[i, j, k] is the coefficient of A[i,j] * B[j,k] in C[i,k].
    # In the decomposition T = sum U_c (tensor) V_c (tensor) W_c (tensor),
    # U_c is n x m, V_c is m x p, W_c is p x n.
    # T[i, j, k] = sum_c U_c[i, j] * V_c[j, k] * W_c[k, i]
    # The standard tensor T has T[i, j, k] = 1 if the indices match the multiplication path?
    # No, the tensor T is defined such that the bilinear form is <T, A \otimes B \otimes C> = trace(C^T A B).
    # Actually, the standard basis tensor for MM is:
    # T[i, j, k] = 1 if i, j, k are the indices for A[i,j], B[j,k], C[i,k].
    # Wait, the decomposition is T = sum U_c \otimes V_c \otimes W_c.
    # U_c is n x m, V_c is m x p, W_c is p x n.
    # The entry T[i, j, k] is the coefficient.
    # For MM, T[i, j, k] is 1 if (i,j) is in A, (j,k) in B, and (i,k) in C?
    # No, the tensor is sparse.
    # T[i, j, k] = 1 if i, j, k form a valid path?
    # Actually, the tensor is:
    # T[i, j, k] = 1 if i, j, k are such that A[i,j] and B[j,k] contribute to C[i,k].
    # This is always true for any i, j, k?
    # No, the tensor is defined on the space of A (n x m), B (m x p), C (p x n).
    # The standard basis tensor is:
    # T[i, j, k] = 1 if the indices are (i, j) for A, (j, k) for B, and (i, k) for C?
    # Wait, the third index of the tensor corresponds to the output C.
    # The output C has indices (i, k).
    # So the tensor T has indices i (from A row), j (from A col / B row), k (from B col).
    # And the output is C[i, k].
    # So T[i, j, k] = 1 if the term A[i, j] * B[j, k] contributes to C[i, k].
    # This is always 1 for all i, j, k?
    # No, the tensor is a sum of basis tensors.
    # The bilinear form is sum_{i,j,k} A[i,j] B[j,k] C[i,k].
    # So T[i, j, k] = 1 for all i, j, k?
    # No, that would mean T is the all-ones tensor.
    # But the decomposition is T = sum U_c \otimes V_c \otimes W_c.
    # U_c[i, j] * V_c[j, k] * W_c[k, i] ?
    # The indices in the product must match.
    # U_c is n x m (indices i, j).
    # V_c is m x p (indices j, k).
    # W_c is p x n (indices k, i).
    # The product is U_c[i, j] * V_c[j, k] * W_c[k, i].
    # This sums over c.
    # The result is a tensor with indices i, j, k.
    # We want this to equal the MM tensor.
    # The MM tensor T_MM has T_MM[i, j, k] = 1 if the term A[i, j] B[j, k] is part of the computation of C[i, k].
    # Since C[i, k] = sum_j A[i, j] B[j, k], the term A[i, j] B[j, k] always appears.
    # So T_MM[i, j, k] = 1 for all i, j, k?
    # No, that's not right.
    # The tensor T is in the space A* \otimes B* \otimes C*.
    # The evaluation is <T, A \otimes B \otimes C> = sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] C[i,k].
    # We want this to be equal to trace(C^T A B) = sum_{i,k} C[i,k] (A B)[i,k] = sum_{i,k} C[i,k] sum_j A[i,j] B[j,k].
    # So <T, A \otimes B \otimes C> = sum_{i,j,k} A[i,j] B[j,k] C[i,k].
    # This implies T[i, j, k] = 1 for all i, j, k.
    # So the tensor is the all-ones tensor?
    # No, the dimensions are different.
    # A is n x m, B is m x p, C is p x n.
    # The sum is over i, j, k.
    # Yes, T[i, j, k] = 1 for all i, j, k.
    # Wait, is that correct?
    # If T is all ones, then rank is 1?
    # No, because the indices are coupled.
    # The tensor is T[i, j, k] = 1.
    # Can we decompose the all-ones tensor of size 3x3x7 into rank 1?
    # U[i,j] V[j,k] W[k,i] = 1?
    # If U=1, V=1, W=1, then sum is 1.
    # So rank 1?
    # No, the matrix multiplication tensor is NOT the all-ones tensor.
    # The tensor is defined by the structure of the multiplication.
    # The standard tensor is:
    # T = sum_{i,j,k} e_{ij} \otimes e_{jk} \otimes e_{ik} ?
    # No, e_{ik} is the basis for C.
    # The term is A[i,j] B[j,k] C[i,k].
    # So the tensor is sum_{i,j,k} e_{ij}^A \otimes e_{jk}^B \otimes e_{ik}^C.
    # This is NOT the all-ones tensor in the sense that the indices are fixed.
    # For a fixed i, j, k, the value is 1.
    # So yes, T[i, j, k] = 1 for all i, j, k.
    # But the rank is not 1 because the "all-ones" tensor of size n x m x p has rank 1 only if n=m=p=1?
    # No, the all-ones tensor of size 2x2x2 has rank 1?
    # U[i,j] V[j,k] W[k,i] = 1.
    # Let U=1, V=1, W=1. Then sum is 1.
    # So rank 1?
    # Why is matrix multiplication rank not 1?
    # Because the tensor is NOT the all-ones tensor.
    # The tensor is T[i, j, k] = 1 if the indices match the multiplication pattern.
    # In MM, A[i,j] multiplies B[j,k] to give C[i,k].
    # So the term is A[i,j] B[j,k] C[i,k].
    # The tensor is sum_{i,j,k} e_{ij} \otimes e_{jk} \otimes e_{ik}.
    # This tensor has T[i, j, k] = 1 for all i, j, k.
    # So it IS the all-ones tensor?
    # No, the basis vectors e_{ik} for C are different from e_{ij} for A.
    # The tensor is in A* \otimes B* \otimes C*.
    # The entry T[i, j, k] is the coefficient of e_{ij}^A \otimes e_{jk}^B \otimes e_{ik}^C.
    # Since the product A B has entry (i, k) = sum_j A[i, j] B[j, k],
    # the bilinear form is sum_{i,k} C[i, k] sum_j A[i, j] B[j, k] = sum_{i,j,k} A[i, j] B[j, k] C[i, k].
    # So the coefficient of A[i, j] B[j, k] C[i, k] is 1.
    # So T[i, j, k] = 1 for all i, j, k.
    # So the tensor is indeed the all-ones tensor?
    # If so, rank is 1.
    # But we know rank of 2x2x2 MM is 7.
    # Where is the error?
    # The error is that the indices i, j, k in the tensor T[i, j, k] are NOT independent.
    # In the sum sum_{i,j,k}, i is from A, j is from A/B, k is from B.
    # And C has indices i, k.
    # So the tensor is T[i, j, k] = 1.
    # But the rank of the all-ones tensor of size 2x2x2 is 1.
    # Why is MM rank 7?
    # Ah, the tensor is NOT the all-ones tensor.
    # The tensor is T = sum_{i,j,k} e_{ij} \otimes e_{jk} \otimes e_{ik}.
    # This is a specific tensor.
    # Is it the all-ones tensor?
    # The all-ones tensor is sum_{i,j,k} e_i \otimes e_j \otimes e_k.
    # Here, the third index is k, but it corresponds to e_{ik} in C.
    # So the basis for C is indexed by pairs (i, k).
    # So the tensor is T[i, j, (i, k)] = 1.
    # If we flatten C to index l = i*p + k, then T[i, j, l] = 1 if l corresponds to (i, k).
    # So T is NOT all ones. It is sparse.
    # For a given i, j, only one k (from B) is allowed? No.
    # For a given i, j, k, the term is A[i, j] B[j, k] C[i, k].
    # So T[i, j, l] = 1 if l = i*p + k.
    # So for fixed i, j, the non-zero entries in the k-dimension are all k?
    # No, for fixed i, j, the sum is over k.
    # So T[i, j, l] is 1 for all l such that l = i*p + k for some k.
    # Since k ranges 0..p-1, l ranges i*p .. i*p + p - 1.
    # So T[i, j, l] = 1 if floor(l/p) == i.
    # So T is a block diagonal structure in the third index?
    # No, it's a block structure in the first index.
    # For i=0, T[0, j, l] = 1 if l < p.
    # For i=1, T[1, j, l] = 1 if p <= l < 2p.
    # Etc.
    # So T is sparse.
    
    # Let's construct T.
    T = np.zeros((n, m, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, i*p + k] = 1.0
                
    # Now we want to find U, V, W such that:
    # sum_c U_c[i, j] * V_c[j, k] * W_c[k, i] = T[i, j, i*p + k]
    # Note: W_c has indices (k, i) because it's p x n.
    # But in the sum, W_c[k, i] is used.
    # So W_c is a matrix of size p x n.
    
    # We will try to find a decomposition of rank r.
    best_rank = 63
    best_U = None
    best_V = None
    best_W = None
    
    # Try ranks from 50 to 62
    for r in range(50, 63):
        if time.time() > deadline:
            break
            
        # Random init
        np.random.seed(42 + r)
        U = np.random.randn(n*m, r)
        V = np.random.randn(m*p, r)
        W = np.random.randn(p*n, r)
        
        # Flatten
        def flatten(U, V, W):
            return np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
        def unflatten(x):
            size_U = n*m*r
            size_V = m*p*r
            U = x[:size_U].reshape(n*m, r)
            V = x[size_U:size_U+size_V].reshape(m*p, r)
            W = x[size_U+size_V:].reshape(p*n, r)
            return U, V, W
            
        def residual(x):
            U, V, W = unflatten(x)
            # Compute sum U_c \otimes V_c \otimes W_c
            # Res[i, j, l] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
            # l = i*p + k
            Res = np.zeros((n, m, p*n))
            for c in range(r):
                # U_c is n x m
                Uc = U[:, c].reshape(n, m)
                # V_c is m x p
                Vc = V[:, c].reshape(m, p)
                # W_c is p x n
                Wc = W[:, c].reshape(p, n)
                # Outer product?
                # We need to add Uc \otimes Vc \otimes Wc to Res.
                # Res[i, j, i*p+k] += Uc[i, j] * Vc[j, k] * Wc[k, i]
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            Res[i, j, i*p + k] += Uc[i, j] * Vc[j, k] * Wc[k, i]
            # Target T
            return (Res - T).flatten()
            
        # Optimize
        x0 = flatten(U, V, W)
        try:
            res = scipy.optimize.least_squares(residual, x0, method='lm', max_nfev=10000)
            if res.cost < 1e-10:
                U, V, W = unflatten(res.x)
                # Snap to half-integers
                U = np.round(U * 2) / 2
                V = np.round(V * 2) / 2
                W = np.round(W * 2) / 2
                # Verify
                Res = np.zeros((n, m, p*n))
                for c in range(r):
                    Uc = U[:, c].reshape(n, m)
                    Vc = V[:, c].reshape(m, p)
                    Wc = W[:, c].reshape(p, n)
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                Res[i, j, i*p + k] += Uc[i, j] * Vc[j, k] * Wc[k, i]
                if np.allclose(Res, T, atol=1e-9):
                    best_rank = r
                    best_U = U
                    best_V = V
                    best_W = W
                    break # Found a valid decomposition, try lower rank? 
                    # But we are iterating from low to high? No, 50 to 62.
                    # So we found the lowest rank in this range.
        except Exception as e:
            pass
            
    if best_U is not None:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
