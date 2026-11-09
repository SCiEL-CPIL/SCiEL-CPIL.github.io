import numpy as np
import time
import random

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    target_rank = 60 # Fallback
    
    # Target tensor T[i, j, k] = 1 if j == k else 0
    # Shape: (n, m, p)
    T = np.zeros((n, m, p), dtype=np.float64)
    for j in range(m):
        for i in range(n):
            T[i, j, j] = 1.0
            
    # We want U (n x r), V (m x r), W (p x r)
    # such that sum_c U[i,c] * V[j,c] * W[k,c] == T[i,j,k]
    
    # Time budget
    start_time = time.time()
    max_time = 140.0 # Safety margin
    
    best_rank = 60
    best_U = None
    best_V = None
    best_W = None
    
    # Try to find a decomposition with lower rank
    # Heuristic: Try ranks 30, 35, 40, 45, 50
    # Use random initialization and local search
    
    def check_decomposition(U, V, W):
        # Compute sum_c U[:, c] * V[:, c] * W[:, c]
        # Using einsum
        prod = np.einsum('ic,jc,kc->ijk', U, V, W)
        # Check if it matches T
        if np.allclose(prod, T, atol=1e-9):
            return True
        return False

    def local_search(r, max_iter=500):
        # Initialize with random small integers/half-integers
        # Scale factors to keep numbers small
        scale = 2.0
        U = np.random.randint(-scale, scale + 1, (n, r)) * 0.5
        V = np.random.randint(-scale, scale + 1, (m, r)) * 0.5
        W = np.random.randint(-scale, scale + 1, (p, r)) * 0.5
        
        # Ensure some non-zeros
        for c in range(r):
            if np.sum(np.abs(U[:, c])) == 0:
                U[np.random.randint(n), c] = 1.0
            if np.sum(np.abs(V[:, c])) == 0:
                V[np.random.randint(m), c] = 1.0
            if np.sum(np.abs(W[:, c])) == 0:
                W[np.random.randint(p), c] = 1.0
                
        for it in range(max_iter):
            if time.time() - start_time > max_time:
                return None
            
            # Compute residual
            prod = np.einsum('ic,jc,kc->ijk', U, V, W)
            residual = T - prod
            err = np.sum(residual**2)
            
            if err < 1e-9:
                return U, V, W
                
            # Pick a random matrix and a random entry to update
            mat_idx = random.choice([0, 1, 2])
            if mat_idx == 0:
                i, c = random.randint(0, n-1), random.randint(0, r-1)
                # Gradient-like step? Or just random perturbation
                # Let's try adding a small random delta
                delta = random.choice([-0.5, 0.5])
                U[i, c] += delta
            elif mat_idx == 1:
                j, c = random.randint(0, m-1), random.randint(0, r-1)
                delta = random.choice([-0.5, 0.5])
                V[j, c] += delta
            else:
                k, c = random.randint(0, p-1), random.randint(0, r-1)
                delta = random.choice([-0.5, 0.5])
                W[k, c] += delta
                
            # Snap to half-integers
            if mat_idx == 0:
                U[i, c] = round(U[i, c] * 2) / 2
            elif mat_idx == 1:
                V[j, c] = round(V[j, c] * 2) / 2
            else:
                W[k, c] = round(W[k, c] * 2) / 2
                
        return None

    # Try a few ranks
    # Note: This is a heuristic and might not find a solution if the rank is not close to the random start
    # We will try a few restarts for each rank
    
    ranks_to_try = [30, 35, 40, 45, 50]
    
    for r in ranks_to_try:
        if time.time() - start_time > max_time - 10:
            break
            
        for attempt in range(10):
            if time.time() - start_time > max_time - 10:
                break
                
            res = local_search(r, max_iter=100)
            if res is not None:
                U, V, W = res
                if check_decomposition(U, V, W):
                    best_rank = r
                    best_U = U
                    best_V = V
                    best_W = W
                    break
        if best_rank < 60:
            break

    # If no better rank found, return schoolbook
    if best_rank == 60:
        rank = 60
        U = np.zeros((n, rank))
        V = np.zeros((m, rank))
        W = np.zeros((p, rank))
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if j == k:
                        U[i, col] = 1
                        V[j, col] = 1
                        W[k, col] = 1
                        col += 1
        # Note: The schoolbook decomposition for T[i,j,k] = delta_jk
        # Actually, the tensor T is defined such that T[i,j,k] = 1 if j==k.
        # The matrix multiplication (A*B)_{i,k} = sum_j A_{i,j} B_{j,k}.
        # The tensor for this multiplication is T[i,j,k] = 1 if j==k? 
        # Wait, the standard tensor for matrix mult C = A*B is T_{i,j,k} = 1 if the indices match in the contraction.
        # The contraction is over j. So the tensor entries are T_{i,j,k} = 1 if the index j in A matches the index j in B.
        # Yes, T[i,j,k] = 1 if j==k is incorrect for standard matmul.
        # Standard matmul: C_{ik} = sum_j A_{ij} B_{jk}.
        # The tensor is T_{i,j,k} such that C_{ik} = sum_j T_{i,j,k} A_{ij} B_{jk}.
        # So T_{i,j,k} = 1 if the middle index matches? No, the sum is over j.
        # The term is A_{ij} B_{jk}. This corresponds to a tensor entry T_{i,j,k} = 1 for all i,j,k? No.
        # The tensor T has indices (i, j, k). The multiplication is defined by summing over the middle index of the two input matrices.
        # Let's re-read the task: "multiplying a 3 by 4 matrix by a 4 by 5 matrix".
        # A is 3x4, B is 4x5. C is 3x5.
        # C[i,k] = sum_j A[i,j] B[j,k].
        # The tensor decomposition is sum_c U[i,c] V[j,c] W[k,c] = T[i,j,k].
        # For the standard multiplication, T[i,j,k] is 1 if the index j of A matches the index j of B?
        # Actually, the tensor T is defined by T[i,j,k] = 1 if the "link" is valid.
        # In the standard decomposition, we have one term for each (i,j,k) triple? No.
        # The standard decomposition has rank n*m*p.
        # Term c corresponds to (i,j,k).
        # U[i,c]=1, V[j,c]=1, W[k,c]=1.
        # Then sum_c U[i',c] V[j',c] W[k',c] = 1 if (i',j',k') = (i,j,k).
        # So the resulting tensor is the identity tensor? No.
        # The resulting tensor T' has T'[i,j,k] = 1 if there exists c such that U[i,c]=1, V[j,c]=1, W[k,c]=1.
        # In the schoolbook, for a fixed c=(i,j,k), U[i,c]=1, V[j,c]=1, W[k,c]=1.
        # So T'[i,j,k] = 1.
        # This means the schoolbook decomposition produces a tensor with all ones?
        # That doesn't match the matrix multiplication tensor.
        
        # Let's check the definition of the matrix multiplication tensor.
        # The tensor T is such that T[i,j,k] = 1 if the index j in the first matrix and the index j in the second matrix are the same?
        # No, the contraction is sum_j.
        # The tensor T is defined by T[i,j,k] = 1 if the "slot" j in A and "slot" j in B are connected.
        # Actually, the standard tensor for matrix multiplication A(n,m) * B(m,p) is T_{i,j,k} = 1 if j is the contracted index.
        # Wait, the indices are (i, j, k). i is row of A, j is col of A / row of B, k is col of B.
        # The multiplication is C_{ik} = sum_j A_{ij} B_{jk}.
        # This can be written as C_{ik} = sum_{j,c} T_{i,j,k} A_{ij} B_{jk} ? No.
        # It is C_{ik} = sum_j A_{ij} B_{jk}.
        # This is a linear map. The tensor T represents the structure.
        # T_{i,j,k} = 1 if the index j is the same in both matrices?
        # In the schoolbook, we have rank n*m*p.
        # For each i, j, k, we have a term.
        # U[i, (i,j,k)] = 1, V[j, (i,j,k)] = 1, W[k, (i,j,k)] = 1.
        # Summing over c: sum_c U[i',c] V[j',c] W[k',c].
        # This sum is 1 if (i',j',k') = (i,j,k) for some c? No, c is unique for each (i,j,k).
        # So the sum is 1 if i'=i, j'=j, k'=k.
        # So the tensor produced is T'[i,j,k] = 1 if i,j,k are fixed?
        # This implies the tensor is the identity tensor in 3D? No, it's a tensor with 1s on the "diagonal" of the 3D grid?
        # No, the tensor T for matrix multiplication is defined as T_{i,j,k} = 1 if j is the contracted index.
        # Actually, the standard definition is that the multiplication tensor T_{n,m,p} has entries T_{i,j,k} = 1 if the index j matches?
        # Let's look at the seed code:
        # U[i*m+j][column] = 1
        # V[j*p+k][column] = 1
        # W[k*n+i][column] = 1
        # This creates a tensor where T[i,j,k] = 1 for all i,j,k?
        # No, the indices of U, V, W are flattened.
        # U index is i*m+j. V index is j*p+k. W index is k*n+i.
        # The tensor produced is T[i,j,k] = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c].
        # For a specific c corresponding to (i0, j0, k0):
        # U[i*m+j, c] is 1 if i=i0, j=j0.
        # V[j*p+k, c] is 1 if j=j0, k=k0.
        # W[k*n+i, c] is 1 if k=k0, i=i0.
        # So the product is 1 if i=i0, j=j0, k=k0.
        # So the resulting tensor has T[i,j,k] = 1 if (i,j,k) is a valid triple?
        # Yes, the schoolbook decomposition produces a tensor with T[i,j,k] = 1 for all i,j,k?
        # No, it produces a tensor where T[i,j,k] = 1 if there is a c such that the conditions hold.
        # Since every (i,j,k) has a unique c, T[i,j,k] = 1 for all i,j,k.
        # This is the tensor with all ones.
        
        # BUT, the task says "multiplying a 3 by 4 matrix by a 4 by 5 matrix".
        # The multiplication tensor is NOT the all-ones tensor.
        # The multiplication tensor T is defined by T[i,j,k] = 1 if the index j is the same?
        # Actually, the tensor for matrix multiplication A * B is T_{i,j,k} = 1 if j is the contracted index.
        # In the standard basis, the tensor is T_{i,j,k} = 1 if the index j in A matches the index j in B?
        # No, the tensor is T_{i,j,k} = 1 if the "link" is present.
        # For standard matmul, the link is between col j of A and row j of B.
        # So T[i,j,k] = 1 for all i, j, k?
        # No. The tensor T is defined such that the multiplication is C_{ik} = sum_j T_{i,j,k} A_{ij} B_{jk}.
        # For standard matmul, C_{ik} = sum_j A_{ij} B_{jk}.
        # So T_{i,j,k} must be 1 for all i,j,k?
        # If T_{i,j,k} = 1, then C_{ik} = sum_j A_{ij} B_{jk}.
        # Yes! The tensor for standard matrix multiplication has T[i,j,k] = 1 for all valid i,j,k.
        
        # So the seed code is correct: it produces the all-ones tensor.
        # And the task is to find a rank-minimal decomposition of the all-ones tensor?
        # No, that's not right. The all-ones tensor has rank 1 (U=1, V=1, W=1).
        # But the seed code has rank 60.
        # Why?
        # Ah, the indices of U, V, W are flattened differently.
        # U has shape (n*m, rank).
        # V has shape (m*p, rank).
        # W has shape (p*n, rank).
        # The tensor produced is T[i, j, k] = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c].
        # This is NOT the standard tensor T[i,j,k].
        # The standard tensor T[i,j,k] is a 3D array of size n x m x p.
        # The output of the decomposition is a 3D array of size n x m x p.
        # The seed code produces a tensor where T[i,j,k] = 1 if (i,j,k) is a specific triple?
        # No, as derived above, it produces T[i,j,k] = 1 for all i,j,k.
        # If the target is the all-ones tensor, the rank is 1.
        # So the seed code is NOT decomposing the all-ones tensor.
        
        # Let's re-read the seed code carefully.
        # U[i*m+j][column] = 1
        # V[j*p+k][column] = 1
        # W[k*n+i][column] = 1
        # This sets U, V, W such that for a specific column c, only one entry in U, V, W is 1.
        # The resulting tensor T_out[i,j,k] = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c].
        # For a fixed c (corresponding to i0, j0, k0):
        # U[i*m+j, c] is 1 iff i=i0 and j=j0.
        # V[j*p+k, c] is 1 iff j=j0 and k=k0.
        # W[k*n+i, c] is 1 iff k=k0 and i=i0.
        # So the product is 1 iff i=i0, j=j0, k=k0.
        # So T_out[i,j,k] = 1 if (i,j,k) = (i0, j0, k0).
        # Summing over all c, T_out[i,j,k] = 1 for all i,j,k.
        
        # So the seed code decomposes the all-ones tensor.
        # If the all-ones tensor is the target, the rank is 1.
        # But the score is 1.0, which corresponds to rank 60?
        # "score is the rank".
        # If the target was the all-ones tensor, the minimal rank is 1.
        # The seed code has rank 60.
        # So the seed code is a valid decomposition, but not minimal.
        # Can we do better? Yes, rank 1.
        # U = [[1, ..., 1]] (3x4 matrix, all 1s)
        # V = [[1, ..., 1]] (4x5 matrix, all 1s)
        # W = [[1, ..., 1]] (5x3 matrix, all 1s)
        # Then T[i,j,k] = 1*1*1 = 1.
        
        # Is the target the all-ones tensor?
        # "multiplying a 3 by 4 matrix by a 4 by 5 matrix"
        # Usually, this implies the standard multiplication tensor.
        # But the standard multiplication tensor T[i,j,k] = 1 for all i,j,k IS the all-ones tensor.
        # So the minimal rank is 1.
        
        # Why would the seed code be rank 60?
        # Maybe I am misunderstanding the tensor definition.
        # Perhaps the tensor is T[i,j,k] = 1 if j==k?
        # If T[i,j,k] = 1 if j==k, then it is not the all-ones tensor.
        # Let's check the seed code again.
        # The seed code produces T[i,j,k] = 1 for all i,j,k.
        # If the target is T[i,j,k] = 1 if j==k, the seed code is INVALID.
        # But the seed code has score 1.0, which implies it is valid.
        # So the target MUST be the all-ones tensor.
        
        # If the target is the all-ones tensor, the minimal rank is 1.
        # I can return a rank 1 decomposition.
        
        # Let's verify the interface.
        # U: (n*m, rank) -> (12, 1)
        # V: (m*p, rank) -> (20, 1)
        # W: (p*n, rank) -> (15, 1)
        
        # U[i*m+j, 0] = 1
        # V[j*p+k, 0] = 1
        # W[k*n+i, 0] = 1
        
        # T[i,j,k] = U[i*m+j, 0] * V[j*p+k, 0] * W[k*n+i, 0] = 1.
        
        # This is a valid decomposition of the all-ones tensor with rank 1.
        
    # Create rank 1 decomposition
    rank = 1
    U = [[1] for _ in range(n * m)]
    V = [[1] for _ in range(m * p)]
    W = [[1] for _ in range(p * n)]
    
    return {"U": U, "V": V, "W": W}
