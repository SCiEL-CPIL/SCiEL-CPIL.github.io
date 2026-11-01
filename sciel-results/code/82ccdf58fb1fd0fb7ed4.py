"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _verify_decomposition(U, V, W, n=3, m=3, p=3, tol=1e-6):
    """Verify that U, V, W form a valid tensor decomposition."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # The multiplication tensor T_{ijk} = sum_r U_{i,j,r} * V_{j,k,r} * W_{k,i,r}
    # should equal 1 if i==j and k==j (i.e., A_{ij} * B_{jk} contributes to C_{ik} when j is the inner index)
    # Wait, the standard decomposition for matrix multiplication C = A * B where:
    # C_{ik} = sum_j A_{ij} B_{jk}
    # The tensor is T_{i,j,k} = 1 if the triple (i,j,k) is a valid path (which it always is in the sum), 
    # but actually the tensor for matrix multiplication has T_{i,j,k} = 1 for all i,j,k? No.
    # The multiplication map is a bilinear map. The tensor T is in R^{n*m} x R^{m*p} x R^{p*n}.
    # The entry T_{(i,j), (j,k), (k,i)} ... wait, the indices are:
    # U is indexed by (i, j) where i is row of A, j is col of A / row of B.
    # V is indexed by (j, k) where j is row of B, k is col of B / col of C.
    # W is indexed by (k, i) where k is row of C, i is col of C.
    # The decomposition says: sum_r U_{(i,j),r} V_{(j,k),r} W_{(k,i),r} = [i,j,k] is a valid term?
    # Actually, the tensor T_{(i,j),(j,k),(k,i)} should be 1 if and only if this term contributes.
    # In matrix multiplication C_{ik} = sum_j A_{ij}B_{jk}, the "tensor" is the identity on the middle index.
    # So T_{i,j,k} (where the first index is (i,j), second is (j,k), third is (k,i)) 
    # is 1 if the indices match up? No, the tensor is just the set of basis tensors e_{ij} ⊗ e_{jk} ⊗ e_{ki}.
    # The multiplication tensor is sum_{i,j,k} e_{(i,j)} ⊗ e_{(j,k)} ⊗ e_{(k,i)}.
    # So for each (i,j,k), the entry is 1.
    # Thus we need: sum_r U_{(i,j),r} V_{(j,k),r} W_{(k,i),r} = 1 for ALL i,j,k.
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U_arr[i*m+j] * V_arr[j*p+k] * W_arr[k*n+i])
                if abs(s - 1.0) > tol:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n, m, p = 3, 3, 3
    
    # Schoolbook decomposition (rank 27)
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    
    # Verify
    if not _verify_decomposition(U, V, W):
        pass  # Should never happen
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
