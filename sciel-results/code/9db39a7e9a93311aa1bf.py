"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
import numpy as np
from itertools import product
import time

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # For each (i, j, k) where the multiplication tensor has a 1,
    # check that sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    # And for all other (i, j, k) combinations that don't exist in the tensor, they should be 0.
    # Actually, the tensor T[i,j,k] = 1 if the (i,j) entry of A and (j,k) entry of B multiply to contribute to (i,k) of C.
    # For standard matrix multiplication, T[i,j,k] = 1 for all valid i,j,k.
    
    # The decomposition should satisfy:
    # For each c: U[:,c], V[:,c], W[:,c] form a rank-1 tensor
    # Sum over c of these rank-1 tensors = multiplication tensor
    
    # Let's verify by checking that the reconstructed tensor matches
    # The multiplication tensor has T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    
    # Actually, we need to be more careful. The tensor is defined as:
    # T[i,j,k] = delta (the entry that multiplies A[i,j] * B[j,k] to give C[i,k])
    # For standard matrix multiplication, T[i,j,k] = 1 for all valid indices.
    
    # Let's reconstruct and check
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for c in range(rank):
                    val += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                if abs(val - 1) > 1e-9:
                    return False
    return True

def _find_3x3x3_decomposition(max_rank=23, timeout=30):
    """Find a low-rank decomposition for 3x3x3 matrix multiplication."""
    start_time = time.time()
    n = m = p = 3
    
    # Try ranks from known lower bounds up
    # The Strassen algorithm gives rank 7 for 2x2x2
    # For 3x3x3, the best known is around 19-23
    
    for rank in range(19, max_rank + 1):
        if time.time() - start_time > timeout:
            break
        
        # Try to find a decomposition using structured patterns
        # Use the fact that we can block the 3x3x3 tensor
        
        # Simple approach: try to find U, V, W with small integer/half-integer entries
        # This is a hard problem, so let's use known constructions
        
        # For now, return a schoolbook decomposition if we can't find better
        pass
    
    # Return schoolbook for 3x3x3 as fallback
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
    return {"U": U, "V": V, "W": W, "rank": rank}

def _combine_3x3_decompositions(decomp1, decomp2):
    """Combine two 3x3x3 decompositions to form a 3x3x6 decomposition."""
    # decomp1 handles columns 0,1,2 of the 3x6 matrix
    # decomp2 handles columns 3,4,5 of the 3x6 matrix
    
    rank1 = decomp1["rank"]
    rank2 = decomp2["rank"]
    total_rank = rank1 + rank2
    
    n, m = 3, 3
    p = 6
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Fill in decomp1 (for k=0,1,2)
    for i in range(n):
        for j in range(m):
            for c in range(rank1):
                U[i*m+j][c] = decomp1["U"][i*m+j][c]
    
    for j in range(m):
        for k in range(3):  # k=0,1,2
            for c in range(rank1):
                V[j*p+k][c] = decomp1["V"][j*3+k][c]
    
    for k in range(3):  # k=0,1,2
        for i in range(n):
            for c in range(rank1):
                W[k*n+i][c] = decomp1["W"][k*n+i][c]
    
    # Fill in decomp2 (for k=3,4,5)
    offset = rank1
    for i in range(n):
        for j in range(m):
            for c in range(rank2):
                U[i*m+j][offset+c] = decomp2["U"][i*m+j][c]
    
    for j in range(m):
        for k in range(3, 6):  # k=3,4,5
            k_local = k - 3
            for c in range(rank2):
                V[j*p+k][offset+c] = decomp2["V"][j*3+k_local][c]
    
    for k in range(3, 6):  # k=3,4,5
        k_local = k - 3
        for i in range(n):
            for c in range(rank2):
                W[k*n+i][offset+c] = decomp2["W"][k_local*n+i][c]
    
    return {"U": U, "V": V, "W": W, "rank": total_rank}

def construct():
    # Schoolbook decomposition of rank n*m*p: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c].
    n, m, p = DIMENSIONS
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
    
    # Try to find a better decomposition by splitting into two 3x3x3 problems
    decomp1 = _find_3x3x3_decomposition(max_rank=23, timeout=20)
    decomp2 = _find_3x3x3_decomposition(max_rank=23, timeout=20)
    
    combined = _combine_3x3_decompositions(decomp1, decomp2)
    
    # Verify the combined decomposition
    if _verify_decomposition(combined["U"], combined["V"], combined["W"], n, m, p):
        if combined["rank"] < rank:
            return {"U": combined["U"], "V": combined["V"], "W": combined["W"]}
    
    # Fall back to schoolbook
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
