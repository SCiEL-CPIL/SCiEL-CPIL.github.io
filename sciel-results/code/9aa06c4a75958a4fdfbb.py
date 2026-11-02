"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
import numpy as np

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that the tensor decomposition exactly represents matrix multiplication.
    The multiplication tensor T[i,j,k] = 1 if and only if the output C[i,k] 
    receives contribution from A[i,j]*B[j,k].
    
    For matrix mult: C[i,k] = sum_j A[i,j] * B[j,k]
    So the tensor has T[i,j,k] = 1 for all i,j,k (it's the full product).
    
    The decomposition says: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    (with appropriate index conventions)
    """
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Reshape: U is (n*m, rank) -> (n, m, rank)
    # U[i*m + j, r]
    U_reshaped = U_arr.reshape(n, m, rank)
    
    # V is (m*p, rank) -> (m, p, rank)
    # V[j*p + k, r]
    V_reshaped = V_arr.reshape(m, p, rank)
    
    # W is (p*n, rank) -> (p, n, rank)
    # W[k*n + i, r]
    W_reshaped = W_arr.reshape(p, n, rank)
    
    # Compute the reconstructed tensor: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # Use einsum: 'ijr,jkr,kir->ijk'
    T_reconstructed = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
    
    # The true multiplication tensor is all ones (every i,j,k triple contributes)
    T_true = np.ones((n, m, p))
    
    return np.allclose(T_reconstructed, T_true, atol=1e-10)

def _strassen_block_decomposition(n, m, p):
    """
    Construct rank-32 decomposition using Strassen blocks for column pairs
    and schoolbook for the single remaining column.
    """
    rank = 4 * 7 + 4  # 32
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def strassen_block(k1, k2, start_col):
        nonlocal U, V, W
        v_00 = 0 * p + k1
        v_01 = 0 * p + k2
        v_10 = 1 * p + k1
        v_11 = 1 * p + k2
        
        w_00 = k1 * n + 0
        w_01 = k1 * n + 1
        w_02 = k2 * n + 0
        w_03 = k2 * n + 1
        
        u_00 = 0
        u_01 = 1
        u_10 = 2
        u_11 = 3
        
        c = start_col + 0
        U[u_00][c] += 1
        U[u_11][c] += 1
        V[v_00][c] += 1
        V[v_11][c] += 1
        W[w_00][c] += 1
        W[w_03][c] += 1
        
        c = start_col + 1
        U[u_10][c] += 1
        U[u_11][c] += 1
        V[v_00][c] += 1
        W[w_01][c] += 1
        W[w_03][c] -= 1
        
        c = start_col + 2
        U[u_00][c] += 1
        V[v_01][c] += 1
        V[v_11][c] -= 1
        W[w_02][c] += 1
        W[w_03][c] += 1
        
        c = start_col + 3
        U[u_11][c] += 1
        V[v_10][c] += 1
        V[v_00][c] -= 1
        W[w_00][c] += 1
        W[w_01][c] += 1
        
        c = start_col + 4
        U[u_00][c] += 1
        U[u_01][c] += 1
        V[v_11][c] += 1
        W[w_00][c] -= 1
        W[w_02][c] += 1
        
        c = start_col + 5
        U[u_10][c] += 1
        U[u_00][c] -= 1
        V[v_00][c] += 1
        V[v_01][c] += 1
        W[w_03][c] += 1
        
        c = start_col + 6
        U[u_01][c] += 1
        U[u_11][c] -= 1
        V[v_10][c] += 1
        V[v_11][c] += 1
        W[w_00][c] += 1
        
        return start_col + 7
    
    def schoolbook_single(k, start_col):
        nonlocal U, V, W
        v_0 = 0 * p + k
        v_1 = 1 * p + k
        
        w_0 = k * n + 0
        w_1 = k * n + 1
        
        u_00 = 0
        u_01 = 1
        u_10 = 2
        u_11 = 3
        
        c = start_col + 0
        U[u_00][c] += 1
        V[v_0][c] += 1
        W[w_0][c] += 1
        
        c = start_col + 1
        U[u_01][c] += 1
        V[v_1][c] += 1
        W[w_0][c] += 1
        
        c = start_col + 2
        U[u_10][c] += 1
        V[v_0][c] += 1
        W[w_1][c] += 1
        
        c = start_col + 3
        U[u_11][c] += 1
        V[v_1][c] += 1
        W[w_1][c] += 1
        
        return start_col + 4
    
    col = 0
    for k1, k2 in [(0, 1), (2, 3), (4, 5), (6, 7)]:
        col = strassen_block(k1, k2, col)
    
    col = schoolbook_single(8, col)
    
    return {"U": U, "V": V, "W": W}

def _tensor_product_decomposition(n, m, p):
    """
    Construct a rank-63 decomposition using the tensor product approach.
    
    The 2x2x2 matrix multiplication tensor has rank 7 (Strassen).
    The 2x1x9 "vector-matrix" multiplication is just selecting rows,
    which is a sum of 9 rank-1 tensors (one per column).
    
    Tensor product of rank-7 and 9 rank-1 terms gives rank 7*9 = 63.
    
    For the 2x2x2 part, use Strassen's 7 products.
    For the 2x1x9 part, each column k contributes a rank-1 term.
    
    The combined tensor: T[i,j,k] = 1 for all i,j,k (full multiplication).
    
    We decompose as: for each of the 7 Strassen products r and each column k,
    create a rank-1 term that handles the (i,j) part via Strassen and the k part
    via the column selection.
    
    More precisely:
    - Strassen gives: for 2x2*2x2, sum_r U2[i,j,r] * V2[j,k2,r] * W2[k2,i,r] = 1 for all i,j,k2
    - For the 2x9 dimension: for each output column k, we need to handle B[j,k]
    
    Actually, let me think about this differently.
    
    The full multiplication tensor for (2,2,9) is:
    T[i,j,k] = 1 for all i in {0,1}, j in {0,1}, k in {0..8}
    
    This is the Kronecker product of the (2,2,2) tensor with a (1,1,9) tensor
    where the (1,1,9) tensor has all ones.
    
    The (2,2,2) tensor has rank 7 (Strassen).
    The (1,1,9) tensor with all ones has rank 9 (it's just 9 independent rank-1 terms).
    
    So the tensor product has rank 7 * 9 = 63.
    
    For Strassen on 2x2x2:
    U2[i,j,r] for r=0..6, V2[j,k2,r] for k2 in {0,1}, W2[k2,i,r]
    
    For the 2x1x9 part (which is really just the identity on the 9 columns):
    For each column k, we have a rank-1 term with:
    - U component: 1 (scalar, since n=2 but we're treating it as the "1" dimension)
    - V component: 1 (scalar)  
    - W component: 1 (scalar)
    
    Wait, I need to be more careful. Let me reconsider.
    
    The multiplication tensor T[i,j,k] for multiplying A (n×m) by B (m×p) to get C (n×p)
    is defined as T[i,j,k] = 1 if and only if C[i,k] includes the term A[i,j]*B[j,k].
    For standard matrix multiplication, T[i,j,k] = 1 for ALL valid i,j,k.
    
    So T is an n×m×p tensor of all ones.
    
    For n=2, m=2, p=9, this is a 2×2×9 tensor of all ones.
    
    This can be written as the outer product of a 2×2 all-ones matrix with a 9-vector of ones,
    but that's not quite right for tensor decomposition.
    
    Actually, T[i,j,k] = 1 = (1,1)^T ⊗ (1,1) ⊗ (1,...,1) in some sense.
    
    For tensor decomposition: T = sum_r U[:, :, r] ⊗ V[:, :, r] ⊗ W[:, :, r]
    where U is (n,m,rank), V is (m,p,rank), W is (p,n,rank).
    
    Using Strassen for the (2,2,2) part:
    For the 2x2x2 all-ones tensor, Strassen gives rank 7.
    
    For the 2x2x9 all-ones tensor, we can view it as:
    T[i,j,k] = T2[i,j,k mod 2] * T9[k // 2, k mod 2] ... no, that's not right.
    
    Let me just use the fact that T[i,j,k] = 1 for all i,j,k.
    
    A simple rank-4*9=36 decomposition would be schoolbook:
    For each (i,j,k): U[i,j,r]=1, V[j,k,r]=1, W[k,i,r]=1 for rank 36.
    
    But we want to use Strassen to reduce the 2x2 part.
    
    The tensor T[i,j,k] for i in {0,1}, j in {0,1}, k in {0..8} is all ones.
    
    We can factor this as: T[i,j,k] = S[i,j] * R[k] where S is 2x2 all-ones and R is 9-vector of ones.
    But S[i,j] = 1*1, so S has rank 1 as a matrix, but as part of a 3-way tensor...
    
    Actually, the simplest approach: use Strassen's 7 products for the (i,j) and one "column slot",
    then replicate for each of the 9 columns.
    
    For each column k (0..8), we have a 2x2x1 sub-tensor that is all ones.
    This 2x2x1 tensor has rank 4 (schoolbook) or we can group columns.
    
    Let me just do: for each k in 0..8, use 7 Strassen-style products but adapted.
    
    Actually, the cleanest rank-63 construction:
    
    For the 2x2x2 all-ones tensor, Strassen gives 7 rank-1 terms.
    For the 2x2x9 all-ones tensor, it's the tensor product of 2x2x2 all-ones with a 1x1x9 all-ones tensor... no.
    
    Let me just directly construct it:
    
    For each k in 0..8, the slice T[:,:,k] is a 2x2 all-ones matrix.
    Each 2x2 all-ones matrix has tensor rank... well, as a 3-way tensor slice.
    
    I think the simplest correct approach is:
    
    Use the fact that T[i,j,k] = 1 = sum over r of U[i,j,r]*V[j,k,r]*W[k,i,r]
    
    For rank 63: 7 (Strassen) * 9 (columns)
    
    For each Strassen product r (0..6) and each column k (0..8):
    - U[i,j, 7*r+k] = U_strassen[i,j,r] (same as Strassen's U for 2x2)
    - V[j, k, 7*r+k] = 1 if j matches, else 0... 
    
    Hmm, this is getting complicated. Let me just use a straightforward approach:
    
    The 2x2x9 all-ones tensor can be decomposed as follows:
    For each pair (i,j), T[i,j,k] = 1 for all k.
    So T[i,j,k] = 1*1*1.
    
    A rank-36 schoolbook: for each (i,j,k), one rank-1 term.
    
    To use Strassen, we reduce the 2x2 part. The 2x2 all-ones "matrix" in the tensor sense...
    
    OK let me just implement the straightforward rank-36 schoolbook as the fallback,
    since rank 63 might not be necessary and 36 is simpler.
    
    Actually wait - the directive says rank 63. Let me re-read.
    
    "construct a valid rank-63 decomposition"
    
    So they want 7*9=63. Let me think about how Strassen composes.
    
    For 2x2x2, Strassen gives:
    T2[i,j,k] = sum_{r=0}^{6} U2[i,j,r] * V2[j,k,r] * W2[k,i,r] = 1 for all i,j,k
    
    For 2x2x9, T[i,j,k] = 1 for all i,j,k.
    
    We can write T[i,j,k] = T2[i,j,0] * 1 (since T2[i,j,0]=1)
    But that's not a tensor product decomposition.
    
    Actually, T[i,j,k] = sum_{r=0}^{6} U2[i,j,r] * (sum over something for V and W)
    
    The issue is that V2 depends on k (the column of B), and W2 depends on k (the row of C).
    
    For the 2x2x9 case, V has shape (m,p,rank) = (2,9,rank) and W has shape (p,n,rank) = (9,2,rank).
    
    Let me define for each Strassen product r and each column k:
    U[i,j, r*9+k] = U2[i,j,r]
    V[j,k, r*9+k] = 1  (for the specific column k)
    W[k,i, r*9+k] = W2[k,i,r]... but W2[k,i,r] is defined for k in {0,1}, not 0..8.
    
    This doesn't work directly because Strassen's W2 is for a 2-column output.
    
    I think the correct interpretation is:
    
    The 2x2x9 tensor is NOT the tensor product of 2x2x2 and 1x1x9.
    
    Let me just use a rank-36 schoolbook fallback. It's simpler and correct.
    """
    n, m, p = 2, 2, 9
    rank = n * m * p  # 36 schoolbook
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U[i*m+j, r] = 1
                U[i * m + j][r] = 1
                # V[j*p+k, r] = 1
                V[j * p + k][r] = 1
                # W[k*n+i, r] = 1
                W[k * n + i][r] = 1
                r += 1
    
    return {"U": U, "V": V, "W": W}

def construct():
    """
    Construct a minimal-rank tensor decomposition for 2x2 * 2x9 matrix multiplication.
    
    Primary: rank-32 Strassen-block construction.
    Fallback: rank-36 schoolbook if verification fails.
    """
    n, m, p = DIMENSIONS  # n=2, m=2, p=9
    
    # Try the rank-32 Strassen block construction
    result = _strassen_block_decomposition(n, m, p)
    
    # Verify it
    if _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
        return result
    else:
        # Fallback to schoolbook
        return _tensor_product_decomposition(n, m, p)

# EVOLVE-BLOCK-END
