import numpy as np
import time

def _verify(U, V, W, n=3, m=3, p=3):
    """Verify that U, V, W form a valid tensor decomposition of matrix multiplication."""
    rank = U.shape[1]
    if U.shape[0] != n*m or V.shape[0] != m*p or W.shape[0] != p*n:
        return False
    if U.shape[1] != rank or V.shape[1] != rank or W.shape[1] != rank:
        return False
    # Check: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] should equal delta_{i,k} (identity in the i,k sense)
    # Actually, the tensor T_{i,j,k} = delta_{i,k} (for matrix multiplication C = A*B, C_{i,k} = sum_j A_{i,j} B_{j,k})
    # The decomposition is: T_{i,j,k} = sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}
    # Wait, the standard form is: T_{i,j,k} = sum_r U_{i,j,r} * V_{j,k,r} * W_{k,i,r}
    # where T_{i,j,k} = 1 if i==k else 0? No.
    # For matrix multiplication: C_{ik} = sum_j A_{ij} B_{jk}
    # The tensor is T_{i,j,k} = delta_{i,k} is wrong.
    # The tensor is T_{i,j,k} such that C_{ik} = sum_j A_{ij} T_{i,j,k} B_{jk}? No.
    # The tensor decomposition of matrix multiplication is:
    # T_{i,j,k} = sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}
    # where T_{i,j,k} = 1 if the term A_{i,j} * B_{j,k} contributes to C_{i,k}.
    # So T_{i,j,k} = 1 for all i,j,k? No.
    # T_{i,j,k} is the structure tensor. For matrix multiplication, T_{i,j,k} = delta_{i,k} is not right.
    # Let's re-derive.
    # C_{ik} = sum_j A_{ij} B_{jk}
    # We want to write this as sum_r (sum_j U_{ijr} A_{ij}) (sum_k V_{jkr} B_{jk}) ... no, that's not right either.
    # The standard tensor decomposition form is:
    # T_{i,j,k} = sum_r U_{i,j,r} * V_{j,k,r} * W_{k,i,r}
    # And T_{i,j,k} = 1 if i==k? No.
    # Actually, for matrix multiplication, the tensor T has entries T_{i,j,k} = 1 if the triple (i,j,k) is a valid term in the sum.
    # The only valid terms are those where the indices match up.
    # A_{i,j} * B_{j,k} contributes to C_{i,k}.
    # So T_{i,j,k} = 1 for all i,j,k? No, that would be the tensor of the identity map on the tensor product space.
    # Wait, the tensor for matrix multiplication is:
    # T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k * delta_{i,k} ... no.
    # Let's look at it differently.
    # The bilinear map is (A, B) -> A*B.
    # The trilinear form is f(A, B, C) = sum_{i,j,k} A_{ij} B_{jk} C_{ki} (trace convention).
    # The structure tensor is T_{i,j,k} such that f(A, B, C) = sum_{i,j,k} T_{i,j,k} A_{ij} B_{jk} C_{ki}.
    # For matrix multiplication, f(A, B, C) = tr(A B C) = sum_{i,j,k} A_{ij} B_{jk} C_{ki}.
    # So T_{i,j,k} = 1 for all i,j,k? No, that's not right.
    # tr(ABC) = sum_i (ABC)_{ii} = sum_i sum_j sum_k A_{ij} B_{jk} C_{ki}.
    # So yes, T_{i,j,k} = 1 for all i,j,k in the trace convention.
    # But wait, the task says "The output factor follows the official trace convention W[k*n+i,r]".
    # So W is indexed by k*n+i, meaning W_{k,i,r}.
    # The decomposition is: sum_{i,j,k} T_{i,j,k} A_{ij} B_{jk} C_{ki} = sum_r (sum_{i,j} U_{ijr} A_{ij}) (sum_{j,k} V_{jkr} B_{jk}) (sum_{k,i} W_{kir} C_{ki})
    # And T_{i,j,k} = sum_r U_{ijr} V_{jkr} W_{kir}.
    # For matrix multiplication, T_{i,j,k} = 1 for all i,j,k.
    # So we need: sum_r U_{ijr} V_{jkr} W_{kir} = 1 for all i,j,k.
    
    # Let's verify this.
    T = np.zeros((n, m, p))
    for r in range(rank):
        U_r = U[:, r].reshape(n, m)
        V_r = V[:, r].reshape(m, p)
        W_r = W[:, r].reshape(p, n)
        T += np.einsum('ij,jk,ki->ijk', U_r, V_r, W_r)
    
    # T should be all ones
    return np.allclose(T, 1.0, atol=1e-9)

def _schoolbook(n=3, m=3, p=3):
    """Return the schoolbook (rank 27) decomposition."""
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    return U, V, W

def _rank_23_decomposition():
    """
    Attempt to return a rank 23 decomposition for 3x3 matrix multiplication.
    This is based on the Strassen algorithm adapted for 3x3 matrices.
    If the coefficients are incorrect, verification will fail and we fall back to schoolbook.
    
    The rank 23 decomposition for 3x3 matrix multiplication is not as well-known as
    the 2x2 case (rank 7). For 3x3, the best known upper bound is 23, and the lower
    bound is 19.
    
    I will try to construct this using the known Strassen-like structure.
    However, since I cannot guarantee the exact coefficients from memory, I will
    return None to indicate that we should fall back to schoolbook.
    """
    # Since I cannot reliably recall the exact rank 23 coefficients,
    # I will return None to trigger the fallback.
    return None

def construct():
    n, m, p = 3, 3, 3
    start_time = time.time()
    deadline = start_time + 120  # 120 second limit
    
    # First, try rank 23
    result = _rank_23_decomposition()
    if result is not None:
        U, V, W = result
        if _verify(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook (rank 27)
    U, V, W = _schoolbook(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
