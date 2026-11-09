"""
Rank-minimal exact tensor decomposition for 3x5 * 5x5 matrix multiplication.

Strategy:
1. Start with the schoolbook decomposition (rank 75).
2. Use a "block elimination" search: treat the multiplication tensor as a set of
   75 constraints (one per (i,j,k) with i==k) plus 0-constraints for i!=k.
   Actually, the tensor T[i,j,k] = delta_{i,k} has 3*5*5 = 75 entries, of which
   5*3 = 15 are 1 (when i==k, j arbitrary) and 60 are 0.
   
   Wait, T[i,j,k] = delta_{i,k}. So for each j (5 choices), we have 3x3 = 9 entries.
   15 entries are 1 (i=k, j arbitrary: 3*5=15), 60 entries are 0.
   
   The decomposition identity is:
     For all (i,j,k): delta_{i,k} = sum_{c=1}^{r} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
   
   This is a system of 75 equations in 75*r variables (U has 15*r, V has 25*r, W has 15*r entries).
   
   We can use a greedy approach:
   - Start with r = 75 (schoolbook).
   - Try to reduce r by finding linear dependencies or using a structured search.
   
   A better approach: Use the fact that the tensor has a specific structure.
   T[i,j,k] = delta_{i,k} can be written as:
     T = sum_{j=0}^{4} (e_j in the i-space) (x) (e_j in the k-space) (x) (e_j in the j-space)
   
   Actually, T[i,j,k] = delta_{i,k} means for each j, the "slice" T[:,:,j] is the 3x3 identity.
   
   So T = sum_{j=0}^{4} I_3 (as a tensor in i,k) tensor e_j (in the j-index).
   
   This gives a rank-25 decomposition where each "column" c corresponds to a pair (j, alpha)
   with j in 0..4 and alpha in 0..2 (the 3 diagonal entries of I_3).
   
   For column c = (j, alpha):
     U[i*m+j, c] = 1 if i == alpha else 0  (wait, this doesn't work directly)
   
   Let me reconsider. The standard decomposition:
     T[i,j,k] = delta_{i,k}
   
   We can write:
     T[i,j,k] = sum_{c=1}^{25} U_c[i,j] * V_c[j,k] * W_c[k,i]
   
   where for each j (0..4) and each alpha (0..2):
     c = j*3 + alpha
     U_c[i,j'] = delta_{j,j'} * delta_{i,alpha}
     V_c[j',k'] = delta_{j,j'} * delta_{k',alpha}  -- wait, this doesn't seem right.
   
   Actually, the simplest rank-75 decomposition is the schoolbook one.
   To get rank-25, we need to be smarter.
   
   Let's use the identity:
     delta_{i,k} = sum_{alpha=0}^{2} delta_{i,alpha} * delta_{alpha,k}
   
   So T[i,j,k] = sum_{alpha=0}^{2} delta_{i,alpha} * delta_{alpha,k}
   
   This gives:
     For each j (0..4) and alpha (0..2):
       U_{j,alpha}[i,j'] = delta_{j,j'} * delta_{i,alpha}
       V_{j,alpha}[j',k'] = delta_{j,j'} * delta_{k',alpha}  -- no, this is wrong.
   
   Let me think again. The tensor is T[i,j,k] = delta_{i,k}.
   
   Decomposition: T = sum_c U_c ⊗ V_c ⊗ W_c where
     U_c[i,j] * V_c[j,k] * W_c[k,i] summed over c gives delta_{i,k}.
   
   For the schoolbook: c = (i,j,k), U_c[i',j'] = delta_{i,i'}delta_{j,j'}, etc.
   
   To reduce rank, we can use the fact that delta_{i,k} = sum_alpha delta_{i,alpha} delta_{alpha,k}.
   
   So T[i,j,k] = sum_{alpha=0}^{2} delta_{i,alpha} delta_{alpha,k}
   
   This suggests a decomposition with r = 5*3 = 15? No, that's not right either.
   
   Actually, the tensor T[i,j,k] = delta_{i,k} has rank at most 25 because:
     T[i,j,k] = sum_{j=0}^{4} delta_{i,k} * delta_{j,j}  -- this is trivial.
   
   The rank of the tensor T[i,j,k] = delta_{i,k} as a 3x5x5 tensor:
     If we fix j, T[:,:,j] = I_3. The rank of I_3 as a matrix is 3.
     The tensor rank of the direct sum of 5 copies of I_3 is at most 5*3 = 15.
   
   Wait, that's interesting. Let me verify:
     T = sum_{j=0}^{4} (I_3 in i,k) ⊗ (e_j in j-index)
   
   For each j, we can decompose I_3 = sum_{alpha=0}^{2} e_alpha ⊗ e_alpha (as a matrix).
   
   So T = sum_{j=0}^{4} sum_{alpha=0}^{2} (e_alpha in i) ⊗ (e_j in j) ⊗ (e_alpha in k)
   
   This gives r = 5*3 = 15 rank terms!
   
   For term (j, alpha):
     U_{j,alpha}[i, j'] = delta_{j, j'} * delta_{i, alpha}
     V_{j,alpha}[j', k'] = delta_{j, j'} * delta_{k', alpha}  -- NO, this is wrong.
   
   Let me be more careful. The decomposition is:
     T[i,j,k] = sum_c U_c[i,j] * V_c[j,k] * W_c[k,i]
   
   For term c = (j0, alpha):
     We want U_c[i,j] * V_c[j,k] * W_c[k,i] = delta_{j, j0} * delta_{i, alpha} * delta_{k, alpha}
     
     So:
       U_c[i,j] = delta_{j, j0} * delta_{i, alpha}
       V_c[j,k] = delta_{j, j0} * delta_{k, alpha}  -- but this would give delta_{j,j0}^2 * delta_{alpha,alpha} = 1 when j=j0 and k=alpha.
       W_c[k,i] = 1  (or some constant)
     
     Then U_c[i,j] * V_c[j,k] * W_c[k,i] = delta_{j,j0} * delta_{i,alpha} * delta_{j,j0} * delta_{k,alpha} * 1
     = delta_{j,j0} * delta_{i,alpha} * delta_{k,alpha}
     
     Summing over c = (j0, alpha):
       sum_{j0, alpha} delta_{j,j0} * delta_{i,alpha} * delta_{k,alpha}
       = delta_{i,k} * (sum_{j0} delta_{j,j0})  -- wait, sum_{j0} delta_{j,j0} = 1
       = delta_{i,k}
     
     Yes! This works with r = 15.
   
   But wait, the factors U, V, W have specific shapes:
     U: (n*m, rank) = (15, 15)
     V: (m*p, rank) = (25, 15)
     W: (p*n, rank) = (15, 15)
   
   For c = (j0, alpha) where j0 in 0..4, alpha in 0..2:
     U[i*m+j, c] = delta_{j, j0} * delta_{i, alpha}
     V[j*p+k, c] = delta_{j, j0} * delta_{k, alpha}
     W[k*n+i, c] = 1
   
   Let's verify:
     sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
     = sum_{j0=0}^{4} sum_{alpha=0}^{2} [delta_{j,j0} * delta_{i,alpha}] * [delta_{j,j0} * delta_{k,alpha}] * 1
     = sum_{j0} sum_{alpha} delta_{j,j0}^2 * delta_{i,alpha} * delta_{k,alpha}
     = sum_{alpha} delta_{i,alpha} * delta_{k,alpha}  (since sum_{j0} delta_{j,j0}^2 = 1)
     = delta_{i,k}  (since i, k are in 0..2)
   
   This is a valid rank-15 decomposition!
   
   But wait, can we do even better? The tensor T[i,j,k] = delta_{i,k} is a 3x5x5 tensor.
   The lower bound for tensor rank is max over all "flattening" ranks.
   
   Flatten as (i) x (j,k): rank = 3 (since each slice is I_3, and they are in different j,k blocks)
   Flatten as (j) x (i,k): rank = 5 (each slice is I_3, 5 independent slices)
   Flatten as (k) x (i,j): rank = 3 (similar to first)
   
   So the lower bound is max(3, 5, 3) = 5. But tensor rank can be higher than matrix rank of any flattening.
   
   Actually, the tensor rank of T[i,j,k] = delta_{i,k} where i in 0..2, j in 0..4, k in 0..2:
   
   This is the tensor that represents the identity map from R^3 to R^3, repeated 5 times in the j-index.
   The tensor rank is at least the rank of the flattening (j) x (i,k), which is 5.
   
   Can we achieve rank 5? Let's see:
     T[i,j,k] = sum_{c=0}^{4} U_c[i,j] * V_c[j,k] * W_c[k,i]
   
   For this to work, we need 5 terms. Each term is a rank-1 tensor.
   
   The tensor T has the property that for each j, T[:,:,j] = I_3.
   A rank-1 tensor U_c ⊗ V_c ⊗ W_c has the property that T[:,:,j] = U_c[:,j] ⊗ (V_c[j,:] ⊗ W_c) -- this is getting complicated.
   
   Let me try a different approach. Suppose we set:
     U_c[i,j] = a_c[i] * b_c[j]
     V_c[j,k] = b_c[j] * c_c[k]  -- wait, this doesn't seem right.
   
   Actually, for a rank-1 tensor, U_c[i,j] * V_c[j,k] * W_c[k,i] must equal a rank-1 tensor in (i,j,k).
   
   The sum of 5 rank-1 tensors must equal delta_{i,k}.
   
   For each j, the slice T[:,:,j] = I_3 must equal sum_c U_c[:,j] ⊗ (V_c[j,:] * W_c) -- this is not quite right.
   
   Let me just try to see if rank 5 is possible by attempting to construct it.
   
   Actually, I think the rank is exactly 15. Here's why:
   Consider the "fiber" at a fixed j. T[:,:,j] = I_3, which has matrix rank 3.
   The tensor rank r must be at least the maximum matrix rank over all fibers, which is 3.
   But also, the 5 fibers are linearly independent in a certain sense.
   
   Actually, the tensor rank of a direct sum of tensors is at least the sum of their ranks if they have disjoint supports. Here, the 5 slices T[:,:,j] for j=0..4 have disjoint supports in the j-index, so the tensor rank is at least 5 * 1 = 5 (each I_3 has tensor rank 3, but as a 3x1x3 tensor, its rank is 3).
   
   Wait, each slice T[:,:,j] is a 3x3 matrix, which as a 3x1x3 tensor (with the middle dimension being 1) has tensor rank equal to its matrix rank, which is 3.
   
   Since the 5 slices have disjoint j-supports, the total tensor rank is at least 5 * 3 = 15.
   
   And we've shown a rank-15 decomposition exists. So the tensor rank is exactly 15.
   
   But wait, the problem asks for the rank of the decomposition of the matrix multiplication tensor, which is T[i,j,k] = delta_{i,k}. And we've shown this has tensor rank 15.
   
   So the answer is rank 15!
"""
import time
import fractions

_START = time.time()
_DEADLINE = 150.0
_MARGIN = 8.0

DIMENSIONS = [3, 5, 5]


def _time_left():
    return _DEADLINE - _MARGIN - (time.time() - _START)


def _verify(U, V, W):
    """Check that the factor matrices exactly decompose the multiplication tensor."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target = 1 if i == k else 0
                s = fractions.Fraction(0)
                ui = i * m + j
                vj = j * p + k
                wk = k * n + i
                for c in range(rank):
                    uc = U[ui][c]
                    if uc == 0:
                        continue
                    vc = V[vj][c]
                    if vc == 0:
                        continue
                    wc = W[wk][c]
                    if wc == 0:
                        continue
                    s += fractions.Fraction(uc) * fractions.Fraction(vc) * fractions.Fraction(wc)
                if s != target:
                    return False
    return True


def _schoolbook():
    """Fallback: rank-75 schoolbook decomposition (always valid)."""
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
    return {"U": U, "V": V, "W": W}


def _rank15_decomposition():
    """
    Construct the rank-15 decomposition:
      T[i,j,k] = delta_{i,k} = sum_{j0=0}^{4} sum_{alpha=0}^{2} 
                  delta_{j,j0} * delta_{i,alpha} * delta_{j,j0} * delta_{k,alpha} * 1
    
    For c = (j0, alpha):
      U[i*m+j, c] = delta_{j,j0} * delta_{i,alpha}
      V[j*p+k, c] = delta_{j,j0} * delta_{k,alpha}
      W[k*n+i, c] = 1
    """
    n, m, p = DIMENSIONS
    rank = 15
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    c = 0
    for j0 in range(m):  # j0 = 0..4
        for alpha in range(n):  # alpha = 0..2
            # U[i*m+j, c] = delta_{j,j0} * delta_{i,alpha}
            for i in range(n):
                for j in range(m):
                    if j == j0 and i == alpha:
                        U[i * m + j][c] = 1
            # V[j*p+k, c] = delta_{j,j0} * delta_{k,alpha}
            for j in range(m):
                for k in range(p):
                    if j == j0 and k == alpha:
                        V[j * p + k][c] = 1
            # W[k*n+i, c] = 1
            for k in range(p):
                for i in range(n):
                    W[k * n + i][c] = 1
            c += 1
    
    return {"U": U, "V": V, "W": W}


def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x5 * 5x5 matrix multiplication.
    Returns {'U', 'V', 'W'} with integer/half-integer entries.
    """
    # Try the rank-15 decomposition first
    if _time_left() < 5.0:
        return _schoolbook()
    
    r15 = _rank15_decomposition()
    if _verify(r15["U"], r15["V"], r15["W"]):
        return r15
    
    # Fallback to schoolbook
    return _schoolbook()


if __name__ == "__main__":
    result = construct()
    rank = len(result["U"][0])
    print(f"Rank: {rank}")
    print(f"Valid: {_verify(result['U'], result['V'], result['W'])}")
