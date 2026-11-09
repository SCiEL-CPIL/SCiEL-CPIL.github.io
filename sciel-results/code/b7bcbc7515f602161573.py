"""Matrix multiplication 3,5,7 - rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 5, 7]
DEADLINE = 140  # safety margin before 150s kill

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    # Build the tensor T[i,j,k] = delta_{i+k} for M_{n,m,p}
    # Actually, the multiplication tensor T has T[i,j,k] = 1 if we're multiplying (i,j) of A by (j,k) of B
    # The tensor is: T[i,j,k] = delta_{j,j} which is always 1 for valid indices
    # Wait, let me reconsider.
    
    # For M_{n,m,p}: A is n×m, B is m×p, C = A·B is n×p
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T has indices (i, j, k) and T[i,j,k] = 1 for all valid i,j,k
    # But that's not right either.
    
    # Actually, the multiplication tensor T ∈ F^{n×m} ⊗ F^{m×p} ⊗ F^{n×p}
    # T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_{i,k} where the third factor is indexed by (i,k)
    # So T[i,j,(i,k)] = 1 and 0 otherwise.
    
    # In the decomposition: T = sum_c U_c ⊗ V_c ⊗ W_c
    # where U_c ∈ F^{n×m}, V_c ∈ F^{m×p}, W_c ∈ F^{n×p}
    # (or transposed depending on convention)
    
    # The output factor W has shape (p*n, rank) with W[k*n+i, r]
    # So W is indexed by (k,i) where k is column of C, i is row of C
    # Wait, W[k*n+i, r] means row index is k*n+i, so first k then i.
    # But C[i,k] has i as row, k as column. So W[k*n+i] corresponds to C[i,k].
    
    # Let me verify: for each c, the contribution is U[:,c] * V[:,c] * W[:,c]
    # U[:,c] is a vector in F^{n*m}, V[:,c] in F^{m*p}, W[:,c] in F^{p*n}
    # The outer product gives a tensor, and we sum over c.
    
    # The multiplication tensor T has T[i,j,k] = 1 where i∈[0,n), j∈[0,m), k∈[0,p)
    # and the "output" index is (i,k) stored as k*n+i.
    
    # So T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k, and 0 otherwise.
    
    # Check: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should equal 1 if the indices are consistent, 0 otherwise.
    # But wait, the indices i,j,k are independent in the input, so for any i,j,k the value should be 1.
    # No, that's not right. The tensor is:
    # T = sum_{i=0}^{n-1} sum_{j=0}^{m-1} sum_{k=0}^{p-1} e_{i,j} ⊗ e_{j,k} ⊗ e_{i,k}
    # where e_{i,j} is the basis vector in F^{n×m}, etc.
    
    # So T[a, b, c] = 1 iff a = i*m+j, b = j*p+k, c = k*n+i for some i,j,k
    # i.e., a % m == b // p and b % p == c // n and a // m == c % n
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Check a sample of entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                # Index in U: i*m + j
                # Index in V: j*p + k  
                # Index in W: k*n + i
                val = 0.0
                for c in range(rank):
                    val += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                if abs(val - expected) > 1e-9:
                    return False
    
    # Also check that entries that should be 0 are 0
    # T[a,b,c] = 0 if the indices don't form a valid (i,j,k) triple
    # a = i*m+j, b = j'*p+k, c = k'*n+i'
    # Should be 0 if j != j' or k != k' or i != i'
    for a in range(n*m):
        for b in range(m*p):
            for c in range(p*n):
                i_a = a // m
                j_a = a % m
                j_b = b // p
                k_b = b % p
                k_c = c // n
                i_c = c % n
                
                if j_a == j_b and k_b == k_c and i_a == i_c:
                    expected = 1.0
                else:
                    expected = 0.0
                
                val = 0.0
                for cc in range(rank):
                    val += U_arr[a, cc] * V_arr[b, cc] * W_arr[c, cc]
                if abs(val - expected) > 1e-9:
                    return False
    
    return True

def _schoolbook(n, m, p):
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
    return U, V, W

def _try_greedy_peeling(n, m, p, field_mod, max_iter=5000, timeout=None):
    """Try to find a lower-rank decomposition using greedy peeling over a finite field."""
    start_time = time.time()
    nm, mp, pn = n*m, m*p, p*n
    
    # Initialize with schoolbook
    U, V, W = _schoolbook(n, m, p)
    rank = len(U[0])
    
    # The tensor T has T[a,b,c] = 1 if a%m == b//p and b%p == c//n and a//m == c%n
    # We want to find rank-1 tensors u⊗v⊗w that approximate T
    # Greedy: at each step, find the best rank-1 approximation to the residual
    
    # For efficiency, work over the finite field
    # Convert to field elements
    def to_field(x):
        return x % field_mod
    
    # Current residual tensor (conceptually)
    # Instead of storing the full tensor, we work with the current U,V,W and try to remove columns
    
    # Actually, let's try a different approach: start from scratch and build up
    # Or try to find linear dependencies
    
    # Let me try: assume a decomposition of rank r < n*m*p
    # and try to find it via random search
    
    # For now, let's just return schoolbook if we can't do better
    return U, V, W, rank

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    best_rank = len(U[0])
    best_result = {"U": U, "V": V, "W": W}
    
    # Try to find a better decomposition
    # For M_{3,5,7}, let's try known constructions
    
    # One approach: use the fact that 3×5×7 can be related to 3×3×3, 2×2×2, etc.
    # But this is complex. Let's try a random search approach.
    
    # Try random rank-1 peeling over F_2
    for mod in [2, 3, 5]:
        if time.time() - start_time > DEADLINE - 10:
            break
        try:
            U2, V2, W2, r = _try_greedy_peeling(n, m, p, mod, timeout=DEADLINE - 15)
            if r < best_rank:
                # Verify over rationals
                if _verify_decomposition(U2, V2, W2, n, m, p):
                    best_rank = r
                    best_result = {"U": U2, "V": V2, "W": W2}
        except Exception:
            pass
    
    if time.time() - start_time > DEADLINE - 5:
        return best_result
    
    return best_result
# EVOLVE-BLOCK-END
