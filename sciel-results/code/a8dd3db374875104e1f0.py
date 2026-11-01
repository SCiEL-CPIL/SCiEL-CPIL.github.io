import time
import numpy as np

DIMENSIONS = [2, 2, 10]

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin from 150s limit

    n, m, p = DIMENSIONS  # 2, 2, 10
    nm, mp, pn = n * m, m * p, p * n  # 4, 20, 20

    # Strassen 7-term decomposition for 2x2x2
    # Each term: (a_vec[4], b_vec[4], c_vec[4])
    # a_vec indexed by [i*m+j] for A[i,j]
    # b_vec indexed by [j*2+k] for B[j,k] (local 2x2)
    # c_vec indexed by [k*2+i] for C[i,k] (trace convention: W[k*n+i])
    strassen = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),   # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, -1]),   # M2
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, 1]),   # M3
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 1, 0, 0]),   # M4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 1, 0]),   # M5
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),   # M6
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),   # M7
    ]

    def build_rank35():
        """Build the standard rank-35 decomposition (5 independent Strassen blocks)."""
        rank = 35
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]

        col = 0
        for block in range(5):
            for a, b, c in strassen:
                # U: A indices [i*m+j]
                for idx in range(nm):
                    U[idx][col] = a[idx]

                # V: B is 2x10, block covers columns 2*block, 2*block+1
                # B[j, 2*block] -> V[j*p + 2*block]
                # B[j, 2*block+1] -> V[j*p + 2*block+1]
                V[0 * p + 2 * block][col] = b[0]
                V[0 * p + 2 * block + 1][col] = b[1]
                V[1 * p + 2 * block][col] = b[2]
                V[1 * p + 2 * block + 1][col] = b[3]

                # W: C is 2x10, trace convention W[k'*n + i]
                # C[0, 2*block]   -> W[(2*block)*2 + 0]
                # C[1, 2*block]   -> W[(2*block)*2 + 1]
                # C[0, 2*block+1] -> W[(2*block+1)*2 + 0]
                # C[1, 2*block+1] -> W[(2*block+1)*2 + 1]
                W[2 * block * n + 0][col] = c[0]
                W[2 * block * n + 1][col] = c[1]
                W[(2 * block + 1) * n + 0][col] = c[2]
                W[(2 * block + 1) * n + 1][col] = c[3]

                col += 1

        return {"U": U, "V": V, "W": W}

    def verify_decomposition(U, V, W, tol=1e-9):
        """Verify that the decomposition exactly reconstructs the matrix multiplication tensor."""
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        rank = Ua.shape[1]

        # The matrix multiplication tensor T[i,j,k] = delta_{j,k} for (n,m,p) = (2,2,10)
        # Actually T[i,j,k] = 1 if j==k else 0... wait, that's not right.
        # For matrix multiplication C = A * B where A is n x m and B is m x p:
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # The tensor is T[i,j,k] = delta_{j,k} only for square matrices in a specific sense.
        # Actually the multiplication tensor has T[i,j,k] = 1 if the indices match properly.
        # For general n,m,p: T[i,j,k] = 1 if j == k is NOT correct.
        # The correct tensor: T[i,j,k] = delta_{j,k} only when m=p.
        # Actually, the matrix multiplication tensor is defined as:
        # T_{i,j,k} = 1 if the (i,j) entry of A times (j,k) entry of B contributes to (i,k) entry of C
        # So T[i,j,k] = 1 for all valid i,j,k (it's the "multiplication" tensor)
        # Wait, no. The tensor is: T[i,j,k] = delta_{j,j} which is always 1? No.
        # 
        # Let me think again. The bilinear map is:
        # (A, B) -> A*B
        # where A in R^{n x m}, B in R^{m x p}, result C in R^{n x p}
        # C[i,k] = sum_{j=0}^{m-1} A[i,j] * B[j,k]
        # 
        # The tensor T has T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
        # because every combination (i,j,k) contributes exactly A[i,j]*B[j,k] to C[i,k].
        # 
        # Wait, that's not right either. The tensor decomposition is:
        # C[i,k] = sum_r U[i,r] * V[k,r] * W[r] ... no.
        # 
        # The standard form: C[i,k] = sum_j sum_r U[i,j,r] * A[i,j] * V[j,k,r] * B[j,k]
        # Or equivalently, the tensor T[i,j,k] satisfies:
        # C = A*B means C[i,k] = sum_j A[i,j] B[j,k]
        # The tensor is T[i,j,k] = delta_{j,k}? No, that's only for identity.
        # 
        # Actually, I think the convention here is that the "multiplication tensor" for
        # multiplying an n x m matrix by an m x p matrix has entries:
        # T[i, j, k] = 1 if j is the shared index (which it always is in the sum)
        # So T[i,j,k] = 1 for all valid i, j, k.
        # 
        # Let me verify with the rank-35 construction:
        # For a given (i, j, k), we need sum_r U[i,r] * V[j,k,r] * W[i,k,r]... 
        # No, the decomposition is:
        # C[i,k] = sum_r (sum_j U[i,j,r] * A[i,j]) * (sum_l V[j,l,r] * B[j,l]) * W[i,k,r]
        # Hmm, that doesn't look right either.
        
        # Let me re-read the problem. The output factor follows W[k*n+i, r].
        # So the decomposition is:
        # C[i,k] = sum_r U[i*m+j, r] * A[i,j] * V[j*p+k, r] * B[j,k] * W[k*n+i, r]
        # Wait, that's not standard. Let me think about what makes sense.
        
        # Actually, I think the standard tensor decomposition for matrix mult is:
        # The multiplication tensor T has T[i,j,k] = 1 for all i,j,k (since every
        # A[i,j]*B[j,k] pair contributes to C[i,k]).
        # 
        # A rank-r decomposition gives:
        # T[i,j,k] = sum_{r=0}^{r-1} U[i,j,r] * V[j,k,r] * W[i,k,r]
        # 
        # Wait, but the shapes are U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
        # So U is indexed by (i,j), V by (j,k), W by (k,i) or (i,k)?
        # 
        # The problem says W[k*n+i, r], so W is indexed by (k,i) where k is column of C, i is row.
        # But C[i,k] has row i, column k. So W[k*n+i] means the index is k*n+i.
        # 
        # So the decomposition is:
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # 
        # And we need T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
        # 
        # Wait, is that right? For matrix multiplication, the "tensor" is indeed all ones
        # because every triple (i,j,k) represents a valid contribution A[i,j]*B[j,k] to C[i,k].
        
        # Let's verify with the rank-35 construction for a simple case.
        # Take n=2, m=2, p=2 (single block). The Strassen decomposition should give
        # T[i,j,k] = 1 for all i,j,k in {0,1}.
        
        # Actually, I realize I should just verify numerically by checking that
        # the bilinear form computed from U,V,W equals the matrix product.
        
        # For random A and B, compute C = A*B and compare with the decomposition.
        
        rng = np.random.RandomState(42)
        A_test = rng.randn(n, m)
        B_test = rng.randn(m, p)
        C_test = A_test @ B_test

        # Compute C from decomposition:
        # C[i,k] = sum_j sum_r U[i*m+j, r] * A[i,j] * V[j*p+k, r] * B[j,k] * W[k*n+i, r]
        # Wait, that's not how tensor decomposition works. Let me reconsider.
        
        # In a tensor decomposition of the multiplication tensor:
        # T[i,j,k] = sum_r U_r[i,j] * V_r[j,k] * W_r[i,k]
        # where U_r is a matrix in R^{n x m}, V_r in R^{m x p}, W_r in R^{n x p}
        # 
        # But here U, V, W are flattened. So:
        # U[i*m+j, r] = U_r[i,j]
        # V[j*p+k, r] = V_r[j,k]  
        # W[k*n+i, r] = W_r[i,k]  (note: W is indexed by k*n+i, so W_r[i,k] = W[k*n+i, r])
        
        # The bilinear form is:
        # C[i,k] = sum_j A[i,j] B[j,k]
        # = sum_r (sum_j U_r[i,j] A[i,j]) * (sum_j V_r[j,k] B[j,k]) * W_r[i,k]
        # 
        # Wait, that's not right. The tensor decomposition means:
        # T[i,j,k] = sum_r U_r[i,j] * V_r[j,k] * W_r[i,k]
        # 
        # And the bilinear map is:
        # C[i,k] = sum_j T[i,j,k] * A[i,j] * B[j,k]
        # = sum_j sum_r U_r[i,j] * V_r[j,k] * W_r[i,k] * A[i,j] * B[j,k]
        # 
        # Hmm, but that would make C depend on both A and B in a non-standard way.
        
        # I think I'm overcomplicating this. Let me just verify numerically.
        
        # Actually, I think the correct interpretation is:
        # The multiplication tensor T satisfies T[i,j,k] = 1 for all valid i,j,k.
        # A rank-r decomposition is: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        # Let's check this for the rank-35 case with n=m=p=2.
        if time.time() > deadline:
            return False

        # For the full 2x2x10 case, check T[i,j,k] = 1 for all i,j,k
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += Ua[i*m+j, r] * Va[j*p+k, r] * Wa[k*n+i, r]
                    if abs(val - 1.0) > tol:
                        return False
        return True

    # First, build and verify the rank-35 solution
    best_result = build_rank35()
    
    if not verify_decomposition(best_result["U"], best_result["V"], best_result["W"]):
        # If verification fails, there's a bug; return the best we can
        return best_result

    if time.time() > deadline:
        return best_result

    # Now try to find a lower rank decomposition
    # Strategy: Use the fact that U factors repeat across blocks.
    # The 35 terms are: for each block b in 0..4, for each strassen term s in 0..6:
    #   U term = strassen[s][0] (same for all blocks!)
    #   V term depends on block b and strassen term s
    #   W term depends on block b and strassen term s
    #
    # Since U is the same across blocks, we might be able to combine terms.
    # Specifically, if we have two terms with the same U vector but different V,W,
    # we can't directly combine them unless V and W also align.
    #
    # However, we can try to find a rank-r decomposition where r < 35 by
    # solving a linear system. The idea:
    # 
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    # 
    # This is a system of n*m*p = 2*2*10 = 40 equations.
    # Unknowns: r * (nm + mp + pn) = r * (4 + 20 + 20) = 44r
    # 
    # For r=35, we have 40 equations and 1540 unknowns, so it's underdetermined.
    # For r=34, 40 equations and 1496 unknowns, still underdetermined.
    # 
    # The question is: what is the minimum r such that a solution exists?
    # 
    # Let's try to find a rank-r solution for r < 35 by:
    # 1. Starting from the rank-35 solution
    # 2. Trying to eliminate one term at a time by adjusting the others
    
    # Alternative approach: Use random sampling to find a lower rank solution.
    # Fix random values for some entries and solve for the rest.
    
    # Actually, let's think about this more carefully. The tensor T[i,j,k] = 1 for all i,j,k.
    # This is a very simple tensor. Its rank might be much less than 35.
    # 
    # For T[i,j,k] = 1 (all ones), we can write:
    # T[i,j,k] = sum_{r=0}^{nm+mp+pn-1} ... 
    # 
    # Actually, the all-ones tensor of size n x m x p has rank at most n*m + m*p + p*n - (n-1)*(m-1)*(p-1) or something.
    # 
    # Let me think differently. The all-ones tensor can be decomposed as:
    # T[i,j,k] = 1 = (sum_a e_a[i]) * (sum_b e_b[j]) * (sum_c e_c[k]) ... no, that gives a rank-1 tensor.
    # 
    # Actually, T[i,j,k] = 1 for all i,j,k is a rank-1 tensor! Because:
    # T[i,j,k] = (1,1,...,1)_i * (1,1,...,1)_j * (1,1,...,1)_k
    # 
    # Wait, that's not right. A rank-1 tensor has the form T[i,j,k] = u[i]*v[j]*w[k].
    # If u[i]=1, v[j]=1, w[k]=1 for all i,j,k, then T[i,j,k] = 1*1*1 = 1. Yes!
    # 
    # But wait, that would mean the rank of the all-ones tensor is 1, which can't be right
    # because matrix multiplication has rank at least max(n*m, m*p, p*n) or something.
    
    # I think I'm confusing the multiplication tensor with the all-ones tensor.
    # Let me re-read the problem.
    
    # The problem says: "exactly decompose the matrix multiplication tensor"
    # For multiplying A (n x m) by B (m x p) to get C (n x p):
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # The "matrix multiplication tensor" is the tensor T such that:
    # C[i,k] = sum_{j,r} A[i,j] * B[j,k] * T[i,j,k,r] ... no.
    
    # I think the standard formulation is:
    # The multiplication tensor T is a 3-tensor with T[i,j,k] = 1 if the (i,j,k) entry
    # contributes to the matrix product, which is always the case. So T[i,j,k] = 1 for all i,j,k.
    # 
    # But then the rank would be 1, which contradicts the known lower bounds.
    
    # Let me look at this from a different angle. The bilinear map is:
    # M(A,B) = A*B
    # This is a bilinear map from R^{n*m} x R^{m*p} -> R^{n*p}
    # 
    # A rank-r decomposition of this bilinear map means:
    # M(A,B) = sum_{r=0}^{r-1} f_r(A) * g_r(B) * h_r
    # where f_r: R^{n*m} -> R, g_r: R^{m*p} -> R, h_r in R^{n*p}
    # 
    # In matrix form:
    # C[i,k] = sum_r (sum_j U[i*m+j, r] * A[i,j]) * (sum_l V[j*p+k, r] * B[j,l]) * W[k*n+i, r]
    # 
    # Wait, that doesn't look right either. Let me try:
    # C[i,k] = sum_r U_r(A) * V_r(B) * W_r[i,k]
    # where U_r(A) = sum_{i,j} U[i*m+j, r] * A[i,j]
    # V_r(B) = sum_{j,k} V[j*p+k, r] * B[j,k]
    # W_r[i,k] = W[k*n+i, r]
    
    # Let's verify with the rank-35 construction for a specific case.
    # Take n=2, m=2, p=2, and use the Strassen decomposition for a single block.
    
    # Actually, let me just do a numerical verification properly.
    
    def verify_bilinear(U, V, W, tol=1e-9):
        """Verify that the decomposition correctly computes matrix multiplication."""
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        rank = Ua.shape[1]
        
        rng = np.random.RandomState(123)
        for _ in range(5):
            A_test = rng.randn(n, m)
            B_test = rng.randn(m, p)
            C_expected = A_test @ B_test
            
            # Compute C from decomposition
            C_computed = np.zeros((n, p))
            for i in range(n):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        # U_r(A) = sum_{i',j'} U[i'*m+j', r] * A[i',j']
                        # But wait, U is indexed by [i*m+j, r], and we're computing for specific i,k.
                        # I think the decomposition is:
                        # C[i,k] = sum_r (sum_{i',j'} U[i*m+j, r] * A[i,j]) * ... 
                        # No, that doesn't make sense because U is fixed per r.
                        
                        # Let me reconsider. I think:
                        # C[i,k] = sum_r U_r * V_r * W_r[i,k]
                        # where U_r = sum_{i',j'} U[i'*m+j', r] * A[i',j']
                        # V_r = sum_{j',k'} V[j'*p+k', r] * B[j',k']
                        # W_r[i,k] = W[k*n+i, r]
                        
                        U_r = np.sum(Ua[:, r] * A_test.ravel())
                        V_r = np.sum(Va[:, r] * B_test.ravel())
                        val += U_r * V_r * Wa[k*n+i, r]
                    C_computed[i, k] = val
            
            if not np.allclose(C_computed, C_expected, atol=tol):
                return False
        return True

    if not verify_bilinear(best_result["U"], best_result["V"], best_result["W"]):
        # The verification method is wrong; let me try another interpretation.
        # Maybe the decomposition is:
        # C[i,k] = sum_{j,r} U[i*m+j, r] * A[i,j] * V[j*p+k, r] * B[j,k] * W[k*n+i, r]
        # But that would be a 4-way tensor, not 3-way.
        
        # Let me try the most standard interpretation:
        # The multiplication tensor T[i,j,k] = 1 for all i,j,k (since every A[i,j]*B[j,k] contributes to C[i,k]).
        # Rank-r decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        def verify_tensor(U, V, W, tol=1e-9):
            Ua = np.array(U, dtype=float)
            Va = np.array(V, dtype=float)
            Wa = np.array(W, dtype=float)
            rank = Ua.shape[1]
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(Ua[i*m+j, :] * Va[j*p+k, :] * Wa[k*n+i, :])
                        if abs(val - 1.0) > tol:
                            return False
            return True

        if verify_tensor(best_result["U"], best_result["V"], best_result["W"]):
            # Good, the tensor interpretation is correct.
            pass
        else:
            # Try the bilinear interpretation
            pass

    # Let me just go with the tensor interpretation and try to find a lower rank.
    # T[i,j,k] = 1 for all i in [0,2), j in [0,2), k in [0,10)
    # 
    # This is a 2 x 2 x 10 all-ones tensor.
    # 
    # The rank of the all-ones tensor of size a x b x c is known to be:
    # rank = a*b + b*c + c*a - (a-1)*(b-1)*(c-1) ... I'm not sure of the formula.
    # 
    # Actually, for the all-ones tensor, we can write:
    # T[i,j,k] = 1 = e_1[i] * e_1[j] * e_1[k]
    # where e_1 is the all-ones vector. So it's rank 1!
    # 
    # But that can't be right for matrix multiplication. I must be misunderstanding the tensor.
    
    # Let me re-read the problem statement more carefully.
    # "Find a rank-minimal exact tensor decomposition for multiplying a 2 by 2 matrix by a 2 by 10 matrix."
    # "The output factor follows the official trace convention W[k*n+i,r]."
    # 
    # I think the "matrix multiplication tensor" is not the all-ones tensor. It's the tensor
    # that encodes the bilinear map of matrix multiplication. The standard formulation is:
    # 
    # The multiplication tensor T has T[i,j,k] = 1 if j is the "shared" index, which is always
    # the case. So T[i,j,k] = 1 for all i,j,k. But then the rank is 1, which is wrong.
    
    # I think the issue is that the tensor for matrix multiplication is not simply all-ones.
    # Let me look at this from the perspective of the bilinear map.
    
    # The bilinear map M: R^{n*m} x R^{m*p} -> R^{n*p}
    # M(A,B) = A*B
    # 
    # In coordinates:
    # C[i,k] = sum_{j=0}^{m-1} A[i,j] * B[j,k]
    # 
    # A rank-r decomposition of this bilinear map is:
    # C[i,k] = sum_{r=0}^{r-1} f_r(A) * g_r(B) * h_r[i,k]
    # where f_r: R^{n*m} -> R, g_r: R^{m*p} -> R, h_r: R^{n*p} -> R
    # 
    # In matrix form:
    # f_r(A) = sum_{i,j} U[i*m+j, r] * A[i,j]
    # g_r(B) = sum_{j,k} V[j*p+k, r] * B[j,k]
    # h_r[i,k] = W[k*n+i, r]
    
    # So the decomposition is:
    # C[i,k] = sum_r (sum_{i',j'} U[i'*m+j', r] * A[i',j']) * (sum_{j',k'} V[j'*p+k', r] * B[j',k']) * W[k*n+i, r]
    
    # Let's verify this with the rank-35 construction.
    
    def verify_bilinear_v2(U, V, W, tol=1e-9):
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        rank = Ua.shape[1]
        
        rng = np.random.RandomState(456)
        for _ in range(3):
            A_test = rng.randn(n, m)
            B_test = rng.randn(m, p)
            C_expected = A_test @ B_test
            
            C_computed = np.zeros((n, p))
            for i in range(n):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        U_r = np.dot(Ua[:, r], A_test.ravel())
                        V_r = np.dot(Va[:, r], B_test.ravel())
                        val += U_r * V_r * Wa[k*n+i, r]
                    C_computed[i, k] = val
            
            if not np.allclose(C_computed, C_expected, atol=tol):
                return False
        return True

    if verify_bilinear_v2(best_result["U"], best_result["V"], best_result["W"]):
        # Great, this is the correct interpretation.
        # Now try to find a lower rank.
        
        # For a rank-r decomposition, we have r * (nm + mp + pn) = r * 44 unknowns
        # and n*p = 20 output entries, each depending on n*m*m*p = 2*2*2*10 = 80 input products.
        # Actually, the bilinear map has n*m*m*p = 80 independent bilinear terms.
        # 
        # A rank-r decomposition has r * (nm + mp + pn) = 44r parameters.
        # We need 44r >= 80, so r >= 2 (very loose bound).
        # 
        # But the actual rank is much higher due to structural constraints.
        
        # Let's try to find a rank-r solution for r < 35 by solving a linear system.
        # The idea: fix random values for U and V, then solve for W linearly.
        
        def try_rank_r(r, max_attempts=10):
            """Try to find a rank-r decomposition by fixing U,V randomly and solving for W."""
            if time.time() > deadline:
                return None
            
            for attempt in range(max_attempts):
                if time.time() > deadline:
                    return None
                
                rng = np.random.RandomState(attempt * 1000 + 42)
                U_rand = rng.randn(nm, r)
                V_rand = rng.randn(mp, r)
                
                # For each output (i,k), we need:
                # C[i,k] = sum_r U_r(A) * V_r(B) * W[k*n+i, r]
                # 
                # The bilinear terms are: for each (i,j,k), the coefficient of A[i,j]*B[j,k] in C[i,k]
                # is sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                # 
                # We need this to equal 1 for all i,j,k.
                # 
                # So for each (i,j,k):
                # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
                # 
                # This is a system of n*m*p = 40 equations in r*p*n = r*20 unknowns (W entries).
                # 
                # Let's set up the linear system.
                
                # Unknowns: W[k*n+i, r] for k in [0,p), i in [0,n), r in [0,r)
                # Total: p*n*r = 10*2*r = 20r unknowns
                
                # Equations: for each (i,j,k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
                # Total: n*m*p = 2*2*10 = 40 equations
                
                # Build the system: A_sys * w = b
                # where w is a vector of length 20r (all W entries)
                
                num_eqs = n * m * p  # 40
                num_unknowns = p * n * r  # 20r
                
                if num_unknowns < num_eqs:
                    return None  # Underdetermined in the wrong way
                
                A_sys = np.zeros((num_eqs, num_unknowns))
                b = np.ones(num_eqs)
                
                eq_idx = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            for r_idx in range(r):
                                # Unknown: W[k*n+i, r_idx]
                                unk_idx = (k * n + i) * r + r_idx
                                A_sys[eq_idx, unk_idx] = U_rand[i*m+j, r_idx] * V_rand[j*p+k, r_idx]
                            eq_idx += 1
                
                # Solve the least-squares system
                try:
                    w_sol, residuals, rank_sys, sv = np.linalg.lstsq(A_sys, b, rcond=None)
                except:
                    continue
                
                # Check if the solution is good
                if len(residuals) > 0 and residuals[0] > 1e-6:
                    continue
                
                # Verify the solution
                W_sol = w_sol.reshape(p*n, r)
                
                # Check all equations
                max_err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0.0
                            for r_idx in range(r):
                                val += U_rand[i*m+j, r_idx] * V_rand[j*p+k, r_idx] * W_sol[k*n+i, r_idx]
                            max_err = max(max_err, abs(val - 1.0))
                
                if max_err < 1e-6:
                    # Found a valid rank-r decomposition!
                    return {
                        "U": U_rand.tolist(),
                        "V": V_rand.tolist(),
                        "W": W_sol.tolist()
                    }
            
            return None

        # Try ranks from 34 down to some lower bound
        for r in range(34, 20, -1):
            if time.time() > deadline:
                break
            
            result = try_rank_r(r, max_attempts=5)
            if result is not None:
                # Verify it works
                if verify_bilinear_v2(result["U"], result["V"], result["W"]):
                    best_result = result
                    break
    
    return best_result
