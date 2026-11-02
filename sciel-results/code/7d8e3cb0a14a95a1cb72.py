"""Program for matrix multiplication tensor decomposition 2x3x7."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2 by 3 matrix by a 3 by 7 matrix.
    Returns a dictionary with keys 'U', 'V', 'W' containing the factor matrices.
    The score is the rank (lower is better).
    """
    start_time = time.time()
    deadline = start_time + 145  # 5 second safety margin
    
    n, m, p = DIMENSIONS  # 2, 3, 7
    rank_schoolbook = n * m * p  # 42
    
    # Schoolbook decomposition as fallback
    def make_schoolbook():
        U = np.zeros((n * m, rank_schoolbook), dtype=int)
        V = np.zeros((m * p, rank_schoolbook), dtype=int)
        W = np.zeros((p * n, rank_schoolbook), dtype=int)
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W
    
    U_school, V_school, W_school = make_schoolbook()
    
    # Try to improve using Strassen-like decomposition
    # For 2x3x7, we can split into blocks and use faster algorithms
    
    # Strategy: Split the 3x7 product into (2x2 * 2x2) + (2x1 * 1x7) or similar
    # Actually, let's try a more systematic approach
    
    # The matrix multiplication tensor M_{2,3,7} can be decomposed by:
    # Splitting A (2x3) = [A1 | A2] where A1 is 2x2, A2 is 2x1
    # Splitting B (3x7) = [B1; B2] where B1 is 2x7, B2 is 1x7
    # Then C = A*B = A1*B1 + A2*B2
    
    # A1*B1 is 2x2 * 2x7, which can use Strassen-like methods
    # A2*B2 is 2x1 * 1x7, which is just outer product (rank 2*7=14)
    
    # For 2x2 * 2x7: split B1 into [B1a | B1b] where each is 2x4... no, 2x7 doesn't split evenly
    # Let's try: B1 is 2x7, we can think of it as 7 columns of 2x1
    # Actually, let's use the fact that 2x2x2 has rank 7 (Strassen)
    
    # Better approach: Use bilinear complexity bounds
    # M_{2,3,7} rank is at most:
    # - Schoolbook: 42
    # - Using Strassen for 2x2x2 blocks:
    #   Split A (2x3) into A_left (2x2) and A_right (2x1)
    #   Split B (3x7) into B_top (2x7) and B_bottom (1x7)
    #   C = A_left * B_top + A_right * B_bottom
    #   A_left * B_top: 2x2 * 2x7 -> can decompose B_top as 7 columns of 2x1
    #   Each 2x2 * 2x1 can be done with rank 3 (since it's just two multiplications)
    #   Actually 2x2 * 2x1 = [a b; c d] * [x; y] = [ax+by; cx+dy]
    #   This has rank 2 (two scalar multiplications ax, ay, bx, by, cx, cy, dx, dy -> no)
    #   Wait, [ax+by; cx+dy] requires computing ax, ay, bx, by, cx, cy, dx, dy? No.
    #   It requires: ax, by, cx, dy, and then additions. But ax, by, cx, dy are 4 products.
    #   Actually, the tensor for 2x2 * 2x1 has rank 4 (each output needs 2 products, but they share inputs)
    #   Let me think again. A = [a b; c d], v = [x; y]
    #   A*v = [ax+by; cx+dy]
    #   Bilinear form: ax, ay, bx, by, cx, cy, dx, dy -> 8 terms, but we can combine
    #   Actually no, these are all independent bilinear forms. The rank is 4? 
    #   ax+by and cx+dy. We need to compute these. 
    #   Let u1 = a, u2 = b, u3 = c, u4 = d (from A)
    #   v1 = x, v2 = y (from v)
    #   Output1 = a*x + b*y = u1*v1 + u2*v2
    #   Output2 = c*x + d*y = u3*v1 + u4*v2
    #   This is a tensor of rank 2: it's (u1⊗v1⊗e1) + (u2⊗v2⊗e1) + (u3⊗v1⊗e2) + (u4⊗v2⊗e2)
    #   That's rank 4! Each term is a rank-1 tensor.
    
    # So A_left * B_top where B_top has 7 columns: 7 * rank(2x2*2x1) = 7*4 = 28
    # A_right * B_bottom: A_right is 2x1, B_bottom is 1x7
    #   [a; b] * [c1 c2 ... c7] = [a*c1 ... a*c7; b*c1 ... b*c7]
    #   This has rank 2*7 = 14 (each output element is one product)
    #   Actually: output (i,k) = A_right[i,0] * B_bottom[0,k]
    #   So 2*7 = 14 rank-1 terms
    #   
    # Total: 28 + 14 = 42. Same as schoolbook!
    
    # Let me try a different split. What about using Strassen for 2x2x2?
    # M_{2,2,2} has rank 7 (Strassen) vs 8 (schoolbook)
    
    # If we can group some of the 3x7 into 2x2 blocks...
    # 3 = 2 + 1, 7 = 7
    # 
    # Alternative: Think of the multiplication as:
    # C[i,k] = sum_j A[i,j] * B[j,k] for i=0,1; j=0,1,2; k=0..6
    #
    # Group the j-sum: j=0,1 (using 2x2 Strassen-like) and j=2 (separate)
    # For j=0,1: C_partial[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    #   This is a 2x2 * 2x7 multiplication
    #   We can view B[0:2, :] as 7 columns of 2x1 vectors
    #   Each column: [B[0,k]; B[1,k]]
    #   A[0:2, 0:2] * [B[0,k]; B[1,k]] = [A[0,0]*B[0,k] + A[0,1]*B[1,k]; A[1,0]*B[0,k] + A[1,1]*B[1,k]]
    #   As computed above, each such column requires rank 4
    #   So 7 columns * 4 = 28
    #
    # For j=2: C_extra[i,k] = A[i,2]*B[2,k]
    #   This is a 2x1 * 1x7 outer product, rank 2*7 = 14
    #
    # Total: 28 + 14 = 42. Still schoolbook.
    
    # The issue is that Strassen saves on 2x2x2 (rank 7 vs 8), but here we're doing 2x2x7
    # which is 7 times 2x2x1, and 2x2x1 has rank 4 (no saving possible since it's already minimal)
    
    # Let me reconsider. Can we do better?
    # 
    # Known results:
    # - M_{2,3,3} has rank 12 (schoolbook is 18, but can do better)
    # - M_{2,2,2} has rank 7
    #
    # For M_{2,3,7}, the lower bound is max(2*7, 3*7, 2*3) = max(14, 21, 6) = 21
    # Upper bound from schoolbook: 42
    #
    # Can we achieve rank < 42?
    # 
    # Idea: Use the fact that some of the products can be shared.
    # 
    # Let's try a direct construction with rank < 42.
    # 
    # Actually, let me try using a known result:
    # M_{n,m,p} <= M_{n,m,p'} * M_{n',m,p''} when p = p' + p'' (not quite)
    #
    # Better: Use the decomposition:
    # M_{2,3,7} <= M_{2,2,7} + M_{2,1,7}
    # M_{2,2,7}: 2x2 * 2x7. 
    #   = 7 * M_{2,2,1}
    #   M_{2,2,1}: rank is 4 (as shown)
    #   So M_{2,2,7} <= 28
    # M_{2,1,7}: 2x1 * 1x7, rank 14
    # Total: 42
    #
    # No improvement. Let me try a different grouping.
    #
    # What if we group the output columns?
    # C is 2x7. Group columns into groups of 2 or 3.
    # 
    # Actually, the minimal rank of M_{2,3,7} is an open problem in some cases,
    # but for small dimensions, we can try to construct better decompositions.
    #
    # Let me try a random search for a lower rank decomposition.
    # Given time constraints, I'll try a few random rank-1 terms and see if we can
    # find a valid decomposition with rank < 42.
    #
    # Actually, that's very hard to do randomly. The space is huge.
    #
    # Let me think about this differently. The tensor T for matrix multiplication
    # M_{2,3,7} has entries T[i,j,k] = 1 for all i in {0,1}, j in {0,1,2}, k in {0..6}.
    # Wait, no. The tensor is defined such that:
    # T[A, B] = A*B where T is a trilinear form.
    # T(e_i, e_j, e_k) = 1 if and only if the (i,k)-th entry of the product uses A[i,j]*B[j,k].
    # So T[i,j,k] = 1 for all valid (i,j,k).
    #
    # The rank of this tensor is what we want to minimize.
    #
    # For the schoolbook decomposition, each term corresponds to one (i,j,k) triple,
    # and we have 2*3*7 = 42 terms.
    #
    # To get a lower rank, we need to find linear combinations of these rank-1 tensors
    # that still span the same tensor.
    #
    # This is equivalent to finding a rank-r decomposition of the tensor.
    #
    # Given the time budget, let me try a simple improvement:
    # Use the fact that for 2x2x2, we can save 1 rank (Strassen).
    # If we can identify a 2x2x2 sub-tensor within our problem, we might save.
    #
    # In M_{2,3,7}, can we find a 2x2x2 sub-problem?
    # We need 2 rows of A, 2 columns of A (i.e., 2 inner indices), and 2 columns of B.
    # Inner indices: we have 3, so we can pick any 2.
    # Output columns: we have 7, so we can pick any 2.
    #
    # If we isolate a 2x2x2 sub-problem, we can use Strassen to do it in rank 7 instead of 8.
    # But we still need to do the rest.
    #
    # Total terms:
    # - 2x2x2 part: rank 7 (Strassen)
    # - Remaining: 
    #   * Inner index 2 (the third one) with all 7 output columns: 2*1*7 = 14 terms
    #   * The 2x2 inner indices with the remaining 5 output columns: 2*2*5 = 20 terms
    #   Total remaining: 14 + 20 = 34
    # Grand total: 7 + 34 = 41
    #
    # That's a saving of 1! Rank 41 instead of 42.
    #
    # Let me construct this explicitly.
    
    def make_strassen_improved():
        """
        Decomposition using Strassen for a 2x2x2 sub-block.
        
        A is 2x3, B is 3x7.
        Split inner index j into j0=0, j1=1 (for Strassen) and j2=2 (separate).
        Split output index k into k0, k1 (for Strassen) and k2..k6 (separate).
        
        For the 2x2x2 part (i in {0,1}, j in {0,1}, k in {0,1}):
        Use Strassen's 7 products.
        
        For j=2, k=0..6: 2*1*7 = 14 terms
        For j=0,1, k=2..6: 2*2*5 = 20 terms
        
        Total: 7 + 14 + 20 = 41
        """
        rank = 41
        U = np.zeros((n * m, rank), dtype=int)  # (6, 41)
        V = np.zeros((m * p, rank), dtype=int)  # (21, 41)
        W = np.zeros((p * n, rank), dtype=int)  # (14, 41)
        
        col = 0
        # Strassen 2x2x2 for j in {0,1}, k in {0,1}
        # Strassen products for A (2x2) * B (2x2):
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B10)
        # M7 = (A01-A11)(B01+B11)
        
        # A indices: A[i, j] where i=0,1 and j=0,1 -> U index = i*m + j = i*3 + j
        # B indices: B[j, k] where j=0,1 and k=0,1 -> V index = j*p + k = j*7 + k
        # C indices: C[i, k] where i=0,1 and k=0,1 -> W index = k*n + i = k*2 + i
        
        # M1
        U[0*3+0, col] += 1  # A[0,0]
        U[1*3+1, col] += 1  # A[1,1]
        V[0*7+0, col] += 1  # B[0,0]
        V[1*7+1, col] += 1  # B[1,1]
        W[0*2+0, col] += 1  # C[0,0]
        W[1*2+1, col] += 1  # C[1,1]
        W[1*2+0, col] -= 1  # C[0,1]
        W[0*2+1, col] -= 1  # C[1,0]
        col += 1
        
        # M2
        U[1*3+0, col] += 1  # A[1,0]
        U[1*3+1, col] += 1  # A[1,1]
        V[0*7+0, col] += 1  # B[0,0]
        W[1*2+0, col] += 1  # C[0,1]
        col += 1
        
        # M3
        U[0*3+0, col] += 1  # A[0,0]
        V[1*7+0, col] += 1  # B[1,0]
        V[1*7+1, col] -= 1  # B[1,1]
        W[0*2+1, col] += 1  # C[1,0]
        col += 1
        
        # M4
        U[1*3+1, col] += 1  # A[1,1]
        V[0*7+1, col] += 1  # B[0,1]
        V[0*7+0, col] -= 1  # B[0,0]
        W[1*2+1, col] += 1  # C[1,1]
        col += 1
        
        # M5
        U[0*3+0, col] += 1  # A[0,0]
        U[0*3+1, col] += 1  # A[0,1]
        V[1*7+1, col] += 1  # B[1,1]
        W[0*2+1, col] += 1  # C[1,0]
        col += 1
        
        # M6
        U[1*3+0, col] += 1  # A[1,0]
        U[0*3+0, col] -= 1  # A[0,0]
        V[0*7+0, col] += 1  # B[0,0]
        V[1*7+0, col] += 1  # B[1,0]
        W[1*2+0, col] += 1  # C[0,1]
        col += 1
        
        # M7
        U[0*3+1, col] += 1  # A[0,1]
        U[1*3+1, col] -= 1  # A[1,1]
        V[0*7+1, col] += 1  # B[0,1]
        V[1*7+1, col] += 1  # B[1,1]
        W[0*2+0, col] += 1  # C[0,0]
        col += 1
        
        # j=2, k=0..6: 14 terms
        for i in range(2):
            for k in range(7):
                U[i*3+2, col] = 1  # A[i,2]
                V[2*7+k, col] = 1  # B[2,k]
                W[k*2+i, col] = 1  # C[i,k]
                col += 1
        
        # j=0,1, k=2..6: 20 terms
        for i in range(2):
            for j in range(2):
                for k in range(2, 7):
                    U[i*3+j, col] = 1  # A[i,j]
                    V[j*7+k, col] = 1  # B[j,k]
                    W[k*2+i, col] = 1  # C[i,k]
                    col += 1
        
        assert col == rank, f"Column count {col} != rank {rank}"
        return U, V, W
    
    U_strassen, V_strassen, W_strassen = make_strassen_improved()
    
    # Verify the Strassen decomposition is correct
    # Check: For each (i,k), sum over j of A[i,j]*B[j,k] should match
    # We'll verify by checking that the tensor is correctly represented
    
    def verify_decomposition(U, V, W, n, m, p):
        """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
        rank = U.shape[1]
        # Check dimensions
        if U.shape[0] != n * m:
            return False
        if V.shape[0] != m * p:
            return False
        if W.shape[0] != p * n:
            return False
        if V.shape[1] != rank:
            return False
        if W.shape[1] != rank:
            return False
        
        # For each (i, k), the sum over j of A[i,j]*B[j,k] should be computed correctly
        # The decomposition says:
        # C[i,k] = sum_r U[i*m + ?] * V[?] * W[k*n + i]
        # Actually, the trilinear form is:
        # T(A, B) = sum_r (sum_{ij} A[i,j] U[i,j,r]) * (sum_{jk} B[j,k] V[j,k,r]) * ... 
        # No, that's not right either.
        #
        # The correct interpretation:
        # The matrix multiplication tensor is a trilinear map from A to B to C.
        # T(A, B) = C where C[i,k] = sum_j A[i,j] B[j,k]
        #
        # A rank-r decomposition is:
        # C[i,k] = sum_r U_r[i, ?] ... 
        #
        # Actually, the standard form is:
        # C = sum_r U_r * (scalar product) * V_r^T
        # where U_r is n x 1, V_r is 1 x p, and the scalar is sum_j A[i,j] * B[j,k] * W_r[i,j,k]?
        #
        # No. Let me be precise.
        #
        # The tensor T has components T[i,j,k] such that:
        # T[A,B]_{i,k} = sum_{j} A_{i,j} B_{j,k}
        #
        # This can be written as:
        # T_{i,k} = sum_{j} T_{i,j,k} A_{i,j} B_{j,k}
        # where T_{i,j,k} = 1 for all i,j,k.
        #
        # A rank-r decomposition of T is:
        # T_{i,j,k} = sum_{r=0}^{r-1} U_{i,j,r} * V_{j,k,r} * W_{i,k,r}
        # Wait, that's not standard either.
        #
        # The standard tensor decomposition:
        # T_{i,j,k} = sum_r a_r[i] * b_r[j] * c_r[k]
        # But for matrix multiplication, the indices are coupled differently.
        #
        # Actually, for the matrix multiplication tensor M_{n,m,p}, the rank-r decomposition is:
        # sum_{j=1}^m A_{i,j} B_{j,k} = sum_{r=1}^r (sum_{i',j'} U_{i',j',r} A_{i',j'}) (sum_{j',k'} V_{j',k',r} B_{j',k'}) W_{i,k,r}
        #
        # Hmm, that's not quite right. Let me look at the seed program.
        #
        # The seed program sets:
        # U[i*m+j, r] = 1, V[j*p+k, r] = 1, W[k*n+i, r] = 1
        # for each (i,j,k) and r = i*m*p + j*p + k
        #
        # This suggests that the decomposition is:
        # C[i,k] = sum_r U_{i*m+j, r} * V_{j*p+k, r} * W_{k*n+i, r} * A_{i,j} * B_{j,k}
        #
        # No, that's circular. 
        #
        # I think the correct interpretation is:
        # The bilinear map (A, B) -> C has the property that:
        # C_{i,k} = sum_r (U_r^T A) * (V_r^T B) * W_{i,k,r}
        # where U_r is a vector of length n*m, V_r is a vector of length m*p,
        # and W_{i,k,r} is a scalar.
        #
        # But U_r^T A doesn't make sense dimensionally unless we reshape A.
        #
        # Let me just trust the seed program's structure and verify numerically.
        #
        # The seed program's decomposition: for each (i,j,k), there's a term with
        # U[i*m+j, r] = 1, V[j*p+k, r] = 1, W[k*n+i, r] = 1
        #
        # This means the r-th term contributes:
        # delta_{i,j} * delta_{j,k} * delta_{k,i} ... no.
        #
        # I think the correct formula is:
        # C_{i,k} = sum_r sum_{i',j'} U_{i',j',r} A_{i',j'} sum_{j',k'} V_{j',k',r} B_{j',k'} * W_{i,k,r}
        #
        # This is getting complicated. Let me just verify numerically by checking
        # that for random A and B, the reconstructed C matches A*B.
        
        U_arr = np.array(U)
        V_arr = np.array(V)
        W_arr = np.array(W)
        rank = U_arr.shape[1]
        
        # Reshape U, V, W to 3D
        U_3d = U_arr.reshape(n, m, rank)  # U[i, j, r]
        V_3d = V_arr.reshape(m, p, rank)  # V[j, k, r]
        W_3d = W_arr.reshape(p, n, rank)  # W[k, i, r]
        
        # For random A, B, check that C = A*B is recovered
        np.random.seed(42)
        for trial in range(5):
            A = np.random.randint(-3, 3, size=(n, m)).astype(float)
            B = np.random.randint(-3, 3, size=(m, p)).astype(float)
            C_true = A @ B
            
            # Reconstruct C from the decomposition
            C_recon = np.zeros((n, p), dtype=float)
            for r in range(rank):
                # The r-th term
                # u_r is a vector of length n*m, reshaped to (n, m)
                u_r = U_3d[:, :, r]  # (n, m)
                v_r = V_3d[:, :, r]  # (m, p)
                w_r = W_3d[:, :, r]  # (p, n)
                
                # The contribution to C[i,k] from term r:
                # sum_{i',j'} u_r[i',j'] * A[i',j'] * sum_{j',k'} v_r[j',k'] * B[j',k'] * w_r[k,i]
                # This doesn't seem right dimensionally.
                
                # Alternative: The term contributes:
                # (u_r^T A_vec) * (v_r^T B_vec) * w_r[i,k]
                # where A_vec is A reshaped to (n*m, 1), etc.
                # But u_r^T A_vec = sum_{i',j'} u_r[i',j'] A[i',j']
                # and v_r^T B_vec = sum_{j',k'} v_r[j',k'] B[j',k']
                # So the contribution is:
                # (sum_{i',j'} u_r[i',j'] A[i',j']) * (sum_{j',k'} v_r[j',k'] B[j',k']) * w_r[i,k]
                
                a_scalar = np.sum(u_r * A)
                b_scalar = np.sum(v_r * B)
                C_recon += a_scalar * b_scalar * w_r.T  # w_r is (p, n), we want (n, p)
            
            if not np.allclose(C_recon, C_true, atol=1e-9):
                return False
        
        return True
    
    # Verify schoolbook
    schoolbook_ok = verify_decomposition(U_school.tolist(), V_school.tolist(), W_school.tolist(), n, m, p)
    
    # Verify Strassen
    strassen_ok = verify_decomposition(U_strassen.tolist(), V_strassen.tolist(), W_strassen.tolist(), n, m, p)
    
    if strassen_ok and not schoolbook_ok:
        # Strassen is correct, schoolbook is not (shouldn't happen)
        return {
            "U": U_strassen.tolist(),
            "V": V_strassen.tolist(),
            "W": W_strassen.tolist()
        }
    elif schoolbook_ok and not strassen_ok:
        # Schoolbook is correct, Strassen is not
        return {
            "U": U_school.tolist(),
            "V": V_school.tolist(),
            "W": W_school.tolist()
        }
    elif schoolbook_ok and strassen_ok:
        # Both correct, prefer Strassen (lower rank)
        return {
            "U": U_strassen.tolist(),
            "V": V_strassen.tolist(),
            "W": W_strassen.tolist()
        }
    else:
        # Neither verified, return schoolbook as fallback
        return {
            "U": U_school.tolist(),
            "V": V_school.tolist(),
            "W": W_school.tolist()
        }

# EVOLVE-BLOCK-END
