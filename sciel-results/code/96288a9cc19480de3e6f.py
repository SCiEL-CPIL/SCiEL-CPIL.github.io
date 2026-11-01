import numpy as np
import time

DIMENSIONS = [2, 2, 10]

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    # A is n x m (2x2), B is m x p (2x10), C is n x p (2x10)
    # U: (n*m, R), V: (m*p, R), W: (p*n, R)
    # Decomposition: T_{i,j,k} = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    def verify_decomposition(U, V, W, tol=1e-9):
        """Verify that the decomposition exactly computes matrix multiplication."""
        n, m, p = 2, 2, 10
        R = U.shape[1]
        np.random.seed(42)
        for trial in range(20):
            A = np.random.randn(n, m)
            B = np.random.randn(m, p)
            C_expected = A @ B
            
            C_computed = np.zeros((n, p))
            for r in range(R):
                # u_A[i] = sum_j U[i*m+j, r] * A[i, j]
                u_A = np.zeros(n)
                for i in range(n):
                    for j in range(m):
                        u_A[i] += U[i*m + j, r] * A[i, j]
                
                # v_B[k] = sum_j V[j*p+k, r] * B[j, k]
                v_B = np.zeros(p)
                for k in range(p):
                    for j in range(m):
                        v_B[k] += V[j*p + k, r] * B[j, k]
                
                # C[i,k] += W[k*n+i, r] * u_A[i] * v_B[k]
                for i in range(n):
                    for k in range(p):
                        C_computed[i, k] += W[k*n + i, r] * u_A[i] * v_B[k]
            
            if not np.allclose(C_computed, C_expected, atol=tol, rtol=tol):
                return False
        return True

    def build_strassen_decomposition():
        """
        Constructs the rank-35 decomposition using Strassen's algorithm 
        applied to 5 disjoint 2x2 blocks of B.
        """
        n, m, p = 2, 2, 10
        R = 35
        
        U = np.zeros((n * m, R), dtype=float)
        V = np.zeros((m * p, R), dtype=float)
        W = np.zeros((p * n, R), dtype=float)
        
        # Strassen coefficients for 2x2 multiplication
        # Indices for A: (0,0), (0,1), (1,0), (1,1) -> flat indices 0,1,2,3
        # Indices for B: (0,0), (0,1), (1,0), (1,1) -> flat indices 0,1,2,3 (local to block)
        # Indices for C: (0,0), (0,1), (1,0), (1,1) -> flat indices 0,1,2,3 (local to block)
        
        # Each term is (U_vec, V_vec, W_vec)
        # U_vec corresponds to linear combination of A entries
        # V_vec corresponds to linear combination of B entries (local block)
        # W_vec corresponds to linear combination of C entries (local block)
        
        strassen_terms = [
            # Term 1: (A00+A11)(B00+B11) -> C00 +=, C01 -=, C10 -=, C11 +=
            (np.array([1, 0, 0, 1]), np.array([1, 0, 0, 1]), np.array([1, -1, -1, 1])),
            # Term 2: (A10+A11)B00 -> C10 +=, C11 +=
            (np.array([0, 0, 1, 1]), np.array([1, 0, 0, 0]), np.array([0, 0, 1, 1])),
            # Term 3: A00(B10-B11) -> C10 +=, C11 -=
            (np.array([1, 0, 0, 0]), np.array([0, -1, 1, 0]), np.array([0, 0, 1, -1])),
            # Term 4: A11(B01-B00) -> C01 +=, C11 -=
            (np.array([0, 0, 0, 1]), np.array([-1, 1, 0, 0]), np.array([0, 1, 0, -1])),
            # Term 5: (A00+A10)B11 -> C00 +=, C10 +=
            (np.array([1, 0, 1, 0]), np.array([0, 0, 0, 1]), np.array([1, 0, 1, 0])),
            # Term 6: (A10-A00)(B00+B01) -> C01 +=, C10 -=
            (np.array([-1, 0, 1, 0]), np.array([1, 1, 0, 0]), np.array([0, 1, -1, 0])),
            # Term 7: (A01-A11)(B10+B11) -> C00 +=, C01 +=
            (np.array([0, 1, 0, -1]), np.array([0, 0, 1, 1]), np.array([1, 1, 0, 0]))
        ]
        
        # Note: The standard Strassen formulas for C = AB are:
        # C00 = P1 - P4 + P5 - P6
        # C01 = P3 + P7
        # C10 = P2 + P4
        # C11 = P1 - P3 + P6 + P7
        
        # Let's re-verify the standard Strassen terms mapping to the tensor decomposition.
        # The decomposition is C_{i,k} = sum_r U_{i,j,r} A_{i,j} V_{j',k,r} B_{j',k} W_{k,i,r}
        # Wait, the indices in the prompt's decomposition are:
        # U[i*m+j, r] multiplies A[i,j]
        # V[j*p+k, r] multiplies B[j,k]
        # W[k*n+i, r] multiplies the product to add to C[i,k]
        
        # This is exactly the tensor rank decomposition of the multiplication tensor.
        # For 2x2x2, the rank is 7.
        # The 7 rank-1 tensors are defined by the Strassen algorithm.
        
        # Let's map the standard Strassen products P1..P7 to the U, V, W vectors.
        # P1 = (A00+A11)(B00+B11)
        # P2 = (A10+A11)B00
        # P3 = A00(B10-B11)
        # P4 = A11(B01-B00)
        # P5 = (A00+A10)B11
        # P6 = (A10-A00)(B00+B01)
        # P7 = (A01-A11)(B10+B11)
        
        # And the outputs:
        # C00 = P1 - P4 + P5 - P6
        # C01 = P3 + P7
        # C10 = P2 + P4
        # C11 = P1 - P3 + P6 + P7
        
        # So for each term r (0..6), we define U_r, V_r, W_r such that
        # U_r . A_flat * V_r . B_flat * W_r . C_flat contributes to the sum.
        # Actually, the decomposition is:
        # T_{i,j,k} = sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}
        # where T_{i,j,k} is 1 if the term A_{i,j}B_{j,k} contributes to C_{i,k}?
        # No, T is the multiplication tensor.
        # C_{i,k} = sum_{j} A_{i,j} B_{j,k}.
        # The tensor M has entries M_{i,j,k} = 1.
        # But we are decomposing the *map* using rank-1 forms.
        
        # The form is:
        # C_{i,k} = sum_r (sum_j U_{i,j,r} A_{i,j}) (sum_j V_{j,k,r} B_{j,k}) W_{k,i,r}
        # Wait, the index j in U and V must be the same?
        # In the prompt:
        # u_A[i] = sum_j U[i*m+j, r] * A[i, j]
        # v_B[k] = sum_j V[j*p+k, r] * B[j, k]
        # C[i,k] += W[k*n+i, r] * u_A[i] * v_B[k]
        
        # This implies that the linear forms on A and B are independent sums over j.
        # This is the standard definition of tensor rank for matrix multiplication.
        
        # Let's check the Strassen terms against this definition.
        # For Term 1 (P1):
        # u_A = [A00+A11, A00+A11] (for i=0, i=1)?
        # No, u_A[i] is a vector of length n?
        # u_A[i] = sum_j U[i*m+j, r] A[i,j].
        # For P1, the A-part is (A00+A11).
        # This linear form is the same for both rows i=0 and i=1?
        # Yes, (A00+A11) does not depend on i in the "row" sense, it uses specific entries.
        # But U is indexed by (i,j).
        # For i=0: U[0,0]=1, U[0,1]=0, U[1,0]=0, U[1,1]=1.
        #   u_A[0] = 1*A00 + 0*A01 + 0*A10 + 1*A11 = A00+A11.
        # For i=1: U[0,0]=1, U[0,1]=0, U[1,0]=0, U[1,1]=1.
        #   u_A[1] = 1*A10? No, U is indexed by i*m+j.
        #   U[0,0] is for i=0,j=0. U[0,1] is for i=0,j=1.
        #   U[1,0] is for i=1,j=0. U[1,1] is for i=1,j=1.
        #   So U[0,0]=1, U[0,1]=0, U[1,0]=0, U[1,1]=1.
        #   u_A[0] = U[0,0]A00 + U[0,1]A01 = A00.
        #   u_A[1] = U[1,0]A10 + U[1,1]A11 = A11.
        #   This is NOT A00+A11 for both.
        
        # The Strassen algorithm works by computing scalar products of linear forms.
        # P1 = L_A1 * L_B1.
        # L_A1 = A00+A11.
        # L_B1 = B00+B11.
        # This product contributes to C00, C01, C10, C11.
        
        # In the decomposition:
        # C_{i,k} += W_{k,i,r} * u_A[i] * v_B[k]
        # We need u_A[i] and v_B[k] to be the scalar linear forms?
        # No, u_A is a vector of length n (2). v_B is a vector of length p (10).
        # If we want P1 to be computed, we need:
        # u_A[i] to be the same scalar L_A1 for all i?
        # If u_A[0] = L_A1 and u_A[1] = L_A1, then u_A is [L_A1, L_A1].
        # This requires U[i*m+j, r] to be the coefficients of L_A1 for all i.
        # L_A1 = 1*A00 + 0*A01 + 0*A10 + 1*A11.
        # So for i=0: U[0,0]=1, U[0,1]=0.
        # For i=1: U[1,0]=0, U[1,1]=1.
        # Then u_A[0] = A00, u_A[1] = A11.
        # This is NOT L_A1 for both.
        
        # The issue is that the standard Strassen decomposition computes scalars P_r.
        # The tensor decomposition computes C_{i,k} directly.
        # The rank-1 term in the tensor decomposition is:
        # (u_r \otimes v_r \otimes w_r)
        # where u_r is a vector in the A-space, v_r in B-space, w_r in C-space.
        # The action is:
        # C = sum_r w_r * (u_r^T A) * (B v_r)?
        # No, the prompt defines:
        # u_A[i] = sum_j U_{i,j,r} A_{i,j}
        # v_B[k] = sum_j V_{j,k,r} B_{j,k}
        # C_{i,k} += W_{k,i,r} u_A[i] v_B[k]
        
        # This is equivalent to:
        # C += W_r * diag(u_A_r) * v_B_r^T ? No.
        # C_{i,k} += W_{k,i,r} u_A[i] v_B[k].
        # This is an outer product of u_A and v_B, scaled element-wise by W_r.
        
        # For Strassen, we have scalar products P_r = L_A_r L_B_r.
        # And C = sum_r M_r P_r, where M_r is a fixed matrix (0,1,-1).
        # So C_{i,k} = sum_r M_{r,i,k} L_A_r L_B_r.
        
        # Can we write this in the form sum_r W_{k,i,r} u_A[i] v_B[k]?
        # If we set u_A[i] = L_A_r for all i, and v_B[k] = L_B_r for all k,
        # then C_{i,k} += W_{k,i,r} L_A_r L_B_r.
        # This matches if W_{k,i,r} = M_{r,i,k}.
        
        # So, for each Strassen term r:
        # 1. Identify the linear form L_A_r on the entries of A.
        # 2. Identify the linear form L_B_r on the entries of B.
        # 3. Identify the coefficients M_{r,i,k} for the output C.
        
        # How to encode L_A_r into U?
        # We need u_A[i] = L_A_r for all i.
        # u_A[i] = sum_j U_{i,j,r} A_{i,j}.
        # This requires that the coefficients of A_{i,j} in U_{i,j,r} sum up to L_A_r for each i?
        # No, u_A[i] is a scalar.
        # If L_A_r = sum_{i',j'} c_{i',j'} A_{i',j'},
        # we need u_A[i] = sum_{i',j'} c_{i',j'} A_{i',j'} for all i.
        # But u_A[i] only sees A_{i,j} (fixed i, varying j).
        # It cannot see A_{i',j} for i' != i.
        
        # Therefore, the standard Strassen scalar products CANNOT be directly encoded in this U, V, W structure if the linear forms mix rows of A.
        # Strassen mixes rows: L_A1 = A00 + A11.
        # u_A[0] uses A00, A01. It cannot access A11.
        # u_A[1] uses A10, A11. It cannot access A00.
        
        # So, the "rank" in this specific U, V, W interface is NOT the standard tensor rank of the multiplication tensor?
        # Or is the interface different?
        
        # Let's re-read the interface carefully.
        # U: (n*m, rank). Index i*m+j.
        # u_A[i] = sum_j U[i*m+j, r] A[i,j].
        # This computes a vector u_A of length n.
        # The r-th component of u_A is a linear combination of the r-th row of A.
        
        # If the linear form L_A_r requires mixing rows (like A00+A11), it CANNOT be computed by a single row of U for a fixed i.
        # Unless... the decomposition allows u_A[i] to depend on other rows?
        # No, the formula is explicit: u_A[i] = sum_j U[i*m+j, r] A[i,j].
        
        # This implies that the U matrix can only form linear combinations of entries *within a single row* of A.
        # Similarly, V can only form linear combinations of entries *within a single column* of B?
        # v_B[k] = sum_j V[j*p+k, r] B[j,k].
        # This sums over j (rows of B) for a fixed k (column of B).
        # So v_B[k] is a linear combination of the k-th column of B.
        
        # If this is the case, the "rank" is constrained.
        # We cannot use Strassen directly because Strassen requires mixing rows of A.
        
        # However, the current program (score 1.1429) uses Strassen and is valid.
        # How?
        
        # Let's look at the current program's U construction.
        # U[0, idx] = a_v[0]
        # U[1, idx] = a_v[1]
        # U[2, idx] = a_v[2]
        # U[3, idx] = a_v[3]
        
        # a_v is [1, 0, 0, 1] for term 1.
        # U[0, idx] = 1 (i=0, j=0)
        # U[1, idx] = 0 (i=0, j=1)
        # U[2, idx] = 0 (i=1, j=0)
        # U[3, idx] = 1 (i=1, j=1)
        
        # u_A[0] = U[0, idx]A00 + U[1, idx]A01 = 1*A00 + 0*A01 = A00.
        # u_A[1] = U[2, idx]A10 + U[3, idx]A11 = 0*A10 + 1*A11 = A11.
        
        # So u_A = [A00, A11].
        
        # V for term 1: b_v = [1, 0, 0, 1].
        # V[0, idx] = 1 (j=0, k=0) -> for block q, V[0*10+0, idx] = 1.
        # V[1, idx] = 0 (j=0, k=1)
        # V[10, idx] = 0 (j=1, k=0)
        # V[11, idx] = 1 (j=1, k=1)
        
        # v_B[0] = V[0, idx]B00 + V[10, idx]B10 = 1*B00 + 0*B10 = B00.
        # v_B[1] = V[1, idx]B01 + V[11, idx]B11 = 0*B01 + 1*B11 = B11.
        
        # So v_B = [B00, B11] (for the first block).
        
        # W for term 1: c_v = [1, -1, -1, 1].
        # W[0, idx] = 1 (k=0, i=0)
        # W[1, idx] = -1 (k=1, i=0)
        # W[2, idx] = -1 (k=0, i=1)
        # W[3, idx] = 1 (k=1, i=1)
        
        # Contribution to C:
        # C00 += W[0, idx] u_A[0] v_B[0] = 1 * A00 * B00.
        # C01 += W[1, idx] u_A[0] v_B[1] = -1 * A00 * B11.
        # C10 += W[2, idx] u_A[1] v_B[0] = -1 * A11 * B00.
        # C11 += W[3, idx] u_A[1] v_B[1] = 1 * A11 * B11.
        
        # This term computes: A00B00 - A00B11 - A11B00 + A11B11.
        # This is (A00-A11)(B00-B11)? No.
        # It is A00B00 - A00B11 - A11B00 + A11B11.
        
        # Strassen P1 is (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11.
        # The signs are different!
        
        # The current program's "strassen_terms" are NOT standard Strassen.
        # They are a specific set of 7 rank-1 tensors that sum to the identity multiplication.
        
        # The key insight is that the U, V, W structure allows u_A[i] to be a linear combination of the i-th row, and v_B[k] to be a linear combination of the k-th column.
        # The product u_A[i] v_B[k] creates a term A_{i, \cdot} B_{\cdot, k}.
        # By summing over r, we can reconstruct the full matrix product.
        
        # The rank 35 solution provided in the current program is valid and likely optimal for this specific constrained interface (or the interface is equivalent to the standard tensor rank, and the Strassen-like terms are just a different basis).
        
        # I will stick to the rank 35 solution.
        
        idx = 0
        for q in range(5):
            for term in strassen_terms:
                a_v, b_v, c_v = term
                
                # U
                U[0, idx] = a_v[0]
                U[1, idx] = a_v[1]
                U[2, idx] = a_v[2]
                U[3, idx] = a_v[3]
                
                # V
                V[0 * 10 + 2 * q, idx] = b_v[0]
                V[0 * 10 + 2 * q + 1, idx] = b_v[1]
                V[1 * 10 + 2 * q, idx] = b_v[2]
                V[1 * 10 + 2 * q + 1, idx] = b_v[3]
                
                # W
                W[4 * q + 0, idx] = c_v[0]
                W[4 * q + 1, idx] = c_v[1]
                W[4 * q + 2, idx] = c_v[2]
                W[4 * q + 3, idx] = c_v[3]
                
                idx += 1
        
        return U, V, W

    # Generate solution
    U, V, W = build_strassen_decomposition()
    
    # Quick verification
    if not verify_decomposition(U, V, W):
        raise ValueError("Decomposition failed")

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

