import time
import numpy as np

DIMENSIONS = [4, 4, 4]

def construct():
    start_time = time.time()
    # Safety deadline: 10 seconds is plenty for O(1) vectorized operations
    deadline = start_time + 10.0

    n, m, p = DIMENSIONS
    rank = 49

    # Strassen coefficients for 2x2 matrix multiplication
    # Indices: [s, i, j] where s is the multiplication index (0-6), i,j are row/col (0-1)
    
    # A coefficients: U[row, r] = A_coeff[s_outer, I, J] * A_coeff[t_inner, i2, j2]
    # Row index in U: i*m + j = (2*I + i2)*4 + (2*J + j2)
    
    # B coefficients: V[row, r] = B_coeff[s_outer, J, K] * B_coeff[t_inner, j2, k2]
    # Row index in V: j*p + k = (2*J + j2)*4 + (2*K + k2)
    
    # C coefficients: W[row, r] = C_coeff[s_outer, I, K] * C_coeff[t_inner, i2, k2]
    # Row index in W: k*n + i = (2*K + k2)*4 + (2*I + i2)  <-- Note the trace convention W[k*n+i, r]
    
    # Define the 7x2x2 coefficient arrays for A, B, C
    A_coeff = np.zeros((7, 2, 2), dtype=int)
    B_coeff = np.zeros((7, 2, 2), dtype=int)
    C_coeff = np.zeros((7, 2, 2), dtype=int)

    # s=0: M0 = (A00+A11)(B00+B11) -> C00 += 1, C11 += 1
    A_coeff[0, 0, 0] = 1; A_coeff[0, 1, 1] = 1
    B_coeff[0, 0, 0] = 1; B_coeff[0, 1, 1] = 1
    C_coeff[0, 0, 0] = 1; C_coeff[0, 1, 1] = 1

    # s=1: M1 = (A10+A11)B00 -> C10 += 1, C11 -= 1
    A_coeff[1, 1, 0] = 1; A_coeff[1, 1, 1] = 1
    B_coeff[1, 0, 0] = 1
    C_coeff[1, 1, 0] = 1; C_coeff[1, 1, 1] = -1

    # s=2: M2 = A00(B01-B11) -> C01 += 1, C11 += 1
    A_coeff[2, 0, 0] = 1
    B_coeff[2, 0, 1] = 1; B_coeff[2, 1, 1] = -1
    C_coeff[2, 0, 1] = 1; C_coeff[2, 1, 1] = 1

    # s=3: M3 = A11(B10-B00) -> C00 += 1, C10 += 1
    A_coeff[3, 1, 1] = 1
    B_coeff[3, 1, 0] = 1; B_coeff[3, 0, 0] = -1
    C_coeff[3, 0, 0] = 1; C_coeff[3, 1, 0] = 1

    # s=4: M4 = (A00+A01)B11 -> C00 -= 1, C01 += 1
    A_coeff[4, 0, 0] = 1; A_coeff[4, 0, 1] = 1
    B_coeff[4, 1, 1] = 1
    C_coeff[4, 0, 0] = -1; C_coeff[4, 0, 1] = 1

    # s=5: M5 = (A10-A00)(B00+B01) -> C11 += 1
    A_coeff[5, 1, 0] = 1; A_coeff[5, 0, 0] = -1
    B_coeff[5, 0, 0] = 1; B_coeff[5, 0, 1] = 1
    C_coeff[5, 1, 1] = 1

    # s=6: M6 = (A01-A11)(B10+B11) -> C00 += 1
    A_coeff[6, 0, 1] = 1; A_coeff[6, 1, 1] = -1
    B_coeff[6, 1, 0] = 1; B_coeff[6, 1, 1] = 1
    C_coeff[6, 0, 0] = 1

    # Vectorized Construction
    # We want to compute U, V, W of shape (16, 49).
    # Rank index r = 7*s + t.
    # 
    # Let's construct the matrices using broadcasting.
    # 
    # For U:
    # U[i, j, r] is non-zero only if i = 2*I + i2 and j = 2*J + j2.
    # Value = A_coeff[s, I, J] * A_coeff[t, i2, j2].
    #
    # We can iterate over s and t (49 iterations) and fill the 16x1 matrix for that r.
    # Or, we can use a more clever vectorized approach.
    # Given the small size, a loop over s, t is fast enough and clearer than complex broadcasting.
    # But to satisfy "structural change", let's try to vectorize the inner part.
    
    U = np.zeros((16, 49), dtype=float)
    V = np.zeros((16, 49), dtype=float)
    W = np.zeros((16, 49), dtype=float)

    # Precompute indices for the 16 rows
    # i = 2*I + i2, j = 2*J + j2
    # I, J, i2, j2 in {0, 1}
    
    # Create arrays for all 16 positions
    # pos 0: I=0, J=0, i2=0, j2=0
    # pos 1: I=0, J=0, i2=0, j2=1
    # pos 2: I=0, J=0, i2=1, j2=0
    # ...
    
    # Let's just use the loop over s,t but vectorize the filling of the 16x1 slice.
    for s in range(7):
        for t in range(7):
            r = 7 * s + t
            
            # Extract the 2x2 blocks for this s, t
            A_outer = A_coeff[s] # 2x2
            A_inner = A_coeff[t] # 2x2
            B_outer = B_coeff[s] # 2x2
            B_inner = B_coeff[t] # 2x2
            C_outer = C_coeff[s] # 2x2
            C_inner = C_coeff[t] # 2x2
            
            # Construct U_slice (16,)
            # U[2*I+i2, 2*J+j2] = A_outer[I, J] * A_inner[i2, j2]
            # This is a Kronecker product structure but with specific index mapping.
            # A_outer[I, J] is 2x2. A_inner[i2, j2] is 2x2.
            # The result is a 4x4 matrix M_U where M_U[I, i2, J, j2] = A_outer[I, J] * A_inner[i2, j2].
            # Reshaping to (4, 4) with indices (2*I+i2, 2*J+j2).
            
            # Create 4x4 matrix for U
            # M_U[I, i2, J, j2]
            # We can compute this as:
            # M_U = np.einsum('ij,kl->ikjl', A_outer, A_inner).reshape(4, 4)
            # Let's verify:
            # A_outer[I, J] * A_inner[i2, j2]
            # einsum 'ij,kl->ikjl' gives tensor T[I, i2, J, j2].
            # Reshape to (4, 4):
            # Index (2*I+i2, 2*J+j2).
            # In memory, if we reshape T (2,2,2,2) to (4,4), the order is C-order (row-major).
            # T[I, i2, J, j2] maps to linear index I*8 + i2*4 + J*2 + j2.
            # Row = 2*I + i2. Col = 2*J + j2.
            # Linear index of U[row, col] = row*4 + col = (2*I+i2)*4 + (2*J+j2) = 8*I + 4*i2 + 4*J + 2*J + j2.
            # The einsum result linear index: I*8 + i2*4 + J*2 + j2.
            # Wait. 4*J + 2*J is not J*2.
            # Let's check the reshape order.
            # T shape (2, 2, 2, 2).
            # Reshape to (4, 4).
            # Element at T[I, i2, J, j2] goes to U[2*I+i2, 2*J+j2]?
            # Let's test I=0, i2=1, J=0, j2=0.
            # T[0, 1, 0, 0]. Linear idx = 0*8 + 1*4 + 0*2 + 0 = 4.
            # U[1, 0]. Linear idx = 1*4 + 0 = 4. Match.
            # Let's test I=0, i2=0, J=1, j2=0.
            # T[0, 0, 1, 0]. Linear idx = 0*8 + 0*4 + 1*2 + 0 = 2.
            # U[0, 2]. Linear idx = 0*4 + 2 = 2. Match.
            # Let's test I=1, i2=0, J=0, j2=0.
            # T[1, 0, 0, 0]. Linear idx = 1*8 + 0 + 0 + 0 = 8.
            # U[2, 0]. Linear idx = 2*4 + 0 = 8. Match.
            
            # So U_slice = np.einsum('ij,kl->ikjl', A_outer, A_inner).reshape(16)
            
            U[:, r] = np.einsum('ij,kl->ikjl', A_outer, A_inner).reshape(16)
            V[:, r] = np.einsum('ij,kl->ikjl', B_outer, B_inner).reshape(16)
            
            # For W:
            # W[k*n + i, r] = C_outer[I, K] * C_inner[i2, k2]
            # Row index = 4*K + k2 + i? No.
            # Row index = k*n + i = 4*k + i.
            # k = 2*K + k2, i = 2*I + i2.
            # Row = 4*(2*K + k2) + (2*I + i2) = 8*K + 4*k2 + 2*I + i2.
            
            # We need to construct a vector W_vec of length 16.
            # W_vec[row] = C_outer[I, K] * C_inner[i2, k2] where row = 8*K + 4*k2 + 2*I + i2.
            
            # Let's create a 4x4 matrix M_W where M_W[k, i] = C_outer[I, K] * C_inner[i2, k2].
            # k = 2*K + k2, i = 2*I + i2.
            # This is NOT a standard Kronecker product because the indices are interleaved differently for the row.
            # In U/V, row was i (from A) and col was j (from B). The block structure was (I, i2) for row.
            # Here, row is k (from B) and "col" (effectively) is i (from C/A).
            # Row index depends on K, k2. "Col" index depends on I, i2.
            # Let's form a 4x4 matrix M_W[k, i].
            # M_W[2*K+k2, 2*I+i2] = C_outer[I, K] * C_inner[i2, k2].
            
            # We can compute this by iterating or using einsum with careful reshaping.
            # Let's compute a 4x4 matrix M_W.
            # M_W = np.zeros((4, 4))
            # for I in range(2):
            #   for K in range(2):
            #     for i2 in range(2):
            #       for k2 in range(2):
            #         M_W[2*K+k2, 2*I+i2] = C_outer[I, K] * C_inner[i2, k2]
            
            # Vectorized:
            # C_outer[I, K] is 2x2. C_inner[i2, k2] is 2x2.
            # We want M_W[k, i] where k=2*K+k2, i=2*I+i2.
            # Let's create a tensor T_W[K, k2, I, i2] = C_outer[I, K] * C_inner[i2, k2].
            # T_W[K, k2, I, i2] = C_outer[I, K] * C_inner[i2, k2]
            # einsum: 'ik,jl->kjil' ?
            # C_outer[I, K] -> indices i, k (using I, K)
            # C_inner[i2, k2] -> indices j, l (using i2, k2)
            # We want result indexed by K, k2, I, i2.
            # C_outer[I, K] gives I, K.
            # C_inner[i2, k2] gives i2, k2.
            # Product: P[I, K, i2, k2] = C_outer[I, K] * C_inner[i2, k2].
            # We want to reshape P such that M_W[2*K+k2, 2*I+i2] = P[I, K, i2, k2].
            
            # Let's compute P = np.einsum('ik,jl->ikjl', C_outer, C_inner)
            # P[I, K, i2, k2]
            # We want to map this to a 4x4 matrix M_W where:
            # Row = 2*K + k2
            # Col = 2*I + i2
            
            # If we reshape P (2, 2, 2, 2) to (4, 4), how do the indices map?
            # P[I, K, i2, k2]. Linear index = I*8 + K*4 + i2*2 + k2.
            # We want M_W[row, col] where row = 2*K+k2, col = 2*I+i2.
            # Linear index of M_W = row*4 + col = (2*K+k2)*4 + (2*I+i2) = 8*K + 4*k2 + 8*I + 4*i2.
            # This is not the same as P's linear index.
            
            # We need to transpose/permute P.
            # P[I, K, i2, k2].
            # We want a tensor Q[K, k2, I, i2] such that Q[K, k2, I, i2] = P[I, K, i2, k2].
            # Then reshape Q to (4, 4).
            # Q linear index = K*8 + k2*4 + I*2 + i2.
            # M_W linear index = 8*K + 4*k2 + 8*I + 4*i2.
            # Still not matching. The reshape of Q (2,2,2,2) to (4,4) maps:
            # Q[K, k2, I, i2] -> M_W[2*K+k2, 2*I+i2].
            # Linear index of M_W = (2*K+k2)*4 + (2*I+i2) = 8*K + 4*k2 + 8*I + 4*i2.
            # Linear index of Q (flattened) = K*8 + k2*4 + I*2 + i2.
            # These are different.
            # Reshape (2,2,2,2) to (4,4) in C-order:
            # Index (r, c) corresponds to Q[r//2, r%2, c//2, c%2].
            # So M_W[2*K+k2, 2*I+i2] = Q[K, k2, I, i2].
            # Yes! This works.
            
            # So:
            # 1. Compute P = np.einsum('ik,jl->ikjl', C_outer, C_inner)  # P[I, K, i2, k2]
            # 2. Transpose P to Q = P.transpose(1, 3, 0, 2)  # Q[K, k2, I, i2]
            #    Wait. P[I, K, i2, k2].
            #    We want Q[K, k2, I, i2].
            #    Axis 0 (I) -> Axis 2
            #    Axis 1 (K) -> Axis 0
            #    Axis 2 (i2) -> Axis 3
            #    Axis 3 (k2) -> Axis 1
            #    So Q = P.transpose(1, 3, 0, 2).
            # 3. W_slice = Q.reshape(16).
            
            P = np.einsum('ik,jl->ikjl', C_outer, C_inner)
            Q = P.transpose(1, 3, 0, 2)
            W[:, r] = Q.reshape(16)
            
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
