def _verify(U, V, W, n, m, p):
    # Check if the decomposition is valid
    # C_ik = sum_r W_ki_r * (sum_j U_ij_r A_ij) * (sum_j V_jk_r B_jk)
    # We can check this by contracting the tensors
    # T_ikj = sum_r W_ki_r U_ij_r V_jk_r
    # This T should be the matrix multiplication tensor
    # T_ikj = 1 if j is the "middle" index? 
    # Actually, M_ijk = 1 if the indices match the multiplication structure.
    # M_ijk = 1 if we are computing C_ik from A_ij and B_jk.
    # So T_ikj should be 1 if j is valid?
    # The tensor M has M_ijk = 1 for all i,j,k? No.
    # M is a 3-tensor of size n x m x p?
    # M_ijk = 1 if the indices correspond to A_ij, B_jk, C_ik.
    # In the decomposition, the "shape" is (n*m, m*p, p*n).
    # The indices are (i,j), (j,k), (k,i).
    # The tensor M_{(i,j),(j,k),(k,i)} = 1.
    # We can construct the target tensor and compare.
    
    import numpy as np
    U_np = np.array(U)
    V_np = np.array(V)
    W_np = np.array(W)
    
    # Reshape U to (n, m, rank)
    U_reshaped = U_np.reshape(n, m, -1)
    V_reshaped = V_np.reshape(m, p, -1)
    W_reshaped = W_np.reshape(p, n, -1)
    
    # Compute T_ikj = sum_r W_ki_r U_ij_r V_jk_r
    # Note: W is indexed by (k, i, r).
    # We want T_ikj.
    # T_ikj = sum_r W_np[k, i, r] * U_np[i, j, r] * V_np[j, k, r]
    
    # Use einsum
    # U: i j r
    # V: j k r
    # W: k i r
    # Result: i k j?
    # sum_r W[k, i, r] U[i, j, r] V[j, k, r]
    # Indices: i, k, j
    T = np.einsum('kir,ijr,jkr->ikj', W_reshaped, U_reshaped, V_reshaped)
    
    # The target tensor M_ikj should be 1 if the indices are consistent?
    # Actually, the multiplication tensor is M_{ijk} = 1 for all i,j,k?
    # No, the tensor M is defined such that M_{ijk} = 1 if the indices form a valid path?
    # For matrix multiplication, the tensor is M_{ijk} = 1 for all i, j, k in the range?
    # Let's check.
    # C_ik = sum_j A_ij B_jk.
    # So the coefficient of A_ij B_jk in C_ik is 1.
    # So M_{(i,j), (j,k), (k,i)} = 1.
    # In our T, T_ikj is the coefficient of A_ij B_jk in C_ik?
    # Yes.
    # So T_ikj should be 1 for all i, k, j?
    # Yes, for all i in 0..n-1, k in 0..p-1, j in 0..m-1.
    
    expected = np.ones((n, p, m))
    
    if not np.allclose(T, expected):
        return False
    return True
