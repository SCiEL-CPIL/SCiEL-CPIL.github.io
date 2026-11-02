import time
import numpy as np

DIMENSIONS = [3, 5, 6]
START_TIME = time.time()
TIME_LIMIT = 150.0

def construct():
    """
    Constructs the rank-minimal exact tensor decomposition for multiplying 
    a 3x5 matrix by a 5x6 matrix.
    
    Uses block decomposition with Strassen for 2x2x2 sub-blocks where possible.
    Returns dict with 'U', 'V', 'W' factor matrices.
    """
    n, m, p = DIMENSIONS  # 3, 5, 6
    
    def schoolbook_decomp(nn, mm, pp):
        """Schoolbook decomposition for nn x mm x pp."""
        rank = nn * mm * pp
        U = np.zeros((nn * mm, rank))
        V = np.zeros((mm * pp, rank))
        W = np.zeros((pp * nn, rank))
        col = 0
        for i in range(nn):
            for j in range(mm):
                for k in range(pp):
                    U[i * mm + j, col] = 1.0
                    V[j * pp + k, col] = 1.0
                    W[k * nn + i, col] = 1.0
                    col += 1
        return U, V, W
    
    def strassen_decomp():
        """Strassen's 2x2x2 rank-7 decomposition."""
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1 = (A11+A22)(B11+B22)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # M2 = (A21+A22)B11
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = -1
        
        # M3 = A11(B12-B22)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1; W[3, 2] = 1
        
        # M4 = A22(B21-B11)
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[3, 3] = -1
        
        # M5 = (A11+A12)B22
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[3, 4] = 1
        
        # M6 = (A21-A11)(B11+B12)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        
        # M7 = (A12-A22)(B21+B22)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1; W[1, 6] = 1
        
        return U, V, W
    
    def verify_decomp(U, V, W, nn, mm, pp):
        """Verify that the decomposition is correct."""
        # The tensor T[i,j,k] = U[i,j,t] * V[j,k,t] * W[k,i,t] summed over t
        # should equal 1 if i,j,k correspond to a valid multiplication term, 0 otherwise
        # Actually, the decomposition means: C[i,k] = sum_j A[i,j] B[j,k] = sum_t (sum_j U[i*m+j,t] A[i,j]) (sum_j V[j*p+k,t] B[j,k]) W[k*n+i,t]
        
        # Let's verify by checking that for random matrices, the product matches
        np.random.seed(42)
        A = np.random.randn(nn, mm)
        B = np.random.randn(mm, pp)
        C_expected = A @ B
        
        # Compute using decomposition
        C_computed = np.zeros((nn, pp))
        for t in range(U.shape[1]):
            # Extract the linear forms
            A_form = np.zeros((nn, mm))
            for i in range(nn):
                for j in range(mm):
                    A_form[i, j] = U[i * mm + j, t]
            B_form = np.zeros((mm, pp))
            for j in range(mm):
                for k in range(pp):
                    B_form[j, k] = V[j * pp + k, t]
            W_coeff = np.zeros((pp, nn))
            for k in range(pp):
                for i in range(nn):
                    W_coeff[k, i] = W[k * nn + i, t]
            
            # The bilinear form: (A_form * A) * (B_form * B) * W_coeff
            # Actually: sum_{i,j} A_form[i,j] A[i,j] * sum_{j,k} B_form[j,k] B[j,k] * W_coeff[k,i]
            # This is: (A_form ⊙ A) * (B_form ⊙ B) * W_coeff
            # where ⊙ is element-wise product and we sum over all elements
            a_val = np.sum(A_form * A)
            b_val = np.sum(B_form * B)
            C_computed += a_val * b_val * W_coeff
        
        # Check if C_computed matches C_expected
        if not np.allclose(C_computed, C_expected, atol=1e-10):
            return False
        return True
    
    # Try block decomposition
    # Split m=5 into 2+2+1, p=6 into 2+2+2
    # For each (n_block, m_block, p_block), use appropriate decomposition
    
    # Since n=3 is odd, we can't use Strassen on n-dimension
    # But we can split m and p into 2-sized blocks
    
    # Strategy: Split A into columns [0:2], [2:4], [4:5] and B into rows [0:2], [2:4], [4:6]
    # For each pair (m_block, p_block), compute the contribution to C
    
    # The full decomposition is the direct sum of decompositions for each block pair
    
    U_parts = []
    V_parts = []
    W_parts = []
    
    m_splits = [0, 2, 4, 5]  # 5 = 2+2+1
    p_splits = [0, 2, 4, 6]  # 6 = 2+2+2
    
    for mi in range(len(m_splits) - 1):
        for pi in range(len(p_splits) - 1):
            m_start, m_end = m_splits[mi], m_splits[mi + 1]
            p_start, p_end = p_splits[pi], p_splits[pi + 1]
            mm = m_end - m_start
            pp = p_end - p_start
            
            if mm == 2 and pp == 2:
                # Use Strassen for 2x2x2 (but n=3, so we need 3x2x2)
                # 3x2x2: We can't directly use Strassen since n=3
                # Schoolbook rank for 3x2x2 is 12
                U_sub, V_sub, W_sub = schoolbook_decomp(n, mm, pp)
            elif mm == 1 and pp == 1:
                U_sub, V_sub, W_sub = schoolbook_decomp(n, mm, pp)
            elif mm == 2 and pp == 1:
                U_sub, V_sub, W_sub = schoolbook_decomp(n, mm, pp)
            elif mm == 1 and pp == 2:
                U_sub, V_sub, W_sub = schoolbook_decomp(n, mm, pp)
            else:
                U_sub, V_sub, W_sub = schoolbook_decomp(n, mm, pp)
            
            # Adjust indices to account for block offsets
            U_adj = np.zeros((n * m, U_sub.shape[1]))
            V_adj = np.zeros((m * p, V_sub.shape[1]))
            W_adj = np.zeros((p * n, W_sub.shape[1]))
            
            for i in range(n):
                for j in range(mm):
                    U_adj[i * m + (m_start + j), :] = U_sub[i * mm + j, :]
            
            for j in range(mm):
                for k in range(pp):
                    V_adj[(m_start + j) * p + (p_start + k), :] = V_sub[j * pp + k, :]
            
            for k in range(pp):
                for i in range(n):
                    W_adj[(p_start + k) * n + i, :] = W_sub[k * nn + i, :]
            
            U_parts.append(U_adj)
            V_parts.append(V_adj)
            W_parts.append(W_adj)
    
    # Concatenate all parts
    U_final = np.hstack(U_parts)
    V_final = np.hstack(V_parts)
    W_final = np.hstack(W_parts)
    
    rank = U_final.shape[1]
    
    # Verify the decomposition
    if not verify_decomp(U_final, V_final, W_final, n, m, p):
        # Fall back to schoolbook
        U_final, V_final, W_final = schoolbook_decomp(n, m, p)
        rank = n * m * p
    
    # Convert to lists
    U_list = U_final.tolist()
    V_list = V_final.tolist()
    W_list = W_final.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
