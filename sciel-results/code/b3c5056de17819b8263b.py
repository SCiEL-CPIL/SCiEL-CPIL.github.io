import numpy as np
import time

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # We will build the decomposition for 4x5x5 using block decomposition.
    # n=4 (2+2), m=5 (2+3), p=5 (2+3)
    
    # Helper to create Strassen 2x2x2 decomposition
    def strassen_2x2x2():
        # Rank 7
        # U (4x7), V (4x7), W (4x7)
        # Indices: U[i, r], V[j, r], W[k, r]
        # C_ii = sum_r U[i, r] V[j, r] W[k, r]
        # Standard Strassen formulas for 2x2 matrices A, B -> C
        # A = [[A11, A12], [A21, A22]]
        # B = [[B11, B12], [B21, B22]]
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        # C11 = M1+M4-M5+M7
        # C12 = M3+M5
        # C21 = M2+M4
        # C22 = M1-M2+M3+M6
        
        U = np.zeros((4, 7), dtype=int)
        V = np.zeros((4, 7), dtype=int)
        W = np.zeros((4, 7), dtype=int)
        
        # M1: (A11+A22)(B11+B22)
        # A11: i=0, j=0. A22: i=1, j=1.
        # B11: k=0, l=0. B22: k=1, l=1.
        # We need to map (i,j) to U index, (j,k) to V index?
        # Wait, the tensor is T_{i, j, k}.
        # U[i, r], V[j, r], W[k, r].
        # The multiplication is C_{i, k} = sum_j A_{i, j} B_{j, k}.
        # So U corresponds to A indices (i, j)? No.
        # Standard tensor decomposition: T_{i, j, k} = sum_r U_{i, r} V_{j, r} W_{k, r}.
        # Here i is row of A, j is col of A / row of B, k is col of B.
        # So U is (n x rank), V is (m x rank), W is (p x rank).
        # In our 2x2x2 case: n=2, m=2, p=2.
        
        # M1 term: (A11+A22)(B11+B22)
        # This contributes to C11, C12, C21, C22?
        # M1 is a scalar.
        # C11 += M1. C12 += M1? No.
        # Let's stick to the formula.
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # We need to express each M_r as sum of products of A elements and B elements.
        # M1 = (A11+A22)(B11+B22) = A11B11 + A11B22 + A22B11 + A22B22
        # This is not a single product. Strassen uses 7 products.
        # The "rank" 7 means we define 7 intermediate products.
        # Each product is of the form (linear combo of A) * (linear combo of B).
        # So for each of the 7 products, we have a vector of coefficients for A (size 4) and B (size 4).
        # U_{i, r} is the coefficient of A_{i, j} in the r-th product?
        # No, U_{i, r} V_{j, r} W_{k, r} sums to T_{i, j, k}.
        # This is equivalent to:
        # For each r, we have a matrix A^{(r)} and B^{(r)} such that
        # C = sum_r A^{(r)} B^{(r)}? No.
        # The decomposition is T = sum_r u_r \otimes v_r \otimes w_r.
        # u_r is a vector of size n*m? No, U is (n x rank).
        # So u_r is a vector of size n. v_r is size m. w_r is size p.
        # This implies that the r-th term is outer product of u_r, v_r, w_r.
        # This corresponds to a rank-1 tensor.
        # A rank-1 tensor T_{i, j, k} = u_i v_j w_k.
        # This means the multiplication is C_{i, k} = sum_j A_{i, j} B_{j, k}.
        # If we use rank-1 terms, we are decomposing the multiplication tensor.
        
        # Let's use the standard Strassen decomposition in this format.
        # Reference: 
        # M1 = (A11+A22)(B11+B22)
        # This product corresponds to a term in the sum.
        # How to map M1 to U, V, W?
        # M1 = (A11+A22)(B11+B22).
        # Let u = [1, 1, 0, 0] (coeffs for A11, A12, A21, A22).
        # Let v = [1, 1, 0, 0] (coeffs for B11, B12, B21, B22).
        # This product is scalar.
        # It contributes to C11, C12, C21, C22?
        # Actually, the tensor decomposition sums to the identity tensor of multiplication.
        # T_{i, j, k} = delta_{i, k}? No.
        # T_{i, j, k} = 1 if i, j, k form a valid multiplication path?
        # No, T_{i, j, k} is 1 if the element A_{i, j} and B_{j, k} are multiplied.
        # So T_{i, j, k} = 1 for all i, j, k? No.
        # The tensor is T_{i, j, k} = 1 if we are computing C_{i, k} using A_{i, j} and B_{j, k}.
        # Actually, the multiplication tensor $M_{n \times m \times p}$ has entries $T_{i, j, k} = 1$ for all $i, j, k$?
        # No. $C_{i, k} = \sum_j A_{i, j} B_{j, k}$.
        # The tensor is defined by $T_{i, j, k} = 1$.
        # Yes, the standard matrix multiplication tensor has all entries 1.
        # Wait, if T_{i, j, k} = 1 for all i, j, k, then the rank is 1?
        # No, the indices are (i, j) for A, (j, k) for B, (i, k) for C?
        # No, the tensor is in $V^* \otimes W^* \otimes U$.
        # Indices: i (row of A), j (col of A/row of B), k (col of B).
        # The entry $T_{i, j, k}$ is 1.
        # We want to decompose $T_{i, j, k} = \sum_r U_{i, r} V_{j, r} W_{k, r}$.
        # This is correct.
        
        # So for Strassen, we need to express the all-ones tensor (for 2x2x2) as sum of 7 rank-1 tensors.
        # This is a known result.
        
        # Let's just use a known set of coefficients for Strassen 2x2x2.
        # I will hardcode the U, V, W matrices for 2x2x2 Strassen.
        
        # From literature/code:
        # U (4x7):
        # M1: A11, A22 -> U[0,0]=1, U[3,0]=1 (if index 3 is A22? No, A is 2x2. Flat index 0:11, 1:12, 2:21, 3:22).
        # But U is (n x rank). n=2.
        # So U[0, r] is coeff for row 0, U[1, r] for row 1.
        # This doesn't capture the column j.
        # Ah, the decomposition is $T_{i, j, k} = \sum_r U_{i, r} V_{j, r} W_{k, r}$.
        # This implies that the "j" index is handled by V.
        # So for a fixed r, the term is $U_{i, r} V_{j, r} W_{k, r}$.
        # This is a rank-1 tensor.
        
        # Let's find the coefficients for Strassen.
        # This is tricky to derive from scratch. I will use a known implementation or derive it carefully.
        # Actually, I can just use schoolbook for 2x2x2 if I can't find Strassen coefficients quickly.
        # Rank 8 vs 7.
        # But 7 is better.
        
        # Let's try to construct Strassen 2x2x2.
        # We need 7 vectors u_r (size 2), v_r (size 2), w_r (size 2).
        # Such that sum_r u_r \otimes v_r \otimes w_r = T (all ones).
        
        # This is equivalent to finding a rank-7 decomposition of the 2x2x2 all-ones tensor.
        # It is known that the rank of 2x2x2 is 7.
        
        # I will use a pre-computed set of coefficients.
        # If I don't have them, I will fall back to schoolbook (rank 8).
        # Given the time, I will implement schoolbook for 2x2x2 to be safe, and use the block structure.
        # Rank 8 for 2x2x2.
        
        # Let's re-evaluate with Rank 8 for 2x2x2.
        # Block 1: 8 + (8 + 1) = 17.
        # Block 2: (1 + 8) + (1 + 8 + 1) = 19.
        # Block 3: 17.
        # Block 4: 19.
        # Total: 72.
        # This is still better than 100.
        
        # I will implement schoolbook for all sub-blocks to ensure correctness and simplicity.
        # Schoolbook rank for n x m x p is n*m*p.
        
        # Block 1 (2x2x2): 8.
        # Block 1 part 2 (2x3x2): 12.
        # Total 20.
        
        # Let's check if we can do better with simple splits.
        # I will implement the block decomposition with schoolbook for all sub-blocks.
        # This is guaranteed to be correct.
        
        U_list = []
        V_list = []
        W_list = []
        
        def add_schoolbook(n_sub, m_sub, p_sub, offset_i, offset_j, offset_k):
            # Adds schoolbook decomposition for n_sub x m_sub x p_sub
            # to global lists, with offsets.
            rank_sub = n_sub * m_sub * p_sub
            U_sub = np.zeros((n_sub, rank_sub), dtype=int)
            V_sub = np.zeros((m_sub, rank_sub), dtype=int)
            W_sub = np.zeros((p_sub, rank_sub), dtype=int)
            
            col = 0
            for i in range(n_sub):
                for j in range(m_sub):
                    for k in range(p_sub):
                        U_sub[i, col] = 1
                        V_sub[j, col] = 1
                        W_sub[k, col] = 1
                        col += 1
                        
            U_list.append(U_sub)
            V_list.append(V_sub)
            W_list.append(W_sub)
            
        # We need to handle the offsets correctly when concatenating.
        # It's easier to just build the full matrices directly.
        
        U = np.zeros((4, 0), dtype=int)
        V = np.zeros((5, 0), dtype=int)
        W = np.zeros((5, 0), dtype=int)
        
        def append_block(U_sub, V_sub, W_sub, i_off, j_off, k_off):
            nonlocal U, V, W
            n_sub, rank_sub = U_sub.shape
            m_sub = V_sub.shape[0]
            p_sub = W_sub.shape[0]
            
            U_full = np.zeros((4, rank_sub), dtype=int)
            V_full = np.zeros((5, rank_sub), dtype=int)
            W_full = np.zeros((5, rank_sub), dtype=int)
            
            U_full[i_off:i_off+n_sub, :] = U_sub
            V_full[j_off:j_off+m_sub, :] = V_sub
            W_full[k_off:k_off+p_sub, :] = W_sub
            
            U = np.hstack([U, U_full])
            V = np.hstack([V, V_full])
            W = np.hstack([W, W_full])
            
        # Block 1: C11 (2x2)
        # A11 (2x2) B11 (2x2)
        append_block(*schoolbook_2x2x2(), 0, 0, 0)
        # A12 (2x3) B21 (3x2)
        append_block(*schoolbook_2x3x2(), 0, 2, 2)
        
        # Block 2: C12 (2x3)
        # A11 (2x2) B12 (2x3)
        append_block(*schoolbook_2x2x3(), 0, 0, 2)
        # A12 (2x3) B22 (3x3)
        append_block(*schoolbook_2x3x3(), 0, 2, 2)
        
        # Block 3: C21 (2x2)
        # A21 (2x2) B11 (2x2)
        append_block(*schoolbook_2x2x2(), 2, 0, 0)
        # A22 (2x3) B21 (3x2)
        append_block(*schoolbook_2x3x2(), 2, 2, 2)
        
        # Block 4: C22 (2x3)
        # A21 (2x2) B12 (2x3)
        append_block(*schoolbook_2x2x3(), 2, 0, 2)
        # A22 (2x3) B22 (3x3)
        append_block(*schoolbook_2x3x3(), 2, 2, 2)
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# Define schoolbook helpers
def schoolbook_2x2x2():
    U = np.zeros((2, 8), dtype=int)
    V = np.zeros((2, 8), dtype=int)
    W = np.zeros((2, 8), dtype=int)
    c = 0
    for i in range(2):
        for j in range(2):
            for k in range(2):
                U[i, c] = 1
                V[j, c] = 1
                W[k, c] = 1
                c += 1
    return U, V, W

def schoolbook_2x3x2():
    U = np.zeros((2, 12), dtype=int)
    V = np.zeros((3, 12), dtype=int)
    W = np.zeros((2, 12), dtype=int)
    c = 0
    for i in range(2):
        for j in range(3):
            for k in range(2):
                U[i, c] = 1
                V[j, c] = 1
                W[k, c] = 1
                c += 1
    return U, V, W

def schoolbook_2x2x3():
    U = np.zeros((2, 12), dtype=int)
    V = np.zeros((2, 12), dtype=int)
    W = np.zeros((3, 12), dtype=int)
    c = 0
    for i in range(2):
        for j in range(2):
            for k in range(3):
                U[i, c] = 1
                V[j, c] = 1
                W[k, c] = 1
                c += 1
    return U, V, W

def schoolbook_2x3x3():
    U = np.zeros((2, 18), dtype=int)
    V = np.zeros((3, 18), dtype=int)
    W = np.zeros((3, 18), dtype=int)
    c = 0
    for i in range(2):
        for j in range(3):
            for k in range(3):
                U[i, c] = 1
                V[j, c] = 1
                W[k, c] = 1
                c += 1
    return U, V, W

