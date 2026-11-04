import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 2, 3, 5
    
    # We aim for Rank 28 using a mix of Strassen and Schoolbook blocks.
    # Decomposition strategy:
    # 1. Split B into B_left (3x2) and B_right (3x3).
    # 2. C_left = A * B_left (2x2 output). Use Schoolbook (Rank 12).
    #    Actually, we can do better. Let's look at the structure.
    #    
    # Better Strategy:
    # Split A into A_top (1x3) and A_bottom (1x3).
    # Split B into B_top (1x5) and B_bottom (2x5).
    # This doesn't seem to help much directly with standard Strassen unless we group them.
    
    # Let's try the specific Rank 28 construction from inspiration 2, 
    # but verify it carefully. 
    # Inspiration 2 used:
    # - Strassen 2x2x2 for A[0:2, 0:2] * B[0:2, 0:2] -> Rank 7
    #   This contributes to C[0:2, 0:2].
    # - Strassen-like or other for the rest.
    
    # Let's construct a known Rank 28 decomposition for 2x3 * 3x5.
    # One way is to split the 3 columns of A and 3 rows of B into 2+1.
    # A = [A1 | a], B = [B1; b] where A1 is 2x2, a is 2x1, B1 is 2x5, b is 1x5.
    # C = A1 B1 + a b
    # 
    # Rank(A1 B1): 2x2 * 2x5.
    # We can split B1 into B11 (2x2) and B12 (2x3).
    # C1 = A1 B11 (2x2) + A1 B12 (2x3)
    # Rank(A1 B11) <= 7 (Strassen 2x2x2)
    # Rank(A1 B12) <= ? 2x2 * 2x3.
    #   Split A1 into a1 (1x2) and a2 (1x2).
    #   C1 = a1 B11 + a2 B12 ... no, C1 = A1 B12.
    #   Let's use Schoolbook for 2x2 * 2x3? Rank 12.
    #   Or Strassen? 
    #   2x2 * 2x3 can be seen as two 2x2 * 2x1 multiplications.
    #   Rank(2x2 * 2x1) = Rank(2x1 * 1x2) ? No.
    #   2x2 * 2x1: A(2x2) * B(2x1) -> C(2x1).
    #   C[i] = sum_j A[i,j] B[j].
    #   This is equivalent to 2x2 matrix mult with a column of B being 1? No.
    #   It is a linear map. 
    #   Known rank of 2x2 * 2x1 is 4? No, 2x2*2x2 is 7.
    #   2x2 * 2x1: 
    #     C0 = A00 B0 + A01 B1
    #     C1 = A10 B0 + A11 B1
    #     This is rank 4 (Schoolbook). Can we do better?
    #     Yes, it's a bilinear map. 
    #     Actually, 2x2 * 2x1 is isomorphic to 2x2 * 2x2 where the second half of B is 0? No.
    #     It is a subspace. The rank is at most 7. 
    #     Actually, 2x2 * 2x1 is just 4 scalar multiplications. 
    #     Wait, A has 4 entries, B has 2 entries. 
    #     C0 = A00 B0 + A01 B1
    #     C1 = A10 B0 + A11 B1
    #     This is exactly 4 terms. Is it reducible?
    #     No, because each A entry is independent. 
    #     So Rank(2x2 * 2x1) = 4.
    #     So Rank(2x2 * 2x3) = Rank(2x2 * 2x1) + Rank(2x2 * 2x1) = 4 + 4 = 8.
    #     
    # So for A1 B12 (2x2 * 2x3), we can use Rank 8.
    # 
    # Now for a b (2x1 * 1x5).
    # C[i, k] = a[i] b[k].
    # This is an outer product. 
    # Rank of outer product tensor is 1? No.
    # The tensor T[i, k, (a, b)] = 1 if ...
    # C[i, k] = a_i b_k.
    # This is a single term in the bilinear expansion? 
    # No, a_i and b_k are variables.
    # T is in F^{2} * F^{5} * F^{2*5}.
    # T[i, k, (i', k')] = delta_{i, i'} delta_{k, k'}.
    # This is the identity tensor of size 2x5. 
    # Rank of identity tensor of size N is N.
    # So Rank(2x1 * 1x5) = 10? 
    # Wait. C[i, k] = a_i b_k.
    # To compute C, we need to multiply a_i by b_k for all i, k.
    # This is 10 multiplications. 
    # Can we do it in fewer?
    # No, because each output depends on a unique pair (a_i, b_k).
    # Actually, if we fix i, C[i, :] = a_i * b. This is a scalar-vector multiplication.
    # We need to compute a_i * b_j for all j.
    # If we precompute nothing, it's 5 mults per row.
    # Total 10 mults.
    # Is there a faster way? 
    # No, the rank of the outer product map is N*M.
    # 
    # So total Rank = Rank(A1 B11) + Rank(A1 B12) + Rank(a b)
    # = 7 + 8 + 10 = 25.
    # 
    # Wait, is Rank(2x1 * 1x5) really 10?
    # Let's check Rank(1x1 * 1x1) = 1.
    # Rank(1x2 * 2x1) = 4?
    # C = a b^T.
    # C[i, j] = a_i b_j.
    # Yes, rank is n*p.
    # 
    # So we have a Rank 25 decomposition?
    # Let's verify the blocks.
    # 
    # Block 1: A1 (2x2) * B11 (2x2) -> C11 (2x2). Rank 7 (Strassen).
    # Block 2: A1 (2x2) * B12 (2x3) -> C12 (2x3). Rank 8 (2x 4).
    # Block 3: a (2x1) * b (1x5) -> C2 (2x5). Rank 10.
    # 
    # Total Rank = 7 + 8 + 10 = 25.
    # 
    # This is better than 28!
    # Let's implement this Rank 25 decomposition.
    
    rank = 25
    U = [[0] * rank for _ in range(n * m)]  # 6 x 25
    V = [[0] * rank for _ in range(m * p)]  # 15 x 25
    W = [[0] * rank for _ in range(p * n)]  # 10 x 25
    
    # Indices:
    # A: i in {0,1}, j in {0,1,2} -> idx = i*m + j
    #   A1: j in {0,1} -> idx 0,1,3,4
    #   a:  j in {2}   -> idx 2,5
    # 
    # B: j in {0,1,2}, k in {0,1,2,3,4} -> idx = j*p + k
    #   B11: j in {0,1}, k in {0,1} -> idx 0,1,5,6
    #   B12: j in {0,1}, k in {2,3,4} -> idx 2,3,4,7,8,9
    #   b:   j in {2},   k in {0,1,2,3,4} -> idx 10,11,12,13,14
    # 
    # C: k in {0,1,2,3,4}, i in {0,1} -> idx = k*n + i
    #   C11: k in {0,1}, i in {0,1} -> idx 0,2,1,3
    #   C12: k in {2,3,4}, i in {0,1} -> idx 4,6,8,5,7,9
    #   C2:  k in {0,1,2,3,4}, i in {0,1} -> idx 0,2,4,6,8,1,3,5,7,9
    #        Note: C2 contributes to ALL outputs?
    #        C = A1 B1 + a b.
    #        A1 B1 = A1 [B11 B12] = [A1 B11, A1 B12].
    #        So A1 B11 contributes to C[:, 0:2].
    #        A1 B12 contributes to C[:, 2:5].
    #        a b contributes to C[:, 0:5].
    #        So yes, C2 (a b) adds to all columns.
    
    col = 0
    
    # --- Block 1: Strassen 2x2x2 for A1 * B11 ---
    # A1 indices in U: 0,1,3,4
    # B11 indices in V: 0,1,5,6
    # C11 indices in W: 0,1,2,3
    
    # M1 = (A00 + A11) * (B00 + B11)
    U[0][col] = 1; U[4][col] = 1
    V[0][col] = 1; V[6][col] = 1
    W[0][col] = 1; W[1][col] = 1; W[2][col] = 1; W[3][col] = 1
    col += 1
    
    # M2 = A11 * (B10 - B00)
    U[4][col] = 1
    V[5][col] = 1; V[0][col] = -1
    W[1][col] = 1; W[3][col] = 1  # Only bottom row of C11
    col += 1
    
    # M3 = (A00 - A10) * B11
    U[0][col] = 1; U[3][col] = -1
    V[6][col] = 1
    W[0][col] = 1; W[1][col] = 1; W[2][col] = 1; W[3][col] = 1
    col += 1
    
    # M4 = A00 * (B01 - B11)
    U[0][col] = 1
    V[1][col] = 1; V[6][col] = -1
    W[0][col] = 1; W[2][col] = 1  # Only top row of C11
    col += 1
    
    # M5 = (A01 + A11) * (B00 + B01)
    U[1][col] = 1; U[4][col] = 1
    V[0][col] = 1; V[1][col] = 1
    W[0][col] = 1; W[1][col] = 1; W[2][col] = 1; W[3][col] = 1
    col += 1
    
    # M6 = (A00 - A10) * (B10 + B11)
    U[0][col] = 1; U[3][col] = -1
    V[5][col] = 1; V[6][col] = 1
    W[1][col] = 1; W[3][col] = 1
    col += 1
    
    # M7 = (A01 - A11) * (B00 - B10)
    U[1][col] = 1; U[4][col] = -1
    V[0][col] = 1; V[5][col] = -1
    W[0][col] = 1; W[1][col] = 1; W[2][col] = 1; W[3][col] = 1
    col += 1
    
    # --- Block 2: Schoolbook 2x2 * 2x3 for A1 * B12 ---
    # A1 indices in U: 0,1,3,4
    # B12 indices in V: 2,3,4,7,8,9
    # C12 indices in W: 4,5,6,7,8,9
    # 
    # We split B12 into two 2x1 blocks: B12a (cols 2,3 of B) and B12b (col 4 of B).
    # Actually, B12 is 2x3. 
    # Let's just do schoolbook for 2x2 * 2x3. Rank 12?
    # No, I argued Rank 8 earlier. 
    # Let's re-verify Rank(2x2 * 2x1).
    # C = A B. A is 2x2, B is 2x1.
    # C0 = A00 B0 + A01 B1
    # C1 = A10 B0 + A11 B1
    # This is 4 terms. 
    # Is it possible to do in 3?
    # If we use Strassen-like?
    # No, 2x2 * 2x1 is a specific case. 
    # The rank of 2x2 * 2x1 is 4.
    # So 2x2 * 2x3 is two such multiplications, Rank 8.
    
    # Block 2a: A1 * B_col2 (2x2 * 2x1) -> C_col2 (2x1)
    # B_col2 indices in V: 2, 7
    # C_col2 indices in W: 4, 5
    
    U[0][col] = 1; V[2][col] = 1; W[4][col] = 1; col += 1
    U[1][col] = 1; V[2][col] = 1; W[5][col] = 1; col += 1
    U[3][col] = 1; V[7][col] = 1; W[4][col] = 1; col += 1
    U[4][col] = 1; V[7][col] = 1; W[5][col] = 1; col += 1
    
    # Block 2b: A1 * B_col3 (2x2 * 2x1) -> C_col3 (2x1)
    # B_col3 indices in V: 3, 8
    # C_col3 indices in W: 6, 7
    
    U[0][col] = 1; V[3][col] = 1; W[6][col] = 1; col += 1
    U[1][col] = 1; V[3][col] = 1; W[7][col] = 1; col += 1
    U[3][col] = 1; V[8][col] = 1; W[6][col] = 1; col += 1
    U[4][col] = 1; V[8][col] = 1; W[7][col] = 1; col += 1
    
    # Block 2c: A1 * B_col4 (2x2 * 2x1) -> C_col4 (2x1)
    # B_col4 indices in V: 4, 9
    # C_col4 indices in W: 8, 9
    
    U[0][col] = 1; V[4][col] = 1; W[8][col] = 1; col += 1
    U[1][col] = 1; V[4][col] = 1; W[9][col] = 1; col += 1
    U[3][col] = 1; V[9][col] = 1; W[8][col] = 1; col += 1
    U[4][col] = 1; V[9][col] = 1; W[9][col] = 1; col += 1
    
    # --- Block 3: Outer product a * b (2x1 * 1x5) ---
    # a indices in U: 2, 5
    # b indices in V: 10, 11, 12, 13, 14
    # C2 indices in W: 0..9
    
    for i in range(2):
        u_idx = 2 + i * 3  # 2, 5
        for k in range(5):
            v_idx = 10 + k
            w_idx = k * 2 + i
            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1
            
    # Check time
    if time.time() > deadline:
        # Return schoolbook as fallback if we ran out of time (unlikely for static construction)
        rank_fb = n * m * p
        U_fb = [[0] * rank_fb for _ in range(n * m)]
        V_fb = [[0] * rank_fb for _ in range(m * p)]
        W_fb = [[0] * rank_fb for _ in range(p * n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j][c] = 1
                    V[j*p+k][c] = 1
                    W[k*n+i][c] = 1
                    c += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}

    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    # Quick self-test
    res = construct()
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    
    n, m, p = DIMENSIONS
    # Verify: T = U W^T V^T ? No.
    # T[i, j, k, l] = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+l, c]
    # Let's check a random case.
    np.random.seed(42)
    A = np.random.rand(n, m)
    B = np.random.rand(m, p)
    C = A @ B
    
    # Reconstruct C from decomposition
    C_recon = np.zeros((n, p))
    for c in range(U.shape[1]):
        u_col = U[:, c]
        v_col = V[:, c]
        w_col = W[:, c]
        
        # u_col is vector of size n*m
        # v_col is vector of size m*p
        # w_col is vector of size p*n
        
        # The term is (u_col * v_col) * w_col?
        # No, the tensor is T = sum_c u_c \otimes v_c \otimes w_c
        # C[i, k] = sum_j A[i,j] B[j,k]
        # In the decomposition:
        # C = sum_c (U[:,c] * A) (V[:,c] * B) * W[:,c]? No.
        
        # Standard form:
        # C = sum_c (U_c^T A) (V_c^T B) W_c
        # Where U_c is a vector in R^{n*m}, V_c in R^{m*p}, W_c in R^{p*n}
        # Let's reshape U_c to (n,m), V_c to (m,p), W_c to (p,n)
        
        U_c = u_col.reshape(n, m)
        V_c = v_col.reshape(m, p)
        W_c = w_col.reshape(p, n)
        
        # The product is:
        # Term_c = (U_c * A) * (V_c * B) ? No.
        # The bilinear form is:
        # <U_c, A> <V_c, B> <W_c, C>? No.
        
        # The decomposition is:
        # T = sum_c u_c \otimes v_c \otimes w_c
        # This means:
        # C = sum_c ( (u_c \cdot A) (v_c \cdot B) ) w_c^T ?
        # Let's check indices.
        # T[i, j, k, l] = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+l, c]
        # We want C[i, k] = sum_j A[i,j] B[j,k]
        # So sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] should equal A[i,j] B[j,k] when summed over j?
        # No, the tensor T represents the multiplication.
        # C[i,k] = sum_j T[i,j,k,i] A[i,j] B[j,k] ? No.
        
        # Let's just verify the identity:
        # For all i, j, k, l:
        # delta_{j, j'} delta_{k, k'} delta_{i, i'} = sum_c U[i*m+j, c] V[j'*p+k, c] W[k'*n+i', c]
        
        # Let's test this identity with the generated U, V, W.
        identity = np.zeros((n, m, m, p, p, n))
        for c in range(U.shape[1]):
            u_c = U[:, c].reshape(n, m)
            v_c = V[:, c].reshape(m, p)
            w_c = W[:, c].reshape(p, n)
            
            # u_c is (n, m)
            # v_c is (m, p)
            # w_c is (p, n)
            
            # The term is u_c[i, j] * v_c[j, k] * w_c[k, i]
            # This is a scalar for fixed i,j,k.
            # Let's build the tensor.
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = u_c[i, j] * v_c[j, k] * w_c[k, i]
                        identity[i, j, j, k, k, i] += val
                        # Wait, the indices in T are (i, j, (k, i))
                        # T[i, j, k, i] should be 1 if the indices match the schoolbook term.
                        # Schoolbook: T[i, j, k, i] = 1.
                        # So we are summing over c to get 1 at T[i, j, k, i].
        
        # Let's check if identity[i, j, j, k, k, i] == 1
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if abs(identity[i, j, j, k, k, i] - 1.0) > 1e-9:
                        valid = False
                        print(f"Failed at i={i}, j={j}, k={k}: got {identity[i, j, j, k, k, i]}")
        
        if valid:
            print("Decomposition is valid.")
        else:
            print("Decomposition is INVALID.")
