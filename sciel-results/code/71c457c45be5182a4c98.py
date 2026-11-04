import time
import sys

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    
    # Blocks for j (rows of B, cols of A)
    # Sizes: 2, 2, 1
    j_blocks = [(0, 2), (2, 2), (4, 1)]
    
    # Blocks for k (cols of B, cols of C)
    # Sizes: 2, 2, 2
    k_blocks = [(0, 2), (2, 2), (4, 2)]
    
    # Define Strassen coefficients for 2x2x2
    # Each term is (u_coeffs, v_coeffs, w_coeffs)
    # u_coeffs: dict {(i, j): val}
    # v_coeffs: dict {(j, k): val}
    # w_coeffs: dict {(i, k): val}
    
    strassen_terms = []
    
    # M1
    strassen_terms.append(({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}))
    # M2
    strassen_terms.append(({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}))
    # M3
    strassen_terms.append(({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}))
    # M4
    strassen_terms.append(({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}))
    # M5
    strassen_terms.append(({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}))
    # M6
    strassen_terms.append(({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}))
    # M7
    strassen_terms.append(({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}))
    
    # Define Schoolbook coefficients for 2x1x2
    # Local j is always 0.
    schoolbook_212_terms = []
    # C00 = A00 B00
    schoolbook_212_terms.append(({(0,0): 1}, {(0,0): 1}, {(0,0): 1}))
    # C01 = A00 B01
    schoolbook_212_terms.append(({(0,0): 1}, {(0,1): 1}, {(0,1): 1}))
    # C10 = A10 B00
    schoolbook_212_terms.append(({(1,0): 1}, {(0,0): 1}, {(1,0): 1}))
    # C11 = A10 B01
    schoolbook_212_terms.append(({(1,0): 1}, {(0,1): 1}, {(1,1): 1}))
    
    # Calculate total rank
    rank = 0
    for j_start, j_size in j_blocks:
        for k_start, k_size in k_blocks:
            if j_size == 2 and k_size == 2:
                rank += 7
            elif j_size == 1 and k_size == 2:
                rank += 4
            else:
                # Should not happen with this partition
                rank += j_size * k_size * n
                
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for j_start, j_size in j_blocks:
        for k_start, k_size in k_blocks:
            if j_size == 2 and k_size == 2:
                terms = strassen_terms
            else:
                terms = schoolbook_212_terms
                
            for u_c, v_c, w_c in terms:
                # Process U
                for (li, lj), val in u_c.items():
                    gi = li
                    gj = j_start + lj
                    U[gi * m + gj][col] += val
                    
                # Process V
                for (lj, lk), val in v_c.items():
                    gj = j_start + lj
                    gk = k_start + lk
                    V[gj * p + gk][col] += val
                    
                # Process W
                for (li, lk), val in w_c.items():
                    gi = li
                    gk = k_start + lk
                    W[gk * n + gi][col] += val
                    
                col += 1
                
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    # Simple verification
    res = construct()
    U = res['U']
    V = res['V']
    W = res['W']
    n, m, p = DIMENSIONS
    rank = len(U[0])
    
    # Check rank
    print(f"Rank: {rank}")
    
    # Verify a few random cases
    import random
    for _ in range(10):
        A = [[random.randint(-10, 10) for _ in range(m)] for _ in range(n)]
        B = [[random.randint(-10, 10) for _ in range(p)] for _ in range(m)]
        
        # Schoolbook C
        C_school = [[0]*p for _ in range(n)]
        for i in range(n):
            for k in range(p):
                s = 0
                for j in range(m):
                    s += A[i][j] * B[j][k]
                C_school[i][k] = s
        
        # Tensor C
        C_tensor = [[0]*p for _ in range(n)]
        for r in range(rank):
            for i in range(n):
                for k in range(p):
                    # Sum over j? No, U, V, W structure:
                    # C[i,k] = sum_r U[i,j]_r * V[j,k]_r * W[k,i]_r ?
                    # Wait, the standard decomposition is:
                    # C_{ik} = \sum_{j, r} A_{ij} B_{jk} T_{ijk}^r ?
                    # The task defines U, V, W such that:
                    # C_{ik} = \sum_r (\sum_j U_{ij, r} A_{ij}) (\sum_j V_{jk, r} B_{jk}) W_{ki, r} ?
                    # No, U has shape (n*m, rank). Row index is i*m+j.
                    # So U_{i*m+j, r} is the coefficient for A_{ij} in term r.
                    # Let a_r = \sum_j U_{i*m+j, r} A_{ij}.
                    # Let b_r = \sum_j V_{j*p+k, r} B_{jk}.
                    # Then C_{ik} = \sum_r a_r b_r W_{k*n+i, r}.
                    
                    # Let's compute a_r and b_r first
                    pass
        
        # Correct verification logic
        # a_r[i] is not a vector, a_r is a scalar for a fixed i?
        # No.
        # U is (n*m, rank).
        # For a fixed r, we form a linear combination of A's entries.
        # But the result C depends on i and k.
        # The formula is:
        # C_{ik} = \sum_{r} W_{k*n+i, r} * ( \sum_{j} U_{i*m+j, r} A_{ij} ) * ( \sum_{j} V_{j*p+k, r} B_{jk} )
        # This is NOT correct. The indices of U and V must align with the summation over j for the product?
        # In matrix multiplication tensor decomposition:
        # T_{ijk} = \sum_r U_{ij, r} V_{jk, r} W_{ki, r}
        # Here U, V, W are the factor matrices.
        # So for a specific r:
        # term_r = ( \sum_{j} U_{i*m+j, r} A_{ij} ) * ( \sum_{j} V_{j*p+k, r} B_{jk} ) * W_{k*n+i, r}
        # Wait, if I sum over j for U, I get a scalar depending on i.
        # If I sum over j for V, I get a scalar depending on k.
        # Then I multiply them and W.
        # This implies that for each r, the contribution to C_{ik} is separable in i and k?
        # No, U depends on i and j. V depends on j and k.
        # The sum over j in U and the sum over j in V are independent?
        # This would mean the tensor is a sum of rank-1 tensors in the space of matrices?
        # No, the tensor is $T_{ijk}$.
        # The decomposition is $T_{ijk} = \sum_r U_{ij, r} V_{jk, r} W_{ki, r}$.
        # So for fixed i, k:
        # $C_{ik} = \sum_r W_{ki, r} \left( \sum_j U_{ij, r} A_{ij} \right) \left( \sum_j V_{jk, r} B_{jk} \right)$
        # This is the standard form.
        
        C_check = [[0]*p for _ in range(n)]
        for i in range(n):
            for k in range(p):
                s = 0
                for r in range(rank):
                    a_r = 0
                    for j in range(m):
                        a_r += U[i*m + j][r] * A[i][j]
                    b_r = 0
                    for j in range(m):
                        b_r += V[j*p + k][r] * B[j][k]
                    s += W[k*n + i][r] * a_r * b_r
                C_check[i][k] = s
                
        if C_check != C_school:
            print("Mismatch!")
            print(f"i={i}, k={k}")
            print(f"School: {C_school[i][k]}, Tensor: {C_check[i][k]}")
            break
    else:
        print("Verification Passed")
