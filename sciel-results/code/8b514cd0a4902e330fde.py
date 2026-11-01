import time
import numpy as np

DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS
    
    rank = 32
    
    U = [[0] * rank for _ in range(n * m)] # 4 x 32
    V = [[0] * rank for _ in range(m * p)] # 18 x 32
    W = [[0] * rank for _ in range(p * n)] # 18 x 32
    
    col = 0
    
    # Helper to add a Strassen block for cols k0, k1
    def add_strassen(k0, k1):
        nonlocal col
        # 7 factors
        
        # M1: (a00+a11)(b00+b11)
        U[0][col] = 1
        U[3][col] = 1
        V[0*p + k0][col] = 1
        V[1*p + k1][col] = 1
        W[k0*n + 0][col] = 1
        W[k1*n + 1][col] = 1
        col += 1
        
        # M2: (a10+a11)b00
        U[2][col] = 1
        U[3][col] = 1
        V[0*p + k0][col] = 1
        W[k0*n + 1][col] = 1
        W[k1*n + 1][col] = -1
        col += 1
        
        # M3: a00(b01-b11)
        U[0][col] = 1
        V[0*p + k1][col] = 1
        V[1*p + k1][col] = -1
        W[k1*n + 0][col] = 1
        W[k1*n + 1][col] = 1
        col += 1
        
        # M4: a11(b10-b00)
        U[3][col] = 1
        V[1*p + k0][col] = 1
        V[0*p + k0][col] = -1
        W[k0*n + 0][col] = 1
        W[k0*n + 1][col] = 1
        col += 1
        
        # M5: (a00+a01)b11
        U[0][col] = 1
        U[1][col] = 1
        V[1*p + k1][col] = 1
        W[k0*n + 0][col] = -1
        W[k1*n + 0][col] = 1
        col += 1
        
        # M6: (a10-a00)(b00+b01)
        U[2][col] = 1
        U[0][col] = -1
        V[0*p + k0][col] = 1
        V[0*p + k1][col] = 1
        W[k1*n + 1][col] = 1
        col += 1
        
        # M7: (a01-a11)(b10+b11)
        U[1][col] = 1
        U[3][col] = -1
        V[1*p + k0][col] = 1
        V[1*p + k1][col] = 1
        W[k0*n + 0][col] = 1
        col += 1

    # Helper to add a Trivial block for col k
    def add_trivial(k):
        nonlocal col
        # 4 factors
        U[0][col] = 1
        V[0*p + k][col] = 1
        W[k*n + 0][col] = 1
        col += 1
        
        U[1][col] = 1
        V[1*p + k][col] = 1
        W[k*n + 0][col] = 1
        col += 1
        
        U[2][col] = 1
        V[0*p + k][col] = 1
        W[k*n + 1][col] = 1
        col += 1
        
        U[3][col] = 1
        V[1*p + k][col] = 1
        W[k*n + 1][col] = 1
        col += 1

    # Block 1: Cols 0-3 (M_224)
    add_strassen(0, 1)
    add_strassen(2, 3)
    
    # Block 2: Cols 4-8 (M_225)
    add_strassen(4, 5)
    add_trivial(6)
    add_strassen(7, 8)
    
    # Convert to numpy arrays for efficient linear algebra
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Verify the decomposition is correct
    def verify_decomposition(U_arr, V_arr, W_arr):
        """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
        n, m, p = DIMENSIONS
        # The tensor T[i, j, k] = sum_r U[i,r] * V[j,r] * W[k,r]
        # For matrix multiplication, T[i, j, k] should be 1 if i//n * m + i%n == j//p * p + j%p 
        # and k//n * m + k%n == i%n... wait, let me think about this more carefully.
        # 
        # Actually, the tensor for multiplying A (n x m) by B (m x p) to get C (n x p) is:
        # T[a, b, c] where a indexes A (0..n*m-1), b indexes B (0..m*p-1), c indexes C (0..n*p-1)
        # T[a, b, c] = 1 if a//n * m + a%n == b//p * p + b%m AND c//p * p + c%n == a%n... 
        # No, that's not right either.
        #
        # Let me reconsider. A[i,j] * B[j,k] contributes to C[i,k].
        # So T[a, b, c] = 1 where:
        #   a = i*m + j (index in A)
        #   b = j*p + k (index in B)  
        #   c = i*p + k (index in C)
        # for valid i, j, k.
        #
        # So T[a, b, c] = 1 iff there exist i, j, k such that:
        #   a = i*m + j, b = j*p + k, c = i*p + k
        # This means: a%m == b//p and b%p == c%p and a//m == c//p
        
        U_arr = np.array(U_arr, dtype=float)
        V_arr = np.array(V_arr, dtype=float)
        W_arr = np.array(W_arr, dtype=float)
        
        rank_val = U_arr.shape[1]
        # Build the tensor from the decomposition
        # T[a, b, c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # This is expensive to build fully, so let's check a random sample
        rng = np.random.RandomState(42)
        for _ in range(1000):
            a = rng.randint(0, n*m)
            b = rng.randint(0, m*p)
            c = rng.randint(0, n*p)
            
            # Compute from decomposition
            val = np.sum(U_arr[a] * V_arr[b] * W_arr[c])
            
            # Expected value
            i_a = a // m
            j_a = a % m
            j_b = b // p
            k_b = b % p
            i_c = c // p
            k_c = c % p
            
            expected = 1.0 if (j_a == j_b and i_a == i_c and k_b == k_c) else 0.0
            
            if abs(val - expected) > 1e-9:
                return False
        return True
    
    # Try to find a linear dependency that allows removing one term
    # We look for coefficients alpha_r such that sum_r alpha_r * (u_r ⊗ v_r ⊗ w_r) = 0
    # This is equivalent to finding a vector in the kernel of a linear map.
    # 
    # Alternative approach: Try removing each term and see if the remaining 31 terms
    # still span the same tensor space.
    
    best_rank = 32
    best_U, best_V, best_W = U_np, V_np, W_np
    
    # Try removing each single term
    for remove_idx in range(rank):
        if time.time() > deadline - 5:
            break
        
        # Create matrices without term remove_idx
        mask = [i for i in range(rank) if i != remove_idx]
        U_test = U_np[:, mask]
        V_test = V_np[:, mask]
        W_test = W_np[:, mask]
        
        # Check if this still works
        if verify_decomposition(U_test, V_test, W_test):
            best_rank = 31
            best_U, best_V, best_W = U_test, V_test, W_test
            break
    
    # If we found rank 31, try to find rank 30
    if best_rank == 31:
        for remove_idx1 in range(31):
            for remove_idx2 in range(remove_idx1 + 1, 31):
                if time.time() > deadline - 3:
                    break
                
                mask = [i for i in range(31) if i != remove_idx1 and i != remove_idx2]
                U_test = best_U[:, mask]
                V_test = best_V[:, mask]
                W_test = best_W[:, mask]
                
                if verify_decomposition(U_test, V_test, W_test):
                    best_rank = 30
                    best_U, best_V, best_W = U_test, V_test, W_test
                    break
            if best_rank == 30:
                break
    
    # Convert back to lists with integer/half-integer entries
    def to_list(arr):
        result = []
        for row in arr:
            row_list = []
            for val in row:
                # Round to nearest 0.5
                rounded = round(val * 2) / 2
                row_list.append(rounded)
            result.append(row_list)
        return result
    
    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
# EVOLVE-BLOCK-END
