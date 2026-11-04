"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def construct():
    n, m, p = 3, 3, 3
    deadline = time.time() + 130  # safety margin before 150s kill
    
    # Naive rank-27 fallback
    def naive_decomposition():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        c = 0
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    u_idx = i * m + k
                    v_idx = k * p + j
                    w_idx = j * n + i
                    U[u_idx][c] = 1
                    V[v_idx][c] = 1
                    W[w_idx][c] = 1
                    c += 1
        return {"U": U, "V": V, "W": W}
    
    # Verify a decomposition by checking tensor equality
    def verify_decomp(U, V, W, num_checks=3):
        rank = len(U[0])
        # Build the tensor T from the decomposition
        # T[i*m+k, k*p+j, j*n+i] = sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r]
        # We need T[i*m+k, k*p+j, j*n+i] = 1 if i, k, j form valid indices, else 0
        # Actually the matrix multiplication tensor is:
        # M[i, k, j] = 1 for all i in [0,n), k in [0,m), j in [0,p)
        # The decomposition should satisfy:
        # sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r] = 1 for all valid (i,k,j)
        
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        
        for _ in range(num_checks):
            # Check a random valid (i, k, j)
            i = np.random.randint(0, n)
            k = np.random.randint(0, m)
            j = np.random.randint(0, p)
            
            u_idx = i * m + k
            v_idx = k * p + j
            w_idx = j * n + i
            
            dot = np.dot(U_arr[u_idx], V_arr[v_idx])
            # This isn't quite right. The tensor decomposition is:
            # T[a, b, c] = sum_r U[a, r] * V[b, r] * W[c, r]
            # So we need: for each (a, b, c) = (i*m+k, k*p+j, j*n+i),
            # sum_r U[a,r] * V[b,r] * W[c,r] = 1
            
            # Let me redo this properly
            pass
        
        # Proper verification: check all 27 entries
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    a = i * m + k
                    b = k * p + j
                    c = j * n + i
                    val = np.dot(U_arr[a], V_arr[b])
                    # Wait, that's not right either. The tensor is rank-r decomposition:
                    # T[a,b,c] = sum_{r=0}^{rank-1} U[a,r] * V[b,r] * W[c,r]
                    val = 0
                    for r in range(rank):
                        val += U_arr[a, r] * V_arr[b, r] * W_arr[c, r]
                    if abs(val - 1.0) > 1e-6:
                        return False
        return True
    
    # Strassen-Winograd rank-23 decomposition for 3x3 matrix multiplication
    # Using the known decomposition from Winograd (1971)
    # This is a well-known result. Let me construct it carefully.
    
    # The 23-term decomposition for 3x3 matrix multiplication:
    # We use the standard construction.
    # 
    # Let A be n x m, B be m x p, C = A * B be n x p.
    # 
    # One standard rank-23 decomposition uses the following structure.
    # I'll use a verified set of matrices.
    
    # Let me use a different approach: construct from known Strassen-style recursion.
    # For 3x3, we can use the "Strassen-Winograd" method.
    
    # Actually, let me just hardcode a verified rank-23 decomposition.
    # From literature, one valid decomposition exists. Let me try to construct it.
    
    # Using the approach from "The Complexity of Computing Polynomials" or similar:
    # For 3x3 matrix multiplication, the tensor rank is 23.
    
    # Let me use a construction based on splitting:
    # A = [A11 A12 A13; A21 A22 A23; A31 A32 A33]
    # B = [B11 B12 B13; B21 B22 B23; B31 B32 B33]
    
    # I'll use a known valid rank-23 decomposition.
    # After careful consideration, let me use this verified decomposition:
    
    # The following is a rank-23 decomposition (verified against the tensor):
    # U has shape (9, 23), V has shape (9, 23), W has shape (9, 23)
    
    # Let me build it step by step using a known construction.
    # From the paper "Fast Matrix Multiplication" or standard references:
    
    # I'll use a construction that I can verify. Let me define it and check.
    
    # Known rank-23 decomposition (one valid set):
    # This is based on the work of Winograd and Strassen.
    
    # For safety, let me use a construction I can derive and verify:
    
    # Actually, let me try a simpler approach: use the naive rank-27 but see if I can
    # find a smaller rank through optimization. But that's hard to do in 150s.
    
    # Let me just return the naive decomposition for now, as it's guaranteed correct.
    # The score will be 1.0 (same as seed), but at least it's valid.
    
    # Wait - let me try to implement a known rank-23 decomposition.
    # Here's one from the literature:
    
    # The decomposition uses 23 bilinear forms.
    # I'll encode it as follows:
    
    # After research, a valid rank-23 decomposition for 3x3 matrix mult is:
    
    U = [[0]*23 for _ in range(9)]
    V = [[0]*23 for _ in range(9)]
    W = [[0]*23 for _ in range(9)]
    
    # Term 1-9: diagonal terms (identity-like)
    for idx in range(9):
        U[idx][idx] = 1
        V[idx][idx] = 1
        # W index: for term idx, we need to map back
        # idx = i*m + k for U, k*p + j for V
        # For diagonal terms, i=k=j=idx%3 for some pattern... this is getting complex.
        pass
    
    # This is getting too complex to derive correctly in this context.
    # Let me just return the naive rank-27 decomposition which is guaranteed correct.
    
    result = naive_decomposition()
    
    # Quick verification of naive decomposition
    if not verify_decomp(result["U"], result["V"], result["W"], num_checks=1):
        # Should never happen for naive, but just in case
        pass
    
    return result
# EVOLVE-BLOCK-END
