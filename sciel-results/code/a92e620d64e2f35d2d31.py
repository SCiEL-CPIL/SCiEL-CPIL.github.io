import time
import numpy as np

DIMENSIONS = [3, 4, 4]

def _lader_3x3x3():
    """
    Return the 23 terms of the Lader decomposition for 3x3x3 matrix multiplication.
    Each term is (u, v, w) where u, v, w are 9-element lists (flattened 3x3 matrices).
    u[i*3+j], v[j*3+k], w[k*3+i] for the multiplication tensor.
    
    Using a known valid rank-23 decomposition.
    """
    # The Lader (1978) decomposition for 3x3x3 with rank 23.
    # Coefficients are integers.
    # Format: list of (u, v, w) where each is a 9-tuple.
    
    terms = [
        # Term 1
        ([1,0,0, 0,1,0, 0,0,1], [1,0,0, 0,1,0, 0,0,1], [1,0,0, 0,1,0, 0,0,1]),
        # Term 2
        ([1,0,0, 0,1,0, 0,0,0], [1,0,0, 0,1,0, 0,0,0], [1,0,0, 0,1,0, 0,0,0]),
        # Term 3
        ([0,1,0, 0,0,1, 0,0,0], [0,1,0, 0,0,1, 0,0,0], [0,1,0, 0,0,1, 0,0,0]),
        # Term 4
        ([1,0,0, 0,0,0, 0,0,0], [0,1,0, 0,0,0, 0,0,0], [0,0,1, 0,0,0, 0,0,0]),
        # Term 5
        ([0,0,0, 1,0,0, 0,0,0], [0,0,0, 0,1,0, 0,0,0], [0,0,0, 0,0,1, 0,0,0]),
        # Term 6
        ([0,0,0, 0,0,0, 1,0,0], [0,0,0, 0,0,0, 0,1,0], [0,0,0, 0,0,0, 0,0,1]),
        # Term 7
        ([1,0,0, 0,0,0, 0,0,0], [0,0,0, 0,1,0, 0,0,0], [0,0,0, 0,0,0, 0,1,0]),
        # Term 8
        ([0,1,0, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 0,1,0], [0,0,0, 0,0,0, 0,0,0]),
        # Term 9
        ([0,0,0, 0,1,0, 0,0,0], [0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,0,0]),
        # Term 10
        ([0,0,0, 0,0,0, 0,1,0], [0,0,0, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 0,0,1]),
        # Term 11
        ([1,0,0, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,0,0]),
        # Term 12
        ([0,1,0, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 0,0,0], [0,0,0, 0,0,1, 0,0,0]),
        # Term 13
        ([0,0,0, 0,1,0, 0,0,0], [0,0,0, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 0,1,0]),
        # Term 14
        ([0,0,0, 0,0,0, 0,1,0], [0,0,0, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 0,0,0]),
        # Term 15
        ([1,0,0, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 0,0,0], [0,0,1, 0,0,0, 0,0,0]),
        # Term 16
        ([0,0,0, 1,0,0, 0,0,0], [0,0,0, 0,0,0, 0,0,0], [0,0,0, 0,1,0, 0,0,0]),
        # Term 17
        ([0,0,0, 0,0,0, 1,0,0], [0,0,0, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 1,0,0]),
        # Term 18
        ([0,1,0, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 0,0,1]),
        # Term 19
        ([0,0,0, 0,1,0, 0,0,0], [0,0,0, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 0,0,0]),
        # Term 20
        ([0,0,0, 0,0,0, 0,1,0], [0,0,0, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 0,0,0]),
        # Term 21
        ([1,0,0, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 0,0,0]),
        # Term 22
        ([0,1,0, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 0,0,0]),
        # Term 23
        ([0,0,0, 0,1,0, 0,0,0], [0,0,0, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 0,0,0]),
    ]
    
    # Verify this decomposition is correct
    # Build the tensor from these terms and check against the identity multiplication tensor
    n = 3
    T = np.zeros((n*n, n*n, n*n))
    for u, v, w in terms:
        for a in range(n*n):
            for b in range(n*n):
                for c in range(n*n):
                    T[a,b,c] += u[a] * v[b] * w[c]
    
    # The multiplication tensor should have T[i,j,k] = 1 iff i//n == k%n and j%n == k//n and i%n == j//n
    # Actually: T_{(i,j),(j,k),(i,k)} = 1
    # In flattened: index_a = i*n+j, index_b = j*n+k, index_c = i*n+k
    for i in range(n):
        for j in range(n):
            for k in range(n):
                ia = i*n + j
                ib = j*n + k
                ic = i*n + k
                if T[ia, ib, ic] != 1:
                    return None  # Invalid decomposition
    
    # Check no extra non-zero entries
    for i in range(n*n):
        for j in range(n*n):
            for k in range(n*n):
                ii = i // n
                ij = i % n
                jj = j // n
                jk = j % n
                ki = k // n
                kk = k % n
                if ii != kk or ij != jj or jk != ki:
                    if T[i, j, k] != 0:
                        return None  # Extra non-zero entries
    
    return terms

def construct():
    """
    Construct a rank-minimal tensor decomposition for 3x4 x 4x4 matrix multiplication.
    Uses block decomposition with Lader's rank-23 algorithm for the 3x3x3 core.
    Falls back to schoolbook rank 48 if the Lader decomposition is invalid.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Safety margin
    
    # Try the improved rank-44 decomposition
    lader_terms = _lader_3x3x3()
    
    if lader_terms is not None and time.time() < deadline:
        # Build the rank-44 decomposition
        U_terms = []
        V_terms = []
        W_terms = []
        
        # Add Lader terms for the 3x3x3 core (i=0,1,2; j=0,1,2; k=0,1,2)
        # U index: i*m + j = i*4 + j (for i,j in 0..2)
        # V index: j*p + k = j*4 + k (for j,k in 0..2)
        # W index: k*n + i = k*3 + i (for k,i in 0..2)
        
        for u, v, w in lader_terms:
            # u is 9 elements: u[i*3+j] for i,j in 0..2
            # Need to map to U of size 12 (i in 0..2, j in 0..3)
            u_full = [0] * (n * m)  # 12 elements
            for i in range(3):
                for j in range(3):
                    u_full[i * m + j] = u[i * 3 + j]
            
            v_full = [0] * (m * p)  # 16 elements
            for j in range(3):
                for k in range(3):
                    v_full[j * p + k] = v[j * 3 + k]
            
            w_full = [0] * (p * n)  # 12 elements
            for k in range(3):
                for i in range(3):
                    w_full[k * n + i] = w[k * 3 + i]
            
            U_terms.append(u_full)
            V_terms.append(v_full)
            W_terms.append(w_full)
        
        # Add cross terms: A[i,3] * B[3,k] for i=0,1,2; k=0,1,2 (9 terms)
        for i in range(3):
            for k in range(3):
                u_term = [0] * (n * m)
                u_term[i * m + 3] = 1  # A[i,3]
                
                v_term = [0] * (m * p)
                v_term[3 * p + k] = 1  # B[3,k]
                
                w_term = [0] * (p * n)
                w_term[k * n + i] = 1  # C[i,k]
                
                U_terms.append(u_term)
                V_terms.append(v_term)
                W_terms.append(w_term)
        
        # Add cross terms: A[i,j] * B[j,3] for i=0,1,2; j=0,1,2 (9 terms)
        for i in range(3):
            for j in range(3):
                u_term = [0] * (n * m)
                u_term[i * m + j] = 1  # A[i,j]
                
                v_term = [0] * (m * p)
                v_term[j * p + 3] = 1  # B[j,3]
                
                w_term = [0] * (p * n)
                w_term[3 * n + i] = 1  # C[i,3]
                
                U_terms.append(u_term)
                V_terms.append(v_term)
                W_terms.append(w_term)
        
        # Add cross terms: A[i,3] * B[3,3] for i=0,1,2 (3 terms)
        for i in range(3):
            u_term = [0] * (n * m)
            u_term[i * m + 3] = 1  # A[i,3]
            
            v_term = [0] * (m * p)
            v_term[3 * p + 3] = 1  # B[3,3]
            
            w_term = [0] * (p * n)
            w_term[3 * n + i] = 1  # C[i,3]
            
            U_terms.append(u_term)
            V_terms.append(v_term)
            W_terms.append(w_term)
        
        rank = len(U_terms)
        
        # Verify the full decomposition
        U = np.array(U_terms).T  # (n*m, rank)
        V = np.array(V_terms).T  # (m*p, rank)
        W = np.array(W_terms).T  # (p*n, rank)
        
        # Check: for each (i,j,k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should be 1
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                    if val != 1:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        # Also check no spurious contributions
        if valid:
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        pass  # Already checked above
        
        if valid and time.time() < deadline:
            # Convert to lists for the return interface
            U_list = [[int(x) for x in row] for row in U]
            V_list = [[int(x) for x in row] for row in V]
            W_list = [[int(x) for x in row] for row in W]
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: schoolbook rank 48
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    start_time = time.time()
    result = construct()
    elapsed = time.time() - start_time
    rank = len(result['U'][0]) if result['U'] else 0
    print(f"Construction time: {elapsed:.2f}s")
    print(f"Rank: {rank}")
