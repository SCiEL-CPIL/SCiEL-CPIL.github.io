import numpy as np
import time

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    # Set a deadline
    start_time = time.time()
    deadline = 150.0  # Seconds
    
    # We will use a recursive decomposition strategy.
    # To avoid deep recursion or infinite loops, we limit the depth or use memoization.
    # Given small dimensions, simple recursion is fine.
    
    def get_decomposition(n, m, p):
        # Base cases
        if m == 1:
            # Rank 1
            # U: (n*1, 1) all 1s
            # V: (1*p, 1) all 1s
            # W: (p*n, 1) all 1s
            U = np.ones((n * 1, 1))
            V = np.ones((1 * p, 1))
            W = np.ones((p * n, 1))
            return U, V, W, 1
        
        if p == 1:
            # Rank 1
            U = np.ones((n * m, 1))
            V = np.ones((m * 1, 1))
            W = np.ones((1 * n, 1))
            return U, V, W, 1
            
        if n == 1:
            # Rank m
            # U: (1*m, m). U[0, j, r] = delta(j, r)
            # V: (m*p, m). V[j, k, r] = delta(j, r)
            # W: (p*1, m). W[k, 0, r] = 1
            U = np.zeros((1 * m, m))
            V = np.zeros((m * p, m))
            W = np.ones((p * 1, m))
            
            for j in range(m):
                U[0 * m + j, j] = 1
                for k in range(p):
                    V[j * p + k, j] = 1
            return U, V, W, m

        # Recursive case: try splits of m and p
        best_rank = float('inf')
        best_split = None
        
        # Try all splits of m
        for m1 in range(1, m):
            m2 = m - m1
            # Try all splits of p
            for p1 in range(1, p):
                p2 = p - p1
                
                # Get decompositions for the 4 sub-problems
                U1, V1, W1, R1 = get_decomposition(n, m1, p1)
                U2, V2, W2, R2 = get_decomposition(n, m1, p2)
                U3, V3, W3, R3 = get_decomposition(n, m2, p1)
                U4, V4, W4, R4 = get_decomposition(n, m2, p2)
                
                R = R1 + R2 + R3 + R4
                
                if R < best_rank:
                    best_rank = R
                    best_split = (m1, p1, U1, V1, W1, U2, V2, W2, U3, V3, W3, U4, V4, W4)
        
        if best_split is None:
            # Fallback to schoolbook if no split found (should not happen for m,p > 1)
            rank = n * m * p
            U = np.zeros((n * m, rank))
            V = np.zeros((m * p, rank))
            W = np.zeros((p * n, rank))
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i * m + j, col] = 1
                        V[j * p + k, col] = 1
                        W[k * n + i, col] = 1
                        col += 1
            return U, V, W, rank

        m1, p1, U1, V1, W1, U2, V2, W2, U3, V3, W3, U4, V4, W4 = best_split
        m2 = m - m1
        p2 = p - p1
        
        R = best_rank
        
        # Combine U
        # U shape: (n*m, R)
        U = np.zeros((n * m, R))
        
        # Block 1: i in [0,n), j in [0,m1) -> U1 (cols 0..R1-1)
        R1 = U1.shape[1]
        R2 = U2.shape[1]
        R3 = U3.shape[1]
        R4 = U4.shape[1]
        
        # U1 rows: i*m1 + j. These map to i*m + j in the full U.
        for i in range(n):
            for j in range(m1):
                U[i * m + j, 0:R1] = U1[i * m1 + j, :]
        
        # U2 rows: i*m1 + j (local). These map to i*m + j in the full U.
        # But U2 is for (n, m1, p2). So indices are same as U1 for U part.
        for i in range(n):
            for j in range(m1):
                U[i * m + j, R1:R1+R2] = U2[i * m1 + j, :]
                
        # U3 rows: i*m2 + j (local). j in [0,m2).
        # In full U, j is offset by m1. So index is i*m + (j + m1).
        for i in range(n):
            for j in range(m2):
                U[i * m + (j + m1), R1+R2:R1+R2+R3] = U3[i * m2 + j, :]
                
        # U4 rows: i*m2 + j (local).
        for i in range(n):
            for j in range(m2):
                U[i * m + (j + m1), R1+R2+R3:R] = U4[i * m2 + j, :]
                
        # Combine V
        # V shape: (m*p, R)
        V = np.zeros((m * p, R))
        
        # V1: (m1*p1, R1). Rows: j*p1 + k.
        # In full V, j is same, k is same (for k<p1).
        # Index: j*p + k.
        for j in range(m1):
            for k in range(p1):
                V[j * p + k, 0:R1] = V1[j * p1 + k, :]
                
        # V2: (m1*p2, R2). Rows: j*p2 + k (local k in [0,p2)).
        # In full V, k is offset by p1. Index: j*p + (k + p1).
        for j in range(m1):
            for k in range(p2):
                V[j * p + (k + p1), R1:R1+R2] = V2[j * p2 + k, :]
                
        # V3: (m2*p1, R3). Rows: j*p1 + k (local j in [0,m2)).
        # In full V, j is offset by m1. Index: (j+m1)*p + k.
        for j in range(m2):
            for k in range(p1):
                V[(j + m1) * p + k, R1+R2:R1+R2+R3] = V3[j * p1 + k, :]
                
        # V4: (m2*p2, R4).
        for j in range(m2):
            for k in range(p2):
                V[(j + m1) * p + (k + p1), R1+R2+R3:R] = V4[j * p2 + k, :]
                
        # Combine W
        # W shape: (p*n, R)
        W = np.zeros((p * n, R))
        
        # W1: (p1*n, R1). Rows: k*n + i.
        # In full W, k is same (for k<p1), i is same.
        for k in range(p1):
            for i in range(n):
                W[k * n + i, 0:R1] = W1[k * n + i, :]
                
        # W2: (p2*n, R2). Rows: k*n + i (local k in [0,p2)).
        # In full W, k is offset by p1. Index: (k+p1)*n + i.
        for k in range(p2):
            for i in range(n):
                W[(k + p1) * n + i, R1:R1+R2] = W2[k * n + i, :]
                
        # W3: (p1*n, R3). Rows: k*n + i.
        # Same k range as W1.
        for k in range(p1):
            for i in range(n):
                W[k * n + i, R1+R2:R1+R2+R3] = W3[k * n + i, :]
                
        # W4: (p2*n, R4).
        for k in range(p2):
            for i in range(n):
                W[(k + p1) * n + i, R1+R2+R3:R] = W4[k * n + i, :]
                
        return U, V, W, R

    U, V, W, rank = get_decomposition(n, m, p)
    
    # Convert to lists of lists as per interface (if required, but numpy arrays are usually accepted if convertible)
    # The prompt says "factor matrices of shapes... with integer/half-integer entries".
    # It returns a dict with U, V, W.
    # To be safe, convert to native Python lists.
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

if __name__ == "__main__":
    # Quick self-test
    result = construct()
    print(f"Rank: {len(result['U'][0]) if result['U'] else 0}")
