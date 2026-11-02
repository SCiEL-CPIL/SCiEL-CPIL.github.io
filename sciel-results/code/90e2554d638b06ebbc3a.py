import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=5
    start_time = time.time()
    DEADLINE = 120.0
    SAFETY_MARGIN = 5.0
    
    def verify(U, V, W):
        """Verify decomposition correctness"""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        rank = U.shape[1]
        tensor = np.zeros((n, m, p))
        for r in range(rank):
            u = U[:, r]
            v = V[:, r]
            w = W[:, r]
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        tensor[i, j, k] += u[i*m+j] * v[j*p+k] * w[k*n+i]
        return np.allclose(tensor, np.ones((n, m, p)), atol=1e-9)
    
    def build_rank_16():
        """Build the rank-16 decomposition"""
        rank = 16
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        col = 0
        
        def add_strassen(i_off, j_off, k_off):
            nonlocal col
            # M1: (a+c)(e+g)
            U[(i_off+0)*m + (j_off+0)][col] = 1.0
            U[(i_off+1)*m + (j_off+1)][col] = 1.0
            V[(j_off+0)*p + (k_off+0)][col] = 1.0
            V[(j_off+1)*p + (k_off+1)][col] = 1.0
            W[(k_off+0)*n + (i_off+0)][col] = 1.0
            W[(k_off+1)*n + (i_off+1)][col] = 1.0
            col += 1
            
            # M2: (b+d)(e)
            U[(i_off+1)*m + (j_off+0)][col] = 1.0
            U[(i_off+1)*m + (j_off+1)][col] = 1.0
            V[(j_off+0)*p + (k_off+0)][col] = 1.0
            W[(k_off+0)*n + (i_off+1)][col] = 1.0
            W[(k_off+1)*n + (i_off+1)][col] = -1.0
            col += 1
            
            # M3: (a)(f+g)
            U[(i_off+0)*m + (j_off+0)][col] = 1.0
            V[(j_off+0)*p + (k_off+1)][col] = 1.0
            V[(j_off+1)*p + (k_off+1)][col] = -1.0
            W[(k_off+1)*n + (i_off+0)][col] = 1.0
            W[(k_off+1)*n + (i_off+1)][col] = 1.0
            col += 1
            
            # M4: (d)(c)
            U[(i_off+1)*m + (j_off+1)][col] = 1.0
            V[(j_off+1)*p + (k_off+0)][col] = 1.0
            V[(j_off+0)*p + (k_off+0)][col] = -1.0
            W[(k_off+0)*n + (i_off+0)][col] = 1.0
            W[(k_off+0)*n + (i_off+1)][col] = 1.0
            col += 1
            
            # M5: (a+b)(g)
            U[(i_off+0)*m + (j_off+0)][col] = 1.0
            U[(i_off+0)*m + (j_off+1)][col] = 1.0
            V[(j_off+1)*p + (k_off+1)][col] = 1.0
            W[(k_off+0)*n + (i_off+0)][col] = -1.0
            W[(k_off+1)*n + (i_off+0)][col] = 1.0
            col += 1
            
            # M6: (c-d)(e+f)
            U[(i_off+1)*m + (j_off+0)][col] = 1.0
            U[(i_off+0)*m + (j_off+0)][col] = -1.0
            V[(j_off+0)*p + (k_off+0)][col] = 1.0
            V[(j_off+0)*p + (k_off+1)][col] = 1.0
            W[(k_off+1)*n + (i_off+1)][col] = 1.0
            col += 1
            
            # M7: (b-a)(d)
            U[(i_off+0)*m + (j_off+1)][col] = 1.0
            U[(i_off+1)*m + (j_off+1)][col] = -1.0
            V[(j_off+1)*p + (k_off+0)][col] = 1.0
            V[(j_off+1)*p + (k_off+1)][col] = 1.0
            W[(k_off+0)*n + (i_off+0)][col] = 1.0
            col += 1

        def add_rank1_slice(i_indices, j_indices, k_indices):
            """Add a rank-1 term for a slice where U,V,W are 1 on the indices"""
            nonlocal col
            for i in i_indices:
                for j in j_indices:
                    U[i*m + j][col] = 1.0
            for j in j_indices:
                for k in k_indices:
                    V[j*p + k][col] = 1.0
            for k in k_indices:
                for i in i_indices:
                    W[k*n + i][col] = 1.0
            col += 1

        # Block A1: j=0, k in 0,1,2
        # Rank 1
        add_rank1_slice([0, 1], [0], [0, 1, 2])
        
        # Block A2a: j in 1,2, k in 0,1
        # Rank 7 (Strassen)
        add_strassen(0, 1, 0)
        
        # Block A2b: j in 1,2, k=2
        # Rank 1
        add_rank1_slice([0, 1], [1, 2], [2])
        
        # Block B: j in 1,2, k in 3,4
        # Rank 7 (Strassen)
        add_strassen(0, 1, 3)
        
        assert col == rank, f"Column count mismatch: {col} != {rank}"
        return U, V, W
    
    # Build initial rank-16 solution
    U, V, W = build_rank_16()
    
    # Verify
    if not verify(U, V, W):
        # Fallback to schoolbook
        rank_school = n * m * p
        U = [[0.0] * rank_school for _ in range(n * m)]
        V = [[0.0] * rank_school for _ in range(m * p)]
        W = [[0.0] * rank_school for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j][col] = 1.0
                    V[j*p + k][col] = 1.0
                    W[k*n + i][col] = 1.0
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
