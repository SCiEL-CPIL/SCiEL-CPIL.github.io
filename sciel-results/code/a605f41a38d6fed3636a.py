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
    
    def build_rank_28():
        """Build the known rank-28 decomposition using Strassen for 2x2x2 blocks"""
        rank = 28
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
        
        # Block 1: j=0 (schoolbook for all i,k)
        for i in range(n):
            for k in range(p):
                U[i*m + 0][col] = 1.0
                V[0*p + k][col] = 1.0
                W[k*n + i][col] = 1.0
                col += 1
        
        # Block 2: j=1,2 with k=0,1 (Strassen)
        add_strassen(0, 1, 0)
        
        # Block 3: j=1,2 with k=2 (schoolbook)
        for i in range(n):
            for j in [1, 2]:
                U[i*m + j][col] = 1.0
                V[j*p + 2][col] = 1.0
                W[2*n + i][col] = 1.0
                col += 1
        
        # Block 4: j=1,2 with k=3,4 (Strassen)
        add_strassen(0, 1, 3)
        
        assert col == rank, f"Column count mismatch: {col} != {rank}"
        return U, V, W
    
    def reduce_rank(U, V, W, max_iterations=10):
        """Try to reduce rank by finding and eliminating linear dependencies"""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        current_rank = U.shape[1]
        
        for _ in range(max_iterations):
            if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
                break
            
            # Build the tensor matrix: each rank-1 term flattened
            terms = []
            for r in range(current_rank):
                t = np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
                terms.append(t.flatten())
            
            M = np.array(terms).T  # shape (900, current_rank)
            
            # Check rank
            rank_M = np.linalg.matrix_rank(M, tol=1e-10)
            
            if rank_M < current_rank:
                # Find null space vector
                U_svd, s, Vt = np.linalg.svd(M, full_matrices=False)
                c = Vt[-1, :]
                
                # Find index with largest coefficient
                k_idx = np.argmax(np.abs(c))
                if np.abs(c[k_idx]) > 1e-10:
                    # Eliminate term k_idx
                    new_U = []
                    new_V = []
                    new_W = []
                    for r in range(current_rank):
                        if r == k_idx:
                            continue
                        coeff = 1 - c[r] / c[k_idx]
                        new_U.append(U[:, r] * coeff)
                        new_V.append(V[:, r] * coeff)
                        new_W.append(W[:, r] * coeff)
                    
                    U = np.array(new_U).T
                    V = np.array(new_V).T
                    W = np.array(new_W).T
                    current_rank = U.shape[1]
                    
                    # Verify
                    if not verify(U.tolist(), V.tolist(), W.tolist()):
                        # Revert
                        U = np.array(U, dtype=float)
                        V = np.array(V, dtype=float)
                        W = np.array(W, dtype=float)
                        break
            else:
                break
        
        return U.tolist(), V.tolist(), W.tolist(), current_rank
    
    # Build initial rank-28 solution
    U, V, W = build_rank_28()
    
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
    
    # Try to reduce rank
    U_reduced, V_reduced, W_reduced, new_rank = reduce_rank(U, V, W)
    
    # Verify reduced solution
    if verify(U_reduced, V_reduced, W_reduced):
        return {"U": U_reduced, "V": V_reduced, "W": W_reduced}
    
    # Return original if reduction failed
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
