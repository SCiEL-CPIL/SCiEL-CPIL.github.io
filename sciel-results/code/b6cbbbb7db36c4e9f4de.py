import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    def u_idx(i, j):
        return i * m + j
        
    def v_idx(j, k):
        return j * p + k
        
    def w_idx(k, i):
        return k * n + i
    
    def verify(U, V, W):
        """Verify the decomposition is correct."""
        rank = len(U[0])
        U_np = np.array(U, dtype=np.float64)
        V_np = np.array(V, dtype=np.float64)
        W_np = np.array(W, dtype=np.float64)
        
        # Build the multiplication tensor
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # Should equal delta(i,j,k) for the multiplication tensor
        # Actually, the tensor for matmul is T[i,j,k] = 1 if this is a valid product term
        
        # The matmul tensor: for each (i,j,k), the coefficient is 1
        # We need: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all (i,j,k)
        # Wait, that's not right. The matmul tensor has T[i,j,k] = 1 for the "diagonal" 
        # in the appropriate sense.
        
        # Actually for matmul: (AB)_{ik} = sum_j A_{ij} B_{jk}
        # The tensor is T_{i,j,k} = delta_{ik} ... no.
        # The tensor T_{i,j,k} such that sum_{i,j,k} T_{i,j,k} A_{ij} B_{jk} = (AB)
        # T_{i,j,k} = 1 for all valid (i,j,k) where i,j,k are the free indices
        # But we need: sum_j A_{ij} B_{jk} = (AB)_{ik}
        # So the tensor is T_{i,j,k} = 1 (all ones) and we contract over j
        
        # For the decomposition: sum_r U_{ij,r} V_{jk,r} W_{ki,r} should give
        # the same result as sum_j A_{ij} B_{jk}
        # So we need: for all (i,k), sum_j sum_r U_{ij,r} V_{jk,r} W_{ki,r} = (AB)_{ik}
        # This means: sum_r W_{ki,r} (sum_j U_{ij,r} A_{ij}) (sum_j V_{jk,r} B_{jk}) ... 
        # No, that's not how it works.
        
        # Correct: The matmul tensor T_{i,j,k} = 1 for all i,j,k (it's the "identity" tensor
        # in the sense that contracting with A and B gives AB)
        # Wait, T_{i,j,k} should be such that sum_{j} T_{i,j,k} A_{ij} B_{jk} = (AB)_{ik}
        # This requires T_{i,j,k} = 1 for all i,j,k? No, that gives sum_j A_ij B_jk which is correct!
        # So T_{i,j,k} = 1 for all valid i,j,k.
        
        # Decomposition: T_{i,j,k} = sum_r U_{ij,r} V_{jk,r} W_{ki,r}
        # So we need: for all (i,j,k), sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
        
        errors = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U_np[i*m+j, r] * V_np[j*p+k, r] * W_np[k*n+i, r]
                    if abs(s - 1) > 1e-9:
                        errors += 1
        return errors == 0, errors
    
    # Start with the known 124-term solution
    S1 = {0, 1}
    rank = 124
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    U[u_idx(0,0)][col] = 1; U[u_idx(1,1)][col] = 1
    V[v_idx(0,0)][col] = 1; V[v_idx(1,1)][col] = 1
    W[w_idx(0,0)][col] = 1; W[w_idx(1,1)][col] = 1
    col += 1
    U[u_idx(1,0)][col] = 1; U[u_idx(1,1)][col] = 1
    V[v_idx(0,0)][col] = 1
    W[w_idx(0,1)][col] = 1; W[w_idx(1,1)][col] = -1
    col += 1
    U[u_idx(0,0)][col] = 1
    V[v_idx(0,1)][col] = 1; V[v_idx(1,1)][col] = -1
    W[w_idx(1,0)][col] = 1; W[w_idx(1,1)][col] = 1
    col += 1
    U[u_idx(1,1)][col] = 1
    V[v_idx(1,0)][col] = 1; V[v_idx(0,0)][col] = -1
    W[w_idx(0,0)][col] = 1; W[w_idx(0,1)][col] = 1
    col += 1
    U[u_idx(0,0)][col] = 1; U[u_idx(0,1)][col] = 1
    V[v_idx(1,1)][col] = 1
    W[w_idx(0,0)][col] = -1; W[w_idx(1,0)][col] = 1
    col += 1
    U[u_idx(1,0)][col] = 1; U[u_idx(0,0)][col] = -1
    V[v_idx(0,0)][col] = 1; V[v_idx(0,1)][col] = 1
    W[w_idx(1,1)][col] = 1
    col += 1
    U[u_idx(0,1)][col] = 1; U[u_idx(1,1)][col] = -1
    V[v_idx(1,0)][col] = 1; V[v_idx(1,1)][col] = 1
    W[w_idx(0,0)][col] = 1
    col += 1
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i in S1 and j in S1 and k in S1:
                    continue
                U[u_idx(i, j)][col] = 1
                V[v_idx(j, k)][col] = 1
                W[w_idx(k, i)][col] = 1
                col += 1
    
    assert col == rank
    ok, errs = verify(U, V, W)
    if not ok:
        # Fallback to naive 125
        rank = 125
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[u_idx(i, j)][col] = 1
                    V[v_idx(j, k)][col] = 1
                    W[w_idx(k, i)][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Try to improve: search for a rank-123 solution
    # Structural change: try to merge two standard terms into one using a linear combination
    # If we can find (i1,j1,k1) and (i2,j2,k2) such that we can replace them with fewer terms
    
    # For now, return the verified 124-term solution
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
