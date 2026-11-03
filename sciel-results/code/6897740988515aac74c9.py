# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """Find rank-minimal exact tensor decomposition for 4x4 * 4x6 matrix multiplication."""
    start_time = time.time()
    deadline = 145.0
    
    n, m, p = 4, 4, 6
    initial_rank = 35
    
    U = [[0] * initial_rank for _ in range(n * m)]
    V = [[0] * initial_rank for _ in range(m * p)]
    W = [[0] * initial_rank for _ in range(p * n)]
    
    # 2x2x2 Strassen (7 terms)
    S222 = [
        (([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)])),
        (([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1), (1,1,-1)])),
        (([(0,0,1)], [(0,1,1), (1,1,-1)], [(0,1,1), (1,1,1)])),
        (([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (1,0,1)])),
        (([(0,0,1), (0,1,1)], [(1,1,1)], [(0,1,1), (0,0,-1)])),
        (([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)])),
        (([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1)]))
    ]
    
    # 2x2x3 rank-5 decomposition
    # A is 2x2, B is 2x3, C is 2x3
    S223 = [
        (([(0,0,1)], [(0,0,1)], [(0,0,1)])),
        (([(1,1,1)], [(1,1,1)], [(1,1,1)])),
        (([(0,0,1), (1,1,1)], [(0,1,1), (1,1,1)], [(0,1,1), (1,1,1)])),
        (([(1,0,1), (1,1,1)], [(0,0,1), (1,0,-1)], [(1,0,1), (1,1,-1)])),
        (([(0,0,1), (1,0,-1)], [(1,0,1), (1,1,1)], [(0,0,1), (0,1,-1)]))
    ]
    
    # Build using Kronecker: outer 2x2x2, inner 2x2x3
    for i, (Ao, Bo, Co) in enumerate(S222):
        for j, (Ai, Bi, Ci) in enumerate(S223):
            r = i * 5 + j
            
            for (ri, si, ci) in Ao:
                for (rj, sj, cj) in Ai:
                    # ri, si: which 2x2 block of A (row block, col block)
                    # rj, sj: local position within that block
                    I = (ri * 2 + rj)
                    J = (si * 2 + sj)
                    U[I * m + J][r] += ci * cj
            
            for (si, ki, ci) in Bo:
                for (sj, kj, cj) in Bi:
                    J = (si * 2 + sj)
                    K = (ki * 3 + kj)
                    V[J * p + K][r] += ci * cj
            
            for (ri, ki, ci) in Co:
                for (rj, kj, cj) in Ci:
                    I = (ri * 2 + rj)
                    K = (ki * 3 + kj)
                    W[K * n + I][r] += ci * cj
    
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    if time.time() > deadline:
        return {"U": U_np.tolist(), "V": V_np.tolist(), "W": W_np.tolist()}
    
    # Remove zero columns
    active = [r for r in range(initial_rank) 
              if not (np.allclose(U_np[:,r],0) or np.allclose(V_np[:,r],0) or np.allclose(W_np[:,r],0))]
    
    if len(active) < initial_rank:
        return {"U": U_np[:,active].tolist(), "V": V_np[:,active].tolist(), "W": W_np[:,active].tolist()}
    else:
        return {"U": U_np.tolist(), "V": V_np.tolist(), "W": W_np.tolist()}

# EVOLVE-BLOCK-END
