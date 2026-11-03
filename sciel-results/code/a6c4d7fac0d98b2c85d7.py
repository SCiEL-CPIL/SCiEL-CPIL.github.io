import time
import sys
import numpy as np

DIMENSIONS = [2, 4, 6]

def construct():
    start_time = time.time()
    deadline = 145.0

    n, m, p = DIMENSIONS  # 2, 4, 6
    nm, mp, pn = n * m, m * p, p * n  # 8, 24, 12

    # Rank 24 construction based on splitting middle dimension 4 into 2+2
    # and using rank 2 decomposition for 2x2x1 blocks.
    
    rank = 24
    U = np.zeros((nm, rank), dtype=float)
    V = np.zeros((mp, rank), dtype=float)
    W = np.zeros((pn, rank), dtype=float)
    
    col_offset = 0
    
    # Part 1: A[:, 0:2] and B[0:2, :]
    # A indices: (0,0)->0, (0,1)->1, (1,0)->4, (1,1)->5
    # B indices: (0,k)->k, (1,k)->6+k
    # C indices: (0,k)->2k, (1,k)->2k+1
    
    # Mapping for Part 1
    # A local (i, j_loc) -> global i*m + (j_loc)
    # B local (j_loc, k) -> global j_loc*p + k
    # C local (i, k) -> global k*n + i
    
    # Decomposition for 2x2x1:
    # r1: A(0,0), A(1,0) active. B(0,k) active. C(0,k), C(1,k) active.
    # r2: A(0,1), A(1,1) active. B(1,k) active. C(0,k), C(1,k) active.
    
    for k in range(p):
        # Term r1 for this k
        # U: A(0,0)=1, A(1,0)=1
        U[0 * m + 0, col_offset] = 1.0
        U[1 * m + 0, col_offset] = 1.0
        # V: B(0,k)=1
        V[0 * p + k, col_offset] = 1.0
        # W: C(0,k)=1, C(1,k)=1
        W[k * n + 0, col_offset] = 1.0
        W[k * n + 1, col_offset] = 1.0
        
        col_offset += 1
        
        # Term r2 for this k
        # U: A(0,1)=1, A(1,1)=1
        U[0 * m + 1, col_offset] = 1.0
        U[1 * m + 1, col_offset] = 1.0
        # V: B(1,k)=1
        V[1 * p + k, col_offset] = 1.0
        # W: C(0,k)=1, C(1,k)=1
        W[k * n + 0, col_offset] = 1.0
        W[k * n + 1, col_offset] = 1.0
        
        col_offset += 1

    # Part 2: A[:, 2:4] and B[2:4, :]
    # A local (i, j_loc) -> global i*m + (2 + j_loc)
    # B local (j_loc, k) -> global (2 + j_loc)*p + k
    # C local (i, k) -> global k*n + i
    
    for k in range(p):
        # Term r1 for this k
        # U: A(0,2)=1, A(1,2)=1
        U[0 * m + 2, col_offset] = 1.0
        U[1 * m + 2, col_offset] = 1.0
        # V: B(2,k)=1
        V[2 * p + k, col_offset] = 1.0
        # W: C(0,k)=1, C(1,k)=1
        W[k * n + 0, col_offset] = 1.0
        W[k * n + 1, col_offset] = 1.0
        
        col_offset += 1
        
        # Term r2 for this k
        # U: A(0,3)=1, A(1,3)=1
        U[0 * m + 3, col_offset] = 1.0
        U[1 * m + 3, col_offset] = 1.0
        # V: B(3,k)=1
        V[3 * p + k, col_offset] = 1.0
        # W: C(0,k)=1, C(1,k)=1
        W[k * n + 0, col_offset] = 1.0
        W[k * n + 1, col_offset] = 1.0
        
        col_offset += 1

    # Verification
    def verify(U, V, W, tol=1e-8):
        T = {}
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T[(a, b, c)] = 1.0
        
        # Check non-zeros
        for (a, b, c), val in T.items():
            s = 0.0
            for r in range(rank):
                s += U[a][r] * V[b][r] * W[c][r]
            if abs(s - val) > tol:
                return False
        # Check zeros
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    if (a, b, c) in T:
                        continue
                    s = 0.0
                    for r in range(rank):
                        s += U[a][r] * V[b][r] * W[c][r]
                    if abs(s) > tol:
                        return False
        return True

    if verify(U.tolist(), V.tolist(), W.tolist()):
        print(f"Found valid rank {rank} decomposition", file=sys.stderr)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        print("Verification failed for rank 24", file=sys.stderr)
        # Fallback to naive rank 48
        rank0 = n * m * p
        U0 = [[0.0] * rank0 for _ in range(nm)]
        V0 = [[0.0] * rank0 for _ in range(mp)]
        W0 = [[0.0] * rank0 for _ in range(pn)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    col = i * (m * p) + j * p + k
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U0[a][col] = 1.0
                    V0[b][col] = 1.0
                    W0[c][col] = 1.0
        return {"U": U0, "V": V0, "W": W0}
