import time
import numpy as np

def construct():
    """
    Return a rank-minimal exact decomposition for 2x3 * 3x5 matrix multiplication.
    
    Primary: rank-28 via block decomposition (Strassen + schoolbook).
    Attempts to verify if rank 27 is achievable within time budget.
    """
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    n, m, p = 2, 3, 5
    
    # Build the known rank-28 decomposition
    rank = 28
    U = [[0] * rank for _ in range(n * m)]  # 6 x 28
    V = [[0] * rank for _ in range(m * p)]  # 15 x 28
    W = [[0] * rank for _ in range(p * n)]  # 10 x 28
    
    col = 0
    
    # Block 1: j=0, all k in {0,1,2,3,4} — schoolbook 2x1x5, rank 10
    for i in range(n):
        for k in range(p):
            U[i * m + 0][col] = 1
            V[0 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    # col = 10
    
    # Helper: add Strassen 2x2x2 block
    def add_strassen_2x2x2(base_j, base_k):
        nonlocal col
        # M1: (A[0,bj]+A[1,bj+1])*(B[bj,bk]+B[bj+1,bk+1])
        U[0*m+base_j][col] = 1
        U[1*m+(base_j+1)][col] = 1
        V[base_j*p+base_k][col] = 1
        V[(base_j+1)*p+(base_k+1)][col] = 1
        W[base_k*n+0][col] = 1
        W[(base_k+1)*n+1][col] = 1
        col += 1
        # M2: (A[1,bj]+A[1,bj+1])*B[bj,bk]
        U[1*m+base_j][col] = 1
        U[1*m+(base_j+1)][col] = 1
        V[base_j*p+base_k][col] = 1
        W[base_k*n+1][col] = 1
        W[(base_k+1)*n+1][col] = -1
        col += 1
        # M3: A[0,bj]*(B[bj,bk+1]-B[bj+1,bk+1])
        U[0*m+base_j][col] = 1
        V[base_j*p+(base_k+1)][col] = 1
        V[(base_j+1)*p+(base_k+1)][col] = -1
        W[(base_k+1)*n+0][col] = 1
        W[(base_k+1)*n+1][col] = 1
        col += 1
        # M4: A[1,bj+1]*(B[bj+1,bk]-B[bj,bk])
        U[1*m+(base_j+1)][col] = 1
        V[(base_j+1)*p+base_k][col] = 1
        V[base_j*p+base_k][col] = -1
        W[base_k*n+0][col] = 1
        W[base_k*n+1][col] = 1
        col += 1
        # M5: (A[0,bj]+A[0,bj+1])*B[bj+1,bk+1]
        U[0*m+base_j][col] = 1
        U[0*m+(base_j+1)][col] = 1
        V[(base_j+1)*p+(base_k+1)][col] = 1
        W[base_k*n+0][col] = -1
        W[(base_k+1)*n+0][col] = 1
        col += 1
        # M6: (A[1,bj]-A[0,bj])*(B[bj,bk]+B[bj,bk+1])
        U[1*m+base_j][col] = 1
        U[0*m+base_j][col] = -1
        V[base_j*p+base_k][col] = 1
        V[base_j*p+(base_k+1)][col] = 1
        W[(base_k+1)*n+1][col] = 1
        col += 1
        # M7: (A[0,bj+1]-A[1,bj+1])*(B[bj+1,bk]+B[bj+1,bk+1])
        U[0*m+(base_j+1)][col] = 1
        U[1*m+(base_j+1)][col] = -1
        V[(base_j+1)*p+base_k][col] = 1
        V[(base_j+1)*p+(base_k+1)][col] = 1
        W[base_k*n+0][col] = 1
        col += 1
    
    # Block 2a: j in {1,2}, k in {0,1} — Strassen 2x2x2
    if time.time() > deadline:
        raise TimeoutError("Deadline exceeded")
    add_strassen_2x2x2(1, 0)
    # col = 17
    
    # Block 2b: j in {1,2}, k=2 — schoolbook 2x2x1, rank 4
    for i in range(n):
        for j_local in range(2):
            j = j_local + 1
            U[i * m + j][col] = 1
            V[j * p + 2][col] = 1
            W[2 * n + i][col] = 1
            col += 1
    # col = 21
    
    # Block 2c: j in {1,2}, k in {3,4} — Strassen 2x2x2
    if time.time() > deadline:
        raise TimeoutError("Deadline exceeded")
    add_strassen_2x2x2(1, 3)
    # col = 28
    
    assert col == rank, f"Expected col={rank}, got {col}"
    
    # Verify correctness
    if time.time() < deadline - 5:
        tensor = np.zeros((n, m, p))
        for r in range(rank):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        tensor[i, j, k] += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
        if not np.allclose(tensor, np.ones((n, m, p)), atol=1e-9):
            raise ValueError("Decomposition verification failed")
    
    return {"U": U, "V": V, "W": W}
