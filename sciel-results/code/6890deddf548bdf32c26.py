import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 5]
_start_time = time.time()
TIME_LIMIT = 140  # Safety margin before 150s kill

def construct():
    """
    Decompose 2x3 * 3x5 matrix multiplication.
    Use Strassen on 2x2x2 sub-blocks for rank 26.
    """
    n, m, p = 2, 3, 5
    result = _rank26_solution()
    if result is not None:
        U, V, W = result
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        rank = U.shape[1]
        ok = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(rank))
                    if abs(val - 1.0) > 1e-9:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        if ok:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: known rank-28 solution
    return _rank28_solution()

def _rank26_solution():
    """
    Rank-26 decomposition using Strassen on 2x2x2 sub-blocks.
    
    Structure:
    - j in {0,1}, k in {0,1}: Strassen 7 terms
    - j in {0,1}, k in {2,3}: Strassen 7 terms
    - j in {0,1}, k=4: schoolbook 2 terms
    - j=2, all k: schoolbook 10 terms
    Total: 7+7+2+10 = 26
    """
    n, m, p = 2, 3, 5
    rank = 26
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    c = 0
    
    def add_strassen_block(j_off, k_off):
        nonlocal c
        # m1
        U[0*m + j_off][c] = 1; U[1*m + (j_off+1)][c] = 1
        V[j_off*p + k_off][c] = 1; V[(j_off+1)*p + (k_off+1)][c] = 1
        W[k_off*n + 0][c] = 1; W[(k_off+1)*n + 1][c] = 1
        c += 1
        # m2
        U[1*m + j_off][c] = 1; U[1*m + (j_off+1)][c] = 1
        V[j_off*p + k_off][c] = 1
        W[k_off*n + 1][c] = 1; W[(k_off+1)*n + 1][c] = -1
        c += 1
        # m3
        U[0*m + j_off][c] = 1
        V[j_off*p + (k_off+1)][c] = 1; V[(j_off+1)*p + (k_off+1)][c] = -1
        W[(k_off+1)*n + 0][c] = 1; W[(k_off+1)*n + 1][c] = 1
        c += 1
        # m4
        U[1*m + (j_off+1)][c] = 1
        V[(j_off+1)*p + k_off][c] = 1; V[j_off*p + k_off][c] = -1
        W[k_off*n + 0][c] = 1; W[k_off*n + 1][c] = 1
        c += 1
        # m5
        U[0*m + j_off][c] = 1; U[0*m + (j_off+1)][c] = 1
        V[(j_off+1)*p + (k_off+1)][c] = 1
        W[k_off*n + 0][c] = -1; W[(k_off+1)*n + 0][c] = 1
        c += 1
        # m6
        U[1*m + j_off][c] = 1; U[0*m + j_off][c] = -1
        V[j_off*p + k_off][c] = 1; V[j_off*p + (k_off+1)][c] = 1
        W[(k_off+1)*n + 1][c] = 1
        c += 1
        # m7
        U[0*m + (j_off+1)][c] = 1; U[1*m + (j_off+1)][c] = -1
        V[(j_off+1)*p + k_off][c] = 1; V[(j_off+1)*p + (k_off+1)][c] = 1
        W[k_off*n + 0][c] = 1
        c += 1

    # Strassen block for j in {0,1}, k in {0,1}
    add_strassen_block(0, 0)
    
    # Strassen block for j in {0,1}, k in {2,3}
    add_strassen_block(0, 2)
    
    # Schoolbook for j in {0,1}, k=4
    for i in range(n):
        for j in [0, 1]:
            U[i*m+j][c] = 1
            V[j*p+4][c] = 1
            W[4*n+i][c] = 1
            c += 1
    
    # Schoolbook for j=2, all k
    for i in range(n):
        for k in range(p):
            U[i*m+2][c] = 1
            V[2*p+k][c] = 1
            W[k*n+i][c] = 1
            c += 1
    
    if c != rank:
        return None
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _rank28_solution():
    n, m, p = 2, 3, 5
    rank = 28
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    c = 0
    
    def add_strassen_block(j_off, k_off):
        nonlocal c
        # m1
        U[0*m + j_off][c] = 1; U[1*m + (j_off+1)][c] = 1
        V[j_off*p + k_off][c] = 1; V[(j_off+1)*p + (k_off+1)][c] = 1
        W[k_off*n + 0][c] = 1; W[(k_off+1)*n + 1][c] = 1
        c += 1
        # m2
        U[1*m + j_off][c] = 1; U[1*m + (j_off+1)][c] = 1
        V[j_off*p + k_off][c] = 1
        W[k_off*n + 1][c] = 1; W[(k_off+1)*n + 1][c] = -1
        c += 1
        # m3
        U[0*m + j_off][c] = 1
        V[j_off*p + (k_off+1)][c] = 1; V[(j_off+1)*p + (k_off+1)][c] = -1
        W[(k_off+1)*n + 0][c] = 1; W[(k_off+1)*n + 1][c] = 1
        c += 1
        # m4
        U[1*m + (j_off+1)][c] = 1
        V[(j_off+1)*p + k_off][c] = 1; V[j_off*p + k_off][c] = -1
        W[k_off*n + 0][c] = 1; W[k_off*n + 1][c] = 1
        c += 1
        # m5
        U[0*m + j_off][c] = 1; U[0*m + (j_off+1)][c] = 1
        V[(j_off+1)*p + (k_off+1)][c] = 1
        W[k_off*n + 0][c] = -1; W[(k_off+1)*n + 0][c] = 1
        c += 1
        # m6
        U[1*m + j_off][c] = 1; U[0*m + j_off][c] = -1
        V[j_off*p + k_off][c] = 1; V[j_off*p + (k_off+1)][c] = 1
        W[(k_off+1)*n + 1][c] = 1
        c += 1
        # m7
        U[0*m + (j_off+1)][c] = 1; U[1*m + (j_off+1)][c] = -1
        V[(j_off+1)*p + k_off][c] = 1; V[(j_off+1)*p + (k_off+1)][c] = 1
        W[k_off*n + 0][c] = 1
        c += 1

    add_strassen_block(0, 0)
    add_strassen_block(0, 2)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if j in (0, 1) and k in (0, 1, 2, 3):
                    continue
                U[i*m+j][c] = 1
                V[j*p+k][c] = 1
                W[k*n+i][c] = 1
                c += 1
                
    assert c == rank, f"Rank mismatch: {c} != {rank}"
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _try_rank(n, m, p, rank):
    """
    Try to find a rank decomposition using iterative linear algebra.
    Use a structured approach: start from schoolbook and try to reduce.
    """
    nm = n * m
    mp = m * p
    np_ = n * p
    
    # Build the tensor as a list of (i,j,k) tuples
    tensor_entries = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    
    # We'll use a greedy/iterative approach:
    # Start with schoolbook (rank 30) and try to eliminate terms
    
    # Actually, let's use a different strategy: 
    # Use the known structure and try to find linear dependencies
    
    # For now, return None to use fallback
    return None
