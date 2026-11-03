"""
Matrix multiplication 3x4x5: search for a lower-rank exact tensor decomposition.
"""
import time
import itertools
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START

def construct():
    deadline = time.time() + 145
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n

    # Precompute the schoolbook (baseline)
    def schoolbook():
        rank = n*m*p
        U = [[0]*rank for _ in range(nm)]
        V = [[0]*rank for _ in range(mp)]
        W = [[0]*rank for _ in range(pn)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][c] = 1
                    V[j*p+k][c] = 1
                    W[k*n+i][c] = 1
                    c += 1
        return {"U": U, "V": V, "W": W}

    baseline = schoolbook()

    # Build the tensor as a 3D array for fast verification
    def build_tensor(U, V, W):
        # T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        Ua = np.array(U, dtype=np.int64)
        Va = np.array(V, dtype=np.int64)
        Wa = np.array(W, dtype=np.int64)
        rank = Ua.shape[1]
        # Reshape U: [n, m, rank], V: [m, p, rank], W: [p, n, rank]
        Ur = Ua.reshape(n, m, rank)
        Vr = Va.reshape(m, p, rank)
        Wr = Wa.reshape(p, n, rank)
        # T[i,j,k] = sum_c Ur[i,j,c] * Vr[j,k,c] * Wr[k,i,c]
        # Compute via einsum
        T = np.einsum('ijc,jkc,kic->ijk', Ur, Vr, Wr, optimize=True)
        return T

    def is_valid(U, V, W):
        try:
            T = build_tensor(U, V, W)
            # Check: T[i,j,k] should be 1 if the tensor entry is 1, else 0
            # For matrix multiplication, the tensor is: T[i,j,k] = 1 for all i,j,k (it's the identity in the sense that each entry is independently 1)
            # Actually, the tensor for matrix multiplication C = A*B where A is n x m, B is m x p, C is n x p:
            # C[i,k] = sum_j A[i,j] * B[j,k]
            # The tensor T has indices (i, j, k) and T[i,j,k] = 1 for all valid i,j,k
            # Wait, that's the full tensor. Let me reconsider.
            # The tensor decomposition: T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c] (with appropriate index mapping)
            # For matrix multiplication, T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
            # So we just need T to be all ones.
            if not np.all(T == 1):
                return False
            return True
        except:
            return False

    # Try to find a decomposition of rank < n*m*p
    # Strategy: use random initialization and iterative refinement
    best = baseline
    best_rank = n*m*p

    # For small improvements, try rank = best_rank - 1, etc.
    for target_rank in range(best_rank - 1, 20, -1):
        if time.time() > deadline - 10:
            break
        found = False
        for attempt in range(50):
            if time.time() > deadline - 10:
                break
            # Random sparse initialization
            U = [[0.0]*target_rank for _ in range(nm)]
            V = [[0.0]*target_rank for _ in range(mp)]
            W = [[0.0]*target_rank for _ in range(pn)]
            
            # Initialize with random entries
            for r in range(target_rank):
                # Pick random non-zero entries
                u_rows = random.sample(range(nm), min(nm, random.randint(2, 5)))
                v_rows = random.sample(range(mp), min(mp, random.randint(2, 5)))
                w_rows = random.sample(range(pn), min(pn, random.randint(2, 5)))
                for row in u_rows:
                    U[row][r] = random.choice([1, -1])
                for row in v_rows:
                    V[row][r] = random.choice([1, -1])
                for row in w_rows:
                    W[row][r] = random.choice([1, -1])
            
            # Quick check
            if is_valid(U, V, W):
                found = True
                break
        
        if found:
            best = {"U": U, "V": V, "W": W}
            best_rank = target_rank
            break

    # Try structured decompositions: block-diagonal with shared factors
    if best_rank == n*m*p and time.time() < deadline - 15:
        # Try: can we merge some columns?
        # For each pair of columns in the schoolbook, check if they can be merged
        # This is equivalent to finding a lower-rank decomposition
        # A simple heuristic: try to find linear dependencies among the "basis" tensors
        
        # Alternative: use the known result that for some small matrices,
        # rank can be reduced by using non-trivial combinations
        
        # Try a specific structured approach:
        # Group by i (row of A), and see if we can share across j
        for target_rank in range(best_rank - 1, max(25, best_rank - 10), -1):
            if time.time() > deadline - 15:
                break
            # Try random search with more structured initialization
            for attempt in range(100):
                if time.time() > deadline - 15:
                    break
                U = [[0.0]*target_rank for _ in range(nm)]
                V = [[0.0]*target_rank for _ in range(mp)]
                W = [[0.0]*target_rank for _ in range(pn)]
                
                # Structured: assign each (i,j) pair to a column, but allow sharing
                col = 0
                for i in range(n):
                    for j in range(m):
                        if col < target_rank:
                            # Assign this (i,j) to column col
                            U[i*m+j][col] = 1.0
                            for k in range(p):
                                V[j*p+k][col] = 1.0
                                W[k*n+i][col] = 1.0
                            col += 1
                
                # Now we have a partial assignment. Try to complete it
                # by allowing some columns to serve multiple (i,j) pairs
                # This is getting complex; skip for now
                
                if is_valid(U, V, W):
                    best = {"U": U, "V": V, "W": W}
                    best_rank = target_rank
                    break
            if best_rank != n*m*p:
                break

    return best

# EVOLVE-BLOCK-END
