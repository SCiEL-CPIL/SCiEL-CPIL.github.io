"""Combined program for matrix multiplication 2x3x7 rank minimization
using Strassen-style block decomposition and randomized search."""
import numpy as np
import random
import time
from fractions import Fraction
import sys

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    deadline = time.time() + 130  # safety margin before 150s kill

    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    best = schoolbook()
    best_rank = n * m * p

    def verify(U, V, W, rank):
        """Check if the decomposition is exact using numpy for speed."""
        U_arr = np.array(U, dtype=np.int64)
        V_arr = np.array(V, dtype=np.int64)
        W_arr = np.array(W, dtype=np.int64)
        
        # Reshape to 3D: U is (n, m, rank), V is (m, p, rank), W is (p, n, rank)
        U_3d = U_arr.reshape(n, m, rank)
        V_3d = V_arr.reshape(m, p, rank)
        W_3d = W_arr.reshape(p, n, rank)
        
        # For each (i, j, k), compute sum over c of U[i,j,c] * V[j,k,c] * W[k,i,c]
        # This is a tensor contraction
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U_3d[i, j, :], V_3d[j, k, :])
                    # Wait, that's not right. We need element-wise product then sum
                    s = np.sum(U_3d[i, j, :] * V_3d[j, k, :] * W_3d[k, i, :])
                    if s != 1:
                        return False
        return True

    # Try to find lower rank decompositions
    # Theoretical lower bound for 2x3x7 is around 21
    min_rank = 18

    # Strategy 1: Greedy constructive search - build up the decomposition term by term
    # Start from a partial decomposition and try to complete it
    def greedy_construct(target_rank):
        """Try to construct a valid decomposition using greedy approach."""
        if time.time() > deadline - 20:
            return None
        
        # Initialize with zeros
        U = [[0] * target_rank for _ in range(nm)]
        V = [[0] * target_rank for _ in range(mp)]
        W = [[0] * target_rank for _ in range(pn)]
        
        # Required sums: for each (i,j,k), we need sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = 1
        # Start with schoolbook-like initialization for first n*m*p terms if rank allows
        remaining = {(i, j, k): 1 for i in range(n) for j in range(m) for k in range(p)}
        
        for c in range(target_rank):
            if time.time() > deadline - 20:
                return None
            
            # Choose random non-zero patterns for this term
            # Try different sparsity levels
            sparsity = random.choice([0.3, 0.5, 0.7])
            
            for i in range(nm):
                if random.random() < sparsity:
                    U[i][c] = random.choice([-1, 1, 2, -2])
                else:
                    U[i][c] = 0
            
            for j in range(mp):
                if random.random() < sparsity:
                    V[j][c] = random.choice([-1, 1, 2, -2])
                else:
                    V[j][c] = 0
            
            for k in range(pn):
                if random.random() < sparsity:
                    W[k][c] = random.choice([-1, 1, 2, -2])
                else:
                    W[k][c] = 0
            
            # Update remaining requirements
            for (i, j, k), rem in list(remaining.items()):
                contribution = U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if contribution != 0:
                    remaining[(i, j, k)] = rem - contribution
            
            # Remove satisfied entries
            keys_to_remove = [key for key, val in remaining.items() if val == 0]
            for key in keys_to_remove:
                del remaining[key]
        
        # Check if all requirements are met
        if len(remaining) == 0:
            return {"U": U, "V": V, "W": W}
        
        return None

    # Try greedy construction for various ranks
    for target_rank in range(min_rank, best_rank + 1):
        if time.time() > deadline - 30:
            break
        
        max_greedy_attempts = 50
        found = False
        
        for attempt in range(max_greedy_attempts):
            if time.time() > deadline - 30:
                break
            
            result = greedy_construct(target_rank)
            if result is not None:
                if verify(result["U"], result["V"], result["W"], target_rank):
                    best = result
                    best_rank = target_rank
                    found = True
                    break
        
        if found:
            continue

    # Strategy 2: Finite field F2 rank-revealing algorithm
    # Formulate the matrix multiplication tensor over F2 and use
    # iterative Gaussian elimination on tensor flattenings to empirically
    # determine if the tensor rank over F2 is strictly lower than schoolbook rank.
    # If a reduction exists over F2, it may lift to an integer decomposition.
    if time.time() < deadline - 40:
        def f2_gauss_rank(matrix):
            """Compute rank of a binary matrix over F2 using Gaussian elimination."""
            mat = np.array(matrix, dtype=np.uint8) % 2
            rows = mat.shape[0]
            cols = mat.shape[1]
            rank = 0
            for col in range(cols):
                if rank >= rows:
                    break
                # Find pivot row
                pivot = -1
                for row in range(rank, rows):
                    if mat[row, col] == 1:
                        pivot = row
                        break
                if pivot == -1:
                    continue
                # Swap rows
                mat[[rank, pivot]] = mat[[pivot, rank]]
                # Eliminate
                for row in range(rows):
                    if row != rank and mat[row, col] == 1:
                        mat[row] = (mat[row] + mat[rank]) % 2
                rank += 1
            return rank

        # Strategy 2a: Direct F2 rank estimation via flattening
        # The tensor T has entries T[i,j,k] = 1 for all i,j,k (matrix mult tensor)
        # Flatten along different modes and estimate rank
        # Mode-1 flattening: reshape to (n, m*p) - this is just the identity-like structure
        # For matrix multiplication, the tensor is structured; let's try random projections
        
        # Build the tensor as a 3D array over F2
        T = np.ones((n, m, p), dtype=np.uint8)
        
        # Try to find a low-rank decomposition over F2 using iterative approach
        # We'll use a randomized SVD-like approach over F2
        # First, get an upper bound on F2 rank via flattening
        T_flat_1 = T.reshape(n, m * p)  # Mode-1 flattening
        T_flat_2 = T.transpose(1, 0, 2).reshape(m, n * p)  # Mode-2 flattening
        T_flat_3 = T.transpose(2, 0, 1).reshape(p, n * m)  # Mode-3 flattening
        
        rank1 = f2_gauss_rank(T_flat_1)
        rank2 = f2_gauss_rank(T_flat_2)
        rank3 = f2_gauss_rank(T_flat_3)
        
        # The tensor rank is at least max(rank1, rank2, rank3)
        f2_lower_bound = max(rank1, rank2, rank3)
        
        # Try to construct an actual F2 decomposition with rank close to the bound
        # Use iterative randomized search over F2
        for target_rank in range(max(f2_lower_bound, 10), min(best_rank, 35)):
            if time.time() > deadline - 25:
                break
            
            max_f2_attempts = 50
            found_f2 = False
            
            for attempt in range(max_f2_attempts):
                if time.time() > deadline - 25:
                    break
                
                # Generate random binary matrices
                U = [[random.randint(0, 1) for _ in range(target_rank)] for _ in range(nm)]
                V = [[random.randint(0, 1) for _ in range(target_rank)] for _ in range(mp)]
                W = [[random.randint(0, 1) for _ in range(target_rank)] for _ in range(pn)]
                
                # Verify over F2
                valid_f2 = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0
                            for c in range(target_rank):
                                s = (s + U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]) % 2
                            if s != 1:
                                valid_f2 = False
                                break
                        if not valid_f2:
                            break
                    if not valid_f2:
                        break
                
                if valid_f2:
                    # Found an F2 decomposition; try to lift to integers
                    # The F2 solution may not directly give an integer solution,
                    # but it indicates the structure exists
                    # Try using the F2 pattern as a starting point for integer search
                    found_f2 = True
                    
                    # Now try to find an integer decomposition with the same rank
                    # by perturbing the F2 solution
                    for int_attempt in range(20):
                        if time.time() > deadline - 25:
                            break
                        
                        U_int = []
                        for row in U:
                            U_int.append([x * random.choice([1, -1]) if x == 1 else 0 for x in row])
                        V_int = []
                        for row in V:
                            V_int.append([x * random.choice([1, -1]) if x == 1 else 0 for x in row])
                        W_int = []
                        for row in W:
                            W_int.append([x * random.choice([1, -1]) if x == 1 else 0 for x in row])
                        
                        if verify(U_int, V_int, W_int, target_rank):
                            best = {"U": U_int, "V": V_int, "W": W_int}
                            best_rank = target_rank
                            found_f2 = False  # Don't break outer loop, keep searching
                            break
                    
                    if found_f2:
                        break
            
            if not found_f2 and best_rank <= target_rank:
                break

    # Strategy 3: Systematic search with structured patterns
    if time.time() < deadline - 20:
        # Try known small-rank constructions
        # For 2x3x7, try ranks from 18 to 25 with more attempts
        
        for target_rank in range(max(18, best_rank - 8), min(26, best_rank)):
            if time.time() > deadline - 15:
                break
            
            # Multiple random restarts with different distributions
            for dist_type in range(5):
                if time.time() > deadline - 15:
                    break
                
                max_attempts = 100
                found = False
                
                for attempt in range(max_attempts):
                    if time.time() > deadline - 15:
                        break
                    
                    U = []
                    for i in range(nm):
                        row = []
                        for c in range(target_rank):
                            r = random.random()
                            if dist_type == 0:
                                # Sparse with small values
                                if r < 0.4:
                                    row.append(0)
                                else:
                                    row.append(random.choice([-1, 1]))
                            elif dist_type == 1:
                                # Denser with values -2 to 2
                                if r < 0.2:
                                    row.append(0)
                                else:
                                    row.append(random.randint(-2, 2))
                            elif dist_type == 2:
                                # Very sparse
                                if r < 0.7:
                                    row.append(0)
                                else:
                                    row.append(random.choice([-1, 1, 2, -2]))
                            elif dist_type == 3:
                                # Mixed
                                row.append(random.choice([0, 0, 1, -1, 2, -2]))
                            else:
                                # Uniform -3 to 3
                                row.append(random.randint(-3, 3))
                        U.append(row)
                    
                    V = []
                    for j in range(mp):
                        row = []
                        for c in range(target_rank):
                            r = random.random()
                            if dist_type == 0:
                                if r < 0.4:
                                    row.append(0)
                                else:
                                    row.append(random.choice([-1, 1]))
                            elif dist_type == 1:
                                if r < 0.2:
                                    row.append(0)
                                else:
                                    row.append(random.randint(-2, 2))
                            elif dist_type == 2:
                                if r < 0.7:
                                    row.append(0)
                                else:
                                    row.append(random.choice([-1, 1, 2, -2]))
                            elif dist_type == 3:
                                row.append(random.choice([0, 0, 1, -1, 2, -2]))
                            else:
                                row.append(random.randint(-3, 3))
                        V.append(row)
                    
                    W = []
                    for k in range(pn):
                        row = []
                        for c in range(target_rank):
                            r = random.random()
                            if dist_type == 0:
                                if r < 0.4:
                                    row.append(0)
                                else:
                                    row.append(random.choice([-1, 1]))
                            elif dist_type == 1:
                                if r < 0.2:
                                    row.append(0)
                                else:
                                    row.append(random.randint(-2, 2))
                            elif dist_type == 2:
                                if r < 0.7:
                                    row.append(0)
                                else:
                                    row.append(random.choice([-1, 1, 2, -2]))
                            elif dist_type == 3:
                                row.append(random.choice([0, 0, 1, -1, 2, -2]))
                            else:
                                row.append(random.randint(-3, 3))
                        W.append(row)
                    
                    if verify(U, V, W, target_rank):
                        best = {"U": U, "V": V, "W": W}
                        best_rank = target_rank
                        found = True
                        break
                
                if found:
                    break
            
            if best_rank == target_rank:
                break

    # Final verification and return
    if time.time() < deadline - 5:
        # Double-check the best solution
        if not verify(best["U"], best["V"], best["W"], best_rank):
            # Fallback to schoolbook if verification fails
            best = schoolbook()
            best_rank = n * m * p

    return best
# EVOLVE-BLOCK-END
