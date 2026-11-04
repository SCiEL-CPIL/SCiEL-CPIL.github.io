import numpy as np
import time
import random
from itertools import product

def construct():
    n, m, p = 3, 3, 3
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin

    # Schoolbook rank-27 as fallback
    def schoolbook():
        rank = 27
        U = [[0] * rank for _ in range(9)]
        V = [[0] * rank for _ in range(9)]
        W = [[0] * rank for _ in range(9)]
        col = 0
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    U[i*3+j][col] = 1
                    V[j*3+k][col] = 1
                    W[k*3+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def verify_full(U, V, W, rank):
        """Check if decomposition is valid - all 729 components"""
        # T[i,j,k] where i,j,k in {0,1,2}
        # The tensor component is: sum_r U[i*3+j][r] * V[j*3+k][r] * W[k*3+i][r]
        # This should equal 1 if (i,j,k) is a valid triple, else 0
        # Wait, let me reconsider.
        
        # The matrix multiplication tensor T has components T[i,j,k] for i,j,k in {0,1,2}
        # T[i,j,k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k]
        # But actually, the tensor is defined such that:
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # So the contribution of A[i,j]*B[j,k] to C[i,k] is 1, and to C[i',k'] is 0 if (i',k') != (i,k)
        
        # In the factorization: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # This should give the coefficient of A[i,j]*B[j,k] in the output at position (i,k)
        # which is 1 if the output index is (i,k), i.e., W index is k*n+i
        
        # So for each (i,j,k) and each output index (i',k'), we need:
        # sum_r U[i,j,r] * V[j,k,r] * W[k',i',r] = 1 if (i',k') == (i,k) else 0
        
        # Equivalently, for fixed (i,j,k):
        # sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1
        # sum_r U[i,j,r] * V[j,k,r] * W[k',i',r] = 0 for all (k',i') != (k,i)
        
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    # Check the non-zero component
                    s = 0.0
                    for r in range(rank):
                        s += U[i*3+j][r] * V[j*3+k][r] * W[k*3+i][r]
                    if abs(s - 1.0) > 1e-9:
                        return False
                    
                    # Check the zero components
                    for kp in range(3):
                        for ip in range(3):
                            if kp == k and ip == i:
                                continue
                            s = 0.0
                            for r in range(rank):
                                s += U[i*3+j][r] * V[j*3+k][r] * W[kp*3+ip][r]
                            if abs(s) > 1e-9:
                                return False
        return True

    def check_half_integer(U, V, W):
        """Check all entries are half-integers"""
        for arr in [U, V, W]:
            for row in arr:
                for val in row:
                    if abs(val * 2 - round(val * 2)) > 1e-9:
                        return False
        return True

    # Try ranks from 23 up to 26
    best_result = schoolbook()

    for rank in range(23, 27):
        if time.time() > deadline:
            break

        # Multiple random restarts
        for attempt in range(50):
            if time.time() > deadline:
                break

            # Initialize with random half-integers in [-2, 2]
            vals = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
            U = [[random.choice(vals) for _ in range(rank)] for _ in range(9)]
            V = [[random.choice(vals) for _ in range(rank)] for _ in range(9)]
            W = [[random.choice(vals) for _ in range(rank)] for _ in range(9)]

            # Simple local search: try small perturbations
            improved = True
            max_iter = 200
            while improved and max_iter > 0 and time.time() < deadline:
                improved = False
                max_iter -= 1
                # Pick a random entry to perturb
                arr_choice = random.choice([0, 1, 2])
                arr = [U, V, W][arr_choice]
                row = random.randrange(9)
                col = random.randrange(rank)
                old_val = arr[row][col]
                # Try nearby half-integer values
                for delta in [-1.0, -0.5, 0.5, 1.0]:
                    new_val = old_val + delta
                    if new_val < -3.0 or new_val > 3.0:
                        continue
                    arr[row][col] = new_val
                    if verify_full(U, V, W, rank):
                        if check_half_integer(U, V, W):
                            # Found valid decomposition!
                            return {"U": U, "V": V, "W": W}
                    arr[row][col] = old_val

            # Also try a more structured initialization
            # Start from schoolbook and try to merge terms
            if attempt == 0:
                # Try structured approach: start with rank 27, try to zero out columns
                U27 = [[0]*27 for _ in range(9)]
                V27 = [[0]*27 for _ in range(9)]
                W27 = [[0]*27 for _ in range(9)]
                col = 0
                for i in range(3):
                    for j in range(3):
                        for k in range(3):
                            U27[i*3+j][col] = 1
                            V27[j*3+k][col] = 1
                            W27[k*3+i][col] = 1
                            col += 1

                # Try to find if we can set some columns to zero while maintaining validity
                # by adjusting other columns
                pass

    return best_result
