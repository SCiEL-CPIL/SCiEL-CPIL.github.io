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
    
    # Try to find lower rank using random initialization + refinement
    # The matrix multiplication tensor T[i,j,k] = sum_r U[i,j,r]*V[j,k,r]*W[k,i,r]
    # We need T[i,j,k] = delta_{i,k} for the (i,j,k) -> output index mapping
    # Actually, the convention is W[k*n+i, r] so output index is k*3+i
    
    # Target tensor: T[i,j,k] where output is C[i,k] = sum_j A[i,j]*B[j,k]
    # So T[i,j,k] = 1 if we're computing C[i,k], i.e., T is indexed by (i,j,k)
    # In the factorization: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = delta_{i,k}... 
    # Wait, no. The output should be C[i,k] = sum_j A[i,j] B[j,k]
    # So the tensor is T[i,j,k] = 1 for all i,j,k? No.
    
    # Let me re-read: "output factor follows the official trace convention W[k*n+i,r]"
    # So W is indexed by [k*n + i, r] = [k*3 + i, r]
    # The decomposition is: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = C[i,k] from A*B
    # where C[i,k] = sum_j A[i,j]*B[j,k]
    # So we need: sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = delta_{i,k} is WRONG
    # We need: sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1 for the coefficient of A[i,j]*B[j,k] in C[i,k]
    # The coefficient of A[i,j]*B[j,k] in C[i,k] is 1, and in C[i',k'] is 0 if (i',k') != (i,k)
    # So T[i,j,k] = 1 always? No, T is a 3-index tensor where T[i,j,k] represents the contribution
    # to C from A[i,j]*B[j,k]. The output index is (i,k), so:
    # sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] should equal 1 for all i,j,k
    # Because A[i,j]*B[j,k] contributes to C[i,k] with coefficient 1.
    
    # So we need: for all i,j,k in {0,1,2}: sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1
    
    # Let's verify with schoolbook: U[i,j,r]=1, V[j,k,r]=1, W[k,i,r]=1 for r=(i,j,k)
    # Then sum_r ... = 1 for the matching r, 0 otherwise. Sum = 1. Correct.
    
    # Now search for lower rank
    def verify(U, V, W, rank):
        """Check if decomposition is valid"""
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*3+j][r] * V[j*3+k][r] * W[k*3+i][r]
                    if abs(s - 1.0) > 1e-9:
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
                    if verify(U, V, W, rank):
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
