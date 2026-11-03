import time
import numpy as np
from fractions import Fraction
import random

DIMENSIONS = [4, 4, 7]
WALL_DEADLINE = 145.0
SAFETY_MARGIN = 5.0

def construct():
    start_time = time.time()
    n, m, p = 4, 4, 7
    
    # Schoolbook rank = n*m*p = 112
    rank_school = n * m * p
    best_result = None
    best_rank = rank_school
    
    def build_schoolbook():
        r = n * m * p
        U = [[0] * r for _ in range(n * m)]
        V = [[0] * r for _ in range(m * p)]
        W = [[0] * r for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j][col] = 1
                    V[j*p + k][col] = 1
                    W[k*n + i][col] = 1
                    col += 1
        return U, V, W
    
    # Start with schoolbook
    U_s, V_s, W_s = build_schoolbook()
    best_result = {"U": U_s, "V": V_s, "W": W_s}
    best_rank = rank_school
    
    def check_valid(U, V, W, tol=1e-9):
        """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
        r = len(U[0])
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(r):
                        s += U_arr[i*m + j, c] * V_arr[j*p + k, c] * W_arr[k*n + i, c]
                    if abs(s - 1.0) > tol:
                        return False
        return True

    def try_rank_reduction(target_rank, max_iter=20, num_restarts=5):
        """
        Attempt to find a decomposition of rank target_rank using ALS.
        Returns (U, V, W) if found, else None.
        """
        if target_rank < max(n*m, m*p, p*n):
            return None
            
        for restart in range(num_restarts):
            if time.time() > start_time + WALL_DEADLINE - SAFETY_MARGIN:
                return None
                
            # Initialize with random values
            U = np.random.randn(n * m, target_rank)
            V = np.random.randn(m * p, target_rank)
            W = np.random.randn(p * n, target_rank)
            
            # Scale to avoid overflow/underflow
            U /= np.linalg.norm(U)
            V /= np.linalg.norm(V)
            W /= np.linalg.norm(W)
            
            for it in range(max_iter):
                if time.time() > start_time + WALL_DEADLINE - SAFETY_MARGIN:
                    return None
                    
                # Update U
                for i in range(n):
                    for j in range(m):
                        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                        # For fixed i,j, we want to solve for U[i*m+j, :]
                        # Let b[k] = 1 (since T[i,j,k]=1 for all valid k? No, T[i,j,k] is the tensor entry)
                        # The tensor T[i,j,k] is 1 if (i,j,k) is a valid index? 
                        # Yes, the tensor for matrix multiplication has T[i,j,k]=1.
                        # So we are fitting T[i,j,k] = 1 for all i,j,k.
                        
                        # Wait, the tensor T[i,j,k] is 1 for all i in 0..n-1, j in 0..m-1, k in 0..p-1.
                        # So the target is an all-ones tensor.
                        
                        # We want sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all k.
                        
                        # Let A[k, r] = V[j*p+k, r] * W[k*n+i, r]
                        # Then A @ U[i*m+j, :] = ones(p)
                        
                        A = np.zeros((p, target_rank))
                        for k in range(p):
                            for r in range(target_rank):
                                A[k, r] = V[j * p + k, r] * W[k * n + i, r]
                        
                        b = np.ones(p)
                        
                        try:
                            # Use least squares
                            U[i * m + j, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                        except:
                            U[i * m + j, :] = 0
                        
                # Update V
                for j in range(m):
                    for k in range(p):
                        A = np.zeros((n, target_rank))
                        for i in range(n):
                            for r in range(target_rank):
                                A[i, r] = U[i * m + j, r] * W[k * n + i, r]
                        
                        b = np.ones(n)
                        
                        try:
                            V[j * p + k, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                        except:
                            V[j * p + k, :] = 0

                # Update W
                for k in range(p):
                    for i in range(n):
                        A = np.zeros((m, target_rank))
                        for j in range(m):
                            for r in range(target_rank):
                                A[j, r] = U[i * m + j, r] * V[j * p + k, r]
                        
                        b = np.ones(m)
                        
                        try:
                            W[k * n + i, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                        except:
                            W[k * n + i, :] = 0

            # Check if the solution is close to integers/half-integers
            # and if it is valid
            U_int = np.round(U * 2) / 2
            V_int = np.round(V * 2) / 2
            W_int = np.round(W * 2) / 2
            
            # Check validity of the rounded solution
            if check_valid(U_int, V_int, W_int, tol=1e-6):
                # Convert to lists of floats (or fractions if needed, but float is fine for return)
                return U_int.tolist(), V_int.tolist(), W_int.tolist()
            
            # Also check the non-rounded solution, just in case
            if check_valid(U, V, W, tol=1e-9):
                # If it's exact but not integer, we might need to rationalize
                # But the problem requires integer/half-integer entries.
                # So we must round.
                pass

        return None

    # Try to find a better decomposition
    # Start from rank 111 down to 50
    # We can skip some ranks to save time
    for target_rank in range(rank_school - 1, 49, -1):
        if time.time() > start_time + WALL_DEADLINE - SAFETY_MARGIN:
            break
            
        result = try_rank_reduction(target_rank, max_iter=10, num_restarts=2)
        if result is not None:
            U, V, W = result
            best_result = {"U": U, "V": V, "W": W}
            best_rank = target_rank
            # We found a better one, continue to see if we can go lower
            # But we should probably break if we find a very good one?
            # No, keep searching for the minimum.
            
    return best_result
