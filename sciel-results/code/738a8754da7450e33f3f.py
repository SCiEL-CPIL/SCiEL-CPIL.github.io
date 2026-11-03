"""Rewritten program: rank-minimal exact tensor decomposition for 3x3 * 3x4 matrix multiplication."""
DIMENSIONS = [3, 3, 4]

import time
import numpy as np
from fractions import Fraction

# We will search for a low-rank decomposition.
# The multiplication tensor T_{i,j,k} = delta(i,j,k) in appropriate indices.
# Specifically, for A (n x m) and B (m x p), C = A B (n x p).
# C_{i,k} = sum_j A_{i,j} B_{j,k}.
# Tensor T_{i,j,k} where i in [0,n), j in [0,m), k in [0,p).
# Decomposition: T_{i,j,k} = sum_c U_{i*m+j, c} V_{j*p+k, c} W_{k*n+i, c}
# Wait, the problem says:
# U shape (n*m, rank), V shape (m*p, rank), W shape (p*n, rank)
# U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
# Product sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should equal delta(i,j,k) ?
# No, matrix multiplication is C_{i,k} = sum_j A_{i,j} B_{j,k}.
# This is a bilinear map (A, B) -> C.
# The tensor of the bilinear map is T_{(i,j), (j',k), (i',k')} ?
# Let's re-read carefully: "exactly decompose the matrix multiplication tensor".
# Standard definition: T_{i,j,k} = 1 if it contributes to C_{i,k} from A_{i,j} B_{j,k}.
# Actually, the bilinear form is <A, B> = sum_{i,j,k} T_{i,j,k} A_{i,j} B_{j,k} ... no.
# The output C_{i,k} = sum_j A_{i,j} B_{j,k}.
# So the tensor has indices (i, j, k) and value 1.
# We want to write T_{i,j,k} = sum_c U_{i*m+j, c} V_{j*p+k, c} W_{k*n+i, c} ??
# This looks like a 3-way tensor decomposition where the modes are:
# Mode 1: index (i, j) -> size n*m = 9
# Mode 2: index (j, k) -> size m*p = 12
# Mode 3: index (k, i) -> size p*n = 12
# Let's check the product: sum_c U_{(i,j), c} V_{(j,k), c} W_{(k,i), c}
# This sum must equal 1 if the indices match the multiplication structure?
# Wait, the standard multiplication tensor T_{i,j,k} has value 1 for all i,j,k? No.
# A_{i,j} B_{j,k} contributes to C_{i,k}.
# The bilinear map is (A, B) -> C.
# This is often represented as a tensor T_{(i,j), (j,k), (i,k)} ?
# Let's look at the indices in the prompt:
# U[i*m+j, c]  -> index (i, j)
# V[j*p+k, c]  -> index (j, k)
# W[k*n+i, c]  -> index (k, i)
# The sum is over c.
# If we set U_{(i,j), c} V_{(j,k), c} W_{(k,i), c}, what does this represent?
# It represents a scalar for each triple (i, j, k).
# For matrix multiplication, we have C_{i,k} = sum_j A_{i,j} B_{j,k}.
# This can be seen as a sum of rank-1 tensors.
# A rank-1 term in the decomposition of the multiplication algorithm is:
# A_{i,j} * B_{j,k} * (contribution to C_{i,k}).
# But the indices in W are (k, i).
# So W_{(k,i), c} corresponds to the output C_{i,k}.
# The condition for a valid decomposition is:
# sum_c U_{(i,j), c} V_{(j,k), c} W_{(k,i), c} = 1  for all i, j, k?
# No. The tensor T_{i,j,k} for matrix multiplication is defined such that
# C_{i,k} = sum_{j} T_{i,j,k} A_{i,j} B_{j,k}.
# Here T_{i,j,k} = 1 if the indices align?
# Actually, the standard "multiplication tensor" is T_{(i,j), (j',k'), (i',k')} where
# T = 1 if i=i', j=j', j'=k', k=k'? No.
# Let's look at the specific indices given:
# U index: i*m+j
# V index: j*p+k
# W index: k*n+i
# Note that the 'j' in U and V is the same. The 'k' in V and W is the same. The 'i' in W and U is the same.
# This forms a cycle: (i,j) -> (j,k) -> (k,i).
# This is exactly the structure of the matrix multiplication tensor T_{i,j,k} = 1.
# The decomposition requires:
# sum_c U_{i,m+j,c} V_{j,p+k,c} W_{k,n+i,c} = 1  for all i, j, k.
# This is a system of equations.
# Number of equations: n * m * p = 3 * 3 * 4 = 36.
# We want to find the minimum rank r such that there exist U, V, W satisfying this.

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    n, m, p = DIMENSIONS
    # Indices:
    # U: (n*m) x r
    # V: (m*p) x r
    # W: (p*n) x r
    
    # We can try to find a solution using randomized search or algebraic methods.
    # Since the dimensions are small, we can try to solve for r = 1, 2, ... up to n*m*p.
    # The schoolbook rank is n*m*p = 36.
    # We know that for 3x3 * 3x4, the rank is likely less than 36.
    # Let's try to find a decomposition of rank r.
    
    # Strategy:
    # For a fixed r, we have U (9x r), V (12x r), W (12x r).
    # The equation is: sum_c U_{i*m+j, c} V_{j*p+k, c} W_{k*n+i, c} = 1.
    # This is a system of 36 polynomial equations in 9r + 12r + 12r = 33r variables.
    # We can use numerical optimization (e.g., gradient descent) or algebraic solving.
    # Given the time limit, a randomized approach with local optimization might work.
    
    # Let's try ranks from 1 to 36.
    # We will use a simple randomized search with refinement.
    
    import random
    import math
    
    def check_solution(U, V, W, r):
        # U: 9 x r, V: 12 x r, W: 12 x r
        # Check if sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] == 1 for all i,j,k
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(r):
                        val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    def try_rank(r, max_iters=10000):
        # Random initialization
        U = [[random.uniform(-1, 1) for _ in range(r)] for _ in range(n*m)]
        V = [[random.uniform(-1, 1) for _ in range(r)] for _ in range(m*p)]
        W = [[random.uniform(-1, 1) for _ in range(r)] for _ in range(p*n)]
        
        # Gradient descent on the loss: sum_{i,j,k} (sum_c ... - 1)^2
        # This is complex to compute gradients manually.
        # Alternative: use scipy.optimize if available?
        # Let's stick to a simpler approach: try to solve linearly if r is large enough?
        # Actually, for small r, we can try to find a solution by fixing two matrices and solving for the third?
        # No, it's nonlinear.
        
        # Let's try a different approach: 
        # The tensor T_{i,j,k} = 1.
        # We can try to find a decomposition of the form:
        # U_{i,j,c} = a_{i,c} b_{j,c}
        # V_{j,k,c} = b_{j,c} d_{k,c}
        # W_{k,i,c} = d_{k,c} a_{i,c}
        # Then sum_c a_{i,c} b_{j,c} b_{j,c} d_{k,c} d_{k,c} a_{i,c} = sum_c a_{i,c}^2 b_{j,c}^2 d_{k,c}^2 = 1.
        # This is a special case.
        
        # General case:
        # Let's use a randomized search with local improvement.
        # We'll define a loss function and do coordinate descent or random restarts.
        
        def loss(U, V, W):
            total = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for c in range(r):
                            s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                        total += (s - 1.0) ** 2
            return total
        
        best_loss = float('inf')
        best_U, best_V, best_W = None, None, None
        
        for restart in range(50):
            if time.time() > deadline - 5:
                break
            U = [[random.uniform(-1, 1) for _ in range(r)] for _ in range(n*m)]
            V = [[random.uniform(-1, 1) for _ in range(r)] for _ in range(m*p)]
            W = [[random.uniform(-1, 1) for _ in range(r)] for _ in range(p*n)]
            
            current_loss = loss(U, V, W)
            if current_loss < best_loss:
                best_loss = current_loss
                best_U, best_V, best_W = U, V, W
            
            # Simple local search: perturb one element at a time
            for _ in range(200):
                if time.time() > deadline - 5:
                    break
                # Pick random matrix, row, col
                mat_idx = random.randint(0, 2)
                if mat_idx == 0:
                    rows = n*m
                elif mat_idx == 1:
                    rows = m*p
                else:
                    rows = p*n
                
                row = random.randint(0, rows-1)
                col = random.randint(0, r-1)
                
                # Save old value
                if mat_idx == 0:
                    old_val = U[row][col]
                elif mat_idx == 1:
                    old_val = V[row][col]
                else:
                    old_val = W[row][col]
                
                # Perturb
                new_val = old_val + random.uniform(-0.1, 0.1)
                
                if mat_idx == 0:
                    U[row][col] = new_val
                elif mat_idx == 1:
                    V[row][col] = new_val
                else:
                    W[row][col] = new_val
                
                new_loss = loss(U, V, W)
                if new_loss < current_loss:
                    current_loss = new_loss
                    if current_loss < best_loss:
                        best_loss = current_loss
                        best_U, best_V, best_W = U, V, W
                else:
                    # Revert
                    if mat_idx == 0:
                        U[row][col] = old_val
                    elif mat_idx == 1:
                        V[row][col] = old_val
                    else:
                        W[row][col] = old_val
            
            if best_loss < 1e-12:
                break
        
        if best_loss < 1e-9:
            # Convert to integers if possible? The problem allows integer/half-integer.
            # We can try to round to nearest 0.5 and check.
            U_int = [[round(U[i][j] * 2) / 2 for j in range(r)] for i in range(n*m)]
            V_int = [[round(V[i][j] * 2) / 2 for j in range(r)] for i in range(m*p)]
            W_int = [[round(W[i][j] * 2) / 2 for j in range(r)] for i in range(p*n)]
            if check_solution(U_int, V_int, W_int, r):
                return U_int, V_int, W_int
            # If rounding fails, return the float version? The problem says "integer/half-integer entries".
            # So we MUST return integers or half-integers.
            # If we can't find a half-integer solution, this rank might not be achievable with half-integers.
            # But the schoolbook solution uses integers.
            # Let's try to find an integer solution if r is large.
            # For now, if we found a float solution, we might need to refine it to half-integers.
            # This is hard. Let's assume we can find a solution that is close to half-integers.
        
        return None
    
    # Try ranks from 1 to 36
    for r in range(1, 37):
        if time.time() > deadline - 10:
            break
        res = try_rank(r)
        if res is not None:
            U, V, W = res
            return {"U": U, "V": V, "W": W}
    
    # Fallback: Schoolbook decomposition
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
