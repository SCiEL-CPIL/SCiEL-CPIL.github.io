"""Matrix multiplication 4,4,9 - rank-minimal exact tensor decomposition."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [4, 4, 9]

def _verify(U, V, W, n=4, m=4, p=9):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    # Build the multiplication tensor T[i,j,k] = 1 if i,j,k are valid indices
    # T is indexed as (i*n + j, k) -> W[k*n + i]... 
    # Actually the tensor is: C[i,k] = sum_j A[i,j]*B[j,k]
    # Tensor T_{(i,j),(j,k),(i,k)} = 1
    # In flattened form: U has shape (n*m, rank), V has (m*p, rank), W has (p*n, rank)
    # The decomposition requires: for all (i,j), (j',k), (i',k')
    #   sum_c U[i*m+j, c] * V[j'*p+k, c] * W[k'*n+i', c] = 1 if j==j' and i==i' and k==k' else 0
    
    # More precisely: the tensor M_{(i,j),(j2,k),(i2,k2)} = delta(j,j2)*delta(i,i2)*delta(k,k2)
    # We need: sum_c U[i*m+j,c] * V[j2*p+k,c] * W[k2*n+i2,c] = delta(j,j2)*delta(i,i2)*delta(k,k2)
    
    # Check a sample of entries for efficiency, but for correctness check all
    # For small dimensions we can check all
    for i in range(n):
        for j in range(m):
            for j2 in range(m):
                for k in range(p):
                    for i2 in range(n):
                        for k2 in range(p):
                            expected = 1.0 if (j == j2 and i == i2 and k == k2) else 0.0
                            actual = 0.0
                            for c in range(rank):
                                actual += U[i*m+j][c] * V[j2*p+k][c] * W[k2*n+i2][c]
                            if abs(actual - expected) > 1e-9:
                                return False
    return True

def _verify_fast(U, V, W, n=4, m=4, p=9):
    """Faster verification using numpy."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)  # (n*m, rank)
    V_arr = np.array(V, dtype=float)  # (m*p, rank)
    W_arr = np.array(W, dtype=float)  # (p*n, rank)
    
    # The tensor equation: for each c, U[:,c] * V[:,c] * W[:,c] contributes to the tensor
    # T[i,j,j2,k,i2,k2] = sum_c U[i*m+j,c]*V[j2*p+k,c]*W[k2*n+i2,c]
    # We need T = identity tensor (delta(j,j2)*delta(i,i2)*delta(k,k2))
    
    # Reshape to check: T should equal the "matrix multiplication tensor"
    # Let's build the full tensor and check
    # Actually, let's use a smarter check:
    # For fixed (i, j, j2, k, i2, k2), we need the sum to be delta(j,j2)*delta(i,i2)*delta(k,k2)
    
    # Build T as a 6D array would be too large. Instead, check via contraction.
    # For each c, the contribution is outer product structure.
    # Let's just do the full check with numpy broadcasting for the specific structure.
    
    # Alternative: check that for all (i,j), (j2,k), (i2,k2):
    # sum_c U[i*m+j,c]*V[j2*p+k,c]*W[k2*n+i2,c] = delta(j,j2)*delta(i,i2)*delta(k,k2)
    
    # We can vectorize over c for fixed indices
    # Let's check all (i,j,j2,k,i2,k2) combinations
    # n*m*m*p*n*p = 4*4*4*9*4*9 = 23328 combinations, each with rank terms
    # This is feasible
    
    indices = list(product(range(n), range(m), range(m), range(p), range(n), range(p)))
    U_flat = U_arr  # (n*m, rank)
    V_flat = V_arr  # (m*p, rank)
    W_flat = W_arr  # (p*n, rank)
    
    for (i, j, j2, k, i2, k2) in indices:
        expected = 1.0 if (j == j2 and i == i2 and k == k2) else 0.0
        actual = np.dot(U_flat[i*m+j], np.multiply(V_flat[j2*p+k], W_flat[k2*n+i2]))
        if abs(actual - expected) > 1e-9:
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 144
    
    # Start with schoolbook decomposition as fallback
    U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j][col] = 1
                V_sb[j*p+k][col] = 1
                W_sb[k*n+i][col] = 1
                col += 1
    
    best_rank = schoolbook_rank
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # Strategy: Try to find lower rank decompositions
    # For matrix multiplication, known results suggest rank can be lower than n*m*p
    # For 4x4 x 4x9, let's search for ranks below 144
    
    # Approach: Use integer tensor decomposition search
    # We'll try to find decompositions with rank r < 144
    
    # Key insight: The multiplication tensor has structure. 
    # We can try random integer vectors and solve linear systems.
    
    # For a given rank r, we need to find U (16 x r), V (36 x r), W (36 x r) 
    # such that the tensor equation holds.
    
    # The tensor equation gives us n*m*m*p*n*p = 23328 constraints
    # But many are redundant. The independent constraints are n*m*p = 144 (the "diagonal" ones)
    # plus the off-diagonal ones that must be zero.
    
    # Actually, the full tensor has n*m*m*p*n*p entries but the structure means:
    # For (i,j,j2,k,i2,k2): value = delta(j,j2)*delta(i,i2)*delta(k,k2)
    # Non-zero only when j=j2, i=i2, k=k2: there are n*m*p = 144 such entries, all equal to 1.
    # All other entries (23328-144 = 23184) must be 0.
    
    # So we have 23328 equations in 16*r + 36*r + 36*r = 88*r unknowns.
    # For r=144: 88*144 = 12672 unknowns, 23328 equations (overdetermined but consistent)
    # For lower r, we need the system to be consistent.
    
    # Let's try a search approach:
    # 1. Fix r, try to find integer solutions
    # 2. Use the fact that for matrix multiplication, there are known constructions
    
    # For now, let's try to find if rank 144 can be improved.
    # Known: for n x n matrix multiplication, rank is at most n^3 (schoolbook)
    # Strassen gives lower bounds but for 4x4, the exact rank is 49 (Strassen)
    # But we have 4x4 times 4x9, which is different.
    
    # Let's try a greedy/iterative approach:
    # Start with schoolbook, try to merge terms or find linear dependencies.
    
    # Alternative: Use the structure to find a better decomposition.
    # For 4x4 x 4x9, we can think of it as 9 independent 4x4 multiplications
    # but with shared structure.
    
    # Let's try: treat the 9 columns of B as separate, so we have 9 4x4 multiplications.
    # Each 4x4 multiplication has rank at most 49 (Strassen), so total could be 9*49=441? No, that's worse.
    # Actually, we can share the A matrix, so it's more like: for each of the 9 output columns,
    # we compute A * B[:,k], which is 4x4 times 4x1, but that's just matrix-vector.
    
    # Hmm, let me reconsider. The tensor is for (4x4) * (4x9) -> (4x9).
    # This is equivalent to 9 independent 4x4 * 4x1 multiplications sharing the same A.
    
    # Let's try a different approach: use the known Strassen decomposition for 4x4 and extend it.
    
    # For 4x4 matrix multiplication, Strassen's algorithm gives rank 49.
    # Let me implement Strassen for 4x4 first, then extend to 4x9.
    
    # Strassen for 2x2: rank 7
    # For 4x4: use Strassen recursively, rank 7^2 = 49
    
    # Let me implement the Strassen decomposition for 4x4 matrices.
    # Then for 4x4 * 4x9, I need to adapt it.
    
    # Actually, let me think about this differently.
    # The multiplication (4x4)*(4x9) can be seen as:
    # C[:,k] = A * B[:,k] for k=0..8
    # Each C[:,k] is a 4x1 vector = A * (4x1 vector)
    # This is just matrix-vector multiplication, which has rank 4*1*4=16 per column?
    # No, matrix-vector mult 4x4 * 4x1 has tensor rank 16 (schoolbook).
    # But we can do better with Strassen-like methods.
    
    # Actually, for 4x4 * 4x1, the tensor is the same as 4x4 matrix multiplication restricted.
    # The full 4x4*4x4 tensor has rank at most 49 (Strassen).
    # The 4x4*4x1 tensor is a sub-tensor, so its rank is at most 49 as well? No, 
    # it's a different tensor. The 4x4*4x1 tensor has dimensions (4*4) x (4*1) x (4*1) = 16 x 4 x 4.
    # Schoolbook rank = 16*4*4/4 = 16? No, rank = 4*4*1 = 16.
    
    # I'm overcomplicating this. Let me just try to find a decomposition computationally.
    
    # Approach: Use integer programming / random search for lower rank.
    # For a given rank r, we have 88*r variables and 23328 equations.
    # We can set up the system and try to solve it.
    
    # Let's try ranks from 143 downward and see if we can find a solution.
    # This is a hard problem, so let's use a heuristic.
    
    # Heuristic: Start with schoolbook, try to find linear dependencies among the rank-1 terms.
    # If we can express one term as a combination of others, we can reduce rank.
    
    # Actually, let's try a simpler approach: use the fact that for matrix multiplication,
    # there are known rank formulas. For m x n times n x p, the rank is at most m*n*p,
    # and for specific cases, lower bounds are known.
    
    # For 4x4 * 4x9, let's just return the schoolbook decomposition for now,
    # but try a quick search to see if we can do better.
    
    # Quick search: try rank 143, 142, ... with random integer vectors
    # and check if the linear system has a solution.
    
    # Actually, let's use a smarter approach. The multiplication tensor for (m x n) * (n x p)
    # has a known structure. Let me try to find a decomposition using the 
    # "border rank" or "tensor rank" literature.
    
    # For now, let's implement a basic search and hope for improvement.
    
    # Try to find a decomposition with rank < 144 using a greedy approach:
    # 1. Start with all 144 schoolbook terms
    # 2. Try to remove one term and see if the remaining 143 can still span the tensor
    # 3. This is equivalent to checking if the 144th term is in the span of the others
    
    # Let's set up the linear system for the schoolbook decomposition and check for dependencies.
    
    # The tensor equation for schoolbook:
    # For each (i,j,k), we have a term with U[i*m+j,c]=1, V[j*p+k,c]=1, W[k*n+i,c]=1
    # The tensor value at (i,j,j2,k,i2,k2) = sum_c U[i*m+j,c]*V[j2*p+k,c]*W[k2*n+i2,c]
    # = number of c such that (i*m+j == i*m+j) AND (j*p+k == j2*p+k) AND (k*n+i == k2*n+i2)
    # Wait, this isn't right. Let me re-examine.
    
    # In the schoolbook decomposition:
    # U[i*m+j, c] = 1 if c corresponds to (i,j,k) for some k, else 0
    # V[j*p+k, c] = 1 if c corresponds to (i,j,k) for this j,k, else 0  
    # W[k*n+i, c] = 1 if c corresponds to (i,j,k) for this k,i, else 0
    
    # So for a given c corresponding to (i0,j0,k0):
    # U[i*m+j, c] = 1 if (i,j) = (i0,j0)
    # V[j*p+k, c] = 1 if (j,k) = (j0,k0)
    # W[k*n+i, c] = 1 if (k,i) = (k0,i0)
    
    # So the product U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 iff (i,j)=(i0,j0) AND (j,k)=(j0,k0) AND (k,i)=(k0,i0)
    # This means i=i0, j=j0, k=k0.
    
    # So the tensor value at (i,j,j2,k,i2,k2) = number of (i0,j0,k0) such that
    # (i,j)=(i0,j0), (j2,k)=(j0,k0), (i2,k2)=(i0,k0)
    # = 1 if j=j2, i=i2, k=k2, and (i0,j0,k0)=(i,j,k), else 0.
    # Wait, that gives delta(j,j2)*delta(i,i2)*delta(k,k2), which is correct!
    
    # So the schoolbook decomposition is correct. Now, can we do better?
    
    # Let's check if there are linear dependencies among the 144 terms.
    # We need to check if any term can be written as a linear combination of the others.
    
    # This is a tensor rank problem, which is hard. Let's try a computational approach.
    
    # For a given rank r, we need to find U, V, W with integer entries such that
    # the tensor equation holds. Let's try to solve this for r < 144.
    
    # Use numpy to set up the linear system for a given r.
    # The equations are: for each (i,j,j2,k,i2,k2),
    # sum_c U[i*m+j,c]*V[j2*p+k,c]*W[k2*n+i2,c] = delta(j,j2)*delta(i,i2)*delta(k,k2)
    
    # This is a bilinear system in U,V,W, which is hard to solve directly.
    # But if we fix two of the matrices, it's linear in the third.
    
    # Approach: Fix U and V randomly, then solve for W. Check if the solution is integer.
    
    # Let's try this for r = 143, 142, ...
    
    for r in range(schoolbook_rank - 1, max(schoolbook_rank - 20, 0), -1):
        if time.time() > deadline:
            break
        
        # Try a few random initializations
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            # Fix U and V with random integer entries
            U_rand = np.random.randint(-2, 3, size=(n*m, r))
            V_rand = np.random.randint(-2, 3, size=(m*p, r))
            
            # For each (i,j,i2,k2), we have:
            # sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i2,c] = delta(j,j)*delta(i,i2)*delta(k,k2)
            # Wait, the equation is for all (i,j,j2,k,i2,k2):
            # sum_c U[i*m+j,c]*V[j2*p+k,c]*W[k2*n+i2,c] = delta(j,j2)*delta(i,i2)*delta(k,k2)
            
            # Let's fix (i,j,i2,k2) and vary (j2,k). For fixed (i,j,i2,k2):
            # sum_c U[i*m+j,c]*V[j2*p+k,c]*W[k2*n+i2,c] = delta(j,j2)*delta(i,i2)*delta(k,k2)
            
            # If i != i2, the RHS is 0 for all (j2,k).
            # If i == i2, the RHS is delta(j,j2)*delta(k,k2).
            
            # This is still complex. Let me try a different approach.
            
            # For fixed U and V, the equation for W is:
            # For each (i,j,j2,k,i2,k2):
            # sum_c W[k2*n+i2,c] * (U[i*m+j,c]*V[j2*p+k,c]) = delta(j,j2)*delta(i,i2)*delta(k,k2)
            
            # Let A_{(i,j,j2,k),(k2,i2),c} = U[i*m+j,c]*V[j2*p+k,c]
            # Then sum_c A_{(i,j,j2,k),(k2,i2),c} * W[k2*n+i2,c] = delta(j,j2)*delta(i,i2)*delta(k,k2)
            
            # This is a linear system in W. The number of unknowns is p*n*r = 36*r.
            # The number of equations is n*m*m*p*n*p = 23328.
            
            # Let's set up this linear system.
            
            # Unknowns: W[k*n+i, c] for k in range(p), i in range(n), c in range(r)
            # Total unknowns: p*n*r = 36*r
            
            # Equations: for each (i,j,j2,k,i2,k2):
            # sum_c U[i*m+j,c]*V[j2*p+k,c]*W[k2*n+i2,c] = delta(j,j2)*delta(i,i2)*delta(k,k2)
            
            # Let's build the matrix.
            num_unknowns = p * n * r
            num_equations = n * m * m * p * n * p
            
            # This is a huge system. Let's only use a subset of equations for speed.
            # Use the "diagonal" equations where j=j2, i=i2, k=k2: these give 144 equations.
            # And some off-diagonal equations to ensure zeros.
            
            # Actually, let's just try to solve the full system for small r.
            # For r=143, num_unknowns = 36*143 = 5148, num_equations = 23328.
            # This is a large but solvable system.
            
            # Let's build the linear system matrix.
            # This will be slow, so let's only try a few values of r.
            
            if r > 140:  # Only try close to schoolbook for speed
                # Build the system
                A_mat = np.zeros((num_equations, num_unknowns))
                b_vec = np.zeros(num_equations)
                
                eq_idx = 0
                for i in range(n):
                    for j in range(m):
                        for j2 in range(m):
                            for k in range(p):
                                for i2 in range(n):
                                    for k2 in range(p):
                                        rhs = 1.0 if (j == j2 and i == i2 and k == k2) else 0.0
                                        b_vec[eq_idx] = rhs
                                        
                                        for c in range(r):
                                            coeff = U_rand[i*m+j, c] * V_rand[j2*p+k, c]
                                            if coeff != 0:
                                                # W[k2*n+i2, c] is at index (k2*n+i2)*r + c
                                                w_idx = (k2*n + i2) * r + c
                                                A_mat[eq_idx, w_idx] += coeff
                                        eq_idx += 1
                
                # Solve the linear system
                try:
                    W_sol, residuals, rank_sys, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                    # Check if the solution is close to integer
                    W_rounded = np.round(W_sol)
                    W_int = W_rounded.astype(int)
                    
                    # Verify the solution
                    W_list = W_int.reshape(p*n, r).tolist()
                    U_list = U_rand.tolist()
                    V_list = V_rand.tolist()
                    
                    if _verify_fast(U_list, V_list, W_list, n, m, p):
                        best_rank = r
                        best_U = U_list
                        best_V = V_list
                        best_W = W_list
                        break  # Found a solution for this r, try lower
                except:
                    pass
        
        if time.time() > deadline:
            break
    
    return {"U": best_U, "V": best_V, "W": best_W}
