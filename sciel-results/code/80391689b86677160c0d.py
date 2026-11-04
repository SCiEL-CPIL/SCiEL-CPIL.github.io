"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START

def construct():
    n, m, p = DIMENSIONS  # 2, 3, 4
    dim_U = n * m  # 6
    dim_V = m * p  # 12
    dim_W = p * n  # 8
    
    # The strategy is to exploit the structure of the 2x3x4 tensor.
    # We can view the multiplication C = A B where A is 2x3 and B is 3x4.
    # We can partition the middle dimension (size 3) or use a block structure.
    # However, a more direct approach is to use a known rank-15 or rank-14 construction if possible.
    # But since I don't have a verified rank-14/15 construction hardcoded, I will implement a robust search.
    # 
    # Wait, the prompt asks to "combine two approaches".
    # Approach 1: The current program (schoolbook rank 24).
    # Approach 2: Inspiration 3 suggests a rank-15 attempt.
    # 
    # Let's try to construct a rank-15 decomposition.
    # The tensor T_{i,j,k} for 2x3x4 has 2*3*4 = 24 entries.
    # We want to represent this as a sum of rank-1 tensors.
    #
    # Actually, there is a known result that R(2,3,4) <= 15.
    # Let's try to find it or generate it.
    #
    # A common technique is to use the "border rank" or specific algebraic identities.
    # However, generating a valid exact decomposition from scratch in Python within 120s is risky.
    #
    # Let's look at the "Inspiration 3" code snippet which was cut off. It seems to have started a rank-15 construction.
    # Since I don't have the full code, I will implement a generic solver that tries to find a low-rank decomposition.
    # But solving a system of polynomial equations is hard.
    #
    # Alternative: Use the known rank-7 Strassen for 2x2x2.
    # Can we embed 2x3x4 into a larger structure?
    # Not directly helpful for reducing rank below 24 without specific identities.
    #
    # Let's try a random search for rank 15, 14, etc. with small integer coefficients.
    # The space is large, but for small ranks, it might be feasible if we use a good initialization.
    #
    # Actually, let's look at the structure again.
    # C_{ik} = sum_{j=0}^{2} A_{ij} B_{jk}
    #
    # If we assume a decomposition of rank R, we have:
    # C_{ik} = sum_{r=0}^{R-1} u_{i,j,r} v_{j,k,r} w_{k,i,r}  -- No, the indices are fixed for U and V columns.
    #
    # The standard form is:
    # C_{ik} = sum_{r} (sum_{j} U_{i,j,r} A_{ij}) (sum_{l} V_{l,k,r} B_{lk}) W_{k,i,r} ? No.
    #
    # The problem statement defines:
    # U: (n*m) x rank. U[i*m+j, r]
    # V: (m*p) x rank. V[j*p+k, r]
    # W: (p*n) x rank. W[k*n+i, r]
    #
    # The reconstruction is:
    # C_{ik} = sum_{r} W[k*n+i, r] * (sum_{j} U[i*m+j, r] A_{ij}) * (sum_{l} V[l*p+k, r] B_{lk})?
    # No, that would be a trilinear form. The matrix multiplication is bilinear.
    #
    # Let's re-read the tensor definition carefully.
    # "exactly decompose the matrix multiplication tensor"
    # The tensor T is in (V_A^*) x (V_B^*) x V_C.
    # T = sum_{i,j,k} A^*_{ij} \otimes B^*_{jk} \otimes C_{ik}
    #
    # A rank-1 decomposition is a term: u \otimes v \otimes w
    # where u is a linear form on A, v is a linear form on B, and w is a vector in C.
    #
    # If u = A_{i0, j0} (basis covector), v = B_{j0, k0} (basis covector), and w = E_{i0, k0} (basis vector),
    # then the term is A_{i0, j0} B_{j0, k0} E_{i0, k0}.
    # This contributes A_{i0, j0} B_{j0, k0} to C_{i0, k0}.
    #
    # In the matrix representation provided by the interface:
    # U is (n*m) x R. Column r is the vector u_r.
    # V is (m*p) x R. Column r is the vector v_r.
    # W is (p*n) x R. Column r is the vector w_r.
    #
    # The reconstruction of C is:
    # C = sum_{r} w_r \otimes (u_r \cdot A) \otimes (v_r \cdot B) ? No.
    #
    # Let's check the schoolbook example.
    # For a specific (i, j, k):
    # U[i*m+j, c] = 1. So u_c extracts A_{ij}.
    # V[j*p+k, c] = 1. So v_c extracts B_{jk}.
    # W[k*n+i, c] = 1. So w_c has a 1 at index k*n+i.
    #
    # The contribution to C is:
    # For each r, we compute scalar_a = sum_{i',j'} U[i'*m+j', r] A_{i'j'}
    # scalar_b = sum_{j',k'} V[j'*p+k', r] B_{j'k'}
    # Then the contribution to C is scalar_a * scalar_b * w_r.
    #
    # If u_c is the basis vector for (i0, j0), scalar_a = A_{i0, j0}.
    # If v_c is the basis vector for (j0, k0), scalar_b = B_{j0, k0}.
    # If w_c is the basis vector for (k0, i0) in the W index space (which maps to C_{i0, k0}),
    # then the term adds A_{i0, j0} B_{j0, k0} to C_{i0, k0}.
    #
    # This matches the schoolbook decomposition.
    #
    # So the condition for a valid decomposition of rank R is:
    # For all i, k:
    # sum_{r=0}^{R-1} W[k*n+i, r] * (sum_{j} U[i*m+j, r] A_{ij}) * (sum_{l} V[l*p+k, r] B_{lk}) = sum_{j} A_{ij} B_{jk}
    #
    # This must hold for ALL A, B.
    # This is a system of polynomial equations in the entries of U, V, W.
    #
    # We can solve this by choosing random A, B and checking if the equation holds,
    # or by equating coefficients.
    #
    # Equating coefficients:
    # The LHS is a bilinear form in A and B.
    # The coefficient of A_{i1, j1} B_{j2, k2} in the LHS is:
    # sum_{r} W[k*n+i, r] U[i1*m+j1, r] V[j2*p+k2, r]
    #
    # The RHS is sum_{j} A_{ij} B_{jk}.
    # The coefficient of A_{i1, j1} B_{j2, k2} in the RHS is:
    # 1 if i1=i, k2=k, and j1=j2.
    # 0 otherwise.
    #
    # So we need:
    # For all i, k, i1, j1, j2, k2:
    # sum_{r} W[k*n+i, r] U[i1*m+j1, r] V[j2*p+k2, r] = delta_{i,i1} delta_{k,k2} delta_{j1,j2}
    #
    # This is a set of (p*n) * (n*m) * (m*p) = 8 * 6 * 12 = 576 equations.
    # The variables are the entries of U, V, W.
    # Total variables: (n*m + m*p + p*n) * R = (6 + 12 + 8) * R = 26 * R.
    #
    # We want to find R such that this system has a solution with integer/half-integer entries.
    #
    # For R=24, we have 24*26 = 624 variables and 576 equations.
    # The schoolbook solution is a specific sparse solution.
    #
    # For R < 24, we need to find a solution.
    # This is a hard problem.
    #
    # However, we can use a heuristic search.
    # 1. Initialize U, V, W with random small integers.
    # 2. Check the error (sum of squared differences of coefficients).
    # 3. Use gradient descent or random perturbations to reduce the error.
    #
    # But the equations are cubic in the variables (product of U, V, W).
    # Gradient descent might be slow.
    #
    # Alternative: Use the fact that the tensor is small.
    # We can try to solve for R=15, 14, etc. using a SAT solver or SMT solver?
    # No, we only have standard library, NumPy, SciPy.
    #
    # Let's try a random search with a clever initialization.
    # We can start with the schoolbook solution (R=24) and try to merge terms.
    #
    # Merging terms:
    # If we have two terms r1, r2 such that their contributions can be combined,
    # we might reduce the rank.
    #
    # Actually, there is a known rank-15 decomposition for 2x3x4.
    # Let's try to find it online or derive it.
    #
    # Since I cannot browse, I will implement a random search that is more likely to find a solution
    # by using a structured initialization.
    #
    # Strategy:
    # 1. Try R=15.
    # 2. Initialize U, V, W with random values from {-1, 0, 1}.
    # 3. Compute the error tensor E_{i,k,i1,j1,j2,k2} = LHS - RHS.
    # 4. If error is 0, return the solution.
    # 5. Use a local search: pick a random entry, perturb it, check if error decreases.
    #
    # This is very slow.
    #
    # Better Strategy:
    # Use the known rank-7 Strassen for 2x2x2.
    # Can we embed 2x3x4 into 2x2x2? No.
    #
    # Let's look at the dimensions again.
    # 2x3x4.
    #
    # Maybe we can use a block decomposition.
    # A = [A1, A2] where A1 is 2x1, A2 is 2x2.
    # B = [B1; B2] where B1 is 1x4, B2 is 2x4.
    # C = A1 B1 + A2 B2.
    #
    # A1 B1 is 2x1 * 1x4 = 2x4. Rank 2 (schoolbook).
    # A2 B2 is 2x2 * 2x4 = 2x4. Rank 8 (schoolbook) or 7 (Strassen).
    # Total rank: 2 + 7 = 9?
    # Wait, the sum of two bilinear maps has rank at most the sum of the ranks.
    # But we need to compute C = A1 B1 + A2 B2.
    #
    # Let's verify this.
    # C_{ik} = A_{i0} B_{0k} + sum_{j=1}^{2} A_{ij} B_{jk}.
    #
    # Term 1: A_{i0} B_{0k}.
    # This is a rank-2 tensor?
    # It depends on i and k.
    # For i=0,1 and k=0,1,2,3.
    # It is the outer product of vector A_col0 and vector B_row0.
    # Rank is 2? No, the rank of the bilinear map (A, B) -> A_{i0} B_{0k} is 1?
    # No, it's a single term in the sum.
    # In the tensor decomposition, it corresponds to 2 terms?
    # No, A_{i0} B_{0k} is one scalar multiplication A_{i0} * B_{0k} for each i?
    # No, A_{i0} is a vector of 2 elements. B_{0k} is a vector of 4 elements.
    # The product is a 2x4 matrix.
    # This requires 2*4 = 8 scalar multiplications in schoolbook?
    # No, A_{i0} B_{0k} is just one entry of A times one entry of B.
    # Wait, A_{i0} is a specific entry.
    # So for each i, k, we have A_{i0} B_{0k}.
    # This is 2 * 4 = 8 terms.
    #
    # Term 2: sum_{j=1}^{2} A_{ij} B_{jk}.
    # This is 2x2 * 2x4.
    # Schoolbook rank: 2 * 2 * 4 = 16.
    # Strassen rank for 2x2x2 is 7.
    # But here we have 2x2x4.
    # We can split B into two 2x2 blocks.
    # B2 = [B21, B22].
    # A2 B2 = A2 B21 + A2 B22.
    # Each is 2x2 * 2x2. Rank 7 each.
    # Total rank for Term 2: 7 + 7 = 14.
    #
    # Total rank: 8 (for Term 1) + 14 (for Term 2) = 22.
    # This is better than 24, but not 15.
    #
    # Can we do better for Term 1?
    # A_{i0} B_{0k}.
    # This is the outer product of a 2-vector and a 4-vector.
    # The rank of the bilinear map is 2*4 = 8?
    # No, the rank of the tensor A_{i0} B_{0k} is 1?
    # No, it's a sum of 8 rank-1 tensors?
    # A_{i0} B_{0k} = sum_{i,k} A_{i0} B_{0k} E_{ik}.
    # This is 8 terms.
    #
    # So total rank 22.
    #
    # Is there a way to combine Term 1 and Term 2?
    # C = A1 B1 + A2 B2.
    #
    # Let's try to find a rank-15 decomposition using a search.
    #
    # I will implement a simple local search for R=15.
    #
    # 1. Initialize U, V, W with random integers in {-1, 0, 1}.
    # 2. Compute error.
    # 3. If error is 0, return.
    # 4. Try to reduce error by changing one entry at a time.
    #
    # This is likely to fail, but it's a valid attempt.
    #
    # To improve the chances, I will use the structure of the problem.
    # The equations are:
    # sum_{r} W[k*n+i, r] U[i1*m+j1, r] V[j2*p+k2, r] = delta_{i,i1} delta_{k,k2} delta_{j1,j2}
    #
    # We can fix U and V, and solve for W?
    # No, it's not linear in W if U and V are fixed?
    # sum_{r} W[k*n+i, r] (U[i1*m+j1, r] V[j2*p+k2, r]) = RHS.
    # This is linear in W!
    #
    # So we can:
    # 1. Randomly initialize U and V.
    # 2. Solve the linear system for W.
    # 3. Check if W has integer/half-integer entries (or close to them).
    # 4. If yes, verify the full solution.
    #
    # This is a much better strategy!
    #
    # The system is:
    # For each (i, k, i1, j1, j2, k2):
    # sum_{r} W[k*n+i, r] M_{r, (i1,j1,j2,k2)} = RHS
    # where M_{r, ...} = U[i1*m+j1, r] * V[j2*p+k2, r].
    #
    # We have 576 equations and 8*R unknowns (entries of W).
    # For R=15, we have 120 unknowns and 576 equations.
    # This is an overdetermined system.
    # We can use least squares to find W.
    # Then check if W is close to integer/half-integer.
    #
    # We can repeat this with different random U, V.
    #
    # Let's implement this.

    def check_solution(U, V, W, n, m, p):
        """Check if U, V, W form a valid decomposition."""
        # Compute the tensor defined by U, V, W
        # T[i, k, i1, j1, j2, k2] = sum_r W[k*n+i, r] U[i1*m+j1, r] V[j2*p+k2, r]
        # We need T[i, k, i, j, j, k] = 1 and T[i, k, i1, j1, j2, k2] = 0 otherwise.
        
        # We only need to check the non-zero terms of the target tensor.
        # The target tensor is 1 if i=i1, k=k2, j1=j2.
        
        # Let's check a few random samples to speed up.
        # But for exactness, we must check all.
        # 576 checks is fast.
        
        for i in range(n):
            for k in range(p):
                for i1 in range(n):
                    for j1 in range(m):
                        for j2 in range(m):
                            for k2 in range(p):
                                # Calculate LHS
                                lhs = 0
                                for r in range(len(W[0])):
                                    lhs += W[k*n+i][r] * U[i1*m+j1][r] * V[j2*p+k2][r]
                                
                                # Calculate RHS
                                if i == i1 and k == k2 and j1 == j2:
                                    rhs = 1
                                else:
                                    rhs = 0
                                    
                                if lhs != rhs:
                                    return False
        return True

    def solve_for_W(U, V, n, m, p, R):
        """Solve for W given U and V using least squares."""
        # Number of equations
        num_eq = n * p * n * m * m * p
        # Number of unknowns (entries of W)
        num_unk = p * n * R
        
        # Build the matrix A and vector b
        # A[idx_eq, idx_unk]
        # idx_eq = ((i*p + k)*n + i1)*m*m + j1*m + j2 ... wait, k2 is also involved.
        # Let's flatten the indices (i, k, i1, j1, j2, k2)
        
        A = np.zeros((num_eq, num_unk))
        b = np.zeros(num_eq)
        
        idx = 0
        for i in range(n):
            for k in range(p):
                for i1 in range(n):
                    for j1 in range(m):
                        for j2 in range(m):
                            for k2 in range(p):
                                # Equation idx
                                # Unknowns are W[k*n+i, r] for r in 0..R-1
                                # The coefficient for W[k*n+i, r] is U[i1*m+j1, r] * V[j2*p+k2, r]
                                
                                for r in range(R):
                                    w_idx = (k*n + i) * R + r
                                    A[idx, w_idx] = U[i1*m+j1][r] * V[j2*p+k2][r]
                                    
                                # RHS
                                if i == i1 and k == k2 and j1 == j2:
                                    b[idx] = 1
                                else:
                                    b[idx] = 0
                                idx += 1
                                
        # Solve least squares
        # A W = b
        # W is (p*n*R, 1)
        try:
            W_flat, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
        except np.linalg.LinAlgError:
            return None
            
        # Reshape W
        W = [[0] * R for _ in range(p * n)]
        for r in range(R):
            for ki in range(p * n):
                W[ki][r] = W_flat[ki * R + r]
                
        return W

    def round_to_half(x):
        """Round x to the nearest multiple of 0.5."""
        return round(2 * x) / 2.0

    def is_half_integer(x, tol=1e-9):
        """Check if x is close to a half-integer."""
        k = round(2 * x)
        return abs(x - k / 2.0) < tol

    def refine_W(W, n, m, p, R):
        """Try to snap W entries to nearest half-integers and check validity."""
        W_snapped = [[0] * R for _ in range(p * n)]
        for i in range(p * n):
            for r in range(R):
                W_snapped[i][r] = round_to_half(W[i][r])
        
        # Check if the snapped W is valid (with the same U, V)
        # We need U and V here, so we pass them in.
        # But this function only has W.
        # Let's do the check in the main loop.
        return W_snapped

    def search_rank(R, n, m, p, max_iters=1000, time_limit=100):
        """Search for a decomposition of rank R."""
        start_time = time.time()
        
        for iter in range(max_iters):
            if time.time() - start_time > time_limit:
                break
                
            # Randomly initialize U and V
            # Use small integers: -1, 0, 1
            U = [[random.choice([-1, 0, 1]) for _ in range(R)] for _ in range(n * m)]
            V = [[random.choice([-1, 0, 1]) for _ in range(R)] for _ in range(m * p)]
            
            # Solve for W
            W = solve_for_W(U, V, n, m, p, R)
            if W is None:
                continue
                
            # Snap W to half-integers
            W_snapped = [[0] * R for _ in range(p * n)]
            for i in range(p * n):
                for r in range(R):
                    W_snapped[i][r] = round_to_half(W[i][r])
                    
            # Check validity
            if check_solution(U, V, W_snapped, n, m, p):
                return U, V, W_snapped
                
            # Optional: Check if W is close to half-integers
            # If not, maybe try a different initialization
            
        return None

    # First, try to find a low rank solution
    # Known lower bound is 14. Let's try 15, 14.
    # If not found, fall back to schoolbook.
    
    best_result = None
    best_rank = n * m * p
    
    # Schoolbook solution
    U_school = [[0] * best_rank for _ in range(n * m)]
    V_school = [[0] * best_rank for _ in range(m * p)]
    W_school = [[0] * best_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
                
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    
    # Try lower ranks
    # We have 120s total. Use 100s for search.
    # Try R=15 first.
    res = search_rank(15, n, m, p, max_iters=500, time_limit=50)
    if res:
        U, V, W = res
        best_result = {"U": U, "V": V, "W": W}
        best_rank = 15
        
    # If 15 failed, try 14
    if best_rank > 14:
        res = search_rank(14, n, m, p, max_iters=500, time_limit=40)
        if res:
            U, V, W = res
            best_result = {"U": U, "V": V, "W": W}
            best_rank = 14
            
    # If 14 failed, try 22 (block decomposition)
    # We can construct the block decomposition explicitly.
    # C = A1 B1 + A2 B2
    # A1 is 2x1, B1 is 1x4.
    # A2 is 2x2, B2 is 2x4.
    #
    # This is complex to implement correctly in the U, V, W format.
    # The schoolbook solution is safer.
    
    return best_result

# EVOLVE-BLOCK-END
