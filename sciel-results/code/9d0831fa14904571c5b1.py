import time
import numpy as np
from fractions import Fraction

def construct():
    n, m, p = 2, 3, 5
    nm, mp, pn = n * m, m * p, p * n  # 6, 15, 10
    
    start_time = time.time()
    deadline = start_time + 115.0

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

    def verify(U, V, W):
        r = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(r):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if s != 1:
                        return False
        return True

    def is_half_integer(x):
        # x is a Fraction
        y = x * 2
        return y.denominator == 1

    def build_decomp():
        """
        Constructs a rank-28 decomposition using a recursive Strassen-like approach.
        
        We partition the 3x5 B matrix into B_left (3x2) and B_right (3x3).
        C = A * B = A * [B_left | B_right] = [A*B_left | A*B_right]
        
        We use a specific algebraic construction for M_{2,3,5}.
        This is derived from the fact that the rank of M_{2,3,5} is 28.
        We can construct this by combining a decomposition of M_{2,3,2} (rank 8) 
        and M_{2,3,3} (rank 14) ? No, 8+14=22 which is too low.
        
        Actually, the known rank of M_{2,3,5} is 28.
        A standard construction for M_{2,3,5} uses the block structure.
        
        Let's use a known explicit rank-28 decomposition for 2x3 * 3x5.
        Since deriving it from scratch in code is complex, we can use a 
        "structured random search" with a much tighter constraint: 
        We fix U and V to be block-diagonal or sparse in a specific pattern 
        that corresponds to the Strassen algorithm on the 2x2x2 core 
        plus auxiliary terms.
        
        However, a more robust "different strategy" that is structurally 
        different from the previous random-lstsq is to use a 
        **Greedy Exact Integer Search** on a reduced subspace.
        
        Actually, let's implement the specific rank-28 solution found in literature 
        or construct it via the following identity:
        
        For 2x3 * 3x5, we can view it as 2x3 * (3x2 + 3x3).
        Rank(M_{2,3,2}) = 8.
        Rank(M_{2,3,3}) = 14 (Strassen for 2x2x2 is 7, but 2x3x3 is 14? 
        Wait, 2x3*3x3. If we split 3x3 into 2x2+2x1+1x2+1x1...
        
        Let's try a different structural approach: 
        **Explicit Rank-28 Construction via Tensor Slicing**
        
        We will define the 28 terms explicitly.
        This is a known result. The decomposition of M_{2,3,5} with rank 28
        can be obtained by taking the tensor product of M_{2,2,2} (Strassen, rank 7)
        and some auxiliary 1x1x1 terms? No.
        
        Let's use the "Block Strassen" method.
        Split A (2x3) into A1 (2x2) and A2 (2x1).
        Split B (3x5) into B1 (2x5) and B2 (1x5).
        
        C = A1*B1 + A2*B2.
        
        Rank(M_{2,2,5})? 
        Rank(M_{2,1,5}) = 5.
        
        This path is tricky to get exactly 28 without a known table.
        
        Alternative Structural Change:
        Instead of random search, we use a **Deterministic Algebraic Construction**.
        
        We know that the rank of M_{2,3,5} is 28.
        We can construct it by combining:
        1. A rank-7 decomposition for the 2x2x2 core (Strassen).
        2. Auxiliary terms to handle the extra dimensions.
        
        Let's implement a specific known rank-28 decomposition for 2x3*3x5.
        Reference: "On the rank of matrix multiplication tensors"
        
        Since I cannot look up the exact 28-term coefficients in real-time easily 
        without risk of error, I will use a **Hybrid Deterministic/Local Search**:
        
        1. Initialize U, V, W with a structured "block Strassen" pattern for the 2x2x2 part.
        2. Use local moves (flipping signs, swapping indices) to satisfy the remaining equations.
        
        However, the prompt asks for a strategy materially different.
        The current one is Random + Lstsq.
        The new one will be **Explicit Block Construction + Exact Integer Completion**.
        
        We will explicitly set up the first 7 terms as Strassen for the top-left 2x2x2 block.
        Then we will solve for the remaining 21 terms using a structured linear system 
        where the variables are constrained to be integers/half-integers by design 
        (e.g., using a basis of the solution space over Q and checking for integrality).
        
        Actually, the simplest "different" and robust approach for this specific size 
        is to use the **Rank-28 decomposition derived from the 2x2x2 Strassen 
        algorithm extended to the rectangular case**.
        
        Let's hardcode a valid rank-28 decomposition if possible, or use a 
        very fast exact solver on a reduced space.
        
        Given the constraints, I will implement a **Structured Random Search with Exact Rationals**.
        Instead of floating point Lstsq, we will:
        1. Fix U and V to be sparse with entries in {-1, 1}.
        2. Solve for W exactly using Gaussian elimination over Fractions.
        3. Check if W is half-integer.
        
        This avoids the numerical instability of the current program.
        
        To make it fast, we will restrict the search space significantly:
        - We only consider U and V where each column has exactly 2 non-zeros in U 
          and 3 non-zeros in V (typical for low rank).
        - We use a pre-computed list of candidate column patterns.
        """
        
        # Define the tensor T_{ijk} = 1 for all i,j,k
        
        # We will try to find a rank-28 solution.
        # Strategy: 
        # 1. Generate random sparse U, V with exact integer entries.
        # 2. Solve A W = b over Q.
        # 3. If W has half-integer entries, return.
        
        # To ensure speed, we limit the number of trials and use a smarter 
        # generation of U, V.
        
        import random
        random.seed(42)
        
        target_rank = 28
        
        # Pre-generate some random sparse matrices? No, do it on the fly.
        
        # We will try 5000 trials.
        # Each trial:
        # 1. Create U (6x28) and V (15x28) with entries in {-1, 0, 1}.
        #    Constraint: Sum of squares in each column is small?
        # 2. Build the 30x280 linear system for W.
        # 3. Solve using Fraction.
        
        # Optimization: 
        # The system is A W = 1.
        # A is 30 x 280.
        # We can reduce the system by noticing that the rows are grouped by (i,k).
        # For a fixed (i,k), there are m=3 equations (for j=0,1,2).
        # The variables are W[k,i,c] for c=0..27.
        # So for each (i,k), we have a 3x28 system.
        # There are 10 such (i,k) pairs.
        # So the system is block diagonal with 10 blocks of size 3x28!
        
        # This is a HUGE simplification.
        # We don't need to solve a 30x280 system.
        # We need to solve 10 independent 3x28 systems.
        
        # For each (i,k) pair:
        # Equations:
        # sum_c U[i,j,c] V[j,k,c] W[k,i,c] = 1 for j=0,1,2
        
        # Let x_c = W[k,i,c].
        # a_jc = U[i,j,c] * V[j,k,c]
        # sum_c a_jc x_c = 1
        
        # We need to find x_c in (1/2)Z^28 such that A_block x = 1.
        
        # This is much faster.
        
        def solve_block(a_mat):
            # a_mat is 3x28
            # Solve a_mat x = 1
            # We want x in (1/2)Z^28.
            
            # We can find the null space of a_mat?
            # Or just find a particular solution and add null space vectors?
            
            # Since 28 vars, 3 eqs, the solution space is 25-dimensional.
            # We want a solution with half-integer coordinates.
            
            # We can try to set 25 variables to 0 and solve for the remaining 3?
            # If we pick 3 columns that form an invertible 3x3 submatrix,
            # we can solve for those 3 variables.
            # If the solution is half-integer, we are done.
            
            # So the strategy for each block:
            # 1. Find a set of 3 columns that are linearly independent.
            # 2. Solve for the 3 variables.
            # 3. Check if they are half-integers.
            # 4. If not, try another set of 3 columns.
            
            # This is very fast.
            
            # Let's implement this.
            
            cols = list(range(28))
            
            # Try all combinations of 3 columns? C(28,3) = 3276.
            # For 10 blocks, that's 32760 checks. Very fast.
            
            for c1, c2, c3 in itertools.combinations(cols, 3):
                # Build 3x3 matrix
                M = np.array([
                    [a_mat[0, c1], a_mat[0, c2], a_mat[0, c3]],
                    [a_mat[1, c1], a_mat[1, c2], a_mat[1, c3]],
                    [a_mat[2, c1], a_mat[2, c2], a_mat[2, c3]]
                ])
                
                det = np.linalg.det(M)
                if abs(det) < 1e-9:
                    continue
                
                # Solve M x = 1
                try:
                    x_sol = np.linalg.solve(M, np.ones(3))
                except:
                    continue
                
                # Check if x_sol are half-integers
                x_check = x_sol * 2
                if np.allclose(x_check, np.round(x_check), atol=1e-5):
                    x_rounded = np.round(x_check) / 2
                    # Verify
                    if np.allclose(M @ x_rounded, np.ones(3)):
                        # Construct full x vector
                        x_full = np.zeros(28)
                        x_full[c1] = x_rounded[0]
                        x_full[c2] = x_rounded[1]
                        x_full[c3] = x_rounded[2]
                        return x_full
            
            return None

        import itertools
        
        num_trials = 1000
        for t in range(num_trials):
            if time.time() > deadline:
                return None
            
            # Generate random U, V
            # To increase chance of success, we want the 3x3 submatrices to be "nice".
            # Entries in {-1, 1} are good.
            
            U = np.zeros((nm, target_rank), dtype=int)
            V = np.zeros((mp, target_rank), dtype=int)
            
            for c in range(target_rank):
                # Randomly assign -1, 1 to some entries
                # Let's use a density that ensures non-zero 3x3 submatrices
                # Each column of U has 6 entries. Let's make them all non-zero?
                # If U and V are dense with +/-1, the 3x3 submatrices are likely invertible.
                
                U[:, c] = np.random.choice([-1, 1], size=nm)
                V[:, c] = np.random.choice([-1, 1], size=mp)
            
            # Now check if we can solve all 10 blocks
            W_sol = np.zeros((pn, target_rank))
            success = True
            
            for i in range(n):
                for k in range(p):
                    # Build 3x28 matrix a_mat
                    a_mat = np.zeros((3, target_rank))
                    for j in range(m):
                        for c in range(target_rank):
                            a_mat[j, c] = U[i * m + j, c] * V[j * p + k, c]
                    
                    x_sol = solve_block(a_mat)
                    if x_sol is None:
                        success = False
                        break
                    W_sol[k * n + i, :] = x_sol
            
            if success:
                # Verify the full solution
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W_sol.tolist()
                
                if verify(U_list, V_list, W_list):
                    return {"U": U_list, "V": V_list, "W": W_list}
        
        return None

    # Try the new structured approach
    res = build_decomp()
    if res:
        return res
        
    # Fallback to schoolbook if it fails
    return schoolbook()

if __name__ == "__main__":
    construct()
