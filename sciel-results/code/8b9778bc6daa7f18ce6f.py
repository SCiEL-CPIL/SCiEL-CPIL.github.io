import time
import numpy as np
from fractions import Fraction
from itertools import product

DIMENSIONS = [4, 5, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Fallback: Schoolbook
    U_fb = np.zeros((n * m, rank_schoolbook))
    V_fb = np.zeros((m * p, rank_schoolbook))
    W_fb = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j, col] = 1.0
                V_fb[j * p + k, col] = 1.0
                W_fb[k * n + i, col] = 1.0
                col += 1
                
    best_result = {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    best_rank = rank_schoolbook

    # Strategy: Try to find a rank 19 decomposition by lifting from 5x5x5 rank 19.
    # The 5x5x5 multiplication tensor has rank 19 (BCLR).
    # We want to restrict this to 4x5x5.
    # Let T be the 5x5x5 tensor. We have T_{ijk} = sum_{r=1}^{19} U_{ij}^r V_{jk}^r W_{ki}^r.
    # We want to find U', V', W' of size 4x5, 5x5, 5x4 such that
    # T'_{ijk} (for i<4) = sum_{r=1}^{19} U'_{ij}^r V'_{jk}^r W'_{ki}^r = 1 for all i<4, j, k.
    
    # This is a system of 4*5*5 = 100 equations.
    # Variables: U' (4*5*19 = 380), V' (5*5*19 = 475), W' (5*4*19 = 380). Total 1235 variables.
    # This is underdetermined. We can set most variables to 0 and solve for the rest.
    
    # A better approach: Use the known 5x5x5 rank 19 decomposition.
    # Let (U5, V5, W5) be a rank 19 decomposition of 5x5x5.
    # We can try to set U'_{ij} = U5_{ij} for i<4, and W'_{ki} = W5_{ki} for i<4.
    # Then we need to find V' such that sum_r U5_{ij}^r V'_{jk}^r W5_{ki}^r = 1 for i<4.
    # This is a linear system for V'.
    # Let A_{(i,k), (j,r)} = U5_{ij}^r W5_{ki}^r. Wait, the sum is over r.
    # For fixed i, j, k: sum_r (U5_{ij}^r * W5_{ki}^r) * V'_{jk}^r = 1.
    # This is not a simple linear system for V' because V'_{jk}^r are multiplied by coefficients that depend on r.
    # Actually, for fixed j, k, the equation is sum_r C_{r} V'_{jk}^r = 1, where C_r = U5_{ij}^r W5_{ki}^r.
    # This must hold for all i<4.
    # So for each (j, k), we have 4 equations:
    # sum_r (U5_{ij}^r W5_{ki}^r) V'_{jk}^r = 1 for i=0..3.
    # This is a system of 4 equations with 19 unknowns (V'_{jk}^r for r=0..18).
    # We can choose 15 of the unknowns to be 0 and solve for the 4 remaining.
    # Or, we can just solve the underdetermined system and pick a sparse solution.
    
    # Let's try to construct the 5x5x5 rank 19 decomposition first.
    # This is complex. Instead, let's try a direct search for 4x5x5 rank 19 using a structured initialization.
    
    # Alternative: Use the fact that rank(4x5x5) <= rank(5x5x5) = 19.
    # We can try to solve for U, V, W with rank 19 directly.
    
    # Let's use a "fix 2, solve 1" strategy.
    # Fix U and V to random small integers. Solve for W.
    # For each (k, i), the equation is sum_r U_{ij}^r V_{jk}^r W_{ki}^r = 1 for all j.
    # Wait, the equation is for all (i, j, k).
    # sum_r U_{ij}^r V_{jk}^r W_{ki}^r = 1.
    # Group by (k, i). For fixed (k, i), we have 5 equations (for j=0..4) and 19 unknowns (W_{ki}^r).
    # Let M_{j, r} = U_{ij}^r V_{jk}^r. Then sum_r M_{j, r} W_{ki}^r = 1.
    # This is a linear system M W_{ki} = 1_vector.
    # If M is 5x19, we can find a solution if the rows are consistent.
    # We can choose U, V such that M has rank 5 and the system is consistent.
    # Actually, we can just set W_{ki}^r = 0 for r>=5 and solve the 5x5 system for the first 5 columns.
    
    # Let's try this:
    # For each (k, i), we will determine W_{ki}^r for r=0..4 by solving a 5x5 system.
    # The system is: sum_{r=0}^4 U_{ij}^r V_{jk}^r W_{ki}^r = 1 for j=0..4.
    # This requires that the 5x5 matrix B_{j, r} = U_{ij}^r V_{jk}^r is invertible.
    # And we set W_{ki}^r = 0 for r=5..18.
    
    # So the problem reduces to finding U and V such that for all (i, k),
    # the matrix B^{(i,k)}_{j, r} = U_{ij}^r V_{jk}^r is invertible.
    # And we want U, V to have small integer entries.
    
    # Let's try to construct U and V with entries in {-1, 0, 1}.
    
    def _try_structured_rank_19():
        if time.time() > deadline - 10:
            return None
            
        # We need U (20x19), V (25x19).
        # We will try to find U, V such that for all i in 0..3, k in 0..4,
        # the 5x5 matrix B_{j, r} = U[i, j, r] * V[j, k, r] is invertible.
        
        # Let's try to set U and V to be random sparse matrices.
        for seed in range(50):
            if time.time() > deadline - 5:
                break
                
            np.random.seed(seed)
            
            # Initialize U, V with -1, 0, 1
            U = np.random.randint(-1, 2, (n * m, 19))
            V = np.random.randint(-1, 2, (m * p, 19))
            
            # Check if we can solve for W
            W = np.zeros((p * n, 19))
            valid = True
            
            for i in range(n):
                for k in range(p):
                    # Construct B (5x5)
                    B = np.zeros((5, 5))
                    for j in range(m):
                        for r in range(5):
                            B[j, r] = U[i * m + j, r] * V[j * p + k, r]
                            
                    # Solve B W_col = 1
                    try:
                        det = np.linalg.det(B)
                        if abs(det) < 1e-9:
                            valid = False
                            break
                        W_col = np.linalg.solve(B, np.ones(5))
                        W[k * n + i, :5] = W_col
                    except np.linalg.LinAlgError:
                        valid = False
                        break
                        
            if not valid:
                continue
                
            # Verify the full decomposition
            if _verify(n, m, p, U, V, W):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                
        return None

    result_19 = _try_structured_rank_19()
    if result_19:
        best_rank = 19
        best_result = result_19

    return best_result

def _verify(n, m, p, U, V, W):
    rank = U.shape[1]
    # Check all i, j, k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if not np.isclose(s, 1.0, atol=1e-9):
                    return False
    return True
