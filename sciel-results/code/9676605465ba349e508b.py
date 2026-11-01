"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import product

_START_TIME = time.time()
_DEADLINE = _START_TIME + 140  # 10s safety margin

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if s != 1:
                    return False
    return True

def _strassen_block_decomposition(n, m, p):
    """Build rank-21 decomposition using Strassen on three 2x2 blocks."""
    rank = 21
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for b in range(3):
        offset = b * 7
        
        U[0][offset+0] = 1
        U[0][offset+2] = 1
        U[0][offset+4] = 1
        U[0][offset+5] = -1
        
        U[1][offset+4] = 1
        U[1][offset+6] = -1
        
        U[2][offset+1] = 1
        U[2][offset+5] = 1
        
        U[3][offset+0] = 1
        U[3][offset+1] = 1
        U[3][offset+3] = 1
        
        V[0 * p + b*2][offset+0] = 1
        V[0 * p + b*2][offset+1] = 1
        V[0 * p + b*2][offset+5] = 1
        
        V[0 * p + b*2 + 1][offset+3] = -1
        V[0 * p + b*2 + 1][offset+6] = 1
        
        V[1 * p + b*2][offset+2] = 1
        V[1 * p + b*2][offset+3] = -1
        V[1 * p + b*2][offset+5] = 1
        
        V[1 * p + b*2 + 1][offset+0] = 1
        V[1 * p + b*2 + 1][offset+4] = 1
        V[1 * p + b*2 + 1][offset+6] = 1
        
        W[(b*2) * n + 0][offset+0] = 1
        W[(b*2) * n + 0][offset+3] = 1
        W[(b*2) * n + 0][offset+4] = -1
        W[(b*2) * n + 0][offset+6] = 1
        
        W[(b*2+1) * n + 0][offset+2] = 1
        W[(b*2+1) * n + 0][offset+4] = 1
        
        W[(b*2) * n + 1][offset+1] = 1
        W[(b*2) * n + 1][offset+3] = 1
        
        W[(b*2+1) * n + 1][offset+0] = 1
        W[(b*2+1) * n + 1][offset+1] = -1
        W[(b*2+1) * n + 1][offset+2] = 1
        W[(b*2+1) * n + 1][offset+5] = 1
    
    return {"U": U, "V": V, "W": W}

def _schoolbook_decomposition(n, m, p):
    """Fallback rank-24 schoolbook decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _try_sparse_decomposition(n, m, p, rank, rng, max_iters=5000):
    """
    Try to find a rank-`rank` decomposition by:
    1. Randomly assigning sparse supports to U, V, W
    2. Solving the linear system for the values
    3. Checking if we get valid (integer/half-integer) entries
    """
    nm, mp, pn = n * m, m * p, p * n
    num_constraints = n * m * p  # 24
    
    # Each rank-1 term contributes to the tensor.
    # The tensor T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # We need T[i,j,k] = 1 for all i,j,k.
    
    # Strategy: fix the pattern of which (i,j,r), (j,k,r), (k,i,r) are nonzero,
    # then solve for the values.
    
    for _ in range(max_iters):
        if time.time() > _DEADLINE - 5:
            break
        
        # Randomly decide for each rank position r and each factor which entries are nonzero
        U_support = set()
        V_support = set()
        W_support = set()
        
        for r in range(rank):
            # U: nm entries, choose ~3-5 nonzeros
            u_nonzero = rng.sample(range(nm), k=rng.randint(2, min(5, nm)))
            for idx in u_nonzero:
                U_support.add((idx, r))
            
            # V: mp entries, choose ~3-5 nonzeros
            v_nonzero = rng.sample(range(mp), k=rng.randint(2, min(5, mp)))
            for idx in v_nonzero:
                V_support.add((idx, r))
            
            # W: pn entries, choose ~3-5 nonzeros
            w_nonzero = rng.sample(range(pn), k=rng.randint(2, min(5, pn)))
            for idx in w_nonzero:
                W_support.add((idx, r))
        
        # Build the linear system
        # Unknowns: values of U[i,j,r], V[j,k,r], W[k,i,r] for supported entries
        # But the equations are nonlinear (product of three variables).
        # 
        # Alternative: fix U and V, then W is determined linearly.
        # Or: use a different approach.
        
        # Let's try: fix U and V randomly to small integers, then solve for W.
        # W[k*n+i, r] must satisfy:
        # For each (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        # This is linear in W if U and V are fixed.
        
        # But we have pn*rank unknowns for W and num_constraints equations.
        # If pn*rank > num_constraints, we might have a solution.
        
        # Actually, let's be smarter. For each r, define:
        # a_r[i,j] = U[i*m+j, r]
        # b_r[j,k] = V[j*p+k, r]  
        # c_r[k,i] = W[k*n+i, r]
        # Constraint: sum_r a_r[i,j] * b_r[j,k] * c_r[k,i] = 1
        
        # If we fix all a_r and b_r, then for each (i,k), the variable c_r[k,i] appears
        # in constraints for all j. Specifically, for fixed (i,k), c_r[k,i] appears in
        # the constraint for (i,j,k) as a_r[i,j]*b_r[j,k]*c_r[k,i].
        
        # So for fixed (i,k), the constraints over j give:
        # sum_r (sum_j a_r[i,j]*b_r[j,k]) * c_r[k,i] = 1  ... no, that's not right.
        
        # The constraint is: for each (i,j,k): sum_r a_r[i,j]*b_r[j,k]*c_r[k,i] = 1
        # For fixed (i,k), we have m constraints (one per j) involving variables c_r[k,i] for r=0..rank-1.
        # That's m equations in rank unknowns. If rank >= m=2, this is feasible.
        
        # Total: n*p*(m) = 2*6*2 = 24 equations, and n*p*rank = 2*6*rank = 12*rank unknowns for W.
        # If rank >= 2, we have more unknowns than equations, so a solution exists generically.
        
        # Let's try: fix U and V to random small integers, solve for W via least squares,
        # then round and check.
        
        # Build U and V with random values from {-1, 0, 1}
        U_vals = np.zeros((nm, rank), dtype=float)
        V_vals = np.zeros((mp, rank), dtype=float)
        
        for idx in U_support:
            U_vals[idx] = rng.choice([-1, 1])
        for idx in V_support:
            V_vals[idx] = rng.choice([-1, 1])
        
        # Build the linear system for W
        # For each (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        # Group by (k, i): for each (k,i), the unknowns are W[k*n+i, r] for r=0..rank-1
        # The equations involving W[k*n+i, *] are for all j:
        #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1  for j=0..m-1
        
        # So for each (k,i), we have m equations in rank unknowns.
        # Total: p*n*m = 24 equations, p*n*rank = 12*rank unknowns.
        
        # Build the full system
        # Unknowns indexed by (k, i, r) -> flat index k*n+i*rank + r
        num_unknowns = pn * rank
        A = np.zeros((num_constraints, num_unknowns), dtype=float)
        b_vec = np.ones(num_constraints, dtype=float)
        
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for r in range(rank):
                        # Unknown W[k*n+i, r] is at flat index (k*n+i)*rank + r
                        unknown_idx = (k * n + i) * rank + r
                        A[eq_idx, unknown_idx] = U_vals[i*m+j, r] * V_vals[j*p+k, r]
                    eq_idx += 1
        
        # Solve least squares
        try:
            W_flat, residuals, rank_A, sv = np.linalg.lstsq(A, b_vec, rcond=None)
        except np.linalg.LinAlgError:
            continue
        
        # Check if the solution is close to integer/half-integer
        W_vals = W_flat.reshape((pn, rank))
        
        # Check if W_vals entries are close to integers or half-integers
        max_dev = 0
        for i in range(pn):
            for r in range(rank):
                val = W_vals[i, r]
                # Check distance to nearest integer or half-integer
                nearest_int = round(val)
                nearest_half = round(val * 2) / 2
                dev = min(abs(val - nearest_int), abs(val - nearest_half))
                max_dev = max(max_dev, dev)
        
        if max_dev < 0.01:
            # Round to nearest integer or half-integer
            U_int = np.zeros((nm, rank), dtype=int)
            V_int = np.zeros((mp, rank), dtype=int)
            W_int = np.zeros((pn, rank), dtype=int)  # Store as doubled values
            
            for i in range(nm):
                for r in range(rank):
                    U_int[i, r] = int(round(U_vals[i, r]))
            for i in range(mp):
                for r in range(rank):
                    V_int[i, r] = int(round(V_vals[i, r]))
            for i in range(pn):
                for r in range(rank):
                    W_int[i, r] = int(round(W_vals[i, r] * 2))  # doubled
            
            # Verify with doubled W
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0
                        for r in range(rank):
                            s += U_int[i*m+j, r] * V_int[j*p+k, r] * W_int[k*n+i, r]
                        if s != 2:  # doubled
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                # Convert back: W values are half-integers
                U_list = U_int.tolist()
                V_list = V_int.tolist()
                W_list = []
                for i in range(pn):
                    row = []
                    for r in range(rank):
                        if W_int[i, r] % 2 == 0:
                            row.append(W_int[i, r] // 2)
                        else:
                            row.append(W_int[i, r] / 2)
                    W_list.append(row)
                
                return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def construct():
    n, m, p = DIMENSIONS
    
    # First, get the known rank-21 solution
    result_21 = _strassen_block_decomposition(n, m, p)
    if _verify(result_21["U"], result_21["V"], result_21["W"], n, m, p):
        best = result_21
    else:
        best = _schoolbook_decomposition(n, m, p)
    
    # Try to find a better (lower rank) decomposition
    rng = random.Random(42)
    
    for rank in [20, 19, 18]:
        if time.time() > _DEADLINE - 10:
            break
        
        result = _try_sparse_decomposition(n, m, p, rank, rng, max_iters=3000)
        if result is not None:
            if _verify(result["U"], result["V"], result["W"], n, m, p):
                return result
    
    return best
# EVOLVE-BLOCK-END
