import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    dim_U = n * m  # 12
    dim_V = m * p  # 16
    dim_W = p * n  # 12
    
    # Precompute tensor indices: T[I,J,K] = 1 for valid matrix mult entries
    T_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                I_idx = i * m + j
                J_idx = j * p + k
                K_idx = k * n + i
                T_indices.append((I_idx, J_idx, K_idx))
    
    # Build the full tensor
    T = np.zeros((dim_U, dim_V, dim_W), dtype=int)
    for (I_idx, J_idx, K_idx) in T_indices:
        T[I_idx, J_idx, K_idx] = 1
    
    def verify_solution(U, V, W, R):
        """Verify exact decomposition with half-integer entries."""
        U_int = np.round(np.array(U) * 2).astype(int)
        V_int = np.round(np.array(V) * 2).astype(int)
        W_int = np.round(np.array(W) * 2).astype(int)
        
        # Check: sum_r U_int[I,r] * V_int[J,r] * W_int[K,r] == 8 for non-zeros, 0 elsewhere
        T_approx = np.einsum('ir, jr, kr -> ijk', U_int, V_int, W_int)
        
        # Build expected tensor (scaled by 8)
        T_full = np.zeros((dim_U, dim_V, dim_W), dtype=int)
        for (I_idx, J_idx, K_idx) in T_indices:
            T_full[I_idx, J_idx, K_idx] = 8
            
        return np.array_equal(T_approx, T_full)
    
    def schoolbook_decomposition():
        """Return the rank-48 schoolbook decomposition."""
        rank = n * m * p
        U = [[0] * rank for _ in range(dim_U)]
        V = [[0] * rank for _ in range(dim_V)]
        W = [[0] * rank for _ in range(dim_W)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return U, V, W
    
    def try_greedy_cover(R, time_limit, seed=42):
        """
        Greedy set cover approach: build rank components one at a time.
        Each component is a rank-1 tensor u ⊗ v ⊗ w.
        We want to cover all 48 tensor positions with R components.
        """
        np.random.seed(seed)
        start_time = time.time()
        
        # Small integer values to try: -2, -1, 0, 1, 2 (scaled by 1/2)
        # So actual values: -1, -0.5, 0, 0.5, 1
        small_vals = [-2, -1, 0, 1, 2]
        
        best_solution = None
        best_rank = R
        
        # Try multiple random initializations
        for attempt in range(100):
            if time.time() > time_limit:
                break
            
            # Initialize with R random rank-1 components
            U = np.zeros((dim_U, R), dtype=int)
            V = np.zeros((dim_V, R), dtype=int)
            W = np.zeros((dim_W, R), dtype=int)
            
            for r in range(R):
                # Random small integer values
                U[:, r] = np.random.choice(small_vals, size=dim_U)
                V[:, r] = np.random.choice(small_vals, size=dim_V)
                W[:, r] = np.random.choice(small_vals, size=dim_W)
            
            # Greedily try to cover all tensor positions
            covered = np.zeros((dim_U, dim_V, dim_W), dtype=int)
            uncovered = T.copy()
            
            for r in range(R):
                if time.time() > time_limit:
                    break
                
                # Compute contribution of this rank-1 component
                contrib = np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
                
                # Check if this helps cover uncovered positions
                # We want contrib[i,j,k] to be 8 where uncovered[i,j,k]=1
                # and 0 where uncovered[i,j,k]=0 (or already covered)
                
                # Try to adjust this component to better cover
                # Simple approach: for each uncovered position, try to set this component to help
                for (I_idx, J_idx, K_idx) in T_indices:
                    if uncovered[I_idx, J_idx, K_idx] == 1:
                        # We need U[I_idx,r]*V[J_idx,r]*W[K_idx,r] to contribute to 8
                        # Current contribution from this rank is contrib[I_idx,J_idx,K_idx]
                        # We want total to be 8
                        pass
            
            # Verify
            if verify_solution(U / 2.0, V / 2.0, W / 2.0, R):
                return U / 2.0, V / 2.0, W / 2.0
        
        return None
    
    def try_local_search(R, time_limit, seed=42):
        """
        Local search: start from schoolbook sub-selection, then try to merge/modify columns.
        """
        np.random.seed(seed)
        start_time = time.time()
        
        U0, V0, W0 = schoolbook_decomposition()
        U0 = np.array(U0, dtype=float)
        V0 = np.array(V0, dtype=float)
        W0 = np.array(W0, dtype=float)
        
        # Take first R columns
        U = U0[:, :R].copy()
        V = V0[:, :R].copy()
        W = W0[:, :R].copy()
        
        # Try to improve by modifying columns
        for it in range(500):
            if time.time() > time_limit:
                break
            
            # Pick a random column to modify
            r = np.random.randint(0, R)
            
            # Try small perturbations
            for delta in [-0.5, 0.5]:
                # Try modifying U
                for i in range(dim_U):
                    old_val = U[i, r]
                    new_val = old_val + delta
                    # Keep values in reasonable range
                    if abs(new_val) <= 2:
                        U[i, r] = new_val
                        if verify_solution(U, V, W, R):
                            return U, V, W
                        U[i, r] = old_val
                
                # Try modifying V
                for j in range(dim_V):
                    old_val = V[j, r]
                    new_val = old_val + delta
                    if abs(new_val) <= 2:
                        V[j, r] = new_val
                        if verify_solution(U, V, W, R):
                            return U, V, W
                        V[j, r] = old_val
                
                # Try modifying W
                for k in range(dim_W):
                    old_val = W[k, r]
                    new_val = old_val + delta
                    if abs(new_val) <= 2:
                        W[k, r] = new_val
                        if verify_solution(U, V, W, R):
                            return U, V, W
                        W[k, r] = old_val
        
        return None
    
    def try_structured_search(R, time_limit, seed=42):
        """
        Try structured solutions based on known patterns for small matrix multiplication.
        """
        np.random.seed(seed)
        start_time = time.time()
        
        # For 3x4 x 4x4, try to find solutions with structure
        # Key insight: each row of U, V, W has limited support
        
        # Try: U[i,j] depends only on i, V[j,k] depends only on j, W[k,i] depends only on i
        # This is too restrictive, but let's try variations
        
        # Alternative: use the fact that the tensor is separable in certain ways
        # T[i,j,k] = 1 iff the indices are consistent with matrix multiplication
        
        # Try random structured initialization
        for attempt in range(200):
            if time.time() > time_limit:
                break
            
            U = np.zeros((dim_U, R), dtype=float)
            V = np.zeros((dim_V, R), dtype=float)
            W = np.zeros((dim_W, R), dtype=float)
            
            # Initialize with random half-integers
            for r in range(R):
                # Each rank-1 component should be "focused" on a subset of indices
                # Pick random subsets
                u_support = set(random.sample(range(dim_U), random.randint(2, 5)))
                v_support = set(random.sample(range(dim_V), random.randint(2, 5)))
                w_support = set(random.sample(range(dim_W), random.randint(2, 5)))
                
                for i in u_support:
                    U[i, r] = random.choice([-1.0, -0.5, 0.5, 1.0])
                for j in v_support:
                    V[j, r] = random.choice([-1.0, -0.5, 0.5, 1.0])
                for k in w_support:
                    W[k, r] = random.choice([-1.0, -0.5, 0.5, 1.0])
            
            # Local search
            for it in range(100):
                if time.time() > time_limit:
                    break
                
                # Pick random element to modify
                mat = random.choice(['U', 'V', 'W'])
                if mat == 'U':
                    idx = random.randint(0, dim_U - 1)
                    r = random.randint(0, R - 1)
                    old_val = U[idx, r]
                    for new_val in [-1.0, -0.5, 0.0, 0.5, 1.0]:
                        if new_val != old_val:
                            U[idx, r] = new_val
                            if verify_solution(U, V, W, R):
                                return U, V, W
                            U[idx, r] = old_val
                elif mat == 'V':
                    idx = random.randint(0, dim_V - 1)
                    r = random.randint(0, R - 1)
                    old_val = V[idx, r]
                    for new_val in [-1.0, -0.5, 0.0, 0.5, 1.0]:
                        if new_val != old_val:
                            V[idx, r] = new_val
                            if verify_solution(U, V, W, R):
                                return U, V, W
                            V[idx, r] = old_val
                else:
                    idx = random.randint(0, dim_W - 1)
                    r = random.randint(0, R - 1)
                    old_val = W[idx, r]
                    for new_val in [-1.0, -0.5, 0.0, 0.5, 1.0]:
                        if new_val != old_val:
                            W[idx, r] = new_val
                            if verify_solution(U, V, W, R):
                                return U, V, W
                            W[idx, r] = old_val
        
        return None
    
    # Main search
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    # Try ranks from low to high
    # Lower bound is max(12, 16, 12) = 16, but realistically we need more
    # Try a range of ranks
    for R in range(20, 49):
        if time.time() > deadline:
            break
        
        # Try structured search first (more likely to find solutions)
        res = try_structured_search(R, time_limit=deadline, seed=42)
        if res is not None:
            U, V, W = res
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        # Try local search
        res = try_local_search(R, time_limit=deadline, seed=42)
        if res is not None:
            U, V, W = res
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        # Try greedy cover
        res = try_greedy_cover(R, time_limit=deadline, seed=42)
        if res is not None:
            U, V, W = res
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook rank 48
    U, V, W = schoolbook_decomposition()
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
