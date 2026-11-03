import numpy as np
import time
import random

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix by a 3x7 matrix.
    Returns: {'U': list of lists, 'V': list of lists, 'W': list of lists}
    """
    n, m, p = 3, 3, 7
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Strategy: Decompose 3x3x7 as 3x3x3 + 3x3x4
    # Use rank-7 Strassen for 3x3x3, rank-12 schoolbook for 3x3x4
    # Total: 7 + 12 = 19, then try to merge redundant terms
    
    # Step 1: Build Strassen factors for 3x3x3 (rank 7)
    # For 3x3x3, we can use the standard 7-term algorithm
    # A = [[a11, a12, a13], [a21, a22, a23], [a31, a32, a33]]
    # B = [[b11, b12, b13], [b21, b22, b23], [b31, b32, b33]]
    
    # Standard 7-term decomposition for 3x3x3:
    # M1 = (a11 + a22 + a33)(b11 + b22 + b33)
    # M2 = (a11 + a22)(b11 + b22)  
    # M3 = (a22 + a33)(b22 + b33)
    # M4 = (a33 + a11)(b33 + b11)
    # M5 = (a12 + a23)(b12 + b23)
    # M6 = (a23 + a31)(b23 + b31)
    # M7 = (a31 + a12)(b31 + b12)
    # ... This is getting complex. Let me use a simpler approach.
    
    # Actually, for 3x3x3, the known minimal rank is 7, but constructing it explicitly
    # is non-trivial. Let me use a different strategy:
    
    # Use block structure: 3x3x7 = [3x3x3 | 3x3x4]
    # For 3x3x3: use schoolbook (rank 9) - safe and correct
    # For 3x3x4: use schoolbook (rank 12)
    # Total: 9 + 12 = 21
    
    # Then try to merge: if two terms have the same U and V factors, we can combine W factors
    # This is a linear algebra problem: find linear dependencies in the combined factor space
    
    # Build schoolbook for 3x3x3 (rank 9)
    U_333 = np.zeros((9, 9), dtype=int)
    V_333 = np.zeros((9, 9), dtype=int)
    W_333 = np.zeros((9, 9), dtype=int)
    
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U_333[i * 3 + j, col] = 1
                V_333[j * 3 + k, col] = 1
                W_333[k * 3 + i, col] = 1
                col += 1
    
    # Build schoolbook for 3x3x4 (rank 12)
    # A is 3x3, B is 3x4, C is 4x3
    U_334 = np.zeros((9, 12), dtype=int)
    V_334 = np.zeros((12, 12), dtype=int)  # m*p = 3*4 = 12
    W_334 = np.zeros((12, 12), dtype=int)  # p*n = 4*3 = 12
    
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(4):
                U_334[i * 3 + j, col] = 1
                V_334[j * 4 + k, col] = 1
                W_334[k * 3 + i, col] = 1
                col += 1
    
    # Combine: 3x3x7 = 3x3x3 (cols 0-2 of B) + 3x3x4 (cols 3-6 of B)
    # U: (9, 9+12) = (9, 21)
    # V: (21, 21) where rows 0-8 are for 3x3x3, rows 9-20 are for 3x3x4
    # W: (21, 21) where rows 0-8 are for 3x3x3, rows 9-20 are for 3x3x4
    
    U_combined = np.hstack([U_333, U_334])  # (9, 21)
    
    # For V: 3x3x3 uses V_333 (9x9) for B cols 0-2
    # 3x3x4 uses V_334 (12x12) for B cols 3-6
    # But we need to map correctly. B is 3x7, so:
    # V row j*7+k corresponds to B[j,k]
    # For 3x3x3 part: B[j,k] where k=0,1,2 -> V_333 row j*3+k
    # For 3x3x4 part: B[j,k] where k=3,4,5,6 -> V_334 row j*4+(k-3)
    
    V_combined = np.zeros((21, 21), dtype=int)
    # First 9 columns: 3x3x3 part
    for j in range(3):
        for k in range(3):
            V_combined[j * 7 + k, :9] = V_333[j * 3 + k, :]
    # Next 12 columns: 3x3x4 part
    for j in range(3):
        for k in range(4):
            V_combined[j * 7 + (k + 3), 9:21] = V_334[j * 4 + k, :]
    
    # For W: 3x3x3 uses W_333 (9x9) for C rows 0-2
    # 3x3x4 uses W_334 (12x12) for C rows 3-6
    # W row k*3+i corresponds to C[k,i]
    # For 3x3x3 part: C[k,i] where k=0,1,2 -> W_333 row k*3+i
    # For 3x3x4 part: C[k,i] where k=3,4,5,6 -> W_334 row (k-3)*3+i
    
    W_combined = np.zeros((21, 21), dtype=int)
    # First 9 columns: 3x3x3 part
    for k in range(3):
        for i in range(3):
            W_combined[k * 3 + i, :9] = W_333[k * 3 + i, :]
    # Next 12 columns: 3x3x4 part
    for k in range(4):
        for i in range(3):
            W_combined[(k + 3) * 3 + i, 9:21] = W_334[k * 3 + i, :]
    
    # Verify the combined decomposition
    def verify(U, V, W):
        """Verify decomposition using NumPy."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_vec = U[i * m + j, :]
                    v_vec = V[j * p + k, :]
                    w_vec = W[k * n + i, :]
                    val = np.dot(u_vec * v_vec, w_vec)
                    if val != 1:
                        return False
        return True
    
    # Check our combined decomposition
    if not verify(U_combined, V_combined, W_combined):
        # Fallback to pure schoolbook
        U_sb = np.zeros((n * m, n * m * p), dtype=int)
        V_sb = np.zeros((m * p, n * m * p), dtype=int)
        W_sb = np.zeros((p * n, n * m * p), dtype=int)
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j, col] = 1
                    V_sb[j * p + k, col] = 1
                    W_sb[k * n + i, col] = 1
                    col += 1
        return {'U': U_sb.tolist(), 'V': V_sb.tolist(), 'W': W_sb.tolist()}
    
    best_U = U_combined.copy()
    best_V = V_combined.copy()
    best_W = W_combined.copy()
    best_rank = 21
    
    # Now try to merge redundant terms
    # Two terms r1 and r2 can be merged if U[:,r1] == U[:,r2] and V[:,r1] == V[:,r2]
    # Then we can replace W[:,r1] and W[:,r2] with W[:,r1]+W[:,r2] and remove one term
    
    # Find groups of terms with identical U and V factors
    def find_mergeable_groups(U, V, W):
        """Find groups of terms with identical U and V columns."""
        groups = {}
        for r in range(U.shape[1]):
            u_key = tuple(U[:, r].tolist())
            v_key = tuple(V[:, r].tolist())
            key = (u_key, v_key)
            if key not in groups:
                groups[key] = []
            groups[key].append(r)
        
        # Return groups with size > 1
        mergeable = {k: v for k, v in groups.items() if len(v) > 1}
        return mergeable
    
    # Try to merge
    mergeable = find_mergeable_groups(best_U, best_V, best_W)
    
    for key, group in mergeable.items():
        if len(group) < 2:
            continue
        
        # Merge: sum W factors, keep first term, remove others
        w_sum = np.zeros_like(best_W[:, group[0]])
        for r in group:
            w_sum += best_W[:, r]
        
        # Create new matrices with merged term
        # Remove all terms in group except group[0]
        keep_indices = [r for r in range(best_U.shape[1]) if r not in group[1:]]
        
        new_U = np.column_stack([best_U[:, r] for r in keep_indices])
        new_V = np.column_stack([best_V[:, r] for r in keep_indices])
        new_W = np.column_stack([best_W[:, r] for r in keep_indices])
        
        # Replace the W factor of the first term with the sum
        first_idx_in_new = keep_indices.index(group[0])
        new_W[:, first_idx_in_new] = w_sum
        
        # Verify
        if verify(new_U, new_V, new_W):
            best_U = new_U
            best_V = new_V
            best_W = new_W
            best_rank = new_U.shape[1]
            
            # Try to merge again with the new structure
            mergeable = find_mergeable_groups(best_U, best_V, best_W)
    
    # Random search for further improvements
    random.seed(42)
    np.random.seed(42)
    
    coeff_set = np.array([-1, 0, 1])
    
    # Try to find lower rank decompositions
    for target_rank in range(15, best_rank):
        if time.time() > deadline - 10:
            break
        
        max_restarts = 500
        found = False
        
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break
            
            # Initialize with random factors
            U = np.random.choice(coeff_set, size=(n * m, target_rank))
            V = np.random.choice(coeff_set, size=(m * p, target_rank))
            W = np.random.choice(coeff_set, size=(p * n, target_rank))
            
            if verify(U, V, W):
                best_U = U
                best_V = V
                best_W = W
                best_rank = target_rank
                found = True
                break
        
        if found:
            break
    
    # Convert to list of lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {'U': U_list, 'V': V_list, 'W': W_list}
