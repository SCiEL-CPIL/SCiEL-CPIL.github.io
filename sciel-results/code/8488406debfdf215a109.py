import time
import numpy as np

DIMENSIONS = [3, 5, 5]
DEADLINE = 130.0  # safety margin before 150s kill

def construct():
    start = time.time()
    n, m, p = DIMENSIONS  # 3, 5, 5

    # Known rank-19 decomposition for 5x5 matrix multiplication
    # Using the Bini et al. / Pan decomposition structure
    # For 5x5x5, the minimal rank is 19 (known result)
    
    # The 5x5x5 multiplication tensor has rank 19
    # We'll use a known explicit decomposition
    
    # Rank-19 decomposition for 5x5 matrix multiplication
    # Each term is (u, v, w) where u is 5x5 (input), v is 5x5 (input), w is 5x5 (output)
    # T[i,j,k] = sum_r u_r[i,j] * v_r[j,k] * w_r[k,i]
    
    # Using a structured approach: 4 disjoint 2x2x2 Strassen blocks + 3 extra terms for 5x5
    # Actually, let's use the known rank-19 decomposition
    
    # For 5x5x5, one known decomposition uses:
    # - 4 Strassen 2x2x2 blocks covering positions (0,0), (0,2), (2,0), (2,2) in the 5x5 grid
    # - Plus additional terms to cover the remaining positions
    
    # Let me use a different approach: start with the trivial rank-25 and try to reduce
    
    # Build target tensor
    target = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i*m+j, j*p+k, k*n+i] = 1.0

    # Start with trivial decomposition (rank = n*m*p = 75)
    trivial_rank = n * m * p
    U_trivial = np.zeros((n*m, trivial_rank), dtype=np.int64)
    V_trivial = np.zeros((m*p, trivial_rank), dtype=np.int64)
    W_trivial = np.zeros((p*n, trivial_rank), dtype=np.int64)
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_trivial[i*m+j, c] = 1
                V_trivial[j*p+k, c] = 1
                W_trivial[k*n+i, c] = 1
                c += 1

    # Now try to reduce rank by finding mergeable column pairs
    # A pair of columns (r1, r2) can be merged if their combined tensor is rank-1
    
    def get_col_tensor(U, V, W, r):
        """Return dict mapping (a,b,c) -> value for column r."""
        t = {}
        for a in range(n*m):
            ua = U[a, r]
            if ua == 0:
                continue
            for b in range(m*p):
                vb = V[b, r]
                if vb == 0:
                    continue
                for c_idx in range(p*n):
                    wc = W[c_idx, r]
                    if wc == 0:
                        continue
                    val = ua * vb * wc
                    if val != 0:
                        t[(a, b, c_idx)] = val
        return t

    def is_rank1_tensor(tensor_dict):
        """Check if a tensor dict represents a rank-1 tensor."""
        if not tensor_dict:
            return True, None
        if len(tensor_dict) == 1:
            (a, b, c), val = list(tensor_dict.items())[0]
            return True, (a, b, c, val)
        
        # For rank-1: T[a,b,c] = u[a]*v[b]*w[c]
        # Pick a reference (a0, b0, c0) with non-zero value
        ref_key = list(tensor_dict.keys())[0]
        a0, b0, c0 = ref_key
        ref_val = tensor_dict[ref_key]
        
        # Check consistency: for all (a,b,c), T[a,b,c] / T[a,b0,c0] should be proportional
        # T[a,b,c] = T[a,b0,c0] * T[a0,b,c0] / T[a0,b0,c0] * T[a0,b0,c] / T[a0,b0,c0]
        # Actually: T[a,b,c] = (T[a,b0,c0] * T[a0,b,c0] * T[a0,b0,c]) / (T[a0,b0,c0] * T[a0,b0,c0])
        # Let's verify this formula
        
        # First, collect all unique a, b, c values
        a_vals = set(k[0] for k in tensor_dict.keys())
        b_vals = set(k[1] for k in tensor_dict.keys())
        c_vals = set(k[2] for k in tensor_dict.keys())
        
        # Check if the tensor is consistent with rank-1
        # T[a,b,c] = u[a]*v[b]*w[c]
        # => T[a,b,c] * T[a0,b0,c0] = T[a,b0,c0] * T[a0,b,c0] * T[a0,b0,c] / T[a0,b0,c0]
        # => T[a,b,c] * T[a0,b0,c0]^2 = T[a,b0,c0] * T[a0,b,c0] * T[a0,b0,c]
        
        # This is complex. Let's use a simpler check:
        # If we fix a and b, T[a,b,c] should be proportional to w[c]
        # If we fix a and c, T[a,b,c] should be proportional to v[b]
        # If we fix b and c, T[a,b,c] should be proportional to u[a]
        
        # Pick two reference points
        keys = list(tensor_dict.keys())
        if len(keys) < 2:
            return True, tensor_dict
        
        # Try to determine u, v, w from the data
        # Set u[a0] = 1, then v[b] = T[a0,b,c0] / w[c0]
        # This is getting complex. Let's use a numerical approach.
        
        # Build the full tensor array
        max_a = max(a_vals) + 1 if a_vals else 1
        max_b = max(b_vals) + 1 if b_vals else 1
        max_c = max(c_vals) + 1 if c_vals else 1
        
        T = np.zeros((max_a, max_b, max_c), dtype=np.float64)
        for (a, b, c), val in tensor_dict.items():
            T[a, b, c] = val
        
        # Check rank-1 by computing the SVD of matricizations
        # Matricize along first mode: shape (max_a, max_b*max_c)
        T1 = T.reshape(max_a, -1)
        if np.linalg.matrix_rank(T1) > 1:
            return False, None
        
        T2 = T.transpose(1, 0, 2).reshape(max_b, -1)
        if np.linalg.matrix_rank(T2) > 1:
            return False, None
        
        T3 = T.transpose(2, 0, 1).reshape(max_c, -1)
        if np.linalg.matrix_rank(T3) > 1:
            return False, None
        
        return True, tensor_dict

    def try_merge(U, V, W, rank, r1, r2):
        """Try to merge columns r1 and r2. Returns (new_U, new_V, new_W, new_rank) or None."""
        t1 = get_col_tensor(U, V, W, r1)
        t2 = get_col_tensor(U, V, W, r2)
        
        combined = {}
        for key, val in t1.items():
            combined[key] = val
        for key, val in t2.items():
            combined[key] = combined.get(key, 0) + val
        
        combined = {k: v for k, v in combined.items() if v != 0}
        
        is_r1, _ = is_rank1_tensor(combined)
        if not is_r1:
            return None
        
        # Create new column
        new_U = U.copy()
        new_V = V.copy()
        new_W = W.copy()
        
        # Remove columns r1 and r2, add new column at position rank-1
        new_rank = rank - 1
        
        if not combined:
            # Both cancel, just remove both
            cols_to_keep = [c for c in range(rank) if c not in (r1, r2)]
            new_U2 = np.zeros((n*m, new_rank), dtype=np.int64)
            new_V2 = np.zeros((m*p, new_rank), dtype=np.int64)
            new_W2 = np.zeros((p*n, new_rank), dtype=np.int64)
            for new_c, old_c in enumerate(cols_to_keep):
                new_U2[:, new_c] = U[:, old_c]
                new_V2[:, new_c] = V[:, old_c]
                new_W2[:, new_c] = W[:, old_c]
            return new_U2, new_V2, new_W2, new_rank
        
        # Find u, v, w for the combined tensor
        # For each (a,b,c) in combined, we need u[a]*v[b]*w[c] = combined[(a,b,c)]
        
        # Pick a reference (a0, b0, c0)
        ref_key = list(combined.keys())[0]
        a0, b0, c0 = ref_key
        ref_val = combined[ref_key]
        
        # Set u[a0] = ref_val, v[b0] = 1, w[c0] = 1
        # Then for other entries: u[a]*v[b]*w[c] = combined[(a,b,c)]
        
        # This is underdetermined. Let's use a simpler approach:
        # If the tensor is rank-1, we can extract u, v, w via SVD
        
        max_a = max(k[0] for k in combined.keys()) + 1
        max_b = max(k[1] for k in combined.keys()) + 1
        max_c = max(k[2] for k in combined.keys()) + 1
        
        T = np.zeros((max_a, max_b, max_c), dtype=np.float64)
        for (a, b, c), val in combined.items():
            T[a, b, c] = val
        
        # For rank-1 tensor, T = u ⊗ v ⊗ w
        # Use the fact that any slice is rank-1
        
        # Extract u, v, w
        # Fix b=b0, c=c0: T[:, b0, c0] = u * v[b0] * w[c0]
        # So u is proportional to T[:, b0, c0]
        
        # Let's just assign: for each (a,b,c), set u[a] = combined[(a,b,c)] / (v[b] * w[c])
        # This is circular. Let's use a different approach.
        
        # Since we verified it's rank-1, we can use:
        # u[a] = T[a, b0, c0] / (v[b0] * w[c0])
        # v[b] = T[a0, b, c0] / (u[a0] * w[c0])
        # w[c] = T[a0, b0, c] / (u[a0] * v[b0])
        
        # Set u[a0] = 1, v[b0] = 1, w[c0] = ref_val
        u = np.zeros(max_a, dtype=np.float64)
        v = np.zeros(max_b, dtype=np.float64)
        w = np.zeros(max_c, dtype=np.float64)
        u[a0] = 1.0
        v[b0] = 1.0
        w[c0] = ref_val
        
        # Now fill in the rest
        for (a, b, c), val in combined.items():
            if a == a0 and b == b0:
                w[c] = val / (u[a0] * v[b0])
            elif a == a0 and c == c0:
                v[b] = val / (u[a0] * w[c0])
            elif b == b0 and c == c0:
                u[a] = val / (v[b0] * w[c0])
        
        # Now we have u, v, w. Convert to integers if possible
        # For now, use float values and round
        
        # Assign to new column
        col_idx = new_rank - 1
        for a in range(max_a):
            if u[a] != 0:
                new_U[a, col_idx] = int(round(u[a]))
        for b in range(max_b):
            if v[b] != 0:
                new_V[b, col_idx] = int(round(v[b]))
        for c in range(max_c):
            if w[c] != 0:
                new_W[c, col_idx] = int(round(w[c]))
        
        # Remove old columns r1, r2
        cols_to_keep = [c for c in range(rank) if c not in (r1, r2)]
        new_U2 = np.zeros((n*m, new_rank), dtype=np.int64)
        new_V2 = np.zeros((m*p, new_rank), dtype=np.int64)
        new_W2 = np.zeros((p*n, new_rank), dtype=np.int64)
        for new_c, old_c in enumerate(cols_to_keep):
            new_U2[:, new_c] = U[:, old_c]
            new_V2[:, new_c] = V[:, old_c]
            new_W2[:, new_c] = W[:, old_c]
        # Add new column
        new_U2[:, new_rank - 1] = new_U[:, col_idx]
        new_V2[:, new_rank - 1] = new_V[:, col_idx]
        new_W2[:, new_rank - 1] = new_W[:, col_idx]
        
        return new_U2, new_V2, new_W2, new_rank

    # Greedy rank reduction
    U_curr = U_trivial.copy()
    V_curr = V_trivial.copy()
    W_curr = W_trivial.copy()
    curr_rank = trivial_rank
    
    best_U, best_V, best_W = U_curr.copy(), V_curr.copy(), W_curr.copy()
    best_rank = curr_rank
    
    # Try all pairs of columns
    for _ in range(20):  # Max 20 rounds of merging
        if time.time() - start > DEADLINE - 10:
            break
        
        merged_any = False
        for r1 in range(curr_rank):
            if time.time() - start > DEADLINE - 10:
                break
            for r2 in range(r1 + 1, curr_rank):
                result = try_merge(U_curr, V_curr, W_curr, curr_rank, r1, r2)
                if result is not None:
                    new_U, new_V, new_W, new_rank = result
                    # Verify
                    T_test = np.einsum('ar,br,cr->abc', new_U.astype(np.float64),
                                       new_V.astype(np.float64), new_W.astype(np.float64))
                    if np.allclose(T_test, target, atol=1e-10):
                        U_curr, V_curr, W_curr = new_U, new_V, new_W
                        curr_rank = new_rank
                        merged_any = True
                        if curr_rank < best_rank:
                            best_rank = curr_rank
                            best_U, best_V, best_W = new_U.copy(), new_V.copy(), new_W.copy()
                        break
            if merged_any:
                break
        
        if not merged_any:
            break

    if time.time() - start > DEADLINE:
        return {"U": U_trivial.tolist(), "V": V_trivial.tolist(), "W": W_trivial.tolist()}

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
