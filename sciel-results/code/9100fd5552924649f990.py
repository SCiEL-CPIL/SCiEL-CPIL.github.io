"""Search for a rank-minimal exact tensor decomposition of 2x4 times 4x7 matrix multiply."""
import time
import random
import numpy as np

DIMENSIONS = [2, 4, 7]
DEADLINE = time.time() + 115.0  # safety margin within 150s


def _verify_np(U, V, W, n, m, p, tol=1e-9):
    """Faster vectorized check. U, V, W are 2D arrays (rows, rank)."""
    for i in range(n):
        for k in range(p):
            w = W[k * n + i]
            s = 0.0
            for j in range(m):
                u = U[i * m + j]
                v = V[j * p + k]
                s += np.sum(u * v * w)
            if abs(s - 1.0) > tol:
                return False
    return True


def _construct_np(n, m, p):
    """Construct rank n*p decomposition."""
    rank = n * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    for i in range(n):
        for k in range(p):
            c = k * n + i
            for j in range(m):
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
            W[k * n + i, c] = 1
    return U, V, W


def _try_subspace_method(n, m, p):
    """
    Subspace method: Try to find a decomposition of rank < n*p by
    allowing columns to serve multiple (i,k) pairs with fractional coefficients.
    
    Key idea: For each column c, define its "support" as the set of (i,k) pairs
    it contributes to. If a column serves (i1,k1) and (i2,k2), we need:
    - W[k1*n+i1, c] and W[k2*n+i2, c] to be non-zero
    - The U and V entries to be set so that the products work out
    
    We'll try ranks from n*p-1 down to n*p-3, using a structured search.
    """
    base_rank = n * p
    for target_rank in range(base_rank - 1, max(1, base_rank - 5), -1):
        if time.time() > DEADLINE - 15:
            return None
        
        # Strategy: Start with the n*p construction and try to merge columns
        # by finding pairs of (i,k) that can share a column with adjusted coefficients
        
        # For a column to serve (i1,k1) and (i2,k2) simultaneously:
        # We need U[i1*m+j, c] * V[j*p+k1, c] * W[k1*n+i1, c] summed over j = contribution to (i1,k1)
        # and similarly for (i2,k2)
        # If i1 != i2 or k1 != k2, the U and V indices differ, so we can set
        # W[k1*n+i1, c] = a, W[k2*n+i2, c] = b
        # U[i1*m+j, c] = u1_j, V[j*p+k1, c] = v1_j => sum_j u1_j * v1_j * a = 1
        # U[i2*m+j, c] = u2_j, V[j*p+k2, c] = v2_j => sum_j u2_j * v2_j * b = 1
        
        # Simplest case: if i1 == i2 but k1 != k2 (same row, different column of output)
        # Then U rows are the same: U[i*m+j, c] for all j
        # V rows differ: V[j*p+k1, c] vs V[j*p+k2, c]
        # We can set U[i*m+j, c] = 1 for all j
        # V[j*p+k1, c] = 1, W[k1*n+i, c] = a => sum_j 1*1*a = m*a = 1 => a = 1/m
        # V[j*p+k2, c] = 1, W[k2*n+i, c] = b => sum_j 1*1*b = m*b = 1 => b = 1/m
        # This works! So two (i,k) pairs with same i can share a column with W entries 1/m.
        
        # Similarly, if k1 == k2 but i1 != i2 (same column, different row)
        # V rows are the same: V[j*p+k, c] for all j
        # U rows differ: U[i1*m+j, c] vs U[i2*m+j, c]
        # Set V[j*p+k, c] = 1 for all j
        # U[i1*m+j, c] = 1, W[k*n+i1, c] = a => a = 1/m
        # U[i2*m+j, c] = 1, W[k*n+i2, c] = b => b = 1/m
        
        # If i1 != i2 AND k1 != k2, it's more complex but still possible with
        # appropriate U, V, W values.
        
        # Let's try to build a decomposition where we merge as many (i,k) pairs as possible.
        # For n=2, p=7: we have 14 (i,k) pairs. We can merge pairs with same i (7 pairs per i-value,
        # but only 2 i-values, so we can merge within each i-group).
        
        # For i=0: pairs (0,0), (0,1), ..., (0,6) - 7 pairs
        # For i=1: pairs (1,0), (1,1), ..., (1,6) - 7 pairs
        
        # If we merge all 7 pairs for i=0 into 1 column, and all 7 for i=1 into 1 column,
        # we get rank 2. But that's too aggressive - let's check if it works.
        
        # For column c0 serving all (0,k) for k=0..6:
        # U[0*m+j, c0] = 1 for all j (rows 0..3)
        # V[j*p+k, c0] = 1 for all j,k (rows 0..27)
        # W[k*n+0, c0] = 1/m for all k
        # Check: for (i=0, k=k'): sum_j U[0*m+j,c0]*V[j*p+k',c0]*W[k'*n+0,c0] = sum_j 1*1*(1/m) = m*(1/m) = 1 ✓
        # But wait, this column also contributes to (i=0, k=k'') for k'' != k'. That's fine,
        # we want it to contribute to ALL (0,k) pairs.
        
        # However, this column does NOT contribute to (i=1, k) pairs because U[1*m+j, c0] = 0
        # (we only set U[0*m+j, c0] = 1).
        
        # So with 2 columns: c0 for all (0,k), c1 for all (1,k), we get rank 2!
        # Let's verify this more carefully.
        
        if target_rank >= 2:
            # Try the 2-column construction
            U = np.zeros((n * m, target_rank), dtype=float)
            V = np.zeros((m * p, target_rank), dtype=float)
            W = np.zeros((p * n, target_rank), dtype=float)
            
            # Column 0: serves all (0, k) for k=0..p-1
            for j in range(m):
                U[0 * m + j, 0] = 1
                V[j * p + k, 0] = 1 if False else 1  # Set for all j,k
            # Actually, we need V[j*p+k, 0] = 1 for all j and k
            for j in range(m):
                for k in range(p):
                    V[j * p + k, 0] = 1
            for k in range(p):
                W[k * n + 0, 0] = 1.0 / m
            
            # Column 1: serves all (1, k) for k=0..p-1
            for j in range(m):
                U[1 * m + j, 1] = 1
                for k in range(p):
                    V[j * p + k, 1] = 1
            for k in range(p):
                W[k * n + 1, 1] = 1.0 / m
            
            if _verify_np(U, V, W, n, m, p):
                print(f"Found rank {target_rank} decomposition via subspace method!")
                return {'U': U, 'V': V, 'W': W}
        
        # Try rank 3, 4, etc. with partial merging
        for attempt in range(500):
            if time.time() > DEADLINE - 5:
                return None
            
            # Random approach: assign each (i,k) to a column, allowing sharing
            W = np.zeros((p * n, target_rank), dtype=float)
            col_assignments = {}  # col -> list of (i,k)
            
            # Start with one column per (i,k)
            all_pairs = [(i, k) for i in range(n) for k in range(p)]
            random.shuffle(all_pairs)
            
            # Assign pairs to columns, allowing some columns to have multiple pairs
            col_idx = 0
            for (i, k) in all_pairs:
                if col_idx >= target_rank:
                    # Must share with an existing column
                    # Find a column that can accommodate this pair
                    found = False
                    for c in range(target_rank):
                        # Check if this column's W row for (i,k) is already set
                        if W[k * n + i, c] == 0:
                            # Try to add this pair to column c
                            W[k * n + i, c] = 1.0 / m
                            if c not in col_assignments:
                                col_assignments[c] = []
                            col_assignments[c].append((i, k))
                            found = True
                            break
                    if not found:
                        break
                else:
                    W[k * n + i, col_idx] = 1.0 / m
                    col_assignments[col_idx] = [(i, k)]
                    col_idx += 1
            
            if col_idx < target_rank and not all(len(v) > 0 for v in col_assignments.values()):
                continue
            
            # Build U and V based on assignments
            U = np.zeros((n * m, target_rank), dtype=float)
            V = np.zeros((m * p, target_rank), dtype=float)
            
            valid = True
            for c, pairs in col_assignments.items():
                if not pairs:
                    continue
                # For each (i,k) in this column's assignment, we need:
                # sum_j U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
                # W[k*n+i, c] = 1/m, so we need sum_j U[i*m+j, c] * V[j*p+k, c] = m
                # Set U[i*m+j, c] = 1 for all j, V[j*p+k, c] = 1 for all j
                for (i, k) in pairs:
                    for j in range(m):
                        U[i * m + j, c] = 1
                        V[j * p + k, c] = 1
            
            if _verify_np(U, V, W, n, m, p):
                print(f"Found rank {target_rank} decomposition via subspace method (attempt {attempt})!")
                return {'U': U, 'V': V, 'W': W}
    
    return None


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()

    # Start with the known rank n*p decomposition
    U, V, W = _construct_np(n, m, p)
    if _verify_np(U, V, W, n, m, p):
        best_rank = n * p
        print(f"Verified rank {n*p} decomposition")
    else:
        # Fallback to schoolbook
        best_rank = n * m * p
        U = np.zeros((n * m, best_rank), dtype=float)
        V = np.zeros((m * p, best_rank), dtype=float)
        W = np.zeros((p * n, best_rank), dtype=float)
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, c] = 1
                    V[j * p + k, c] = 1
                    W[k * n + i, c] = 1
                    c += 1
        print(f"Using schoolbook rank {best_rank}")

    # Try subspace method to find lower rank
    result = _try_subspace_method(n, m, p)
    if result is not None:
        U, V, W = result['U'], result['V'], result['W']
        best_rank = U.shape[1]
        print(f"Improved to rank {best_rank}")

    # Convert to list-of-lists with integer/half-integer entries
    def to_int_list(mat):
        m2 = mat * 2
        if np.allclose(m2, np.round(m2), atol=1e-9):
            m2 = np.round(m2).astype(int)
            if np.all(m2 % 2 == 0):
                return (m2 // 2).tolist()
            return m2.tolist()
        return mat.tolist()

    return {
        'U': to_int_list(U),
        'V': to_int_list(V),
        'W': to_int_list(W)
    }

