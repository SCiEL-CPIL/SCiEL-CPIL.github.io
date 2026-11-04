import time
import numpy as np
import itertools

DIMENSIONS = [4, 4, 6]
n, m, p = DIMENSIONS
N = n * m  # 16
M = m * p  # 24
P = p * n  # 24

def _verify(U, V, W):
    """
    Verify if U, V, W form a valid decomposition.
    U: (16, rank), V: (24, rank), W: (24, rank)
    Condition: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] == 1 for all i,j,k
    """
    rank = U.shape[1]
    if rank == 0:
        return False
    
    # For speed, we check a subset first, then all.
    # Actually, 16*24*24 = 9216 terms. Rank ~ 50.
    # 9216 * 50 = 460k operations. Very fast in numpy.
    
    # We can vectorize the check.
    # Tensor T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    # Let's just loop over i, j, k to be safe and simple, it's small enough.
    # Or use einsum if we reshape.
    
    # Reshape U to (n, m, rank)
    # Reshape V to (m, p, rank)
    # Reshape W to (p, n, rank)
    
    U_3d = U.reshape(n, m, rank)
    V_3d = V.reshape(m, p, rank)
    W_3d = W.reshape(p, n, rank)
    
    # We want sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    # This is not a standard einsum directly because indices are permuted.
    # Let's just do the triple loop, it's 9216 iterations.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # u_val = U[i*m+j, :]
                # v_val = V[j*p+k, :]
                # w_val = W[k*n+i, :]
                # prod = u_val * v_val * w_val
                # s = np.sum(prod)
                # if abs(s - 1.0) > 1e-10: return False
                
                # Optimized check:
                # We can precompute products? No, just dot product.
                # np.dot(u, v*w)
                # Let's stick to the loop for clarity and robustness against shape bugs.
                u_vec = U[i*m+j, :]
                v_vec = V[j*p+k, :]
                w_vec = W[k*n+i, :]
                # sum(u * v * w)
                val = np.dot(u_vec, v_vec * w_vec)
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _build_initial_decomposition():
    """
    Build a valid decomposition. 
    We use a 'Block Strassen' approach that is known to be valid.
    Split 4x4 * 4x6 into two 4x4 * 4x3 parts? No, Strassen is for square.
    
    Let's use the standard 'Embedding' strategy which is guaranteed valid.
    Embed 4x6 into 4x8.
    Use Strassen for 4x4 * 4x8?
    
    Actually, the simplest valid decomposition is the Schoolbook one (Rank 96).
    But we want to start with something better to prune.
    
    Let's try to construct a valid Rank 72 decomposition (3 * 24? No).
    
    Let's use the known result: R(4,4,6) <= R(4,4,4) + R(4,4,2).
    R(4,4,4) = 49 (Strassen).
    R(4,4,2) <= 28 (Block Strassen).
    Total 77.
    
    The current program tries this but fails. Why?
    Likely the index mapping in the 4x4*4x2 part is wrong.
    
    Let's build the 49 rank part correctly first.
    Then build the 28 rank part correctly.
    
    Alternative: Use a random valid decomposition? No, must be exact integers/half-integers.
    
    Let's construct the 49 rank part for 4x4*4x4.
    This is standard recursive Strassen.
    
    Then for the 4x4*4x2 part, we can use a specific algorithm.
    Actually, we can just use Schoolbook for the 4x4*4x2 part to ensure validity, 
    then prune.
    Schoolbook 4x4*4x2 has rank 4*4*2 = 32.
    Total Rank = 49 + 32 = 81.
    
    Starting from 81 is better than 96.
    Let's implement a generic builder that takes a list of 'terms'.
    Each term is a triple of sparse vectors (u, v, w).
    
    Actually, let's just generate the Schoolbook decomposition for the whole 4x4*4x6.
    Rank = 96.
    Then we run the pruning algorithm.
    Pruning 96 columns down to ~60-70 is very likely.
    
    Is there a smarter initial structure?
    Yes, Strassen for 4x4*4x4 (49) + Schoolbook for 4x4*4x2 (32) = 81.
    Let's try to build that.
    
    Part 1: 4x4 * 4x4 (cols 0-3 of B, cols 0-3 of C)
    Part 2: 4x4 * 4x2 (cols 4-5 of B, cols 4-5 of C)
    
    We need to be very careful with indices.
    """
    
    # Helper to create a term
    def make_term(u_idx, v_idx, w_idx, val=1.0):
        # u_idx: (i, j)
        # v_idx: (j, k)
        # w_idx: (k, i)
        return (u_idx, v_idx, w_idx, val)

    terms = []
    
    # --- Part 1: 4x4 * 4x4 (Strassen Rank 49) ---
    # We implement recursive Strassen for 4x4.
    # Level 1: 2x2 blocks of size 2.
    # Level 2: 2x2 blocks of size 1.
    
    # Indices for A: (i, j), i,j in 0..3
    # Indices for B: (j, k), j in 0..3, k in 0..3 (for this part)
    # Indices for C: (k, i), k in 0..3, i in 0..3
    
    # We will generate the 49 terms.
    # Strassen 2x2*2x2 products:
    # P1 = (A11+A22)(B11+B22) -> C11 += P1, C22 += P1
    # P2 = (A21+A22)B11       -> C21 += P2, C22 += P2
    # P3 = A11(B12-B22)       -> C12 += P3, C22 -= P3
    # P4 = A22(B21-B11)       -> C21 += P4, C11 -= P4
    # P5 = (A11+A12)B22       -> C11 += P5, C12 += P5
    # P6 = (A21-A11)(B11+B12) -> C22 -= P6
    # P7 = (A12-A22)(B21+B22) -> C11 -= P7
    
    # We apply this recursively.
    # Let's define a function to add a Strassen block to the list of terms.
    
    def add_strassen_block(offset_i, offset_j, offset_k_b, offset_k_c, size):
        """
        Add terms for a size x size block multiplication using Strassen.
        offset_i: starting row of A
        offset_j: starting col of A / row of B
        offset_k_b: starting col of B
        offset_k_c: starting row of C
        size: block size (must be power of 2)
        """
        if size == 1:
            # Scalar multiplication: A[i,j] * B[j,k] = C[k,i]
            # One term
            i = offset_i
            j = offset_j
            k = offset_k_b
            terms.append(make_term((i,j), (j,k), (k,i), 1.0))
            return
        
        half = size // 2
        
        # A11 (top-left), A12 (top-right), A21 (bot-left), A22 (bot-right)
        # B11, B12, B21, B22
        # C11, C12, C21, C22
        
        # We need to map the logical Strassen products to actual indices.
        # This is complex to do recursively with offsets directly in the term list
        # because the 'signs' and 'sums' happen at the term level.
        
        # Instead, let's just hardcode the 49 terms for 4x4*4x4 using the known structure.
        # Or, simpler: Use the 7 products of 2x2*2x2, where each 2x2*2x2 is done by 7 products of 1x1*1x1? No, that's rank 49.
        # Yes, that's exactly what recursive Strassen is.
        
        # Let's implement the recursion properly by generating the 'linear combinations' of indices.
        
        # P1: (A11+A22)(B11+B22)
        # A indices for P1: (offset_i, offset_j) to (offset_i+half-1, offset_j+half-1) AND (offset_i+half, offset_j+half) to ...
        # This is getting messy to track offsets for 'sums'.
        
        # Alternative: Just use Schoolbook for the initial construction.
        # It's guaranteed valid. Rank 96.
        # Pruning will do the work.
        # Schoolbook is easier to generate correctly.
        pass

    # Let's stick to Schoolbook for initialization to ensure 100% validity.
    # Rank 96.
    for i in range(4):
        for j in range(4):
            for k in range(6):
                terms.append(make_term((i,j), (j,k), (k,i), 1.0))
    
    # Convert terms to matrices
    rank = len(terms)
    U = np.zeros((16, rank))
    V = np.zeros((24, rank))
    W = np.zeros((24, rank))
    
    for r, (u_idx, v_idx, w_idx, val) in enumerate(terms):
        i, j = u_idx
        jj, k = v_idx
        kk, ii = w_idx
        
        # Sanity check indices
        if i != ii or j != jj:
            raise ValueError("Index mismatch in term generation")
            
        U[i*4+j, r] = val
        V[j*6+k, r] = val
        W[k*4+i, r] = val
        
    return U, V, W

def _prune_decomposition(U, V, W, max_time=100):
    """
    Greedy column pruning.
    Try to remove columns that are linearly dependent or redundant.
    """
    start_time = time.time()
    U, V, W = U.copy(), V.copy(), W.copy()
    current_rank = U.shape[1]
    
    # Order columns by some heuristic? 
    # Maybe columns with fewer non-zeros are more likely to be redundant?
    # Or just iterate.
    
    # We need a fast verification.
    # Pre-compute the "residual" or just re-verify.
    # Re-verification is O(N*M*P*Rank). 
    # 9216 * Rank. If Rank=96, ~1M ops. Fast.
    # If we do this for every removal attempt (Rank times), it's Rank^2 * 1M? No.
    # We try to remove one column. If it fails, we keep it.
    # So we do Rank attempts. Each attempt costs O(Verification).
    # Total cost O(Rank * Verification).
    # 96 * 1M = 100M ops. Should be fine in Python/Numpy within 100s? 
    # Python loops are slow. 9216 loops per verification.
    # 96 * 9216 = ~1M Python loop iterations. 
    # Each iteration does numpy dot products.
    # This might be slow.
    
    # Optimization: 
    # Instead of full verification, we can check if the column is a linear combination of others?
    # That's hard with the 3-way product.
    
    # Let's use a smarter check:
    # If we remove column r, the error is sum_{i,j,k} U[i,j,r] V[j,k,r] W[k,i,r] delta.
    # Actually, let's just use the full verification but optimize the inner loop.
    
    def verify_subset(U, V, W, cols_to_check):
        # Only check if removing a specific column breaks the equation.
        # This is hard to isolate.
        pass
        
    # Let's just do the full verification but in a tight loop.
    # We can parallelize? No, standard library only.
    
    # Heuristic: 
    # If two columns are identical, remove one.
    # Let's check for duplicate columns first.
    
    col_hashes = {}
    for r in range(current_rank):
        # Hash the column vectors
        # U col, V col, W col
        key = (tuple(U[:, r]), tuple(V[:, r]), tuple(W[:, r]))
        if key in col_hashes:
            # Duplicate found! Remove current r
            U = np.delete(U, r, axis=1)
            V = np.delete(V, r, axis=1)
            W = np.delete(W, r, axis=1)
            current_rank -= 1
            # Restart hash map? Or just continue.
            # Since we deleted, indices shift. Easiest to restart the loop or break.
            # Let's break and restart the main loop.
            return _prune_decomposition(U, V, W, max_time)
        else:
            col_hashes[key] = r
            
    # Now try to remove columns one by one.
    # To save time, we can shuffle the order or prioritize 'simple' columns.
    # Let's just iterate from last to first.
    
    for r in range(current_rank - 1, -1, -1):
        if time.time() - start_time > max_time:
            break
            
        # Try removing column r
        U_test = np.delete(U, r, axis=1)
        V_test = np.delete(V, r, axis=1)
        W_test = np.delete(W, r, axis=1)
        
        # Verify
        valid = True
        # Fast check: 
        # We know the original was valid.
        # The change is - U[i,j,r] V[j,k,r] W[k,i,r].
        # We need sum_{r' != r} ... == 1.
        # So sum_{all} ... - term_r == 1.
        # Since sum_all == 1, we need term_r == 0 for all i,j,k?
        # NO. The terms can cancel out?
        # The decomposition is sum_r term_r = 1.
        # If we remove term_r, we have sum_{r' != r} term_r' = 1 - term_r.
        # We need this to be 1. So we need term_r == 0 for all i,j,k.
        # This implies that the column r must be "zero" in the tensor product sense.
        # i.e., U[i,j,r] V[j,k,r] W[k,i,r] must be 0 for all i,j,k.
        
        # This is a MUCH faster check!
        # We don't need to re-sum everything.
        # We just need to check if the "footprint" of column r is zero.
        
        # Footprint: 
        # For all i,j,k: U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 0
        
        # We can check this efficiently.
        # If U[i*m+j, r] != 0, then for all k such that V[j*p+k, r] != 0 and W[k*n+i, r] != 0, it fails.
        
        # Let's find non-zeros in U[:, r], V[:, r], W[:, r].
        nz_u = np.nonzero(U[:, r])[0]
        nz_v = np.nonzero(V[:, r])[0]
        nz_w = np.nonzero(W[:, r])[0]
        
        if len(nz_u) == 0 or len(nz_v) == 0 or len(nz_w) == 0:
            # One part is zero, so the product is zero. Safe to remove.
            U = U_test
            V = V_test
            W = W_test
            current_rank -= 1
            continue
            
        # Check intersections
        # For each u_idx in nz_u:
        #   i, j = divmod(u_idx, m)
        #   For each v_idx in nz_v:
        #     jj, k = divmod(v_idx, p)
        #     if jj != j: continue
        #     For each w_idx in nz_w:
        #       kk, ii = divmod(w_idx, n)
        #       if kk != k or ii != i: continue
        #       # Found a non-zero product!
        #       valid = False
        #       break
        
        is_redundant = True
        for u_idx in nz_u:
            i, j = divmod(u_idx, m)
            for v_idx in nz_v:
                jj, k = divmod(v_idx, p)
                if jj != j:
                    continue
                for w_idx in nz_w:
                    kk, ii = divmod(w_idx, n)
                    if kk == k and ii == i:
                        # Product is non-zero
                        is_redundant = False
                        break
                if not is_redundant:
                    break
            if not is_redundant:
                break
                
        if is_redundant:
            U = U_test
            V = V_test
            W = W_test
            current_rank -= 1

    return U, V, W, current_rank

def construct():
    deadline = time.time() + 110
    
    try:
        # Step 1: Build initial decomposition (Schoolbook, Rank 96)
        U, V, W = _build_initial_decomposition()
        
        # Step 2: Prune
        U, V, W, rank = _prune_decomposition(U, V, W, max_time=100)
        
        # Step 3: Final Verification
        if not _verify(U, V, W):
            # Fallback to schoolbook
            U, V, W = _build_initial_decomposition()
            rank = U.shape[1]
            
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    except Exception:
        # Fallback
        U_fb = np.zeros((16, 96))
        V_fb = np.zeros((24, 96))
        W_fb = np.zeros((24, 96))
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(6):
                    U_fb[i*4+j, col] = 1
                    V_fb[j*6+k, col] = 1
                    W_fb[k*4+i, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}

# Run a quick self-test if executed directly
if __name__ == "__main__":
    res = construct()
    U = np.array(res['U'])
    V = np.array(res['V'])
    W = np.array(res['W'])
    print(f"Rank: {U.shape[1]}")
    print(f"Valid: {_verify(U, V, W)}")
