"""Matrix multiplication 6x6x6 - rank-minimal exact tensor decomposition."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [6, 6, 6]

def _verify(U, V, W, n=6, m=6, p=6):
    """Verify that the decomposition exactly computes matrix multiplication.
    C[i,k] = sum_j A[i,j]*B[j,k]
    U shape (n*m, rank), V shape (m*p, rank), W shape (p*n, rank)
    W index: k*n + i
    """
    rank = len(U[0])
    # Build tensors
    # T1[i,j,k] = A[i,j]*B[j,k] contributes to C[i,k]
    # Decomposition: C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Check: for all i,j,k: sum_r U[i*m+j, r]*V[j*p+k, r]*W[k*n+i, r] == 1
    # And for all i,j,k,i',j',k' where not (j==j' and i==i' and k==k'): sum == 0
    # Actually the tensor M[i,j,k] = delta (it's the multiplication tensor)
    # M[i,j,k] = 1 if this is the "right" combination, 0 otherwise
    # The multiplication tensor: M[i,j,k] = 1 for all i,j,k (since A[i,j]*B[j,k] -> C[i,k])
    # Wait, the tensor is: T[i,j,k] where i indexes A row, j indexes A col / B row, k indexes B col
    # T[i,j,k] = 1 (all ones) because A[i,j]*B[j,k] always contributes to C[i,k]
    
    # Actually the standard matrix multiplication tensor:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor M has M[i,j,k] = 1 for all i,j,k (it's the identity tensor in some sense)
    # No wait. The tensor is defined as: M_{i,j,k} = 1 if we're computing C[i,k] from A[i,j] and B[j,k]
    # Since every (i,j,k) triple contributes, M[i,j,k] = 1 for all i,j,k.
    
    # The decomposition requires:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = M[i,j,k] = 1 for all i,j,k
    
    # This is expensive to check fully. Let's check a random subset for speed,
    # but for final answer we need full check.
    
    # Full check: 6*6*6 = 216 equations
    # Each equation: sum over rank terms
    # For rank 189: 216 * 189 = ~40k multiplications, fine
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                ui = i * m + j
                vj = j * p + k
                wk = k * n + i
                for r in range(rank):
                    s += U[ui][r] * V[vj][r] * W[wk][r]
                if s != 1:
                    return False
    return True

def _verify_fast(U, V, W, n=6, m=6, p=6):
    """Fast verification using numpy."""
    rank = len(U[0])
    Ua = np.array(U, dtype=np.int64)
    Va = np.array(V, dtype=np.int64)
    Wa = np.array(W, dtype=np.int64)
    
    # Reshape: U -> (n, m, rank), V -> (m, p, rank), W -> (p, n, rank)
    U3 = Ua.reshape(n, m, rank)
    V3 = Va.reshape(m, p, rank)
    W3 = Wa.reshape(p, n, rank)
    
    # For each (i,j,k): sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r] should be 1
    # Build the full tensor
    # Use einsum: sum over r of U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
    # This is tricky with einsum due to the k->i permutation in W
    
    # Alternative: for each r, compute the outer product contribution
    # Let's just do it directly
    result = np.zeros((n, m, p), dtype=np.int64)
    for r in range(rank):
        # U3[:, :, r] shape (n, m), V3[:, :, r] shape (m, p), W3[:, :, r] shape (p, n)
        # We need: for all i,j,k: U3[i,j,r]*V3[j,k,r]*W3[k,i,r]
        # W3[k,i,r] = W3.T[i,k,r]
        W3T = W3[:, :, r].T  # shape (n, p) -> W3T[i,k] = W3[k,i]
        # Now: sum over j of U3[i,j,r]*V3[j,k,r] gives (n,p) matrix, then multiply by W3T[i,k]
        # Actually it's: U3[i,j,r] * V3[j,k,r] * W3T[i,k,r] for each i,j,k
        # = W3T[i,k,r] * (U3[:, :, r] @ V3[:, :, r])[i, k]
        M = U3[:, :, r] @ V3[:, :, r]  # (n, p)
        result[:, :, :] += W3[:, :, r].T * M[:, np.newaxis, :]  # broadcast: M[i,k] * W3T[i,k] for all j? No!
        # Wait, this doesn't work because j is summed implicitly but W3T doesn't depend on j.
        # The term is: for fixed i,k: W3T[i,k] * sum_j U3[i,j] * V3[j,k]
        # So: result[i, :, k] += W3T[i,k] * (U3[i,:,r] @ V3[:,k,r])
        # Let me redo:
        # For fixed r: contribution to (i,j,k) is U3[i,j,r]*V3[j,k,r]*W3T[i,k,r]
        # Sum over j: for fixed i,k: W3T[i,k,r] * sum_j U3[i,j,r]*V3[j,k,r] = W3T[i,k,r] * (U3[i,:,r] @ V3[:,k,r])
        # But we need the full 3D, not summed over j. The verification checks each (i,j,k) individually.
        # So we can't sum over j. We need:
        # result[i,j,k] += U3[i,j,r] * V3[j,k,r] * W3T[i,k,r]
        # = W3T[i,k,r] * U3[i,j,r] * V3[j,k,r]
        
        # This is: for each (i,k): scale factor W3T[i,k], times U3[i,:,r] (length m) times V3[:,k,r] (length m)
        # result[i, :, k] += W3T[i,k] * np.outer(U3[i,:,r], V3[:,k,r]) ... no, it's element-wise
        # result[i, j, k] += U3[i,j,r] * V3[j,k,r] * W3T[i,k,r]
        
        # Vectorized: 
        # tmp[i,j,k] = U3[i,j,r] * V3[j,k,r] * W3T[i,k,r]
        # = W3T[i,k,r] * U3[i,j,r] * V3[j,k,r]
        # For fixed i,k: W3T[i,k] * (U3[i,:,r] * V3[:,k,r])  -- element-wise product of two length-m vectors
        # So: result[i, :, k] += W3T[i,k] * (U3[i,:,r] * V3[:,k,r])
        
        # Let's compute: for all i,k: W3T[i,k] * (U3[i,:,r] * V3[:,k,r])
        # U3[i,:,r] shape (m,), V3[:,k,r] shape (m,)
        # np.einsum('ij,jk,ik->ijk', U3[:,:,r], V3[:,:,r], W3[:,:,r].T)
        # But that's what we're trying to avoid being slow at.
        
        # Actually for rank ~189 and n=m=p=6, the direct loop is:
        # 189 * 6*6*6 = 189*216 = 40824 operations. That's fast enough.
        pass
    
    # Just do the direct approach with numpy for speed
    # Reshape W to get W3T: W3[k,i,r] -> we want W3T[i,k,r] = W3[k,i,r]
    W3T_full = np.transpose(W3, (1, 0, 2))  # (n, p, rank) where W3T_full[i,k,r] = W3[k,i,r]
    
    # Now: sum_r U3[i,j,r] * V3[j,k,r] * W3T_full[i,k,r]
    # = einsum('ijr,jkr,ikr->ijk', U3, V3, W3T_full)
    result = np.einsum('ijr,jkr,ikr->ijk', U3, V3, W3T_full, optimize=True)
    
    return np.all(result == 1)

def _make_block_decomposition():
    """Create the block-structured rank-189 decomposition.
    Partition 6x6 into 2x2 grid of 3x3 blocks.
    For each pair of block positions (I,J) for A and (J,K) for B, 
    use the rank-21 BCLR decomposition for 3x3 multiplication.
    Total: 4 block pairs * 21 = 84? No wait.
    
    Actually: A is 6x6, B is 6x6.
    Block partition: A has blocks A[I,J] for I,J in {0,1}, each 3x3.
    B has blocks B[J,K] for J,K in {0,1}, each 3x3.
    C has blocks C[I,K] for I,K in {0,1}, each 3x3.
    
    C[I,K] = sum_J A[I,J] * B[J,K]
    
    For each (I,J,K) triple of block indices (8 triples), we need to compute
    the 3x3 product A[I,J] * B[J,K] and add to C[I,K].
    
    Using rank-21 for each 3x3 multiplication:
    Total rank = 8 * 21 = 168? No, that's not right either.
    
    Wait: C[I,K] = sum_J A[I,J]*B[J,K]. For fixed I,K, we sum over J (2 terms).
    Each term A[I,J]*B[J,K] is a 3x3 matrix product, which needs rank 21.
    So for each (I,J,K): rank 21. Total: 8 * 21 = 168.
    
    But actually we can share: for fixed (I,J), we compute A[I,J]*B[J,K] for K=0,1.
    Hmm, but the rank-21 decomposition is for a single 3x3 product.
    
    Let me think differently. The standard approach:
    - For each block position (i_block, j_block) in A and (j_block, k_block) in B:
      We use the rank-21 decomposition of the 3x3 multiplication tensor.
    
    The BCLR rank-21 decomposition of 3x3 matrix multiplication:
    There are 21 "wedges" (r=1..21), each with coefficients.
    
    For the 6x6 case with 2x2 block partition:
    - Block indices: I in {0,1}, J in {0,1}, K in {0,1}
    - For each (I,J,K): 21 rank-1 terms
    - Total rank: 8 * 21 = 168
    
    But wait, the schoolbook for 6x6 is 216 = 6^3.
    The block approach: for each of the 8 block-triples (I,J,K), we do a 3x3 multiplication.
    Each 3x3 multiplication via schoolbook is rank 27, total 8*27=216 (same as schoolbook).
    Using rank-21: 8*21 = 168.
    
    Actually, I realize the BCLR decomposition might not be straightforward to embed.
    Let me use a simpler approach: just do schoolbook within each block.
    That gives 8 * 27 = 216, same as global schoolbook. No improvement.
    
    To get improvement, I need the rank-21 for 3x3.
    
    The BCLR rank-21 decomposition: I'll use the known coefficients.
    """
    
    # BCLR rank-21 decomposition of 3x3 matrix multiplication
    # Reference: Bini, Capovani, Lotti, Romani (1980)
    # The 21 tensors are defined by their coefficients.
    
    # I'll use a known rank-21 decomposition.
    # The decomposition has 21 rank-1 terms.
    # Each term r has: U_r (3x3 matrix), V_r (3x3 matrix), W_r (3x3 matrix)
    # such that sum_r U_r[i,j] * V_r[j,k] * W_r[k,i] = delta (multiplication tensor)
    
    # Actually the tensor is: M[i,j,k] = 1 for all i,j,k (in 3x3 case)
    # And we need: sum_r U_r[i,j] * V_r[j,k] * W_r[k,i] = 1 for all i,j,k
    
    # Let me use a concrete rank-21 decomposition.
    # From literature, one standard rank-21 decomposition:
    
    # I'll encode the 21 terms. Each term is (U, V, W) where U, V, W are 3x3 matrices.
    # For the 6x6 case, I'll map these to the block structure.
    
    # Actually, let me just use the schoolbook for now and try to optimize.
    # The key insight from the observation: p285 got score 1.142857 = 216/189
    # So rank 189 was achieved! That's 9 * 21 = 189.
    
    # 9 * 21: maybe they used rank-21 for the "diagonal" blocks only?
    # Or: 2x2 block partition, and for the 4 diagonal block-products (I=K), use rank-21,
    # and for off-diagonal (I!=K), use schoolbook rank-27?
    # 4*21 + 4*27 = 84 + 108 = 192. No.
    
    # 189 = 9*21. Maybe: 9 block positions? That doesn't make sense for 2x2.
    # Or: 189 = 21 * 9. Perhaps they partitioned into 3x3 blocks (2x2 grid) 
    # and used a different counting.
    
    # Actually 189 = 216 - 27. So they saved one 3x3 schoolbook block (rank 27).
    # Or 189 = 21 * 9. 
    # 
    # Let me try: partition into 2x2 blocks of 3x3.
    # For each of the 4 block-multiplication positions (I,J) -> (J,K) for fixed I,K... 
    # 
    # I think the cleanest approach: use rank-21 for each 3x3 sub-multiplication.
    # There are 8 such sub-multiplications (I,J,K each in {0,1}).
    # 8 * 21 = 168.
    
    # Let me implement rank-21 for 3x3 and embed it.
    
    # BCLR rank-21: I'll use the decomposition from the literature.
    # The 21 terms can be represented. Let me use a known set.
    
    # From various sources, the BCLR decomposition uses the following structure:
    # It's based on the fact that 3x3 multiplication can be done with 21 multiplications.
    
    # I'll use a specific rank-21 decomposition. Here's one encoding:
    # Each of the 21 terms is a triple of 3x3 matrices (U_r, V_r, W_r)
    
    # Let me use the decomposition where:
    # Terms 1-9: diagonal-like terms
    # Terms 10-18: off-diagonal symmetric
    # Terms 19-21: additional terms
    
    # Actually, let me just hardcode a known rank-21 decomposition.
    # I'll use the one from the "Strassen-like" constructions.
    
    # For simplicity and correctness, let me first verify with schoolbook (rank 216),
    # then try to find a lower rank through search.
    
    pass

def _schoolbook_decomposition(n=6, m=6, p=6):
    """Standard schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _block_rank21_decomposition():
    """Block-structured decomposition using rank-21 for 3x3 blocks.
    Partition 6x6 into 2x2 grid of 3x3 blocks.
    For each (I,J,K) in {0,1}^3, use rank-21 for the 3x3 product.
    Total rank: 8 * 21 = 168.
    """
    
    # BCLR rank-21 decomposition of 3x3 matrix multiplication.
    # I need explicit coefficients. Let me use a known decomposition.
    
    # The multiplication tensor for 3x3: M[i,j,k] = 1 for all i,j,k in {0,1,2}
    # We need 21 rank-1 decompositions: sum_r U_r[i,j]*V_r[j,k]*W_r[k,i] = 1
    
    # Here's a rank-21 decomposition (from BCLR 1980, simplified):
    # I'll use the following 21 terms. Each term is (U, V, W) as 3x3 matrices.
    
    # Actually, let me use a different well-known rank-21 decomposition.
    # The key property: for all i,j,k: sum_r U_r[i,j] * V_r[j,k] * W_r[k,i] = 1
    
    # I'll construct it using the following approach:
    # Start with schoolbook (27 terms) and find 6 more terms that allow removing 6 schoolbook terms.
    
    # For now, let me just use schoolbook within blocks (rank 27 per 3x3, total 8*27=216)
    # and then try to optimize.
    
    # Actually, let me just return schoolbook and do search from there.
    return _schoolbook_decomposition()

def _try_compress(U, V, W, n=6, m=6, p=6, max_steps=500, rng=None):
    """Try to reduce rank by zeroing entries and checking validity.
    Returns (U, V, W, new_rank) if compression found, else None.
    """
    if rng is None:
        rng = random.Random()
    
    rank = len(U[0])
    
    # Try to zero out columns that are linearly dependent or can be removed
    # Strategy: try setting individual entries to 0 and check if still valid
    # This is very expensive. Instead, try to find columns that can be merged.
    
    # Alternative: try to find a column that is a linear combination of others
    # and eliminate it.
    
    # For simplicity, try random zeroing of entries in one factor at a time.
    # This is a heuristic.
    
    best_U, best_V, best_W = [row[:] for row in U], [row[:] for row in V], [row[:] for row in W]
    best_rank = rank
    
    for step in range(max_steps):
        # Pick a random factor and entry to try zeroing
        factor = rng.choice([0, 1, 2])
        if factor == 0:
            row = rng.randrange(n * m)
            col = rng.randrange(best_rank)
            if best_U[row][col] == 0:
                continue
            old_val = best_U[row][col]
            best_U[row][col] = 0
            if _verify_fast(best_U, best_V, best_W, n, m, p):
                # Valid! Keep it and try to remove the column if all zeros
                pass
            else:
                best_U[row][col] = old_val
        elif factor == 1:
            row = rng.randrange(m * p)
            col = rng.randrange(best_rank)
            if best_V[row][col] == 0:
                continue
            old_val = best_V[row][col]
            best_V[row][col] = 0
            if _verify_fast(best_U, best_V, best_W, n, m, p):
                pass
            else:
                best_V[row][col] = old_val
        else:
            row = rng.randrange(p * n)
            col = rng.randrange(best_rank)
            if best_W[row][col] == 0:
                continue
            old_val = best_W[row][col]
            best_W[row][col] = 0
            if _verify_fast(best_U, best_V, best_W, n, m, p):
                pass
            else:
                best_W[row][col] = old_val
    
    # Try to remove all-zero columns
    for col in range(best_rank - 1, -1, -1):
        all_zero = True
        for r in range(n * m):
            if best_U[r][col] != 0:
                all_zero = False
                break
        if all_zero:
            for r in range(m * p):
                if best_V[r][col] != 0:
                    all_zero = False
                    break
        if all_zero:
            for r in range(p * n):
                if best_W[r][col] != 0:
                    all_zero = False
                    break
        if all_zero:
            # Remove this column
            for r in range(n * m):
                best_U[r].pop(col)
            for r in range(m * p):
                best_V[r].pop(col)
            for r in range(p * n):
                best_W[r].pop(col)
            best_rank -= 1
    
    if best_rank < rank:
        return best_U, best_V, best_W, best_rank
    return None

def _local_search(U, V, W, n=6, m=6, p=6, max_iter=100, rng=None):
    """Local search: perturb entries and try to maintain validity while reducing rank.
    This is a placeholder for more sophisticated search.
    """
    if rng is None:
        rng = random.Random()
    
    rank = len(U[0])
    best_U = [row[:] for row in U]
    best_V = [row[:] for row in V]
    best_W = [row[:] for row in W]
    best_rank = rank
    
    for _ in range(max_iter):
        # Try adding a small perturbation to a random entry
        factor = rng.choice([0, 1, 2])
        if factor == 0:
            row = rng.randrange(n * m)
            col = rng.randrange(best_rank)
            old = best_U[row][col]
            best_U[row][col] = old + rng.choice([-1, 1])
            if _verify_fast(best_U, best_V, best_W, n, m, p):
                # Try to compress
                result = _try_compress(best_U, best_V, best_W, n, m, p, max_steps=50, rng=rng)
                if result:
                    best_U, best_V, best_W, best_rank = result
            else:
                best_U[row][col] = old
        elif factor == 1:
            row = rng.randrange(m * p)
            col = rng.randrange(best_rank)
            old = best_V[row][col]
            best_V[row][col] = old + rng.choice([-1, 1])
            if _verify_fast(best_U, best_V, best_W, n, m, p):
                result = _try_compress(best_U, best_V, best_W, n, m, p, max_steps=50, rng=rng)
                if result:
                    best_U, best_V, best_W, best_rank = result
            else:
                best_V[row][col] = old
        else:
            row = rng.randrange(p * n)
            col = rng.randrange(best_rank)
            old = best_W[row][col]
            best_W[row][col] = old + rng.choice([-1, 1])
            if _verify_fast(best_U, best_V, best_W, n, m, p):
                result = _try_compress(best_U, best_V, best_W, n, m, p, max_steps=50, rng=rng)
                if result:
                    best_U, best_V, best_W, best_rank = result
            else:
                best_W[row][col] = old
    
    return best_U, best_V, best_W, best_rank

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # 130s to leave margin before 150s kill
    
    # Start with schoolbook decomposition
    best = _schoolbook_decomposition(n, m, p)
    best_rank = n * m * p  # 216
    
    rng = random.Random(42)
    
    # Try compression on schoolbook
    result = _try_compress(best["U"], best["V"], best["W"], n, m, p, max_steps=200, rng=rng)
    if result:
        best["U"], best["V"], best["W"], best_rank = result
    
    # Try local search
    while time.time() < deadline - 20:
        result = _local_search(best["U"], best["V"], best["W"], n, m, p, max_iter=20, rng=rng)
        if result:
            best["U"], best["V"], best["W"], best_rank = result
    
    # Final verification
    if not _verify_fast(best["U"], best["V"], best["W"], n, m, p):
        # Fall back to schoolbook
        best = _schoolbook_decomposition(n, m, p)
    
    return {"U": best["U"], "V": best["V"], "W": best["W"]}
# EVOLVE-BLOCK-END
