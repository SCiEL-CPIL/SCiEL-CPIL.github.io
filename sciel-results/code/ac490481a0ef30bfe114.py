import time
import numpy as np
from fractions import Fraction
from itertools import product

N, M, P = 3, 4, 4
NM, MP, PN = N * M, M * P, P * N


def _verify(U, V, W, tol=1e-10):
    rank = len(U[0])
    U_arr = np.array(U, dtype=float).reshape(N, M, rank)
    V_arr = np.array(V, dtype=float).reshape(M, P, rank)
    W_arr = np.array(W, dtype=float).reshape(P, N, rank)
    T = np.einsum('ijc,jkc,kic->ijk', U_arr, V_arr, W_arr)
    max_err = np.max(np.abs(T - 1.0))
    return max_err < tol, max_err


def _verify_fast(U, V, W):
    rank = len(U[0])
    U_arr = np.array(U, dtype=float).reshape(N, M, rank)
    V_arr = np.array(V, dtype=float).reshape(M, P, rank)
    W_arr = np.array(W, dtype=float).reshape(P, N, rank)
    T = np.einsum('ijc,jkc,kic->ijk', U_arr, V_arr, W_arr)
    return np.max(np.abs(T - 1.0))


def _schoolbook():
    rank = NM * MP * PN
    U = [[0] * rank for _ in range(NM)]
    V = [[0] * rank for _ in range(MP)]
    W = [[0] * rank for _ in range(PN)]
    c = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U[i * M + j][c] = 1
                V[j * P + k][c] = 1
                W[k * N + i][c] = 1
                c += 1
    return {'U': U, 'V': V, 'W': W}


def _strassen_44():
    """Original Strassen-based rank-44 decomposition."""
    rank = 44
    U = [[0] * rank for _ in range(NM)]
    V = [[0] * rank for _ in range(MP)]
    W = [[0] * rank for _ in range(PN)]
    
    s_steps = [
        ({(0, 0): 1, (1, 1): 1}, {(0, 0): 1, (1, 1): 1}, {(0, 0): 1, (1, 1): 1}),
        ({(1, 0): 1, (1, 1): 1}, {(0, 0): 1}, {(1, 0): 1, (1, 1): -1}),
        ({(0, 0): 1}, {(0, 1): 1, (1, 1): -1}, {(0, 1): 1, (1, 1): 1}),
        ({(1, 1): 1}, {(1, 0): 1, (0, 0): -1}, {(0, 0): 1, (1, 0): 1}),
        ({(0, 0): 1, (0, 1): 1}, {(1, 1): 1}, {(0, 1): 1, (0, 0): -1}),
        ({(1, 0): 1, (0, 0): -1}, {(0, 0): 1, (0, 1): 1}, {(1, 1): 1}),
        ({(0, 1): 1, (1, 1): -1}, {(1, 0): 1, (1, 1): 1}, {(0, 0): 1})
    ]
    
    blocks = [(0, 0, 0, 0, 0, 0), (0, 2, 2, 0, 0, 0), (0, 0, 0, 2, 0, 2), (0, 2, 2, 2, 0, 2)]
    
    col = 0
    for (xi, xj, yj, yk, zi, zk) in blocks:
        for (u_c, v_c, w_c) in s_steps:
            for (a, b), val in u_c.items():
                U[(a + xi) * M + (b + xj)][col] += val
            for (b, c), val in v_c.items():
                V[(b + yj) * P + (c + yk)][col] += val
            for (a, c), val in w_c.items():
                W[(c + zk) * N + (a + zi)][col] += val
            col += 1
    
    for j in range(M):
        for k in range(P):
            U[2 * M + j][col] = 1
            V[j * P + k][col] = 1
            W[k * N + 2][col] = 1
            col += 1
    
    assert col == rank
    return {'U': U, 'V': V, 'W': W}


def _strassen_42():
    """Try a rank-42 decomposition: 2 Strassen blocks (14) + 2 naive rows (32) = 46? No.
    Actually: 2 Strassen blocks = 14, remaining 2 rows naively = 2*16=32, total 46.
    Let me try: 1 Strassen block for top 2x2 of output (rows 0-1, cols 0-1) = 7
    Then handle the rest. Actually let me try a different approach.
    
    Alternative: Use Strassen on 2x2 block for output rows 0-1, cols 0-3 (a 2x4 output block)
    A 2x4 = (2x2) * (2x2) where A is 2x4, B is 4x4.
    Split A into [A1 A2] (each 2x2), B into [[B11 B12],[B21 B22]] (each 2x2).
    Output = [A1*B11 + A2*B21, A1*B12 + A2*B22]
    Each 2x2 product needs 7 Strassen mults, so 2*7 = 14 for the two 2x2 output blocks.
    Then rows 2 of output: 4x4 naive = 16.
    Total: 14 + 16 = 30? That seems too low. Let me reconsider.
    
    Actually for 2x4 * 4x4:
    A is 2x4, B is 4x4. Output is 2x4.
    Split A = [A1 | A2] where A1, A2 are 2x2.
    Split B = [[B11, B12], [B21, B22]] where each is 2x2.
    Output col 0-1: A1*B11 + A2*B21 (sum of two 2x2 products)
    Output col 2-3: A1*B12 + A2*B22 (sum of two 2x2 products)
    
    Each 2x2 * 2x2 = 7 Strassen mults.
    But the sums mean we need to add the results. In tensor decomposition,
    each Strassen mult contributes to both output blocks if they share submatrices.
    
    Actually, the four 2x2 products are: A1*B11, A2*B21, A1*B12, A2*B22
    Each needs 7 mults, total 28. But that's worse than schoolbook for this block (32).
    
    Hmm, let me try a different strategy: use the 3x4x4 problem structure more directly.
    """
    pass


def _build_decomposition(block_rows, block_cols, s_steps, col_offset, U, V, W):
    """Build Strassen decomposition for a block of the output.
    
    block_rows: (start_row, num_rows) for the A matrix rows
    block_cols: (start_col, num_cols) for the B matrix cols / output cols
    """
    col = col_offset
    for (xi, xj, yj, yk, zi, zk) in [(0, 0, 0, 0, 0, 0)]:
        for (u_c, v_c, w_c) in s_steps:
            for (a, b), val in u_c.items():
                U[(a + block_rows) * M + (b + block_cols)][col] += val
            for (b, c), val in v_c.items():
                V[(b + block_cols) * P + (c + yk)][col] += val
            for (a, c), val in w_c.items():
                W[(c + zk) * N + (a + block_rows)][col] += val
            col += 1
    return col


def _try_strassen_variants(deadline):
    """Try different Strassen-based decompositions and find the best one."""
    results = []
    
    # Variant 1: Original rank 44
    r1 = _strassen_44()
    err1 = _verify_fast(r1['U'], r1['V'], r1['W'])
    if err1 < 1e-10:
        results.append((len(r1['U'][0]), r1))
    
    # Variant 2: Try Strassen on different block positions
    # Output is 3x4. Try Strassen on rows 0-1, cols 0-1 (2x2 block)
    # Then handle remaining parts
    s_steps = [
        ({(0, 0): 1, (1, 1): 1}, {(0, 0): 1, (1, 1): 1}, {(0, 0): 1, (1, 1): 1}),
        ({(1, 0): 1, (1, 1): 1}, {(0, 0): 1}, {(1, 0): 1, (1, 1): -1}),
        ({(0, 0): 1}, {(0, 1): 1, (1, 1): -1}, {(0, 1): 1, (1, 1): 1}),
        ({(1, 1): 1}, {(1, 0): 1, (0, 0): -1}, {(0, 0): 1, (1, 0): 1}),
        ({(0, 0): 1, (0, 1): 1}, {(1, 1): 1}, {(0, 1): 1, (0, 0): -1}),
        ({(1, 0): 1, (0, 0): -1}, {(0, 0): 1, (0, 1): 1}, {(1, 1): 1}),
        ({(0, 1): 1, (1, 1): -1}, {(1, 0): 1, (1, 1): 1}, {(0, 0): 1})
    ]
    
    # Try: Strassen for output[0:2, 0:2], then Strassen for output[0:2, 2:4], then naive for output[2:3, :]
    # That's 7 + 7 + 16 = 30? No, that's not right either.
    # The issue is that the Strassen decomposition for 2x2*2x2 gives a 2x2 output.
    # For 3x4 * 4x4, we need to think about it differently.
    
    # Let me try: treat it as (3x4) * (4x4)
    # Split A (3x4) into rows: A_top (2x4), A_bottom (1x4)
    # Split B (4x4) into cols: B_left (4x2), B_right (4x2)
    # Output = [A_top * B_left, A_top * B_right; A_bottom * B_left, A_bottom * B_right]
    # Each block: 2x2 or 1x2 times 4x2... this doesn't map cleanly to Strassen.
    
    # Alternative approach: use the fact that 3x4*4x4 can be seen as
    # computing 12 output entries, each a dot product of length 4.
    # Schoolbook: 12 * 4 = 48 mults.
    # Strassen on 2x2 subblocks can reduce this.
    
    # Let me try a more systematic approach:
    # Partition the 3x4 output into blocks and apply Strassen where possible.
    
    # Block structure:
    # Output[0:2, 0:2] = A[0:2, :] * B[:, 0:2]  -> 2x4 * 4x2
    # Output[0:2, 2:4] = A[0:2, :] * B[:, 2:4]  -> 2x4 * 4x2
    # Output[2:3, 0:2] = A[2:3, :] * B[:, 0:2]  -> 1x4 * 4x2
    # Output[2:3, 2:4] = A[2:3, :] * B[:, 2:4]  -> 1x4 * 4x2
    
    # For 2x4 * 4x2:
    # A = [a1 a2 a3 a4] (2x4), B = [b1 b2] (4x2)
    # Output = a1*b1 + a2*b2 (2x2)
    # This is a sum of two 2x2 products.
    # Using Strassen for each: 7 + 7 = 14, but we can potentially share.
    # Actually, a1*b1 and a2*b2 are independent 2x2*2x2 products.
    # So 14 mults for each 2x4*4x2 block.
    # Total for top 2 rows: 14 + 14 = 28
    # Bottom row: 1x4 * 4x2 = two 1x2*2x2... 
    # A_bottom (1x4) * B_left (4x2) = sum of 4 terms, each 1x2 * 2x2? No.
    # It's just 1x4 dot 4x2 = 1x2, which is 4 mults per column, 8 total for 2 cols.
    # Or: A_bottom * B_left = a3*b1 + a4*b2 where a3,a4 are 1x2, b1,b2 are 2x2.
    # Each 1x2 * 2x2 = 1x2, needs 2 mults (or can we do better? No, 1x2*2x2 is 2 mults).
    # So 2 + 2 = 4 mults for bottom-left, 4 for bottom-right, total 8.
    # Grand total: 28 + 8 = 36? That seems too optimistic.
    
    # Let me just verify the original 44 and try to reduce it.
    
    # Variant 3: Try to find if we can do better with a different Strassen arrangement
    # Use Strassen for output[0:2, :] (2x4 output) by treating it as two 2x2 blocks
    # A_top (2x4) * B (4x4) = [A_top[:,0:2]*B[0:2,:], A_top[:,2:4]*B[2:4,:]]
    # = [A1*B_top, A2*B_bottom] where A1,A2 are 2x2, B_top,B_bottom are 2x4
    # Hmm, this is getting complicated. Let me just focus on reducing the rank-44.
    
    return results


def _try_reduce_by_merge(U, V, W, deadline):
    """
    Try to reduce rank by finding pairs of columns that can be merged.
    Two columns c1, c2 can be merged if U[:,c2]=a*U[:,c1], V[:,c2]=b*V[:,c1], W[:,c2]=c*W[:,c1] with a*b*c=1.
    """
    rank = len(U[0])
    best_U, best_V, best_W = U[:], V[:], W[:]
    best_rank = rank
    
    U_frac = [[Fraction(x) for x in row] for row in U]
    V_frac = [[Fraction(x) for x in row] for row in V]
    W_frac = [[Fraction(x) for x in row] for row in W]
    
    deadline = deadline - 5.0
    
    for attempt in range(100):
        if time.time() > deadline:
            break
        
        found_merge = False
        for c1 in range(best_rank - 1):
            if time.time() > deadline:
                break
            for c2 in range(c1 + 1, best_rank):
                u1 = [U_frac[i][c1] for i in range(NM)]
                u2 = [U_frac[i][c2] for i in range(NM)]
                v1 = [V_frac[i][c1] for i in range(MP)]
                v2 = [V_frac[i][c2] for i in range(MP)]
                w1 = [W_frac[i][c1] for i in range(PN)]
                w2 = [W_frac[i][c2] for i in range(PN)]
                
                a = None
                for i in range(NM):
                    if u1[i] != 0:
                        if u2[i] != 0 and a is None:
                            a = u2[i] / u1[i]
                        elif u2[i] == 0:
                            a = None
                            break
                        elif u2[i] / u1[i] != a:
                            a = None
                            break
                    elif u2[i] != 0:
                        a = None
                        break
                
                if a is None or a == 0:
                    continue
                
                b = None
                for i in range(MP):
                    if v1[i] != 0:
                        if v2[i] != 0 and b is None:
                            b = v2[i] / v1[i]
                        elif v2[i] == 0:
                            b = None
                            break
                        elif v2[i] / v1[i] != b:
                            b = None
                            break
                    elif v2[i] != 0:
                        b = None
                        break
                
                if b is None or b == 0:
                    continue
                
                c = None
                for i in range(PN):
                    if w1[i] != 0:
                        if w2[i] != 0 and c is None:
                            c = w2[i] / w1[i]
                        elif w2[i] == 0:
                            c = None
                            break
                        elif w2[i] / w1[i] != c:
                            c = None
                            break
                    elif w2[i] != 0:
                        c = None
                        break
                
                if c is None or c == 0:
                    continue
                
                if a * b * c != 1:
                    continue
                
                # Found a merge!
                for i in range(NM):
                    U_frac[i][c1] = U_frac[i][c1] + U_frac[i][c2]
                for i in range(MP):
                    V_frac[i][c1] = V_frac[i][c1] + V_frac[i][c2]
                for i in range(PN):
                    W_frac[i][c1] = W_frac[i][c1] + W_frac[i][c2]
                
                U_frac = [row[:c2] + row[c2+1:] for row in U_frac]
                V_frac = [row[:c2] + row[c2+1:] for row in V_frac]
                W_frac = [row[:c2] + row[c2+1:] for row in W_frac]
                
                best_rank = best_rank - 1
                best_U = [[float(x) for x in row] for row in U_frac]
                best_V = [[float(x) for x in row] for row in V_frac]
                best_W = [[float(x) for x in row] for row in W_frac]
                
                found_merge = True
                break
            if found_merge or time.time() > deadline:
                break
        
        if not found_merge:
            break
    
    return {'U': best_U, 'V': best_V, 'W': best_W}, best_rank


def _try_general_reduction(U, V, W, deadline):
    """
    Try to reduce rank by finding if any column's tensor is a linear combination of others.
    This is more general than the merge check.
    """
    rank = len(U[0])
    best_U, best_V, best_W = U[:], V[:], W[:]
    best_rank = rank
    
    U_frac = [[Fraction(x) for x in row] for row in U]
    V_frac = [[Fraction(x) for x in row] for row in V]
    W_frac = [[Fraction(x) for x in row] for row in W]
    
    deadline = deadline - 5.0
    
    # Check for duplicate or negated columns
    for attempt in range(50):
        if time.time() > deadline:
            break
        
        found = False
        for c1 in range(best_rank - 1):
            if time.time() > deadline:
                break
            for c2 in range(c1 + 1, best_rank):
                # Check if col c2 = -col c1 in U and V, but same in W
                # Then product is same, so we can remove c2
                u_match = all(U_frac[i][c1] + U_frac[i][c2] == 0 for i in range(NM))
                v_match = all(V_frac[i][c1] + V_frac[i][c2] == 0 for i in range(MP))
                w_match = all(W_frac[i][c1] - W_frac[i][c2] == 0 for i in range(PN))
                
                if u_match and v_match and w_match:
                    # Remove c2
                    U_frac = [row[:c2] + row[c2+1:] for row in U_frac]
                    V_frac = [row[:c2] + row[c2+1:] for row in V_frac]
                    W_frac = [row[:c2] + row[c2+1:] for row in W_frac]
                    best_rank -= 1
                    best_U = [[float(x) for x in row] for row in U_frac]
                    best_V = [[float(x) for x in row] for row in V_frac]
                    best_W = [[float(x) for x in row] for row in W_frac]
                    found = True
                    break
                
                # Check if col c2 = col c1 exactly
                u_match2 = all(U_frac[i][c1] - U_frac[i][c2] == 0 for i in range(NM))
                v_match2 = all(V_frac[i][c1] - V_frac[i][c2] == 0 for i in range(MP))
                w_match2 = all(W_frac[i][c1] - W_frac[i][c2] == 0 for i in range(PN))
                
                if u_match2 and v_match2 and w_match2:
                    # Remove c2
                    U_frac = [row[:c2] + row[c2+1:] for row in U_frac]
                    V_frac = [row[:c2] + row[c2+1:] for row in V_frac]
                    W_frac = [row[:c2] + row[c2+1:] for row in W_frac]
                    best_rank -= 1
                    best_U = [[float(x) for x in row] for row in U_frac]
                    best_V = [[float(x) for x in row] for row in V_frac]
                    best_W = [[float(x) for x in row] for row in W_frac]
                    found = True
                    break
            
            if found or time.time() > deadline:
                break
        
        if not found:
            break
    
    return {'U': best_U, 'V': best_V, 'W': best_W}, best_rank


def _try_strassen_40():
    """
    Try to construct a rank-40 decomposition.
    
    Idea: Use Strassen for a 2x2 block of the output (7 mults),
    then handle the remaining 3x4 - 2x2 = 8 entries more efficiently.
    
    Output is 3x4. If we use Strassen for output[0:2, 0:2]:
    A[0:2, :] (2x4) * B[:, 0:2] (4x2) -> 2x2 output
    This requires: A[0:2,0:2]*B[0:2,0:2] + A[0:2,2:4]*B[2:4,0:2]
    Each is a 2x2 * 2x2 product, needing 7 Strassen mults each.
    Total: 14 mults for this block.
    
    Remaining output:
    output[0:2, 2:4]: A[0:2,:] * B[:,2:4] = A[0:2,0:2]*B[0:2,2:4] + A[0:2,2:4]*B[2:4,2:4]
    output[2:3, 0:2]: A[2:3,:] * B[:,0:2] = A[2:3,0:2]*B[0:2,0:2] + A[2:3,2:4]*B[2:4,0:2]
    output[2:3, 2:4]: A[2:3,:] * B[:,2:4] = A[2:3,0:2]*B[0:2,2:4] + A[2:3,2:4]*B[2:4,2:4]
    
    Total 2x2*2x2 products needed: 8 (each 7 mults) = 56, which is worse.
    
    Let me try a different approach: use the 3x4x4 problem directly.
    """
    pass


def _try_rank_42():
    """
    Attempt a rank-42 decomposition.
    
    Strategy: 
    - Use Strassen for output rows 0-1, cols 0-1 (2x2 block): 7 mults
    - Use Strassen for output rows 0-1, cols 2-3 (2x2 block): 7 mults  
    - Handle output row 2, cols 0-3 (1x4 block): 16 mults naively
    - But we need to be more clever about the 1x4 block.
    
    Actually, let me try:
    - Strassen for output[0:2, 0:2]: 7
    - Strassen for output[0:2, 2:4]: 7
    - Naive for output[2:3, 0:4]: 16
    Total: 30? That can't be right. The issue is that the Strassen decomposition
    for 2x2*2x2 only works when both inputs are 2x2.
    
    For output[0:2, 0:2] = A[0:2,0:4] * B[0:4,0:2]:
    This is (2x4) * (4x2) = 2x2.
    Split: A[0:2,0:4] = [A1 | A2] (each 2x2), B[0:4,0:2] = [[B1],[B2]] (each 2x2)
    Output = A1*B1 + A2*B2
    Each 2x2*2x2 needs 7 Strassen mults, total 14.
    
    Similarly for other blocks. So the minimum with this approach is:
    4 blocks * 14 = 56, which is worse than schoolbook.
    
    The key insight is that Strassen works best when you can group the operations.
    Let me try a completely different approach.
    """
    pass


def _try_crapo_3x4x4():
    """
    Try to use known results for 3x4x4 matrix multiplication.
    
    According to the literature, the rank of 3x4x4 matrix multiplication
    is known to be at most 44 (via Strassen) and at least 37.
    The best known upper bound is 44.
    
    Let me try to construct a better decomposition by hand.
    """
    # For now, return the rank-44 result
    return _strassen_44()


def construct():
    start = time.time()
    
    # Start with the known Strassen-based decomposition with rank 44
    result = _strassen_44()
    
    # Verify correctness
    ok, err = _verify(result['U'], result['V'], result['W'])
    if not ok:
        result = _schoolbook()
        return result
    
    best_result = result
    best_rank = len(result['U'][0])
    
    # Deadline: 140 seconds (10s safety margin)
    deadline = start + 140.0
    
    # Try to reduce rank using merge heuristic
    if time.time() < deadline - 30:
        merged_result, merged_rank = _try_reduce_by_merge(result['U'], result['V'], result['W'], deadline)
        if merged_rank < best_rank:
            ok, err = _verify(merged_result['U'], merged_result['V'], merged_result['W'])
            if ok:
                best_result = merged_result
                best_rank = merged_rank
    
    # Try general reduction
    if time.time() < deadline - 10:
        reduced_result, reduced_rank = _try_general_reduction(best_result['U'], best_result['V'], best_result['W'], deadline)
        if reduced_rank < best_rank:
            ok, err = _verify(reduced_result['U'], reduced_result['V'], reduced_result['W'])
            if ok:
                best_result = reduced_result
                best_rank = reduced_rank
    
    # Final verification
    ok, err = _verify(best_result['U'], best_result['V'], best_result['W'])
    if not ok:
        best_result = result
    
    return best_result
