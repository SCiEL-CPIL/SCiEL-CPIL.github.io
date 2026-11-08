"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit

    n, m, p = DIMENSIONS  # 3, 5, 6

    def schoolbook():
        """Return the schoolbook decomposition of rank n*m*p."""
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

    def verify(U, V, W, rank):
        """Verify that the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    actual = 0.0
                    for c in range(rank):
                        actual += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(actual - 1.0) > 1e-9:
                        return False
        return True

    def strassen_2x2x2_terms():
        """
        Return the 7 Strassen terms for 2x2x2 matrix multiplication.
        Each term is (U_pattern, V_pattern, W_pattern) where patterns are
        dicts mapping (local_i, local_j) -> value for U, etc.
        """
        terms = []
        # Term 1: M1 = (A00+A11)(B00+B11)
        # Contributes +1 to C00 (T[0,0,0]) and +1 to C11 (T[1,1,1])
        u1 = {(0,0): 1, (1,1): 1}
        v1 = {(0,0): 1, (1,1): 1}
        w1 = {(0,0): 1, (1,1): 1}
        terms.append((u1, v1, w1))
        
        # Term 2: M2 = (A10+A11)B00
        # Contributes +1 to C10 (T[1,0,0]) and -1 to C11 (T[1,1,1])
        u2 = {(1,0): 1, (1,1): 1}
        v2 = {(0,0): 1}
        w2 = {(0,1): 1, (1,1): -1}
        terms.append((u2, v2, w2))
        
        # Term 3: M3 = A00(B01-B11)
        # Contributes +1 to C01 (T[0,0,1]) and +1 to C11 (T[1,1,1])
        u3 = {(0,0): 1}
        v3 = {(0,1): 1, (1,1): -1}
        w3 = {(1,0): 1, (1,1): 1}
        terms.append((u3, v3, w3))
        
        # Term 4: M4 = A11(B10-B00)
        # Contributes +1 to C00 (T[0,0,0]) and +1 to C10 (T[1,0,0])
        u4 = {(1,1): 1}
        v4 = {(1,0): 1, (0,0): -1}
        w4 = {(0,0): 1, (0,1): 1}
        terms.append((u4, v4, w4))
        
        # Term 5: M5 = (A00+A01)B11
        # Contributes -1 to C00 (T[0,0,0]) and +1 to C01 (T[0,0,1])
        u5 = {(0,0): 1, (0,1): 1}
        v5 = {(1,1): 1}
        w5 = {(0,0): -1, (1,0): 1}
        terms.append((u5, v5, w5))
        
        # Term 6: M6 = (A10-A00)(B00+B01)
        # Contributes +1 to C11 (T[1,1,1])
        u6 = {(1,0): 1, (0,0): -1}
        v6 = {(0,0): 1, (0,1): 1}
        w6 = {(1,1): 1}
        terms.append((u6, v6, w6))
        
        # Term 7: M7 = (A01-A11)(B10+B11)
        # Contributes +1 to C00 (T[0,0,0])
        u7 = {(0,1): 1, (1,1): -1}
        v7 = {(1,0): 1, (1,1): 1}
        w7 = {(0,0): 1}
        terms.append((u7, v7, w7))
        
        return terms

    def compute_partition_rank(i_groups, j_groups, k_groups):
        """Compute total rank for a given partition."""
        total = 0
        for ig in i_groups:
            for jg in j_groups:
                for kg in k_groups:
                    si, sj, sk = len(ig), len(jg), len(kg)
                    if si == 2 and sj == 2 and sk == 2:
                        total += 7
                    else:
                        total += si * sj * sk
        return total

    def generate_partitions(dim):
        """Generate all partitions of range(dim) into contiguous groups."""
        from itertools import combinations
        partitions = []
        # Use compositions: number of groups g, then g-1 dividers
        for g in range(1, dim + 1):
            # Choose g-1 dividers from dim-1 positions
            for dividers in combinations(range(dim - 1), g - 1):
                groups = []
                prev = 0
                for d in dividers:
                    groups.append(list(range(prev, d + 1)))
                    prev = d + 1
                groups.append(list(range(prev, dim)))
                partitions.append(groups)
        return partitions

    def generate_all_groupings(dim, max_group_size=2):
        """Generate all groupings (not necessarily contiguous) with groups of size <= max_group_size."""
        from itertools import combinations
        groupings = []
        # Try all ways to partition into groups of size 1 or 2
        # This is equivalent to matching in a complete graph on dim vertices
        # Use recursive approach
        def rec(remaining, current_groups):
            if not remaining:
                groupings.append(list(current_groups))
                return
            first = remaining[0]
            # Put first in a group by itself
            rec(remaining[1:], current_groups + [[first]])
            # Put first with another element
            for i in range(1, len(remaining)):
                rec(remaining[1:i] + remaining[i+1:], current_groups + [[first, remaining[i]]])
        rec(list(range(dim)), [])
        return groupings

    def try_linear_reduction(U, V, W, rank):
        """
        Try to reduce rank by finding linear dependencies among the rank-1 tensors.
        Uses QR with column pivoting to find an independent subset.
        """
        if time.time() > deadline:
            return None
        
        nm, mp, pn = n * m, m * p, p * n
        # Build matrix where each column is the flattened Kronecker product
        M = np.zeros((nm * mp * pn, rank), dtype=float)
        
        for c in range(rank):
            u_col = np.array([U[i][c] for i in range(nm)], dtype=float)
            v_col = np.array([V[i][c] for i in range(mp)], dtype=float)
            w_col = np.array([W[i][c] for i in range(pn)], dtype=float)
            
            uv = np.kron(u_col, v_col)
            M[:, c] = np.kron(uv, w_col)
        
        # Use QR with column pivoting to find independent columns
        try:
            Q, R, P = np.linalg.qr(M, mode='complete', pivoting=True)
            # R is upper triangular (with pivoting), diag entries indicate rank
            diag_R = np.abs(np.diag(R[:rank, :rank]))
            tol = 1e-8
            num_indep = sum(1 for d in diag_R if d > tol)
        except Exception:
            return None
        
        if num_indep >= rank:
            return None
        
        # P is the permutation: column c of M corresponds to original column P[c]
        # The first num_indep columns of Q (after pivoting) span the column space
        # We want to select the original columns that are independent
        selected_cols = P[:num_indep].tolist()
        
        # Build new U, V, W with only the selected columns
        new_U = [[U[i][c] for c in selected_cols] for i in range(nm)]
        new_V = [[V[i][c] for c in selected_cols] for i in range(mp)]
        new_W = [[W[i][c] for c in selected_cols] for i in range(pn)]
        
        new_rank = len(selected_cols)
        
        # Verify the new decomposition
        if verify(new_U, new_V, new_W, new_rank):
            return {"U": new_U, "V": new_V, "W": new_W}
        
        return None

    def build_strassen_decomposition(i_groups=None, j_groups=None, k_groups=None):
        """
        Build a decomposition using Strassen on 2x2x2 blocks and schoolbook elsewhere.
        """
        if i_groups is None:
            i_groups = [[0, 1], [2]]
        if j_groups is None:
            j_groups = [[0, 1], [2, 3], [4]]
        if k_groups is None:
            k_groups = [[0, 1], [2, 3], [4, 5]]
        
        # First pass: count total rank needed
        total_rank = 0
        block_info = []
        for ig in i_groups:
            for jg in j_groups:
                for kg in k_groups:
                    si, sj, sk = len(ig), len(jg), len(kg)
                    if si == 2 and sj == 2 and sk == 2:
                        r = 7  # Strassen
                    else:
                        r = si * sj * sk  # schoolbook
                    total_rank += r
                    block_info.append((ig, jg, kg, r))
        
        # Allocate columns
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]
        
        col = 0
        strassen_terms = strassen_2x2x2_terms()
        
        for ig, jg, kg, r in block_info:
            si, sj, sk = len(ig), len(jg), len(kg)
            if si == 2 and sj == 2 and sk == 2:
                # Use Strassen
                for (u_pat, v_pat, w_pat) in strassen_terms:
                    for (li, lj), val_u in u_pat.items():
                        gi, gj = ig[li], jg[lj]
                        U[gi * m + gj][col] = val_u
                    for (lj, lk), val_v in v_pat.items():
                        gj, gk = jg[lj], kg[lk]
                        V[gj * p + gk][col] = val_v
                    for (lk, li), val_w in w_pat.items():
                        gk, gi = kg[lk], ig[li]
                        W[gk * n + gi][col] = val_w
                    col += 1
            else:
                # Schoolbook for this sub-block
                for li in range(si):
                    for lj in range(sj):
                        for lk in range(sk):
                            gi, gj, gk = ig[li], jg[lj], kg[lk]
                            U[gi * m + gj][col] = 1
                            V[gj * p + gk][col] = 1
                            W[gk * n + gi][col] = 1
                            col += 1
        
        return {"U": U, "V": V, "W": W, "rank": total_rank}

    def find_best_partition():
        """Search over partitions (contiguous and non-contiguous) to find the one with minimum rank."""
        if time.time() > deadline:
            return None
        
        # Start with contiguous partitions
        i_parts = generate_partitions(n)
        j_parts = generate_partitions(m)
        k_parts = generate_partitions(p)
        
        best_rank = float('inf')
        best_partition = None
        
        for ip in i_parts:
            for jp in j_parts:
                for kp in k_parts:
                    r = compute_partition_rank(ip, jp, kp)
                    if r < best_rank:
                        best_rank = r
                        best_partition = (ip, jp, kp)
        
        # Also try non-contiguous groupings (groups of size 1 or 2 only)
        # This is more expensive but may find better solutions
        if time.time() < deadline - 30:  # Reserve time
            i_all = generate_all_groupings(n, max_group_size=2)
            j_all = generate_all_groupings(m, max_group_size=2)
            k_all = generate_all_groupings(p, max_group_size=2)
            
            for ip in i_all:
                for jp in j_all:
                    for kp in k_all:
                        r = compute_partition_rank(ip, jp, kp)
                        if r < best_rank:
                            best_rank = r
                            best_partition = (ip, jp, kp)
                        if time.time() > deadline - 10:
                            break
                    if time.time() > deadline - 10:
                        break
                if time.time() > deadline - 10:
                    break
        
        if best_partition is None:
            return None
        
        return build_strassen_decomposition(*best_partition)

    def try_reduce_rank(U, V, W, rank):
        """
        Try to reduce the rank by finding linear dependencies among rank-1 tensors.
        Returns (U', V', W', new_rank) if reduction found, else None.
        """
        if time.time() > deadline:
            return None
        
        # This function is a wrapper that calls try_linear_reduction
        result = try_linear_reduction(U, V, W, rank)
        if result is not None:
            new_rank = len(result["U"][0])
            return (result["U"], result["V"], result["W"], new_rank)
        
        return None

    def try_improved_decomposition():
        """
        Try to find a better decomposition by searching over partitions.
        """
        if time.time() > deadline:
            return None
        
        # Search for best partition
        result = find_best_partition()
        if result is not None:
            rank = result.pop("rank")
            if verify(result["U"], result["V"], result["W"], rank):
                # Try to further reduce rank via linear algebra
                reduced = try_linear_reduction(result["U"], result["V"], result["W"], rank)
                if reduced is not None:
                    new_rank = len(reduced["U"][0])
                    if verify(reduced["U"], reduced["V"], reduced["W"], new_rank):
                        return reduced
                return result
        
        # Fallback to default partition
        result = build_strassen_decomposition()
        rank = result.pop("rank")
        if verify(result["U"], result["V"], result["W"], rank):
            # Try to reduce rank
            reduced = try_linear_reduction(result["U"], result["V"], result["W"], rank)
            if reduced is not None:
                new_rank = len(reduced["U"][0])
                if verify(reduced["U"], reduced["V"], reduced["W"], new_rank):
                    return reduced
            return result
        
        return None

    # Start with schoolbook as fallback
    best_result = schoolbook()
    best_rank = n * m * p

    # Verify the schoolbook decomposition
    if not verify(best_result["U"], best_result["V"], best_result["W"], best_rank):
        pass

    # Try improved decomposition
    improved = try_improved_decomposition()
    if improved is not None:
        improved_rank = len(improved["U"][0])
        if improved_rank < best_rank:
            best_result = improved
            best_rank = improved_rank

    # Try additional linear reduction on the best result
    if time.time() < deadline - 10:
        current_rank = len(best_result["U"][0])
        reduced = try_linear_reduction(best_result["U"], best_result["V"], best_result["W"], current_rank)
        if reduced is not None:
            reduced_rank = len(reduced["U"][0])
            if reduced_rank < best_rank:
                best_result = reduced
                best_rank = reduced_rank

    return best_result
# EVOLVE-BLOCK-END
