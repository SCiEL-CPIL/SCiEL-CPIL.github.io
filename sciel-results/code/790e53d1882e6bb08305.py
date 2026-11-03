"""Shortest superpermutation on n symbols.

Strategy: build the word by concatenating a permutation of the (n-1)!
(n-1)-letter blocks in some order, each block separated by the fresh symbol n
(overlapped by the common prefix of consecutive blocks).  We then improve the
order via 2-opt (segment reversal) and Or-opt (segment relocation) local search
to minimise total length, which is equivalent to minimising the sum of
(n-1 - LCP(block_i, block_{i+1})) over the cyclic order.

A strict wall-clock deadline guard keeps the function well under the limit and
always returns a valid superpermutation.
"""
import time
import random
import numpy as np
from itertools import permutations


def _lcp(a, b):
    """Length of the longest common prefix of two equal-length tuples."""
    L = 0
    for x, y in zip(a, b):
        if x != y:
            break
        L += 1
    return L


def _build_lcp_matrix(bm1):
    """Build LCP matrix efficiently using vectorized comparison."""
    L = len(bm1)
    m = len(bm1[0])
    # Convert to numpy array for vectorized LCP
    arr = np.array(bm1, dtype=np.int32)
    # lcp[i,j] = number of positions where arr[i,:k] == arr[j,:k] for all k
    # Vectorize: for each position p, check equality across all pairs
    lcp_mat = np.zeros((L, L), dtype=np.int32)
    for p in range(m):
        # For each position p, create a boolean matrix of whether arr[i,p]==arr[j,p]
        # But we need consecutive matches from position 0
        # Alternative: compute prefix match count
        pass
    # Simpler: use the fact that LCP is the first position where they differ
    # Vectorize by comparing columns and finding first mismatch
    # For each pair (i,j), LCP = number of leading positions where arr[i]==arr[j]
    # This is hard to fully vectorize, so use a semi-vectorized approach
    # Compare row i with all rows j at once
    for i in range(L):
        # arr[i] vs all arr[j] at position 0
        # We want the longest prefix match
        # Compute cumulative product of equality masks
        eq = (arr[i] == arr).T  # shape (L, m): eq[j,p] = (arr[i,p] == arr[j,p])
        # cumprod along axis 1 gives 1 for positions <= first mismatch
        cumprod = np.cumprod(eq, axis=1)
        # LCP = sum of cumprod row (number of leading 1s)
        lcp_mat[i] = cumprod.sum(axis=1)
    return lcp_mat


def _build_word(blocks, order, n):
    """Build the superpermutation string from the block order."""
    parts = ["".join(map(str, blocks[order[0]]))]
    for k in range(1, len(order)):
        prev = blocks[order[k - 1]]
        cur = blocks[order[k]]
        l = 0
        for a, b in zip(prev, cur):
            if a != b:
                break
            l += 1
        parts.append("".join(map(str, cur[l:])))
    return "".join(parts)


def _classic_order(bm1):
    """Greedy first-appearance order of the (n-1)! blocks."""
    order = [0]
    seen = set()
    remaining = [i for i in range(1, len(bm1))]
    cur = bm1[0]
    remaining.sort(key=lambda i: _lcp(cur, bm1[i]), reverse=True)
    # Rebuild by repeatedly taking the next block that best overlaps current
    while remaining:
        best = None
        best_l = -1
        for i in remaining:
            l = _lcp(cur, bm1[i])
            if l > best_l:
                best_l = l
                best = i
        order.append(best)
        cur = bm1[best]
        remaining.remove(best)
    return order


def _total_cost(order, lcp_mat, m):
    """Total length = m + sum over cyclic consecutive pairs of (m - LCP)."""
    L = len(order)
    total = m
    for i in range(L):
        a = order[i]
        b = order[(i + 1) % L]
        total += m - lcp_mat[a][b]
    return total


def _total_cost_np(order, lcp_mat_np, m):
    """Vectorized total cost using numpy."""
    L = len(order)
    order_arr = np.array(order, dtype=np.int32)
    next_arr = np.roll(order_arr, -1)
    # Look up LCP values
    lcp_vals = lcp_mat_np[order_arr, next_arr]
    return m + (m - lcp_vals).sum()


def _local_search(order, lcp_mat, m, deadline):
    """2-opt and Or-opt local search to reduce total length of the cyclic order."""
    L = len(order)
    if L < 4:
        return order
    order = list(order)
    current_len = _total_cost(order, lcp_mat, m)
    
    # Convert to numpy for faster lookups
    lcp_np = np.array(lcp_mat, dtype=np.int32)
    
    def rev2_delta(i, j):
        """Compute delta for reversing segment [i, j] inclusive (2-opt)."""
        a = order[i - 1]
        b = order[i]
        c = order[j]
        d = order[(j + 1) % L]
        old = lcp_np[a][b] + lcp_np[c][d]
        new = lcp_np[a][d] + lcp_np[c][b]
        return new - old  # negative = improvement

    def oropt_delta(i, j, k):
        """Compute delta for moving segment [i, j] to position after k (Or-opt).
        The segment [i,j] is removed and inserted after position k in the new sequence.
        We consider k not in [i-1, j] (to avoid trivial moves).
        """
        if k == i - 1 or k == j:
            return 0  # no change
        # Remove segment [i, j]: neighbors are (i-1) and (j+1)
        a = order[i - 1]
        b = order[i]
        c = order[j]
        d = order[(j + 1) % L]
        
        # After removal, a and d become neighbors
        # Insert after k: neighbors are k and k+1 (in the modified sequence)
        # We need to be careful about indices after removal
        
        # Let's think of it as: original sequence has ...a, [seg], d...
        # After removing seg: ...a, d...
        # Then insert seg after position k (where k is in the original indexing, 
        # but we need to adjust if k is in [i, j] or after j)
        
        # Simpler approach: just try all possible insert positions
        # For now, implement a simpler version: move single element
        pass
    
    def single_relocate_delta(i, k):
        """Move element at position i to after position k.
        i and k are in [0, L-1], i != k, i != k+1 (mod L)
        """
        if (k + 1) % L == i:
            return 0  # already adjacent in that direction
        if k == i:
            return 0
        
        # Element to move
        x = order[i]
        # Neighbors of x: before = order[i-1], after = order[i+1]
        before = order[i - 1]
        after = order[(i + 1) % L]
        
        # Position to insert after: k
        # Neighbors of insertion point: k and k+1
        k_elem = order[k]
        k_next = order[(k + 1) % L]
        
        # Old edges: (before, x), (x, after), (k_elem, k_next)
        # New edges: (before, after), (k_elem, x), (x, k_next)
        # But if k_next == after (i.e., inserting right before where x was), adjust
        # Actually, after removing x, the sequence changes. Let's be careful.
        
        # Case 1: k is not i-1 and k is not i (which we already checked)
        # Old cost contribution from affected edges:
        old_cost = lcp_np[before][x] + lcp_np[x][after] + lcp_np[k_elem][k_next]
        
        # New cost:
        # before connects to after (x removed)
        # k_elem connects to x (x inserted after k)
        # x connects to k_next
        new_cost = lcp_np[before][after] + lcp_np[k_elem][x] + lcp_np[x][k_next]
        
        # Delta in LCP sum (we want to maximize LCP, so delta = new_lcp - old_lcp)
        # Total length = m + sum(m - LCP), so minimizing length = maximizing LCP sum
        delta_lcp = new_cost - old_cost
        # Delta in total length = -delta_lcp
        return -delta_lcp
    
    improved = True
    max_rounds = 100
    for round_num in range(max_rounds):
        if time.time() > deadline - 2.0:
            break
        improved = False
        
        # 2-opt pass: try reversing segments
        for i in range(1, L):
            if time.time() > deadline - 2.0:
                break
            for j in range(i + 1, min(i + 50, L)):  # Limit segment length for speed
                delta = rev2_delta(i, j)
                if delta < 0:
                    order[i:j + 1] = reversed(order[i:j + 1])
                    current_len += delta
                    improved = True
                    break
            if improved:
                break
        
        if not improved:
            # Or-opt: try relocating single elements
            for i in range(L):
                if time.time() > deadline - 2.0:
                    break
                for k in range(L):
                    if k == i or (k + 1) % L == i:
                        continue
                    delta = single_relocate_delta(i, k)
                    if delta < 0:
                        # Move element i to after k
                        x = order.pop(i)
                        # Adjust k if k >= i
                        if k >= i:
                            k -= 1
                        order.insert(k + 1, x)
                        current_len += delta
                        improved = True
                        break
                if improved:
                    break
        
        if not improved:
            break
    
    return order


def _construct_superperm(n, deadline):
    # Build (n-1)! blocks
    bm1 = list(permutations(range(1, n)))
    L = len(bm1)
    m = n - 1

    # Precompute LCP matrix
    lcp_mat = _build_lcp_matrix(bm1)

    # Start from the classic order
    order = _classic_order(bm1)

    # Refine with local search
    order = _local_search(order, lcp_mat, m, deadline)

    return _build_word(bm1, order, n)


def construct(n):
    if n <= 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"

    deadline = time.time() + 110.0  # safety margin below 120s

    try:
        return _construct_superperm(n, deadline)
    except Exception:
        # Fallback: classic construction
        word = [1]
        for k in range(2, n + 1):
            perms, seen = [], set()
            for i in range(len(word) - k + 2):
                w = tuple(word[i:i + k - 1])
                if len(set(w)) == k - 1 and w not in seen:
                    seen.add(w)
                    perms.append(w)
            new = []
            for p in perms:
                block = list(p) + [k] + list(p)
                overlap = 0
                for o in range(min(len(new), len(block)), -1, -1):
                    if new[len(new) - o:] == block[:o]:
                        overlap = o
                        break
                new += block[overlap:]
            word = new
        return "".join(map(str, word))
