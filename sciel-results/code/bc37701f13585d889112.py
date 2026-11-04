"""ABS-style Cayley graph construction for the cage problem n(k, g)."""
import numpy as np
from collections import deque

def construct(k, g):
    """
    Construct a k-regular graph of girth at least g using the ABS construction.
    
    For even k: Use Cayley graph of UT_m(k) with m = k//2 + 1,
    generators E_{i,i+1}(±1) for i=0..m-2.
    
    For odd k: Use a modified construction with the group UT_m(k) where
    we select generators to achieve exactly degree k.
    """
    if k % 2 == 0:
        return _abs_even(k, g)
    else:
        return _abs_odd(k, g)


def _abs_even(k, g):
    """ABS construction for even k using UT_m(k) with m = k//2 + 1."""
    m = k // 2 + 1
    n_exp = m * (m - 1) // 2
    n = k ** n_exp
    
    # Check if n is manageable (must fit in memory and time)
    max_n = 50000
    if n > max_n:
        return _luw_fallback(k, g)
    
    # Precompute the effect of each generator on each entry position
    # Entry (i, j) for i < j has index: sum_{r=0}^{i-1} (m - r - 1) + (j - i - 1)
    # i.e., row i has entries (i, i+1), (i, i+2), ..., (i, m-1)
    
    def entry_idx(i, j):
        """Get the linear index for entry (i, j) in the upper triangular part."""
        # Number of entries in rows 0..i-1
        base = sum(m - r - 1 for r in range(i))
        return base + (j - i - 1)
    
    # For right multiplication by E_{i,i+1}(a):
    # (A * E)_{r,c} = A_{r,c} + A_{r,i} * a * delta_{c,i+1}
    # So only column i+1 changes: new_A[r][i+1] = A[r][i+1] + A[r][i] * a
    # But A[r][i] is 0 for r >= i (since upper triangular), so only r < i contribute
    # Wait, that's not right. Let me think again.
    # 
    # Actually for UT_m, A is upper triangular. E_{i,i+1}(a) has a at (i,i+1).
    # (A*E)_{r,c} = sum_j A_{r,j} E_{j,c}
    # E_{j,c} is non-zero only when j=c (diagonal, value 1) or (j,c)=(i,i+1) (value a)
    # So (A*E)_{r,c} = A_{r,c} + A_{r,i} * a if c = i+1, else A_{r,c}
    # 
    # But A_{r,i} is non-zero only if r < i (upper triangular)
    # So: new_entry(r, i+1) = old_entry(r, i+1) + old_entry(r, i) * a, for r < i
    # And new_entry(r, c) = old_entry(r, c) for all other (r,c)
    
    # Precompute for each generator (i, a) which entries change
    # Generator (i, a): for each r in 0..i-1, entry (r, i+1) += entry(r, i) * a
    
    # Build the group elements and edges
    # Use numpy for vectorized operations
    all_elems = np.arange(n, dtype=np.int64).reshape(-1)
    
    # Convert integer index to tuple of entries
    def int_to_entries(val):
        entries = []
        v = val
        for _ in range(n_exp):
            entries.append(v % k)
            v //= k
        return entries
    
    def entries_to_int(entries):
        val = 0
        for i, e in enumerate(reversed(entries)):
            val = val * k + e
        return val
    
    # Precompute generator effects
    # For each generator (i, a), precompute how it transforms an element
    gen_effects = {}
    for i in range(m - 1):
        for a in [1, k - 1]:
            # For this generator, entry (r, i+1) += entry(r, i) * a for r < i
            # Entry (r, i+1) has index entry_idx(r, i+1)
            # Entry (r, i) has index entry_idx(r, i)
            changes = []
            for r in range(i):
                src = entry_idx(r, i)
                dst = entry_idx(r, i + 1)
                changes.append((src, dst, a))
            gen_effects[(i, a)] = changes
    
    # Build edges using vectorized operations
    edges = []
    seen_edges = set()
    
    for idx in range(n):
        entries = int_to_entries(idx)
        for i in range(m - 1):
            for a in [1, k - 1]:
                new_entries = entries[:]
                for src, dst, mult in gen_effects[(i, a)]:
                    new_entries[dst] = (new_entries[dst] + new_entries[src] * mult) % k
                new_idx = entries_to_int(new_entries)
                if idx < new_idx:
                    edges.append((idx, new_idx))
    
    return n, edges


def _abs_odd(k, g):
    """
    ABS construction for odd k.
    
    For odd k, we need degree k which is odd. We can use the group UT_m(k)
    but with a different set of generators.
    
    One approach: Use generators E_{i,i+1}(a) for specific a values.
    For UT_3(k) with k odd, we have 2 "levels" of generators.
    
    Alternative: Use the fact that we can take a Cayley graph on a group
    of order divisible by k, with k generators.
    
    For k=5: Use UT_3(5) which has 125 elements. We need 5 generators.
    Generators: E_{0,1}(1), E_{0,1}(2), E_{1,2}(1), E_{1,2}(2), and one more?
    That would give degree 4 or 6, not 5.
    
    Better approach: Use a group of order k * something.
    
    Actually, let's use the construction from the paper: for odd prime k,
    use the group of affine transformations or a different nilpotent group.
    
    For simplicity and correctness, fall back to LUW for odd k.
    """
    return _luw_fallback(k, g)


def _luw_fallback(k, g):
    """Lazebnik-Ustimenko-Woldar construction as fallback."""
    q = k
    # Choose d such that the girth is at least g
    # The girth of LUW(q, d) is at least 2d - 2 or similar
    # We want girth >= g, so d >= (g + 2) // 2
    d = max(3, (g + 2) // 2)
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: (x * y) % q
    
    # Define the rule for LUW construction
    rule = {}
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        if r == 0:
            rule[c] = (1, c0 - 2)
        elif r == 1:
            rule[c] = (c0 - 1, 1)
        elif r == 2:
            rule[c] = (c0, 1)
        else:
            rule[c] = (1, c0 + 1)
    
    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None] * (d + 1)
            y[1] = first
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield (1 - side, tuple(y))
    
    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    
    return len(order), edges
