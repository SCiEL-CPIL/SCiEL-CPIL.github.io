"""Size-efficient sorting network on 17 wires: Batcher Odd-Merge seed + fast prune."""
import time
import numpy as np

N = 17


def _build_batcher_odd_merge(n):
    """Build Batcher Odd-Merge sort network recursively. Returns list of (i, j)."""
    if n <= 1:
        return []
    if n == 2:
        return [(0, 1)]
    k = n // 2
    # Recursively sort even and odd subsequences
    # Even indices: 0, 2, 4, ...
    # Odd indices: 1, 3, 5, ...
    # For odd-merge, we split into two halves of size k and n-k
    # Actually, standard odd-merge: split into two halves, recursively odd-merge each,
    # then compare-exchange between corresponding elements, then recursively sort the
    # interleaved even/odd positions.
    
    # Let's use the standard recursive definition:
    # odd_merge(n):
    #   if n <= 1: return []
    #   k = n // 2
    #   A = odd_merge(k)
    #   B = odd_merge(n - k)
    #   # A sorts positions 0..k-1, B sorts positions k..n-1
    #   # But in odd-merge, we first sort the two halves, then do a "merge" step
    #   # which compares (i, i+k) for i in 0..min(k, n-k)-1
    #   # then recursively sort the even and odd interleaved sequences
    
    # Actually, let me use a cleaner formulation. Batcher's odd-merge network:
    # OM(n):
    #   if n <= 1: return []
    #   k = n // 2
    #   # Recursively apply OM to first k elements and last n-k elements
    #   left = OM(k)
    #   right = OM(n - k)
    #   # Shift right by k
    #   right_shifted = [(i + k, j + k) for (i, j) in right]
    #   # Merge step: compare (i, i+k) for i in range(min(k, n-k))
    #   merge = [(i, i + k) for i in range(min(k, n - k))]
    #   # Then recursively sort the even-indexed and odd-indexed subsequences of the merged result
    #   # Even positions in the full n-element array: 0, 2, 4, ...
    #   # Odd positions: 1, 3, 5, ...
    #   # The even subsequence has size ceil(n/2), odd has floor(n/2)
    #   # We need to map these back to the original positions
    #   # After the merge step, the even positions contain elements that need sorting
    #   # and the odd positions contain elements that need sorting
    #   # But these are not contiguous in the original array
    #   # So we apply OM to the subsequence of even positions and the subsequence of odd positions
    #   # Even positions: 0, 2, 4, ..., 2*(ceil(n/2)-1)
    #   # Odd positions: 1, 3, 5, ..., 2*(floor(n/2)-1)
    #   even_size = (n + 1) // 2
    #   odd_size = n // 2
    #   even_net = OM(even_size)
    #   odd_net = OM(odd_size)
    #   even_shifted = [(2*i, 2*j) for (i, j) in even_net]
    #   odd_shifted = [(2*i + 1, 2*j + 1) for (i, j) in odd_net]
    #   return left + right_shifted + merge + even_shifted + odd_shifted
    
    if n <= 1:
        return []
    k = n // 2
    left = _build_batcher_odd_merge(k)
    right = _build_batcher_odd_merge(n - k)
    right_shifted = [(i + k, j + k) for (i, j) in right]
    merge = [(i, i + k) for i in range(min(k, n - k))]
    even_size = (n + 1) // 2
    odd_size = n // 2
    even_net = _build_batcher_odd_merge(even_size)
    odd_net = _build_batcher_odd_merge(odd_size)
    even_shifted = [(2 * i, 2 * j) for (i, j) in even_net]
    odd_shifted = [(2 * i + 1, 2 * j + 1) for (i, j) in odd_net]
    return left + right_shifted + merge + even_shifted + odd_shifted


def _build_seed():
    """Build a good initial sorting network. Uses Batcher Odd-Merge for 17 wires."""
    # Batcher Odd-Merge for 17 wires
    comp = _build_batcher_odd_merge(N)
    # Verify it sorts
    if not _verify_fast(comp):
        # Fallback to odd-even transposition
        return [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
    return comp


def _verify_fast(comp):
    """Zero-one principle over all 2^N Boolean inputs using big-integer trick.
    Returns True iff the network sorts all inputs."""
    m = len(comp)
    if m == 0:
        return True
    a = np.fromiter((i for i, _ in comp), dtype=np.int32, count=m)
    b = np.fromiter((j for _, j in comp), dtype=np.int32, count=m)
    total = 1 << N
    last = np.arange(total, dtype=np.int32)
    for k in range(m):
        ai = a[k]
        bi = b[k]
        va = (last >> ai) & 1
        vb = (last >> bi) & 1
        last = np.where(va != vb, last ^ (1 << ai) ^ (1 << bi), last)
    return bool((last == np.arange(total, dtype=np.int32)).all())


def _prune(comp, deadline):
    """Remove redundant comparators via greedy deletion."""
    comp = list(comp)
    progress = True
    while progress and time.time() < deadline:
        progress = False
        for idx in range(len(comp) - 1, -1, -1):
            if time.time() > deadline:
                return comp
            cand = comp[:idx] + comp[idx + 1:]
            if _verify_fast(cand):
                comp = cand
                progress = True
    return comp


def construct():
    t_start = time.time()
    deadline = t_start + 80.0  # safety margin before 90s kill
    comp = _build_seed()
    # Sanity check
    if not _verify_fast(comp):
        # Fallback
        comp = [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
    # Prune redundant comparators
    comp = _prune(comp, deadline)
    return comp
