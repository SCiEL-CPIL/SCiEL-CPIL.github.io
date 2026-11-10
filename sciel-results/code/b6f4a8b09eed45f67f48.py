<<<<<<< SEARCH
import time
import numpy as np
import random

def _compute_max_discrepancy(signs):
    """
    Compute the maximum absolute signed sum over any arithmetic progression.
    An AP is defined by start a, step d, and length L.
    The sum is sum_{k=0}^{L-1} signs[a + k*d].
    We need to check all prefixes of all maximal APs for all steps d and starts a.
    This is equivalent to:
    For each step d:
      For each start r in [0, d):
        Consider the sequence signs[r], signs[r+d], signs[r+2d], ...
        Compute the maximum absolute prefix sum of this sequence.
    """
    n = len(signs)
    arr = np.array(signs, dtype=np.int64)
    max_disc = 1  # Singletons
    
    # We only need to check steps d up to n.
    # For a given d, the starts are r in [0, d).
    for d in range(1, n + 1):
        # For this step d, we have d sequences starting at 0, 1, ..., d-1.
        # We can process them together or individually.
        # To be fast, we can create a 2D array.
        # Max length is ceil(n/d).
        max_len = (n + d - 1) // d
        
        # Create a matrix of shape (d, max_len)
        # Initialize with 0s to handle uneven lengths
        # However, cumsum on padded zeros is fine as long as we ignore the tail
        # Or we can just loop r, which is at most 256 times.
        # Let's try the loop first, it's simpler and likely fast enough for N=256.
        # 256 steps * 256 starts = 65536 cumsums.
        # Each cumsum is on an array of size <= 256.
        # This might be ~1-5ms in Python/NumPy.
        
        for r in range(d):
            if r >= n:
                break
            subseq = arr[r::d]
            if len(subseq) == 0:
                continue
            prefix_sums = np.cumsum(subseq)
            current_max = np.max(np.abs(prefix_sums))
            if current_max > max_disc:
                max_disc = int(current_max)
                
    return max_disc

def _find_worst_ap(signs):
    """
    Returns (max_disc, start, step, length) for one of the APs achieving max_disc.
    """
    n = len(signs)
    arr = np.array(signs, dtype=np.int64)
    max_disc = 1
    best_start = 0
    best_step = 1
    best_len = 1
    
    for d in range(1, n + 1):
        for r in range(d):
            if r >= n:
                break
            subseq = arr[r::d]
            if len(subseq) == 0:
                continue
            prefix_sums = np.cumsum(subseq)
            abs_sums = np.abs(prefix_sums)
            current_max = np.max(abs_sums)
            if current_max > max_disc:
                max_disc = int(current_max)
                best_step = d
                best_start = r
                # Find the length (index in prefix_sums + 1) that achieves this max
                # There might be multiple, pick the first one
                idx = np.argmax(abs_sums)
                best_len = idx + 1
                
    return max_disc, best_start, best_step, best_len

def _thue_morse(n):
    """Generate Thue-Morse sequence of length n."""
    seq = []
    for i in range(n):
        # Count number of 1s in binary representation
        # Python 3.1+ has bin(i).count('1')
        # Or use bit_count() in 3.10+
        # Let's use a fast method
        c = 0
        x = i
        while x:
            c += x & 1
            x >>= 1
        seq.append(1 if c % 2 == 0 else -1)
    return seq
=======
import time
import numpy as np
import random

def _compute_max_discrepancy(signs):
    """
    Fast vectorized computation of max |prefix sum| over all APs.
    For each step d in [1, n), we build a (d, max_len) matrix of the d
    residue-class sequences and take the max abs cumsum.
    """
    n = len(signs)
    arr = np.array(signs, dtype=np.int64)
    max_disc = 1  # Singletons always contribute 1

    for d in range(1, n):
        max_len = (n + d - 1) // d
        # Build matrix: row r contains arr[r], arr[r+d], arr[r+2d], ...
        # Padded with 0s for uneven lengths (safe: cumsum of 0s stays 0)
        # Use advanced indexing for speed
        cols = np.arange(max_len)
        rows = np.arange(d)[:, None]
        idx = rows + cols[None, :] * d
        # Mask out-of-bounds
        mask = idx < n
        # Use np.where to get 0 for out-of-bounds
        mat = np.where(mask, arr[idx], 0)
        cs = np.cumsum(mat, axis=1)
        cur = np.max(np.abs(cs))
        if cur > max_disc:
            max_disc = int(cur)

    # d = n: only singletons, already covered
    return max_disc

def _find_worst_ap(signs):
    """
    Returns (max_disc, start, step, length) for one of the APs achieving max_disc.
    """
    n = len(signs)
    arr = np.array(signs, dtype=np.int64)
    max_disc = 1
    best_start = 0
    best_step = 1
    best_len = 1

    for d in range(1, n):
        max_len = (n + d - 1) // d
        cols = np.arange(max_len)
        rows = np.arange(d)[:, None]
        idx = rows + cols[None, :] * d
        mask = idx < n
        mat = np.where(mask, arr[idx], 0)
        cs = np.cumsum(mat, axis=1)
        abs_cs = np.abs(cs)
        cur = np.max(abs_cs)
        if cur > max_disc:
            max_disc = int(cur)
            # Find position
            flat_idx = np.argmax(abs_cs)
            r = flat_idx // max_len
            c = flat_idx % max_len
            best_start = r
            best_step = d
            best_len = c + 1

    return max_disc, best_start, best_step, best_len

def _legendre_init(n):
    """
    Initialize using the Legendre symbol modulo a prime near n.
    For a prime p, the Legendre symbol (a/p) is +1 if a is a quadratic residue mod p,
    -1 if non-residue, 0 if a ≡ 0 mod p. We map 0 -> +1 to keep signs in {-1,+1}.
    This gives bounded discrepancy on short APs (classical number theory).
    """
    # Find a prime p >= n
    def is_prime(x):
        if x < 2:
            return False
        if x < 4:
            return True
        if x % 2 == 0 or x % 3 == 0:
            return False
        i = 5
        while i * i <= x:
            if x % i == 0 or x % (i + 2) == 0:
                return False
            i += 6
        return True

    p = n
    while not is_prime(p):
        p += 1

    signs = []
    for i in range(n):
        # Legendre symbol of (i+1) mod p, using Euler's criterion
        a = (i + 1) % p
        if a == 0:
            signs.append(1)
        else:
            # a^((p-1)/2) mod p is 1 if residue, p-1 if non-residue
            val = pow(a, (p - 1) // 2, p)
            signs.append(1 if val == 1 else -1)
    return signs
>>>>>>> REPLACE

