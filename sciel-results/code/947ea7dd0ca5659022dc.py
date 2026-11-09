import time
import numpy as np

# EVOLVE-BLOCK-START

def construct():
    deadline = time.time() + 85.0
    p = 1009
    m = 5

    # 1. Compute quadratic residues using Legendre symbol in parallel
    # a = [1, 2, ..., p-1]
    a = np.arange(1, p, dtype=np.int64)
    
    # Legendre symbol (a/p) = a^((p-1)/2) mod p
    # Result is 1 if a is a quadratic residue, p-1 (-1) otherwise
    exp = (p - 1) // 2
    leg = np.power(a, exp, p)
    
    # is_res[i] is True if a[i] (which is i+1) is a quadratic residue
    # We create a boolean array of size p for easy 1-based indexing access
    # is_res[0] is unused (or False), is_res[v] corresponds to value v
    is_res = np.zeros(p, dtype=bool)
    is_res[1:p] = (leg == 1)

    # 2. Compute smallest positive roots for all residues
    # For x in [1, p//2], x^2 mod p gives the residue.
    # Since we iterate x in increasing order, the first assignment is the smallest root.
    # The mapping x -> x^2 mod p is a bijection from [1, (p-1)/2] to QRs.
    xs = np.arange(1, p // 2 + 1, dtype=np.int64)
    rs = (xs * xs) % p
    
    roots = np.zeros(p, dtype=np.int32)
    # Assign smallest root
    # Note: rs contains unique values for the QRs
    roots[rs] = xs

    # 3. Find minimal s
    # We need s, s+1, s+2, s+3, s+4 to be residues.
    # s ranges from 1 to p-m (1004)
    # is_res[v] is True if v is a residue.
    
    best_s = None
    best_roots = None

    # Vectorized check for all s at once?
    # We can use a sliding window sum or logical AND of shifted arrays.
    # Let's construct an array `valid_s` where valid_s[s-1] is True if run starting at s is valid.
    
    # Create a view or array for the check
    # We need is_res[s-1] & is_res[s] & is_res[s+1] & is_res[s+2] & is_res[s+3]
    # Indices in is_res: s-1, s, s+1, s+2, s+3
    
    # Since p is small, a simple loop is very fast and less prone to indexing errors.
    # However, we can do it vectorized for elegance and speed.
    
    # valid[s-1] = is_res[s-1] and ... and is_res[s+3]
    # Let's build the boolean array for valid starts
    # max_s = p - m
    # indices for is_res needed: 0 to max_s-1 + m-1 = max_s + m - 2 = p - 2
    
    valid_starts = np.ones(p - m + 1, dtype=bool) # 1-based s: 1..p-m -> index 0..p-m-1
    
    # For each offset i in 0..m-1, we check is_res[s-1+i]
    # s-1+i ranges from (1-1+0)=0 to (p-m-1-1+m-1) = p-3?
    # Let's just use a loop for the 5 offsets, it's negligible.
    
    for i in range(m):
        # s-1+i is the index in is_res for value s+i
        # s ranges 1..p-m, so s-1 ranges 0..p-m-1
        # We need is_res[s-1+i] to be True
        # This corresponds to indices (0..p-m-1) + i in is_res
        # We AND this into valid_starts
        valid_starts &= is_res[i : p - m + i]
        
    # Find the first True in valid_starts
    # valid_starts[0] corresponds to s=1
    # valid_starts[k] corresponds to s=k+1
    
    idx = np.argmax(valid_starts)
    if valid_starts[idx]:
        best_s = idx + 1
        # Extract roots for s, s+1, ..., s+4
        # Values are best_s, best_s+1, ..., best_s+4
        # Roots are roots[value]
        start_val = best_s
        end_val = best_s + m
        best_roots = roots[start_val:end_val].tolist()
    else:
        # Fallback: should not happen
        raise RuntimeError("No run found")

    return {'start': best_s, 'roots': best_roots}

# EVOLVE-BLOCK-END
