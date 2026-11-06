import time

# EVOLVE-BLOCK-START
def construct():
    """
    Find the smallest s >= 1 such that s, s+1, s+2, s+3, s+4 are all
    non-zero quadratic residues mod 1009, with roots for each.
    """
    p = 1009
    deadline = time.time() + 80.0  # safety margin before 90s kill

    # Build prefix sum of QR indicators and root list
    # is_res[i] is 1 if i is a non-zero QR, else 0
    is_res = [0] * (p + 1)
    roots = [0] * p
    
    for x in range(1, p):
        sq = (x * x) % p
        if is_res[sq] == 0: # Only store the first root found
            roots[sq] = x
        is_res[sq] = 1
        
    # Build prefix sum array for O(1) range sum queries
    # prefix[i] = sum(is_res[0..i-1])
    prefix = [0] * (p + 2)
    for i in range(1, p + 1):
        prefix[i] = prefix[i-1] + is_res[i]
        
    # Find smallest s in [1, 1004] where s..s+4 are all QR
    # We need sum(is_res[s..s+4]) == 5
    # This is equivalent to prefix[s+5] - prefix[s] == 5
    best_s = None
    for s in range(1, p - 4): # 1 to 1004
        if prefix[s + 5] - prefix[s] == 5:
            best_s = s
            break

    if best_s is None:
        best_s = p - 5  # fallback, shouldn't happen

    # Extract roots
    res_roots = [roots[best_s + i] for i in range(5)]

    return {'start': best_s, 'roots': res_roots}
# EVOLVE-BLOCK-END
