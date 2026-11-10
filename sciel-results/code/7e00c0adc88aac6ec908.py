import time

def construct():
    deadline = time.time() + 85.0
    p = 1009
    
    # Build the bitset of non-zero quadratic residues
    # bit i is 1 if i is a QR mod p and i != 0
    res_mask = 0
    for x in range(1, p):
        r = (x * x) % p
        if r != 0:
            res_mask |= (1 << r)
            
    # We need s, s+1, s+2, s+3, s+4 to be QRs.
    # This means bit s, s+1, s+2, s+3, s+4 in res_mask must be 1.
    # Let M be the mask where bit s is 1 if s..s+4 are all QRs.
    # M = res_mask & (res_mask >> 1) & (res_mask >> 2) & (res_mask >> 3) & (res_mask >> 4)
    # Note: shifting right moves bit i to i-1.
    # So (res_mask >> 1) has bit s set if res_mask had bit s+1 set.
    # So bit s in the AND result is 1 iff res_mask has bits s, s+1, s+2, s+3, s+4 set.
    
    M = res_mask
    for i in range(1, 5):
        M &= (res_mask >> i)
        
    # We need the smallest s >= 1 such that bit s is set in M.
    # Also s <= 1004.
    # Since we want the smallest s, we look for the lowest set bit.
    # But we must ensure s >= 1. Bit 0 might be set if 0..4 are QRs?
    # 0 is not a non-zero QR, so bit 0 in res_mask is 0.
    # So bit 0 in M is 0.
    # So the lowest set bit will be >= 1.
    
    # Find lowest set bit
    # If M is 0, no solution.
    if M == 0:
        # Fallback to linear scan (should not happen)
        # But for safety
        pass
        
    # Lowest set bit index:
    # (M & -M) isolates the lowest set bit.
    # bit_length() - 1 gives the index.
    lowest_bit = M & -M
    s = lowest_bit.bit_length() - 1
    
    # Verify s is within bounds
    if s < 1 or s > 1004:
        # If s is out of bounds, we might need to find the next one?
        # But s=1 is expected.
        # If s=0, it means 0..4 were QRs, but 0 is not, so s cannot be 0.
        # If s > 1004, we failed.
        pass
        
    # Find roots for s..s+4
    roots = []
    # We need roots for s, s+1, s+2, s+3, s+4
    # We can find them by iterating x or by precomputing a map.
    # Since p is small, we can just compute them on the fly or precompute a small map.
    # Precomputing a map is safer for "stronger algorithm" vibe?
    # Or just use pow(r, (p+1)//4, p) for p = 3 mod 4?
    # 1009 = 4*252 + 1. So p = 1 mod 4.
    # Tonelli-Shanks or simple search.
    # Simple search is fine for 5 numbers.
    
    # Actually, let's just precompute the roots for all residues to be clean.
    root_of_res = [0] * (p + 1)
    for x in range(1, p):
        r = (x * x) % p
        if r != 0 and root_of_res[r] == 0:
            root_of_res[r] = x
            
    for i in range(5):
        val = s + i
        # val should be a QR
        r = root_of_res[val]
        if r == 0:
            # Should not happen
            raise RuntimeError(f"No root for {val}")
        roots.append(r)
        
    return {'start': s, 'roots': roots}
