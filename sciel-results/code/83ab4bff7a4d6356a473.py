def priority(vector, n):
    # Convert vector to a base-3 integer
    val = 0
    for x in vector:
        val = val * 3 + x
    
    # Apply a strong mixing function (MurmurHash3 finalizer style)
    # We assume val fits in 64 bits for typical n. If n is larger, Python big ints work but we can mask to 64 bits for speed if we want, 
    # but keeping full precision for the mix is safer for uniqueness if n is large.
    # However, the mixing function relies on 64-bit arithmetic behavior for good distribution.
    # If val is larger than 64 bits, the lower bits are what get mixed effectively if we don't reduce.
    # Let's mask to 64 bits to ensure the mixing function behaves as intended.
    val &= (1 << 64) - 1
    
    # MurmurHash3 finalizer
    val = (val ^ (val >> 30)) * 0xbf58476d1ce4e5b9
    val &= (1 << 64) - 1
    val = (val ^ (val >> 27)) * 0x94d049bb133111eb
    val &= (1 << 64) - 1
    val = val ^ (val >> 31)
    
    # Return as float
    return float(val)
