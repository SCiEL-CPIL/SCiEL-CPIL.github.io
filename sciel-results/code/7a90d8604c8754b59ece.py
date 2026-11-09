def _find_k5(adj, n):
    # Returns a list of 5 vertices forming a K5, or None
    # Use bitsets for speed
    for i in range(n):
        for j in range(i+1, n):
            if not (adj[i] & (1 << j)): continue
            common = adj[i] & adj[j]
            # We need a K3 in 'common'
            # Iterate k in common > j
            mask = common & ~((1 << (j+1)) - 1)
            while mask:
                lsb = mask & -mask
                k = lsb.bit_length() - 1
                mask ^= lsb
                # Need K2 in (adj[k] & common) > k
                c2 = adj[k] & common & ~((1 << (k+1)) - 1)
                # Check if c2 has an edge
                # Iterate l in c2
                mask_l = c2
                while mask_l:
                    lsb_l = mask_l & -mask_l
                    l = lsb_l.bit_length() - 1
                    mask_l ^= lsb_l
                    # Need m in (adj[l] & c2) > l
                    c3 = adj[l] & c2 & ~((1 << (l+1)) - 1)
                    if c3:
                        # Found K5: i, j, k, l, m
                        lsb_m = c3 & -c3
                        m = lsb_m.bit_length() - 1
                        return [i, j, k, l, m]
    return None
