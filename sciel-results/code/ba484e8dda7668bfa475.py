def construct():
    N = 210
    MAX = 10210
    MAX_SIZE = 100
    start_time = time.time()
    deadline = start_time + 85.0
    
    target_sf = _squarefree_kernel(N)  # 210
    
    def sf_product(v, b):
        """Compute sf(v*b) where v and b are squarefree."""
        g = _gcd(v, b)
        return v * b // (g * g)
    
    def reduce_kernel_full(k, basis):
        """Reduce squarefree kernel k by basis vectors until stable. Return (reduced_k, rep_mask)."""
        rep = 0
        changed = True
        while changed:
            changed = False
            for idx in range(len(basis)):
                if k == 1:
                    return 1, rep
                b_k, b_rep = basis[idx]
                g = _gcd(k, b_k)
                if g > 1:
                    k = sf_product(k, b_k)
                    rep ^= b_rep
                    changed = True
        return k, rep
    
    basis = []  # list of (sf_kernel, rep_mask)
    
    for a in range(211, MAX + 1):
        if time.time() > deadline:
            break
        
        ker = _squarefree_kernel(a)
        rep = 1 << (a - 211)
        
        # Reduce ker by existing basis
        reduced_k, reduced_rep = reduce_kernel_full(ker, basis)
        
        if reduced_k != 1:
            basis.append((reduced_k, reduced_rep))
        
        # Check if target is in span
        if len(basis) >= 1:
            target_reduced, target_rep = reduce_kernel_full(target_sf, basis)
            if target_reduced == 1:
                selected = []
                for i in range(a - 210):
                    if (target_rep >> i) & 1:
                        selected.append(211 + i)
                if 1 <= len(selected) <= MAX_SIZE:
                    prod = 1
                    for x in selected:
                        prod *= x
                    if _is_perfect_square(prod * N):
                        return [int(x) for x in selected]
    
    return [4 * N]
