def construct():
    import numpy as np
    from fractions import Fraction
    
    denominators = list(range(2, 49))
    n = len(denominators)
    mid = n // 2
    
    left_denoms = denominators[:mid]
    right_denoms = denominators[mid:]
    
    def gen_sums(denoms):
        n = len(denoms)
        invs = np.array([1.0/d for d in denoms])
        size = 1 << n
        masks = np.arange(size, dtype=np.uint64)
        sums = np.zeros(size)
        for i in range(n):
            bit = (masks >> i) & 1
            sums += bit * invs[i]
        return sums
    
    left_sums = gen_sums(left_denoms)
    right_sums = gen_sums(right_denoms)
    
    right_order = np.argsort(right_sums)
    right_sorted = right_sums[right_order]
    
    max_right = 1.0 - left_sums
    right_idx = np.searchsorted(right_sorted, max_right, side='right') - 1
    
    valid = right_idx >= 0
    if not np.any(valid):
        selected, total = [], Fraction()
        for i, d in enumerate(denominators):
            if total + Fraction(1, d) < 1:
                selected.append(i)
                total += Fraction(1, d)
        return selected
    
    totals = np.full(len(left_sums), -1.0)
    valid_indices = np.where(valid)[0]
    totals[valid_indices] = left_sums[valid_indices] + right_sorted[right_idx[valid_indices]]
    
    top_k = min(100, len(totals))
    top_indices = np.argsort(totals)[::-1][:top_k]
    
    for idx in top_indices:
        if totals[idx] < 0:
            continue
        left_mask = idx
        right_mask = right_order[right_idx[idx]]
        
        selected = []
        for i in range(mid):
            if left_mask & (1 << i):
                selected.append(i)
        for i in range(n - mid):
            if right_mask & (1 << i):
                selected.append(mid + i)
        
        if not selected:
            continue
        
        total_frac = Fraction()
        for i in selected:
            total_frac += Fraction(1, denominators[i])
        
        if total_frac < 1:
            return selected
    
    # Fallback greedy
    selected, total = [], Fraction()
    for i, d in enumerate(denominators):
        if total + Fraction(1, d) < 1:
            selected.append(i)
            total += Fraction(1, d)
    return selected
