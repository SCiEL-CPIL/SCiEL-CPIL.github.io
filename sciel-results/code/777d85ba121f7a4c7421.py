    mask_cache = {}
    def get_progression_mask(p, r):
        key = (p, r)
        if key in mask_cache:
            return mask_cache[key]
        mask = 0
        start = r
        if start == 0:
            start = p
        k = start
        while k <= CAP:
            mask |= (1 << k)
            k += p
        mask_cache[key] = mask
        return mask
