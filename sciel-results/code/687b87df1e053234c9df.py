def _find_witness_random(trips_mask, t_idx):
    cands = _candidates[t_idx]
    if not cands:
        return None
    # Try up to 10 random witnesses
    for _ in range(10):
        cand = random.choice(cands)
        all_present = True
        for idx in cand:
            if not (trips_mask >> idx) & 1:
                all_present = False
                break
        if all_present:
            return cand
    return None
