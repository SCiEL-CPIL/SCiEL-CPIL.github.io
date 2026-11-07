def _check_divisibility_incremental(old_sums, new_sums):
    max_old = max(old_sums) if old_sums else 0
    max_new = max(new_sums)
    
    for a in new_sums:
        # a divides something in old
        for k in range(2, max_old // a + 1):
            if k * a in old_sums:
                return False
        # a divides something in new
        for k in range(2, max_new // a + 1):
            if k * a in new_sums:
                return False
                
    for a in old_sums:
        # a divides something in new
        for k in range(2, max_new // a + 1):
            if k * a in new_sums:
                return False
                
    return True
