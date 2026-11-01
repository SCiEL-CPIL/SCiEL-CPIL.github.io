def _backtrack_construct(n_rows, k, max_entry, time_limit=10.0):
    # ...
    used = set()
    
    def backtrack(row_idx, row):
        # ...
        # used is assigned here:
        # used |= _row_diffs(new_row)
        # so 'used' is local to backtrack
        # but read here:
        # used_copy = set(used)
        # hence error.
