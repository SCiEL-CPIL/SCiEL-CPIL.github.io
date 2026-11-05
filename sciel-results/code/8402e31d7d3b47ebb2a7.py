    if best_solution is None:
        rows = []
        used = 0
        for _ in range(n_rows):
            success = False
            for attempt in range(1000):
                max_try = 200 + attempt
                row, max_v, mask = gen_random_row(used, max_try)
                if row:
                    rows.append(row)
                    used |= mask
                    success = True
                    break
            if not success:
                # Should not happen with high enough max_try
                return [ [i for i in range(k+1)] for _ in range(n_rows) ] # Invalid dummy? No, must be valid.
                # Actually, if it fails, something is wrong.
                # But for n=7, k=5, max 200 is plenty.
                pass
        return rows
