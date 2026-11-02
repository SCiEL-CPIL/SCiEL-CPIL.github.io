            new_row = [0]
            new_row_diffs = set()
            x = 0
            success = False
            for _ in range(200):
                x += 1
                new = {x - a for a in new_row}
                # Check against other rows AND current new_row diffs
                if not (new & other_diffs) and not (new & new_row_diffs):
                    new_row.append(x)
                    new_row_diffs |= new
                    if len(new_row) == k + 1:
                        success = True
                        break
