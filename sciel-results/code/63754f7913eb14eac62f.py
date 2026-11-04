        def backtrack(i, current_max):
            if i == k:
                return row[:]
            
            if M - current_max < (k - i):
                return None
            
            for x in range(current_max + 1, M + 1):
                new_diffs = {x - a for a in row}
                # Check internal distinctness
                all_diffs = get_diffs(row + [x])
                if len(all_diffs) != (len(row) + 1) * len(row) // 2:
                    continue
                # Check external distinctness
                if all_diffs & used_diffs:
                    continue
                
                row.append(x)
                res = backtrack(i + 1, x)
                if res:
                    return res
                row.pop()
            return None
