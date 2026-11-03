import time
import itertools

def construct(n_rows, k):
    def greedy():
        used, rows = set(), []
        for _ in range(n_rows):
            row, x = [0], 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows

    diffs_per_row = (k + 1) * k // 2
    min_M = n_rows * diffs_per_row
    
    start_time = time.time()
    deadline = start_time + 115
    
    greedy_rows = greedy()
    greedy_M = max(max(row) for row in greedy_rows)
    
    if min_M >= greedy_M:
        return greedy_rows

    # Search M from min_M to greedy_M - 1
    for M in range(min_M, greedy_M):
        if time.time() > deadline:
            break
            
        # DFS
        def search(row_idx, last_combo, used_mask, current_rows):
            if time.time() > deadline:
                return None
            
            if row_idx == n_rows:
                return current_rows
            
            # Iterate combinations
            # Optimization: If k is small, we can unroll? No.
            
            # To speed up, we can check if the remaining capacity is sufficient.
            # Max diffs remaining = n_rows - row_idx
            # Each row needs diffs_per_row distinct diffs.
            # Total needed = (n_rows - row_idx) * diffs_per_row
            # Available diffs = M - len(used_mask) (roughly)
            # If available < needed, prune.
            
            # A tighter bound:
            # The smallest unused diff is at least 1.
            # The largest is M.
            # Count of unused diffs in [1, M] is M - popcount(used_mask).
            # If M - popcount(used_mask) < (n_rows - row_idx) * diffs_per_row:
            #     return None
            
            # popcount is fast in Python 3.8+ (int.bit_count())
            
            if M - used_mask.bit_count() < (n_rows - row_idx) * diffs_per_row:
                return None
            
            for combo in itertools.combinations(range(1, M + 1), k):
                if combo <= last_combo:
                    continue
                
                row = [0] + list(combo)
                mask = 0
                valid = True
                # Calculate mask
                # We can do this faster if we pre-compute diffs for the combo?
                # No, combo is new.
                
                for i in range(len(row)):
                    ai = row[i]
                    for j in range(i + 1, len(row)):
                        d = row[j] - ai
                        bit = 1 << d
                        if mask & bit:
                            valid = False
                            break
                        mask |= bit
                    if not valid:
                        break
                
                if not valid:
                    continue
                
                if used_mask & mask:
                    continue
                
                res = search(row_idx + 1, combo, used_mask | mask, current_rows + [row])
                if res is not None:
                    return res
            
            return None

        res = search(0, (), 0, [])
        if res is not None:
            return res

    return greedy_rows
