import sys

def construct(n_rows, k):
    n_elem = k + 1
    diffs_per_row = n_elem * (n_elem - 1) // 2
    total_diffs = n_rows * diffs_per_row
    
    M_min = total_diffs
    M_max = M_min + 100
    
    def generate_rulers(k, M):
        rulers = []
        
        def backtrack(pos, current_row, used_diffs):
            if pos == n_elem:
                # Symmetry break: a_1 <= a_k - a_{k-1}
                if current_row[1] <= current_row[k] - current_row[k-1]:
                    rulers.append((tuple(current_row), frozenset(used_diffs)))
                return
            
            start = current_row[-1] + 1
            for val in range(start, M + 1):
                valid = True
                new_diffs = []
                for prev in current_row:
                    d = val - prev
                    if d in used_diffs:
                        valid = False
                        break
                    new_diffs.append(d)
                
                if valid:
                    backtrack(pos + 1, current_row + [val], used_diffs | set(new_diffs))
        
        backtrack(1, [0], set())
        # Sort by max value to prefer shorter rulers
        rulers.sort(key=lambda x: x[0][-1])
        return rulers

    def find_disjoint_rulers(rulers, n_rows):
        if len(rulers) < n_rows:
            return None
        
        def backtrack(idx, used_diffs, selected_rows):
            if len(selected_rows) == n_rows:
                return selected_rows[:]
            
            for i in range(idx, len(rulers)):
                row, diffs = rulers[i]
                if diffs.isdisjoint(used_diffs):
                    new_used = used_diffs | diffs
                    result = backtrack(i + 1, new_used, selected_rows + [list(row)])
                    if result is not None:
                        return result
            return None
        
        return backtrack(0, set(), [])

    for M in range(M_min, M_max + 1):
        rulers = generate_rulers(k, M)
        if not rulers:
            continue
        result = find_disjoint_rulers(rulers, n_rows)
        if result is not None:
            return result

    # Fallback
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
