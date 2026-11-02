import time

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140.0
    
    for scope in range(105, 130):
        if time.time() > deadline:
            break
        result = _solve(n_rows, k, scope, start_time, deadline)
        if result is not None:
            return result
    
    return _greedy(n_rows, k)


def _solve(n_rows, k, scope, start_time, deadline):
    used = 0
    rows = []
    
    def build_row():
        if time.time() > deadline:
            return None
        row = [0]
        return _build_row_bt(row, k, scope, used, start_time, deadline)
    
    def _build_row_bt(row, k, scope, used_mask, start_time, deadline):
        if time.time() > deadline:
            return None
        if len(row) == k + 1:
            return row
        last = row[-1]
        for val in range(last + 1, scope + 1):
            new_mask = 0
            ok = True
            for a in row:
                d = val - a
                if d <= 0 or (used_mask >> d) & 1:
                    ok = False
                    break
                new_mask |= (1 << d)
            if not ok:
                continue
            row.append(val)
            res = _build_row_bt(row, k, scope, used_mask | new_mask, start_time, deadline)
            if res is not None:
                return res
            row.pop()
        return None
    
    for _ in range(n_rows):
        row = build_row()
        if row is None:
            return None
        # Update used
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                used |= (1 << (row[j] - row[i]))
        rows.append(row)
    
    return rows


def _greedy(n_rows, k):
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
        rows.append(row)
    return rows
