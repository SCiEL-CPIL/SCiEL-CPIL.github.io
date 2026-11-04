import random
import time

def construct(n_rows, k):
    start_time = time.time()
    time_limit = 110.0  # Leave 10s margin
    
    best_rows = None
    best_score = float('inf')
    
    # Helper to check if a row is valid given used diffs
    def is_valid_row(row, used):
        diffs = set()
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                d = row[j] - row[i]
                if d in used or d in diffs:
                    return False
                diffs.add(d)
        return True

    # Helper to get valid next elements
    def get_valid_next(row, used):
        last = row[-1]
        valids = []
        # We only need to check up to a reasonable bound, e.g., best_score
        # But initially best_score is inf, so we need a cap.
        # Let's cap at 200 for initial search, then tighten.
        cap = min(200, best_score if best_score < float('inf') else 200)
        for x in range(last + 1, cap + 1):
            new_diffs = {x - a for a in row}
            if not (new_diffs & used):
                # Check internal uniqueness
                if len(new_diffs) == len(row):
                    valids.append(x)
        return valids

    # Randomized greedy construction
    def randomized_greedy():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                valids = get_valid_next(row, used)
                if not valids:
                    return None
                # Bias towards smaller values
                # Pick with probability proportional to 1/x or just uniform from first few
                # Let's pick uniformly from the first 3 valids if available, else the smallest
                if len(valids) > 3:
                    # Weighted choice: smaller x has higher weight
                    weights = [1.0 / (v + 1) for v in valids[:10]]
                    total_w = sum(weights)
                    r = random.uniform(0, total_w)
                    acc = 0
                    chosen = valids[0]
                    for v, w in zip(valids[:10], weights):
                        acc += w
                        if r <= acc:
                            chosen = v
                            break
                else:
                    # If few options, pick smallest to keep scope tight
                    chosen = valids[0]
                
                row.append(chosen)
                used.update({chosen - a for a in row[:-1]})
            rows.append(row)
        return rows

    # Local search: try to reduce the max element
    def local_search(rows):
        current_max = max(max(r) for r in rows)
        improved = True
        while improved:
            improved = False
            # Try to move each row to a smaller configuration
            for i in range(n_rows):
                # Remove row i from used
                row_diffs = set()
                for a in rows[i]:
                    for b in rows[i]:
                        if a != b:
                            row_diffs.add(abs(a - b))
                used_wo_i = set()
                for j in range(n_rows):
                    if j == i: continue
                    for a in rows[j]:
                        for b in rows[j]:
                            if a != b:
                                used_wo_i.add(abs(a - b))
                
                # Try to reconstruct row i with smaller max
                # This is expensive. Maybe just try to shift the whole row?
                # Or try to replace the last element with a smaller one?
                pass # Local search is complex, let's rely on restarts first.
        
        return rows, current_max

    # Main loop
    attempts = 0
    while time.time() - start_time < time_limit:
        rows = randomized_greedy()
        if rows is not None:
            score = max(max(r) for r in rows)
            if score < best_score:
                best_score = score
                best_rows = rows
                # If we hit the theoretical bound, we can stop
                if best_score <= 105:
                    break
        attempts += 1
        if attempts % 1000 == 0:
            # print(f"Attempt {attempts}, best score {best_score}")
            pass

    if best_rows is None:
        # Fallback to deterministic greedy if random failed (unlikely)
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                x = row[-1] + 1
                while True:
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used) and len(new_diffs) == len(row):
                        break
                    x += 1
                row.append(x)
                used.update({x - a for a in row[:-1]})
            rows.append(row)
        best_rows = rows

    return best_rows
