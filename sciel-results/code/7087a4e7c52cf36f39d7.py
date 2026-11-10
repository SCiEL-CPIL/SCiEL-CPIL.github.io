import time
import random

def construct(n_rows, k):
    deadline = time.time() + 115
    
    def get_diffs(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    def build_row(used, limit, temp=0.0, time_limit=0.5):
        """
        Build a row of k+1 integers starting at 0, strictly increasing, max <= limit.
        All diffs must be disjoint from 'used'.
        Returns (row, used_diffs_set) or (None, None) if not found.
        """
        start_time = time.time()
        
        def dfs(current_row, current_used):
            if time.time() - start_time > time_limit:
                return None, None
            
            if len(current_row) == k + 1:
                return current_row, current_used
            
            last = current_row[-1]
            start = last + 1
            candidates = list(range(start, limit + 1))
            
            if temp > 0:
                random.shuffle(candidates)
            else:
                # Ascending order to prefer small max
                pass

            for x in candidates:
                new_diffs = {x - a for a in current_row}
                if new_diffs & current_used:
                    continue
                
                new_used = current_used | new_diffs
                res = dfs(current_row + [x], new_used)
                if res[0] is not None:
                    return res
            return None, None

        row, used_set = dfs([0], set())
        return row, used_set

    best_rows = None
    best_max = float('inf')
    
    attempts = 0
    # We will try to find a solution.
    # We start with a limit of 112 (best known). If we fail, we increase limit?
    # Or just let the solver find the min max.
    # The solver prefers small numbers, so it will find small max if possible.
    
    while time.time() < deadline and attempts < 30:
        attempts += 1
        random.seed(attempts * 12345)
        
        used = set()
        rows = []
        success = True
        
        for i in range(n_rows):
            if time.time() > deadline - 5:
                success = False
                break
            
            # Use a limit of 112 first. If it fails, try higher?
            # Actually, if we can't find a row with limit 112, we might need 113.
            # But if we find all 7 rows with limit 112, we are done.
            # Let's try limit 112 first.
            row, new_used = build_row(used, 112, temp=0.1, time_limit=1.0)
            
            if row is None:
                # Try with higher limit
                row, new_used = build_row(used, 150, temp=0.1, time_limit=1.0)
                if row is None:
                    success = False
                    break
            
            rows.append(row)
            used = new_used
        
        if success:
            mv = max(max(r) for r in rows)
            if mv < best_max:
                best_max = mv
                best_rows = list(rows)
                
            if mv <= 112:
                # Try to improve further (find 111 or 105)
                for _ in range(10):
                    if time.time() > deadline - 5: break
                    random.seed(random.randint(0, 100000))
                    used = set()
                    rows = []
                    success = True
                    for i in range(n_rows):
                        row, new_used = build_row(used, 111, temp=0.2, time_limit=0.5)
                        if row is None:
                            success = False
                            break
                        rows.append(row)
                        used = new_used
                    if success:
                        mv = max(max(r) for r in rows)
                        if mv < best_max:
                            best_max = mv
                            best_rows = list(rows)
                break # Stop main loop
    
    if best_rows is None:
        # Fallback
        used = set()
        rows = []
        for _ in range(n_rows):
            row, used = build_row(used, 200, temp=0.0, time_limit=1.0)
            rows.append(row)
        best_rows = rows
        
    return best_rows
