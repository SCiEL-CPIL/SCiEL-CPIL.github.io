import time
import random

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    best_rows = None
    best_scope = float('inf')
    
    def greedy_solve(seed):
        """Greedy approach with randomization in tie-breaking."""
        rng = random.Random(seed)
        used = set()
        rows = []
        
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                # Find the smallest x that works, but with some randomization
                # Collect the first few valid candidates
                candidates = []
                x += 1
                found_any = False
                while not found_any:
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used:
                        candidates.append(x)
                        if len(candidates) >= 5:
                            found_any = True
                    x += 1
                
                if not candidates:
                    return None, float('inf')
                
                # Pick randomly from the first few candidates
                chosen = rng.choice(candidates)
                row.append(chosen)
                used |= {chosen - a for a in row[:-1]}
                x = chosen
            
            rows.append(row)
        
        scope = max(max(row) for row in rows)
        return rows, scope
    
    # First try the deterministic greedy (seed 0)
    rows, scope = greedy_solve(0)
    if rows is not None:
        best_rows = rows
        best_scope = scope
    
    # Try multiple random restarts
    attempts = 0
    while time.time() < deadline:
        seed = attempts + 1
        rows, scope = greedy_solve(seed)
        if rows is not None and scope < best_scope:
            best_rows = rows
            best_scope = scope
        attempts += 1
    
    if best_rows is None:
        # Fallback: pure greedy
        used = set()
        best_rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            best_rows.append(row)
        best_scope = max(max(row) for row in best_rows)
    
    return best_rows
