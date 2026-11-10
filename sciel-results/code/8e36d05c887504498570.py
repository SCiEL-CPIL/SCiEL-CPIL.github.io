import time
import random

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    best_rows = None
    best_scope = float('inf')
    
    # Set random seed for reproducibility within a run, but change it for restarts
    # Actually, we want different restarts to be different, so we don't fix the seed globally.
    
    while time.time() < deadline:
        used = set()
        rows = []
        success = True
        
        for _ in range(n_rows):
            row = [0]
            current_max = 0
            while len(row) < k + 1:
                # Find candidates
                candidates = []
                x = current_max + 1
                # Limit search window to avoid infinite loops if impossible (though it should always be possible)
                # A reasonable upper bound for x in a greedy step is maybe current_max + k + 10 or similar, 
                # but since we are greedy, x won't jump too far.
                # Let's just scan upwards.
                while True:
                    # Check if x is valid
                    valid = True
                    for a in row:
                        if x - a in used:
                            valid = False
                            break
                    if valid:
                        candidates.append(x)
                        if len(candidates) >= 5: # Take first 5 valid candidates
                            break
                    x += 1
                    # Safety: if x gets too large, break to prevent hanging
                    # The scope is bounded by the counting bound ~ n_rows * k * (k+1)/2 + something
                    # For n_rows=4, k=8, max diff is 36 per row, 144 total.
                    # x should not exceed ~200.
                    if x > 500:
                        break
                
                if not candidates:
                    success = False
                    break
                
                # Pick a random candidate from the small set
                next_x = random.choice(candidates)
                row.append(next_x)
                for a in row[:-1]: # Exclude the new element itself
                    used.add(next_x - a)
                current_max = next_x
            
            if not success:
                break
            rows.append(row)
        
        if success:
            scope = max(max(r) for r in rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = rows
                # If we reach the theoretical lower bound or a very good bound, we can stop early
                # Counting bound: n_rows * k * (k+1) / 2 is the number of differences.
                # Max diff in one row is m. Number of diffs in one row is k(k+1)/2.
                # So m must be at least k(k+1)/2.
                # Total diffs is n_rows * k(k+1)/2.
                # If max scope is M, total possible diffs is M(M+1)/2.
                # So M(M+1)/2 >= n_rows * k(k+1)/2 => M ~ sqrt(n_rows * k(k+1)).
                # For n_rows=4, k=8: 4*36 = 144. sqrt(144)=12. But diffs are within rows, so they don't overlap with other rows' diffs?
                # Wait, "all differences ... are distinct over all rows".
                # So the set of ALL differences from ALL rows must be distinct.
                # Total number of differences is n_rows * k * (k+1) / 2.
                # These differences are integers between 1 and M.
                # So M must be at least n_rows * k * (k+1) / 2.
                # For n_rows=4, k=8: 4 * 36 = 144.
                # So M >= 144?
                # The task says "simple counting bound is 112".
                # Let's re-read carefully.
                # "all differences between two entries of the same row are distinct over all rows"
                # This means if d1 is a diff in row 1 and d2 is a diff in row 2, d1 != d2.
                # So yes, all differences across all rows must be unique.
                # Total diffs = n_rows * C(k+1, 2) = n_rows * k(k+1)/2.
                # Max diff is M.
                # So we need M(M+1)/2 >= n_rows * k(k+1)/2.
                # M(M+1) >= n_rows * k(k+1).
                # For n_rows=4, k=8: 4*72 = 288.
                # M(M+1) >= 288. 16*17=272, 17*18=306. So M >= 17.
                # This contradicts the "counting bound is 112".
                
                # Let's re-read the task description.
                # "Choose 4 rows of 8 increasing integers starting at 0" -> k=7?
                # "list of k + 1 integers" -> if k=8, list has 9 integers.
                # "4 rows of 8 increasing integers" usually means 8 integers.
                # But the interface says `construct(n_rows, k)` and "Each row is a list of k + 1 integers".
                # If the task says "4 rows of 8 increasing integers", it likely means 8 integers per row.
                # So k+1 = 8 => k=7.
                # Let's check the bound for k=7.
                # Total diffs = 4 * 7*8/2 = 4 * 28 = 112.
                # M(M+1)/2 >= 112 => M(M+1) >= 224.
                # 14*15=210, 15*16=240. So M >= 15.
                # Still not 112.
                
                # Maybe the bound refers to the sum of max entries? No, "largest entry as small as possible".
                # Maybe the counting bound is on the *sum* of differences? No.
                # Let's look at the "simple counting bound is 112".
                # If k=8 (9 integers), diffs per row = 36. Total = 144.
                # If the bound is 112, maybe it's related to something else.
                
                # Regardless, we just minimize the scope.
                if best_scope <= 134: # Best known is 134. If we hit this, we can stop.
                    break

    if best_rows is None:
        # Fallback: return a valid, if not optimal, set
        # Use the original greedy to ensure we return *something* valid
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
        
    return best_rows
