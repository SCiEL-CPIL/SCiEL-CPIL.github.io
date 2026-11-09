import time
import random

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin for 150s limit
    
    best_score = float('inf')
    best_rows = None
    
    # Number of restarts with different random orderings
    # We can adjust this based on time, but let's start with a fixed reasonable number
    # and break if time is up.
    
    for restart in range(1000):
        if time.time() > deadline:
            break
            
        # Random order of rows to interleave
        order = list(range(n_rows))
        random.shuffle(order)
        
        rows = [[0] for _ in range(n_rows)]
        used = set()
        success = True
        
        # We need to fill rows to length k+1
        # We will iterate through the order repeatedly until all are full or stuck
        # To avoid infinite loops if a row is stuck, we track how many times we've tried
        # or just break if no row could be extended in a full pass.
        
        max_iter = n_rows * (k + 1) * 5 # Safety cap
        iter_count = 0
        
        while any(len(r) < k + 1 for r in rows):
            if time.time() > deadline:
                success = False
                break
            if iter_count > max_iter:
                success = False
                break
            iter_count += 1
            
            extended_any = False
            for i in order:
                if len(rows[i]) >= k + 1:
                    continue
                    
                current_last = rows[i][-1]
                # Try to find next x
                # We can start searching from current_last + 1
                # Optimization: The new differences are x - a for a in rows[i].
                # The largest new difference is x - 0 = x.
                # The smallest is x - current_last.
                
                # Heuristic: try x starting from current_last + 1
                # But we might need to jump.
                # Let's try a range.
                found = False
                # Upper bound for x in this step: maybe current global max + some margin
                # Or just try sequentially.
                # Since k is small (5 more elements), sequential is fast.
                
                # To speed up, we can check conflicts quickly.
                # new_diffs = {x - a for a in rows[i]}
                # if new_diffs & used: conflict
                
                # We can precompute that x must not be in used (diff with 0)
                # and x - a not in used for other a.
                
                x = current_last + 1
                # We can limit the search to avoid getting stuck in a huge loop
                # if the current ordering is bad.
                # Let's say up to 200 steps or until we find one.
                limit = current_last + 200
                while x <= limit:
                    if x not in used:
                        # Check other differences
                        ok = True
                        for a in rows[i]:
                            if a == 0:
                                continue # x - 0 = x already checked
                            if (x - a) in used:
                                ok = False
                                break
                        if ok:
                            rows[i].append(x)
                            # Update used
                            for a in rows[i]:
                                if a == 0:
                                    used.add(x)
                                else:
                                    used.add(x - a)
                            extended_any = True
                            found = True
                            break
                    x += 1
                
                if not found:
                    # If we couldn't extend this row, we might be stuck.
                    # In a simple greedy, we might just fail this restart.
                    # Or we can try to "jump" further? 
                    # For now, let's mark as failed if we can't find within limit.
                    # Actually, if we can't find it, this ordering is bad.
                    success = False
                    break
            
            if not success:
                break
                
            if not extended_any:
                # No row could be extended in this pass
                success = False
                break
        
        if success:
            # Calculate score
            score = max(r[-1] for r in rows)
            if score < best_score:
                best_score = score
                best_rows = [r[:] for r in rows]
                
                # If we hit the theoretical lower bound or close to it, we can stop
                # Counting bound is 105. If we get <= 105, it's optimal (or close).
                # Actually, the problem says "best known set has largest entry 112".
                # If we find 112 or better, we should try to improve.
                # If we find 105, we are done.
                if best_score <= 105:
                    break

    if best_rows is None:
        # Fallback to simple greedy if all random interleaves failed
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
