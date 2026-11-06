import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    n = 128
    target_lo = 128
    target_hi = 256
    target_range = list(range(target_lo, target_hi + 1))
    
    # Helper to calculate counts for a given set A
    # Returns a list of counts for sums in [0, 256] (or just target range)
    def calc_counts(A):
        counts = [0] * (target_hi + 1)
        A_list = list(A)
        for i in range(len(A_list)):
            a = A_list[i]
            for j in range(i, len(A_list)):
                b = A_list[j]
                s = a + b
                if target_lo <= s <= target_hi:
                    counts[s] += 1
        return counts

    # Evaluate the score: (uncovered, peak)
    # Lower is better.
    def get_score(counts):
        uncovered = 0
        peak = 0
        for s in range(target_lo, target_hi + 1):
            c = counts[s]
            if c == 0:
                uncovered += 1
            elif c > peak:
                peak = c
        return (uncovered, peak)

    # Greedy construction to get a valid initial solution
    def greedy_construct():
        # Start with a set that likely covers the range.
        # {0, 128} covers 128.
        # We need to cover [128, 256].
        # Let's try a simple greedy:
        # 1. Start with A = {0, 128}
        # 2. While there are uncovered sums:
        #    a. Find the uncovered sum s with minimum current count (0).
        #    b. Find candidate x in [0, 128] \ A such that adding x covers s.
        #       x covers s if there exists y in A such that x+y=s, or 2x=s.
        #    c. Among candidates, pick the one that minimizes the resulting peak.
        #    d. Add x to A.
        
        A = {0, 128}
        counts = calc_counts(A)
        
        while True:
            if time.time() > deadline:
                break
                
            uncovered, peak = get_score(counts)
            if uncovered == 0:
                break
            
            # Find first uncovered sum
            target_s = -1
            for s in range(target_lo, target_hi + 1):
                if counts[s] == 0:
                    target_s = s
                    break
            
            if target_s == -1:
                break
                
            # Find candidates x
            candidates = []
            for x in range(n + 1):
                if x in A:
                    continue
                # Check if x covers target_s
                covers = False
                for y in A:
                    if x + y == target_s:
                        covers = True
                        break
                if not covers and 2 * x == target_s:
                    covers = True
                
                if covers:
                    candidates.append(x)
            
            if not candidates:
                # Should not happen if we can always extend, but just in case
                # Try to add a random element? Or break.
                break
            
            best_x = None
            best_score = (float('inf'), float('inf'))
            
            for x in candidates:
                # Simulate adding x
                # We can update counts incrementally for speed, but full recalc is safer and fast enough for small A
                # However, as A grows, full recalc is O(|A|^2). |A| might be ~50. 2500 ops. 128 targets.
                # Let's do a fast simulation.
                
                new_counts = counts[:]
                # Add pairs (x, y) for y in A, and (x, x)
                for y in A:
                    s = x + y
                    if target_lo <= s <= target_hi:
                        new_counts[s] += 1
                s_2x = 2 * x
                if target_lo <= s_2x <= target_hi:
                    new_counts[s_2x] += 1
                
                score = get_score(new_counts)
                if score < best_score:
                    best_score = score
                    best_x = x
            
            if best_x is None:
                break
                
            A.add(best_x)
            counts = calc_counts(A) # Recalculate for consistency
            
        return A, get_score(calc_counts(A))

    # Run greedy to get a valid starting point
    A_best, score_best = greedy_construct()
    
    # If greedy didn't cover everything, we might need to force coverage
    # But greedy should always find a way unless it's stuck in a local minimum of "no candidates for uncovered sum".
    # With 0 and 128 in A, we can always add x to cover s if s-x is in A or x=s/2.
    # Since 0 is in A, s-0=s. If s<=128, x=s is candidate.
    # Since 128 is in A, s-128=x. If s>=128, x=s-128 is candidate (if x<=128).
    # So for any s in [128, 256], x=s-128 is in [0, 128].
    # So candidates are never empty.
    
    # Now, try to improve the peak using local search (hill climbing / simulated annealing)
    # Moves:
    # 1. Add an element x not in A.
    # 2. Remove an element x from A (if coverage is maintained).
    # 3. Swap: remove x, add y.
    
    # We want to minimize peak, subject to uncovered=0.
    
    current_A = set(A_best)
    current_counts = calc_counts(current_A)
    current_score = get_score(current_counts)
    
    best_A = set(current_A)
    best_score = current_score
    
    # To allow removing elements, we need to ensure coverage is not broken.
    # Let's do a simple hill climbing with restarts or just local moves.
    
    # Time budget for local search
    ls_deadline = deadline - 2.0
    
    while time.time() < ls_deadline:
        # Pick a random move
        # 1. Add
        if random.random() < 0.4:
            # Add a random element
            x = random.randint(0, n)
            if x in current_A:
                continue
            new_A = set(current_A)
            new_A.add(x)
            new_counts = calc_counts(new_A)
            new_score = get_score(new_counts)
            
            # Accept if better, or if same uncovered and lower peak
            if new_score < current_score:
                current_A = new_A
                current_counts = new_counts
                current_score = new_score
                if current_score < best_score:
                    best_A = set(current_A)
                    best_score = current_score
            # Accept worse moves with some probability? 
            # For now, strict improvement only to be safe, or simple SA.
            # Let's try a bit of SA.
            else:
                # If uncovered is same, and peak is higher, accept with prob exp(-delta/T)
                if current_score[0] == new_score[0]:
                    delta = new_score[1] - current_score[1]
                    if delta > 0:
                        # Temperature
                        T = 1.0
                        if random.random() < (2.718 ** (-delta / T)):
                            current_A = new_A
                            current_counts = new_counts
                            current_score = new_score
                            if current_score < best_score:
                                best_A = set(current_A)
                                best_score = current_score
                            
        # 2. Remove
        elif random.random() < 0.4:
            if len(current_A) <= 2:
                continue
            x = random.choice(list(current_A))
            new_A = set(current_A)
            new_A.remove(x)
            new_counts = calc_counts(new_A)
            new_score = get_score(new_counts)
            
            # Only accept if uncovered is still 0 (or same) and peak is lower
            if new_score[0] == current_score[0] and new_score < current_score:
                current_A = new_A
                current_counts = new_counts
                current_score = new_score
                if current_score < best_score:
                    best_A = set(current_A)
                    best_score = current_score
            # Or if it breaks coverage, we might want to reject immediately to stay valid?
            # The problem requires EVERY integer to be a sum. So uncovered must be 0.
            # If we accept a move that increases uncovered, we are in an invalid state.
            # Let's enforce validity: only accept moves that keep uncovered=0.
            # So the condition above is correct: new_score[0] must be 0 (assuming current is 0).
            # If current is invalid (uncovered>0), we should try to fix it.
            # But greedy should give a valid start.
            
        # 3. Swap
        else:
            if len(current_A) == 0:
                continue
            x = random.choice(list(current_A))
            y = random.randint(0, n)
            if y == x:
                continue
            if y in current_A:
                continue
            
            new_A = set(current_A)
            new_A.remove(x)
            new_A.add(y)
            
            new_counts = calc_counts(new_A)
            new_score = get_score(new_counts)
            
            if new_score[0] == current_score[0] and new_score < current_score:
                current_A = new_A
                current_counts = new_counts
                current_score = new_score
                if current_score < best_score:
                    best_A = set(current_A)
                    best_score = current_score
            else:
                # Allow some exploration?
                # If we break validity, we are stuck.
                # Let's stick to valid moves only for now.
                pass

    # Ensure best_A is valid
    final_counts = calc_counts(best_A)
    final_score = get_score(final_counts)
    if final_score[0] > 0:
        # Fallback to greedy result if best_A became invalid somehow
        return list(A_best)
        
    return list(best_A)

# Test run
# if __name__ == "__main__":
#     res = construct()
#     print(len(res))
#     # Verify
#     counts = [0]*257
#     res_set = set(res)
#     for i, a in enumerate(res):
#         for j in range(i, len(res)):
#             b = res[j]
#             s = a+b
#             if 128 <= s <= 256:
#                 counts[s] += 1
#     peak = max(counts[128:257])
#     unc = sum(1 for c in counts[128:257] if c==0)
#     print(f"Peak: {peak}, Uncovered: {unc}")

