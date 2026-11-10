"""Optimized program: Beam Search + Targeted Backtracking to maximize sequence length."""
import time
import sys

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 75  # safety margin before 90s kill

    # Greedy baseline (fast, gives a valid starting point)
    def greedy():
        values, sums = [], set()
        for x in range(1, 257):
            tails, total = [x], x
            for a in reversed(values):
                total += a
                tails.append(total)
            if not sums.intersection(tails):
                values.append(x)
                sums.update(tails)
        return values

    best = greedy()
    best_len = len(best)

    def check_add(values, sums, x):
        """Check if x can be appended; return new sums list or None."""
        total = x
        new_sums = [total]
        for a in reversed(values):
            total += a
            new_sums.append(total)
            if total in sums:
                return None
        return new_sums

    def beam_search(width, heuristic):
        """
        Perform beam search.
        Returns the longest valid sequence found.
        """
        # state: (values, sums, total_sum)
        beam = [([], set(), 0)]
        best_found = []
        
        for depth in range(1, 30): # Max depth guess
            if time.time() > deadline:
                break
            if not beam:
                break
            
            next_beam = []
            
            for values, sums, t_sum in beam:
                start_x = values[-1] + 1 if values else 1
                
                for x in range(start_x, 257):
                    if time.time() > deadline:
                        return best_found
                    
                    new_sums = check_add(values, sums, x)
                    if new_sums is None:
                        continue
                        
                    new_values = values + [x]
                    new_sums_set = sums.union(new_sums)
                    new_t_sum = t_sum + x
                    
                    score = heuristic(new_values, new_sums_set, new_t_sum)
                    next_beam.append((score, new_values, new_sums_set, new_t_sum))
            
            if not next_beam:
                break
                
            next_beam.sort(key=lambda item: item[0])
            
            # Update best found
            if depth > len(best_found):
                best_found = next_beam[0][1]
                
            beam = [(v, s, t) for _, v, s, t in next_beam[:width]]
            
        return best_found

    # Heuristic 1: Minimize total sum
    def h_min_sum(values, sums, t_sum):
        return t_sum

    # Heuristic 2: Minimize sum of last 3 elements
    def h_last_sum(values, sums, t_sum):
        if len(values) < 3:
            return sum(values)
        return sum(values[-3:])

    # Run Beam Search with different widths and heuristics
    # Width 100, Heuristic: Min Sum
    b1 = beam_search(100, h_min_sum)
    if len(b1) > best_len:
        best = b1
        best_len = len(best)

    if time.time() < deadline - 10:
        # Width 200, Heuristic: Min Last Sum
        b2 = beam_search(200, h_last_sum)
        if len(b2) > best_len:
            best = b2
            best_len = len(best)
            
    # Final polish: Try to extend the best sequence found using backtracking
    # Start from the best sequence found and try to add more elements
    # This is a targeted search
    def extend_best(values, sums, t_sum):
        nonlocal best, best_len
        if time.time() > deadline:
            return
        
        if len(values) > best_len:
            best = list(values)
            best_len = len(values)
            
        start_x = values[-1] + 1 if values else 1
        for x in range(start_x, 257):
            if time.time() > deadline:
                return
            new_sums = check_add(values, sums, x)
            if new_sums is None:
                continue
            new_values = values + [x]
            new_sums_set = sums.union(new_sums)
            new_t_sum = t_sum + x
            
            extend_best(new_values, new_sums_set, new_t_sum)

    # Get sums for best sequence
    if best:
        # Reconstruct sums for best
        best_sums = set()
        for i in range(len(best)):
            for j in range(i, len(best)):
                s = sum(best[i:j+1])
                best_sums.add(s)
        # We need total sum
        best_t_sum = sum(best)
        # To avoid deep recursion issues, we limit depth or use iterative?
        # Python recursion limit is ~1000. Sequence length is < 50. Safe.
        extend_best(list(best), best_sums, best_t_sum)

    return best
# EVOLVE-BLOCK-END
