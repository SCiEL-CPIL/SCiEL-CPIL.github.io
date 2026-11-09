import time
import random

def construct():
    start_time = time.time()
    time_limit = 55.0  # Safety margin
    
    # Fallback: Power of 2 sequence
    # 1, 2, 4, 8, 16, 32, 64, 128, 256
    best_seq = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    best_len = len(best_seq)
    
    def is_valid_convex_golomb(prefix_sums):
        """
        prefix_sums: list [P_0, P_1, ..., P_k]
        Checks if differences P_j - P_i are distinct for i < j.
        Also checks if gaps P_i - P_{i-1} are strictly increasing.
        """
        # Check strictly increasing gaps
        for i in range(2, len(prefix_sums)):
            gap_curr = prefix_sums[i] - prefix_sums[i-1]
            gap_prev = prefix_sums[i-1] - prefix_sums[i-2]
            if gap_curr <= gap_prev:
                return False
        
        # Check distinct differences
        diffs = set()
        n = len(prefix_sums)
        for i in range(n):
            for j in range(i+1, n):
                d = prefix_sums[j] - prefix_sums[i]
                if d in diffs:
                    return False
                diffs.add(d)
        return True

    def search_backtrack(target_len, max_val=256):
        """
        Finds a convex Golomb ruler of length target_len (number of marks = target_len + 1? 
        No, sequence length L means L gaps, L+1 marks.
        Let's define L as the length of the sequence a.
        Marks P_0...P_L.
        """
        # P_0 = 0
        # P_1 = a_1 >= 1
        # ...
        # P_L = a_1 + ... + a_L <= 256 * L (roughly)
        
        prefix_sums = [0]
        used_diffs = set()
        
        def backtrack(k, last_gap):
            if time.time() - start_time > time_limit:
                return None
            
            if k == target_len:
                return prefix_sums[:]
            
            # Next gap a_k+1 must be > last_gap
            # Max possible value for a_k+1?
            # The total sum P_L must be such that differences fit.
            # Heuristic: don't go too high too fast, but must increase.
            
            # Upper bound for next gap:
            # If we pick gap g, P_k = P_{k-1} + g.
            # We need to ensure we can still complete the sequence.
            
            # Simple bound: g <= 256
            # Also, to allow strictly increasing gaps, if we are at step k,
            # we need at least target_len - k gaps remaining, all > g.
            # Min sum of remaining gaps: (k+1) + (k+2) + ...
            # This bound is loose.
            
            # Let's iterate g from last_gap + 1 to 256
            # To improve search, we can try larger gaps first or smaller?
            # Smaller gaps allow more room for future large gaps?
            # Actually, smaller gaps pack marks closer, increasing collision risk for differences.
            # Larger gaps spread marks, reducing collision risk for differences.
            # So try larger gaps first?
            
            # However, we are constrained by max_val 256 for each gap.
            
            for g in range(256, last_gap, -1): # Try large gaps first to spread out
                new_p = prefix_sums[-1] + g
                # Check new differences
                valid = True
                new_diffs = []
                for i in range(len(prefix_sums)):
                    d = new_p - prefix_sums[i]
                    if d in used_diffs:
                        valid = False
                        break
                    new_diffs.append(d)
                
                if not valid:
                    continue
                
                # Also check if g is strictly increasing (guaranteed by loop)
                
                prefix_sums.append(new_p)
                used_diffs.update(new_diffs)
                
                res = backtrack(k+1, g)
                if res is not None:
                    return res
                
                prefix_sums.pop()
                used_diffs.difference_update(new_diffs)
                
            return None
        
        return backtrack(1, 0) # Start with P_0=0, next gap > 0

    # Try to find longer sequences
    # The power of 2 sequence has length 9.
    # Let's try 10, 11, 12...
    
    # We can also try to optimize the existing best_seq
    # But backtracking from scratch with heuristics might find better ones.
    
    # To avoid getting stuck, we do multiple random restarts with different orderings?
    # The current backtrack is deterministic.
    # Let's add a bit of randomness to the gap selection order or start points.
    
    def search_randomized(target_len, num_restarts=10):
        best_found = None
        for _ in range(num_restarts):
            if time.time() - start_time > time_limit:
                break
            
            # Randomize the search order slightly?
            # Or just run the deterministic one with a slight perturbation in bounds?
            # For now, let's just run the deterministic one.
            # If we want randomness, we can shuffle the candidate gaps.
            
            # Let's modify backtrack to accept a random seed or shuffle logic.
            # For simplicity, we'll just run the deterministic one.
            # If it fails, we try smaller lengths.
            
            res = search_backtrack(target_len)
            if res:
                # Convert prefix sums to gaps
                gaps = [res[i] - res[i-1] for i in range(1, len(res))]
                # Verify strictly increasing
                if all(gaps[i] < gaps[i+1] for i in range(len(gaps)-1)):
                    if all(1 <= g <= 256 for g in gaps):
                        if best_found is None or len(gaps) > len(best_found):
                            best_found = gaps
                break # Found one, no need to restart for same length
        
        return best_found

    # Try lengths 15 down to 10
    for L in range(15, 9, -1):
        if time.time() - start_time > time_limit:
            break
        res = search_randomized(L)
        if res:
            if len(res) > best_len:
                best_seq = res
                best_len = len(res)
            break # Found the longest possible in this range? 
            # Actually, we should continue to see if we can go higher?
            # The loop goes 15, 14... so if 15 works, we stop.
            # If 15 fails, we try 14.
            
    # If we didn't find anything better than 9, return best_seq
    # But wait, the seed program might have found length 10 or 11.
    # My backtracking might be too slow or fail to find it if the search space is tricky.
    
    # Let's also run the greedy algorithm from the seed to compare?
    # No, I should just return the best I found.
    
    # One more check: Is the power of 2 sequence the best?
    # Often, "sparse" sequences are good for Golomb rulers.
    
    # Let's add a specific check for the seed's greedy result to ensure we don't regress.
    # Re-implementing seed greedy quickly:
    def greedy_construct():
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

    greedy_seq = greedy_construct()
    if len(greedy_seq) > best_len:
        best_seq = greedy_seq
        best_len = len(best_seq)

    return best_seq
