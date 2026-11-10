"""Optimized program: hybrid greedy-prefix and fresh backtracking with strong pruning."""
import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin

    def greedy():
        values, sums = [], set()
        for x in range(1, 257):
            # Compute new sums if we add x
            # Sums are: x, x+last, x+last+prev, ...
            new_sums = []
            total = x
            new_sums.append(total)
            for a in reversed(values):
                total += a
                new_sums.append(total)
            
            # Check for collision
            if not sums.intersection(new_sums):
                values.append(x)
                sums.update(new_sums)
        return values

    best = greedy()
    best_len = len(best)
    
    # Pre-compute sums for best
    def get_sums(vals):
        s = set()
        for i in range(len(vals)):
            t = 0
            for j in range(i, len(vals)):
                t += vals[j]
                s.add(t)
        return s

    best_sums = get_sums(best)

    def check_add(vals, sums, x):
        """
        Check if x can be appended to vals.
        Returns the list of new sums if valid, else None.
        """
        total = x
        new_sums = [total]
        for a in reversed(vals):
            total += a
            new_sums.append(total)
        
        # Check collisions with existing sums
        # Also, new_sums are strictly increasing because vals are positive and increasing.
        for s in new_sums:
            if s in sums:
                return None
        return new_sums

    def backtrack(vals, sums, lo, target_len, tlimit):
        """
        Recursive backtracking.
        vals: current sequence
        sums: set of all consecutive block sums so far
        lo: lower bound for next value
        target_len: desired length
        tlimit: time limit for this branch
        """
        if time.time() > tlimit:
            return None
        
        if len(vals) == target_len:
            return list(vals)
        
        # Pruning: not enough numbers left
        if 256 - lo + 1 < target_len - len(vals):
            return None
            
        # Heuristic: We can also prune if the minimum possible future sums
        # exceed the maximum possible distinct sums available, but that's complex.
        # Simple count check is usually sufficient for this size.

        for x in range(lo, 257):
            if time.time() > tlimit:
                return None
            
            ns = check_add(vals, sums, x)
            if ns is None:
                continue
                
            vals.append(x)
            sums2 = sums.union(ns)
            
            res = backtrack(vals, sums2, x + 1, target_len, tlimit)
            if res is not None:
                return res
                
            vals.pop()
            
        return None

    # Strategy:
    # 1. Try to extend the greedy sequence by backtracking the last k elements.
    # 2. If that fails, try shorter prefixes.
    # 3. If that fails, try a fresh search (from scratch) for the target length.
    
    # We iterate target lengths from best_len + 1 upwards.
    # We stop if we run out of time.
    
    for target_len in range(best_len + 1, best_len + 10):
        if time.time() > deadline:
            break
            
        remaining_time = deadline - time.time()
        if remaining_time < 0.5:
            break
            
        found = None
        
        # Phase 1: Prefix-based extension
        # Try prefixes of the greedy sequence, from longest to shortest.
        # A prefix of length P means we fix the first P elements of the greedy sequence
        # and backtrack the rest.
        
        # Maximum prefix length we can use is target_len - 1 (we need at least 1 new element)
        # But we also can't use a prefix longer than the greedy sequence itself.
        max_prefix = min(best_len, target_len - 1)
        
        # We try prefixes of length: max_prefix, max_prefix-1, ..., down to some minimum.
        # Let's try down to 0 (fresh start) but with limited time for short prefixes.
        
        for plen in range(max_prefix, -1, -1):
            if time.time() > deadline:
                break
                
            prefix = best[:plen]
            prefix_sums = get_sums(prefix) if prefix else set()
            lo = prefix[-1] + 1 if prefix else 1
            
            # Allocate time based on prefix length.
            # Longer prefixes are more likely to succeed quickly.
            # Shorter prefixes (especially 0) are expensive.
            if plen > 0:
                # Give more time to longer prefixes? Or just a fixed chunk?
                # Let's give a fraction of remaining time.
                # If plen is close to target_len, it's very constrained.
                # If plen is small, it's open.
                tlimit = min(deadline, time.time() + remaining_time * 0.2)
            else:
                # Fresh start: give it a significant chunk, but not all.
                tlimit = min(deadline, time.time() + remaining_time * 0.5)
                
            if time.time() >= tlimit:
                continue
                
            # Perform backtracking
            result = backtrack(prefix, prefix_sums, lo, target_len, tlimit)
            
            if result is not None:
                # Validate just in case
                if len(set(result)) == len(result) and all(1 <= v <= 256 for v in result):
                    if all(result[i] < result[i+1] for i in range(len(result)-1)):
                        # Check sums
                        s = set()
                        ok = True
                        for i in range(len(result)):
                            t = 0
                            for j in range(i, len(result)):
                                t += result[j]
                                if t in s:
                                    ok = False
                                    break
                                s.add(t)
                            if not ok:
                                break
                        if ok and len(s) == len(result) * (len(result) + 1) // 2:
                            found = result
                            break
                else:
                    # Should not happen if backtrack is correct
                    continue
        
        if found is not None:
            best = found
            best_len = target_len
            # We can continue to try for even longer sequences
        else:
            # If we couldn't find a sequence of this length, 
            # it's unlikely we'll find a longer one quickly, 
            # but we can try one more step if time permits?
            # Usually, if L+1 fails, L+2 is harder. 
            # Let's break if we failed on a fresh start (plen=0) or if time is low.
            # If we only failed on long prefixes, shorter ones might work? 
            # The loop above already tried down to plen=0. 
            # So if found is None, we tried all prefix lengths including 0.
            # Thus, we likely cannot achieve this target_len.
            # It is generally safe to stop, but let's allow one more attempt 
            # if we have plenty of time and haven't tried a fresh start for this length?
            # No, the loop `for plen in range(max_prefix, -1, -1)` covers 0.
            # So if found is None, we exhausted options for this target_len.
            # We should probably stop increasing target_len if we fail, 
            # because the search space grows.
            # However, sometimes L+1 is hard but L+2 is easier? Unlikely.
            # Let's break.
            break

    return best
# EVOLVE-BLOCK-END
