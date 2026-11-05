import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    # 1. Greedy Initialization
    seq = greedy()
    best_seq = seq[:]
    best_len = len(seq)
    
    # Helper: Check if a sequence is valid
    def is_valid(s):
        if not s:
            return True
        sums = set()
        for i in range(len(s)):
            t = 0
            for j in range(i, len(s)):
                t += s[j]
                if t in sums:
                    return False
                sums.add(t)
        return True

    # Helper: Get set of all subarray sums
    def get_sums(s):
        sums = set()
        for i in range(len(s)):
            t = 0
            for j in range(i, len(s)):
                t += s[j]
                sums.add(t)
        return sums

    # Helper: Greedy construction (smallest available)
    def greedy():
        values = []
        sums = set()
        for x in range(1, 257):
            tails = [x]
            total = x
            for a in reversed(values):
                total += a
                tails.append(total)
            if not sums.intersection(tails):
                values.append(x)
                sums.update(tails)
        return values

    # Helper: Try to extend sequence by appending a valid number
    def try_extend(s):
        if not s:
            return [1]
        last = s[-1]
        if last >= 256:
            return None
        sums = get_sums(s)
        
        # Try smallest available first to keep sequence compact
        for x in range(last + 1, 257):
            tails = [x]
            total = x
            for a in reversed(s):
                total += a
                tails.append(total)
            
            if not sums.intersection(tails):
                return s + [x]
        return None

    # Helper: Local search to compact the sequence (minimize sum)
    def local_search_compact(s, limit_time):
        current_seq = s[:]
        current_sum = sum(current_seq)
        
        # Simple Hill Climbing
        while time.time() < limit_time:
            improved = False
            n = len(current_seq)
            # Try to change a random element to a smaller valid value
            # Or just iterate through all elements and try to reduce them?
            # Random is better to avoid bias.
            i = random.randint(0, n - 1)
            
            low = 1
            high = 256
            if i > 0:
                low = current_seq[i-1] + 1
            if i < n - 1:
                high = current_seq[i+1] - 1
                
            if low > high:
                continue
                
            # Try to find a smaller value that is valid
            # We want to minimize sum, so we prefer smaller values.
            # Try the smallest possible value first?
            # If we always try the smallest, we might get stuck.
            # Let's try a few random candidates, biased towards low.
            
            # Candidate 1: The absolute minimum (low)
            # Candidate 2: A random value
            candidates = [low]
            # Add a few randoms
            for _ in range(3):
                candidates.append(random.randint(low, high))
                
            for v in candidates:
                if v == current_seq[i]:
                    continue
                if v < low or v > high:
                    continue
                    
                new_seq = current_seq[:]
                new_seq[i] = v
                
                # Check validity
                if is_valid(new_seq):
                    new_sum = sum(new_seq)
                    if new_sum < current_sum:
                        current_seq = new_seq
                        current_sum = new_sum
                        improved = True
                        break
                    elif new_sum == current_sum:
                        # Tie-breaking: keep the one with smaller max element?
                        # Or just accept to escape plateaus?
                        # Let's accept to explore.
                        current_seq = new_seq
                        improved = True
                        break

        return current_seq

    # Main Loop
    while time.time() < deadline:
        # 1. Try to extend
        extended = try_extend(seq)
        if extended:
            seq = extended
            if len(seq) > best_len:
                best_len = len(seq)
                best_seq = seq[:]
            # After extension, the sequence might be less "compact" than necessary
            # but it's valid.
            # We can try to compact it a bit before trying to extend again
            # to make room for the next element.
            # Let's do a short compaction pass.
            t_limit = min(time.time() + 1.0, deadline)
            seq = local_search_compact(seq, t_limit)
        else:
            # 2. Cannot extend. Try to compact to enable extension.
            t_limit = min(time.time() + 2.0, deadline)
            seq = local_search_compact(seq, t_limit)
            # After compaction, try to extend again
            extended = try_extend(seq)
            if extended:
                seq = extended
                if len(seq) > best_len:
                    best_len = len(seq)
                    best_seq = seq[:]

    return best_seq
