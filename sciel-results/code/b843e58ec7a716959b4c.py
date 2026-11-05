import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 5s margin

    def greedy_extend(seq, sums, last, start_val):
        """Extend sequence greedily from start_val."""
        new_seq = list(seq)
        new_sums = set(sums)
        new_last = last
        for x in range(start_val, 257):
            tails, total = [x], x
            for a in reversed(new_seq):
                total += a
                tails.append(total)
            if not new_sums.intersection(tails):
                new_seq.append(x)
                new_sums.update(tails)
                new_last = x
        return new_seq, new_sums, new_last

    # 1. Beam Search
    beam_width = 1000
    beam = [([], set(), 0)]
    best_seq = []
    best_sums = set()
    
    while time.time() < deadline - 10:  # Leave 10s for local search
        next_beam = []
        
        for seq, sums, last in beam:
            # Try candidates
            # Optimization: If last is large, fewer candidates.
            for x in range(last + 1, 257):
                # Quick check: if x is in sums, skip? No, x is a sum itself.
                # But x must be distinct from ALL sums.
                
                # Calculate tails
                tails = [x]
                total = x
                for a in reversed(seq):
                    total += a
                    tails.append(total)
                    
                if sums.intersection(tails):
                    continue
                
                # Valid
                # Heuristic for keeping in beam:
                # We want to keep states with small 'last' to allow more growth.
                # We also want to keep states that are "dense" or "sparse"?
                # Let's just keep the ones with smallest 'last'.
                
                next_beam.append((seq + [x], sums.union(tails), x))
                
        if not next_beam:
            break
            
        # Select top beam_width
        # Sort by last value ascending
        next_beam.sort(key=lambda s: s[2])
        
        beam = next_beam[:beam_width]
        
        # Update best
        if len(beam) > 0:
            if len(beam[0][0]) > len(best_seq):
                best_seq = beam[0][0]
                best_sums = beam[0][1]

    # 2. Local Search / Hill Climbing
    if best_seq and time.time() < deadline:
        current_seq = best_seq
        current_sums = best_sums
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try removing each element
            n = len(current_seq)
            for i in range(n):
                # Remove element at i
                new_seq = current_seq[:i] + current_seq[i+1:]
                # Recalculate sums for the prefix?
                # It's easier to just re-run greedy extend from the point of removal.
                
                # We need the sums of the prefix current_seq[:i]
                # And the last value of the prefix.
                
                prefix = current_seq[:i]
                if not prefix:
                    prefix_sums = set()
                    prefix_last = 0
                    start_val = 1
                else:
                    prefix_last = prefix[-1]
                    # Calculate sums of prefix
                    prefix_sums = set()
                    for j in range(len(prefix)):
                        s = 0
                        for k in range(j, len(prefix)):
                            s += prefix[k]
                            prefix_sums.add(s)
                    # We need to restart from prefix_last + 1?
                    # No, the element after i was current_seq[i+1].
                    # We can try to extend from prefix_last.
                    # But we must maintain strictly increasing.
                    # So next element must be > prefix_last.
                    
                    # However, the original sequence had current_seq[i+1] which was > prefix_last.
                    # We can try to extend greedily from prefix_last.
                    
                # Greedy extend
                new_seq, new_sums, _ = greedy_extend(prefix, prefix_sums, prefix_last, prefix_last + 1)
                
                if len(new_seq) > len(current_seq):
                    current_seq = new_seq
                    current_sums = new_sums
                    improved = True
                    break  # Restart loop with new best
                    
        best_seq = current_seq

    # Final fallback
    if not best_seq:
        values, sums = [], set()
        for x in range(1, 257):
            tails, total = [x], x
            for a in reversed(values):
                total += a
                tails.append(total)
            if not sums.intersection(tails):
                values.append(x)
                sums.update(tails)
        best_seq = values

    return best_seq
