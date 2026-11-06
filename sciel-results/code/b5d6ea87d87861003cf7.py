import time
import sys
import random

sys.setrecursionlimit(10000)

def construct():
    start_time = time.time()
    deadline = start_time + 55.0

    def is_time_up():
        return time.time() > deadline

    def greedy():
        values, sums = [], set()
        for x in range(1, 257):
            tails = []
            total = 0
            for a in reversed(values + [x]):
                total += a
                tails.append(total)
            if not sums.intersection(tails):
                values.append(x)
                sums.update(tails)
        return values, sums

    base_seq, base_sums = greedy()
    best_seq = base_seq
    base_len = len(base_seq)
    
    # Beam Search to try to exceed base_len
    # State: (last_val, seq_tuple, sums_frozenset)
    # Using frozenset for sums to make them hashable and immutable for the beam
    # But checking validity requires iterating tails.
    
    beam = [(0, (), frozenset())]
    max_len = 0
    best_state = (0, (), frozenset())
    
    # We try to expand the beam step by step
    # If the beam becomes empty, we stop.
    # We keep the best sequence (longest, then smallest last) found so far.
    
    while beam and not is_time_up():
        new_beam = []
        for last, seq, sums in beam:
            # Try all possible next values
            # Optimization: We can skip x if x is in sums (sum of length 1)
            # We need to check all tails.
            for x in range(last + 1, 257):
                if x in sums:
                    continue
                
                # Calculate new tails
                # Tails of new seq = [x] + [x + a for a in reversed(seq)]
                # Actually, tails are sums of consecutive blocks ending at x.
                # tail_1 = x
                # tail_2 = x + seq[-1]
                # ...
                # tail_k = x + seq[-1] + ... + seq[-k]
                
                # We can check validity incrementally or just compute the list.
                # Since seq length is small (< 30), computing the list is fine.
                
                valid = True
                total = x
                # Check tail 1
                if total in sums:
                    valid = False
                else:
                    # Check other tails
                    s = total
                    for a in reversed(seq):
                        s += a
                        if s in sums:
                            valid = False
                            break
                
                if valid:
                    new_seq = seq + (x,)
                    # New sums = old sums union new tails
                    # We need to construct the new frozenset.
                    # This is the expensive part.
                    # Number of new sums = len(new_seq)
                    # Size of sums ~ L^2/2
                    # Creating a new set of size L^2/2 + L is O(L^2).
                    # With B=100, L=20, this is 100 * 400 = 40,000 ops per step.
                    # 20 steps -> 800,000 ops. Very fast.
                    
                    new_sums_set = set(sums)
                    total = x
                    new_sums_set.add(total)
                    for a in reversed(seq):
                        total += a
                        new_sums_set.add(total)
                    
                    new_beam.append((x, new_seq, frozenset(new_sums_set)))
        
        if not new_beam:
            break
            
        # Select top B states from new_beam
        # Heuristic: Smaller last value is generally better (more room).
        # Also, we want to keep diverse paths?
        # Simple heuristic: Sort by last value, then by length of seq (all same length in this layer), then by sum of seq?
        # Let's just sort by last value.
        # To avoid duplicates, we might need to be careful, but last+seq uniquely identifies the path?
        # No, different paths can lead to same last and seq? No, seq determines last.
        # Can two different paths lead to the same seq? No, the sequence is the path.
        # So (last, seq) is unique.
        
        # We want to keep the "best" candidates.
        # Smaller last is better.
        # We can also look at the "density" of sums?
        # Let's stick to sorting by last value.
        
        # Limit beam size
        B = 100
        if len(new_beam) > B:
            # Sort by last value (ascending)
            new_beam.sort(key=lambda s: s[0])
            new_beam = new_beam[:B]
            
        beam = new_beam
        
        # Update best
        if len(beam[0][1]) > max_len:
            max_len = len(beam[0][1])
            best_state = beam[0]
            best_seq = list(best_state[1])
            
    # If beam search didn't find anything better, return greedy
    if max_len > base_len:
        return best_seq
    else:
        return base_seq
