import time

def construct():
    start_time = time.time()
    N = 256
    deadline = start_time + 85
    
    # Fast Greedy to get a baseline
    def greedy():
        values = []
        sums = set()
        for x in range(1, N + 1):
            tails = [x]
            total = x
            valid = True
            if total in sums:
                valid = False
            else:
                for a in reversed(values):
                    total += a
                    tails.append(total)
                    if total in sums:
                        valid = False
                        break
            if valid:
                values.append(x)
                sums.update(tails)
        return values

    best_seq = greedy()
    best_len = len(best_seq)
    
    # Beam Search
    # State: (seq, sums, suffix_sums)
    beam = [([], set(), [])]
    K = 200 # Beam width
    
    while time.time() < deadline:
        next_beam = []
        for seq, sums, suffix_sums in beam:
            last_val = seq[-1] if seq else 0
            
            for x in range(last_val + 1, N + 1):
                # Check validity
                new_sums = [x]
                total = x
                valid = True
                if total in sums:
                    valid = False
                else:
                    for s in suffix_sums:
                        total += s
                        new_sums.append(total)
                        if total in sums:
                            valid = False
                            break
                
                if valid:
                    new_seq = seq + [x]
                    # Optimization: Check if we should update best
                    if len(new_seq) > best_len:
                        best_len = len(new_seq)
                        best_seq = new_seq
                    
                    # Create new state
                    # Using set union is O(N). 
                    # We can do: new_sums_set = sums.copy(); new_sums_set.update(new_sums)
                    # But copy is also O(N).
                    # Let's just use union.
                    new_sums_set = sums.union(new_sums)
                    new_suffix_sums = [x + s for s in suffix_sums] + [x]
                    
                    next_beam.append((new_seq, new_sums_set, new_suffix_sums))
                    
                    # Early exit if we hit deadline
                    if time.time() > deadline:
                        return best_seq

        if not next_beam:
            break
        
        # Rank and select top K
        # Key: (last_element, total_sum)
        # Smaller last element is preferred.
        # Smaller total sum is preferred (leaves more room in sum space?).
        
        def rank_state(state):
            seq, sums, suffix_sums = state
            return (seq[-1], suffix_sums[0])
            
        next_beam.sort(key=rank_state)
        beam = next_beam[:K]
        
    return best_seq
