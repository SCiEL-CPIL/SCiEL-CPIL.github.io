"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    time_limit = 55.0  # 5s buffer before 60s limit

    def get_valid_candidates(values, sums, last_val):
        """
        Returns a list of all x in (last_val, 256] such that appending x 
        to values does not create any duplicate consecutive block sums.
        """
        valid = []
        n = len(values)
        # Precompute suffix sums of values for faster tail computation
        # suffix[i] = values[i] + values[i+1] + ... + values[n-1]
        suffix = [0] * (n + 1)
        for i in range(n - 1, -1, -1):
            suffix[i] = suffix[i + 1] + values[i]
        
        for x in range(last_val + 1, 257):
            # Check if x creates any collision
            # New tails: x, x+values[-1], x+values[-1]+values[-2], ...
            # = x + suffix[n-1], x + suffix[n-2], ..., x + suffix[0]
            collision = False
            for i in range(n + 1):
                tail = x + suffix[i]
                if tail in sums:
                    collision = True
                    break
            if not collision:
                valid.append(x)
        
        return valid

    def greedy_increasing():
        values = []
        sums = set()
        last_val = 0
        while last_val < 256:
            candidates = get_valid_candidates(values, sums, last_val)
            if not candidates:
                break
            # Greedy: pick smallest
            x = candidates[0]
            values.append(x)
            last_val = x
            # Update sums
            total = x
            sums.add(total)
            for a in reversed(values[:-1]):
                total += a
                sums.add(total)
        return values

    def greedy_decreasing():
        """
        Construct a strictly decreasing sequence b_1 > b_2 > ... > b_k
        such that consecutive sums are distinct.
        Then reverse it to get increasing a_1 < ... < a_k.
        The set of consecutive sums is invariant under reversal.
        """
        values = []
        sums = set()
        next_val = 257
        while next_val > 1:
            candidates = []
            n = len(values)
            # Precompute prefix sums for the decreasing sequence
            # When appending x to the right, new sums are x, x+values[-1], x+values[-1]+values[-2], ...
            # = x, x+values[-1], ...
            for x in range(next_val - 1, 0, -1):
                collision = False
                total = x
                if total in sums:
                    collision = True
                else:
                    for i in range(n - 1, -1, -1):
                        total += values[i]
                        if total in sums:
                            collision = True
                            break
                if not collision:
                    candidates.append(x)
            
            if not candidates:
                break
            
            x = candidates[0]
            values.append(x)
            next_val = x
            # Update sums
            total = x
            sums.add(total)
            for a in reversed(values[:-1]):
                total += a
                sums.add(total)
        
        return values[::-1]

    def randomized_trial(strategy, rand):
        values = []
        sums = set()
        last_val = 0
        
        while last_val < 256:
            candidates = get_valid_candidates(values, sums, last_val)
            if not candidates:
                break
            
            if strategy == 'random':
                x = rand.choice(candidates)
            elif strategy == 'large':
                k = min(10, len(candidates))
                x = rand.choice(candidates[-k:])
            elif strategy == 'small':
                k = min(10, len(candidates))
                x = rand.choice(candidates[:k])
            elif strategy == 'mid':
                k = min(10, len(candidates))
                mid_start = len(candidates) // 2 - k // 2
                mid_end = mid_start + k
                x = rand.choice(candidates[mid_start:mid_end])
            else:
                x = rand.choice(candidates)
                
            values.append(x)
            last_val = x
            
            total = x
            sums.add(total)
            for a in reversed(values[:-1]):
                total += a
                sums.add(total)
                
        return values

    def local_search_improve(seq, time_deadline):
        """
        Try to improve a sequence by removing an element and re-extending.
        """
        best_seq = seq[:]
        best_len = len(seq)
        
        for attempt in range(50):
            if time.time() > time_deadline:
                break
            
            # Try removing each element (except first and last to keep it interesting)
            # and re-extending greedily
            for remove_idx in range(1, len(best_seq) - 1):
                if time.time() > time_deadline:
                    break
                
                # Remove element at remove_idx
                trial_seq = best_seq[:remove_idx] + best_seq[remove_idx + 1:]
                
                # Recompute sums
                sums = set()
                for i in range(len(trial_seq)):
                    total = 0
                    for j in range(i, len(trial_seq)):
                        total += trial_seq[j]
                        sums.add(total)
                
                # Now extend greedily from the end
                values = trial_seq[:]
                last_val = values[-1] if values else 0
                
                while last_val < 256:
                    candidates = get_valid_candidates(values, sums, last_val)
                    if not candidates:
                        break
                    x = candidates[0]
                    values.append(x)
                    last_val = x
                    total = x
                    sums.add(total)
                    for a in reversed(values[:-1]):
                        total += a
                        sums.add(total)
                
                if len(values) > best_len:
                    best_len = len(values)
                    best_seq = values[:]
        
        return best_seq

    def beam_search(width, time_deadline):
        """
        Beam search: keep top 'width' partial sequences at each depth.
        """
        # Each beam entry: (values, sums, last_val)
        beams = [([], set(), 0)]
        best_seq = []
        best_len = 0
        
        while time.time() < time_deadline:
            new_beams = []
            for values, sums, last_val in beams:
                if len(values) > best_len:
                    best_len = len(values)
                    best_seq = values[:]
                
                candidates = get_valid_candidates(values, sums, last_val)
                if not candidates:
                    continue
                
                # For each candidate, create a new beam entry
                for x in candidates:
                    new_values = values + [x]
                    new_sums = set(sums)
                    total = x
                    new_sums.add(total)
                    for a in reversed(values):
                        total += a
                        new_sums.add(total)
                    new_beams.append((new_values, new_sums, x))
            
            if not new_beams:
                break
            
            # Sort by length (descending), then by some heuristic (e.g., last_val ascending to leave more room)
            new_beams.sort(key=lambda b: (-len(b[0]), b[2]))
            beams = new_beams[:width]
            
            # If all beams have same length and we can't extend further, stop
            if all(len(b[0]) == best_len for b in beams):
                # Check if any can be extended
                can_extend = False
                for values, sums, last_val in beams:
                    if get_valid_candidates(values, sums, last_val):
                        can_extend = True
                        break
                if not can_extend:
                    break
        
        return best_seq

    def dfs_construct(depth_limit, time_deadline):
        """
        Depth-first search with a limit on depth to explore non-greedy branches.
        """
        best_seq = []
        
        def dfs(values, sums, last_val, depth):
            nonlocal best_seq
            if time.time() > time_deadline:
                return
            if len(values) > len(best_seq):
                best_seq = values[:]
            
            if depth >= depth_limit:
                return
                
            candidates = get_valid_candidates(values, sums, last_val)
            if not candidates:
                return
            
            for x in candidates:
                if len(values) + (256 - last_val) <= len(best_seq):
                    break
                    
                new_sums = set(sums)
                total = x
                new_sums.add(total)
                for a in reversed(values):
                    total += a
                    new_sums.add(total)
                
                dfs(values + [x], new_sums, x, depth + 1)
                
                if time.time() > time_deadline:
                    return

        dfs([], set(), 0, 0)
        return best_seq

    # 1. Baselines
    best_seq = greedy_increasing()
    best_len = len(best_seq)
    
    dec_seq = greedy_decreasing()
    if len(dec_seq) > best_len:
        best_seq = dec_seq
        best_len = len(dec_seq)
        
    # 2. Randomized Search
    strategies = ['random', 'large', 'small', 'mid', 'random', 'large', 'small', 'mid']
    
    trial_count = 0
    while time.time() - start_time < time_limit - 15.0:  # Reserve 15s for local search + beam + DFS
        strategy = strategies[trial_count % len(strategies)]
        rand = random.Random(trial_count)
        
        seq = randomized_trial(strategy, rand)
        
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = seq
            
        trial_count += 1
    
    # 3. Local Search: try to improve best_seq by removing elements and re-extending
    if time.time() - start_time < time_limit - 10.0:
        improved = local_search_improve(best_seq, time.time() + (time_limit - 10.0 - (time.time() - start_time)))
        if len(improved) > best_len:
            best_len = len(improved)
            best_seq = improved
    
    # 4. Beam Search
    if time.time() - start_time < time_limit - 5.0:
        beam_seq = beam_search(width=20, time_deadline=time.time() + (time_limit - 5.0 - (time.time() - start_time)))
        if len(beam_seq) > best_len:
            best_len = len(beam_seq)
            best_seq = beam_seq
    
    # 5. DFS Attempt
    if time.time() - start_time < time_limit - 1.0:
        dfs_seq = dfs_construct(best_len + 1, time.time() + (time_limit - 1.0 - (time.time() - start_time)))
        if len(dfs_seq) > best_len:
            best_seq = dfs_seq
            best_len = len(best_seq)
            
    return best_seq

# EVOLVE-BLOCK-END
