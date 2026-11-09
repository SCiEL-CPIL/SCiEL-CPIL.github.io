"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def _compute_tails(values, x):
    """Compute the new tail sums if we append x to values."""
    tails = [x]
    total = x
    for a in reversed(values):
        total += a
        tails.append(total)
    return tails

def _is_valid_extension(values, sums, x):
    """Check if appending x to values creates no duplicate block sums."""
    tails = _compute_tails(values, x)
    return not sums.intersection(tails)

def _extend_greedy(values, sums, candidates, deadline, mode='smallest'):
    """Extend values greedily using the given mode."""
    ext = list(values)
    ext_sums = set(sums)
    if mode == 'smallest':
        cand_iter = candidates
    elif mode == 'largest':
        cand_iter = list(reversed(candidates))
    else:
        cand_iter = candidates
    
    for x in cand_iter:
        if time.time() > deadline:
            break
        if _is_valid_extension(ext, ext_sums, x):
            tails = _compute_tails(ext, x)
            ext.append(x)
            ext_sums.update(tails)
    return ext, ext_sums

def _beam_search(deadline, beam_width=20, max_steps=50):
    """Beam search over sequence prefixes."""
    # Each beam entry: (values, sums, score)
    # Score = current length (we want to maximize)
    beams = [([], set())]
    best_result = []
    
    for step in range(max_steps):
        if time.time() > deadline:
            break
        
        new_beams = []
        for values, sums in beams:
            if not values and step > 0:
                continue
            # Try adding each candidate
            for x in range(1, 257):
                if values and x <= values[-1]:
                    continue
                if _is_valid_extension(values, sums, x):
                    tails = _compute_tails(values, x)
                    new_values = values + [x]
                    new_sums = sums | set(tails)
                    new_beams.append((new_values, new_sums))
        
        if not new_beams:
            break
        
        # Keep top beam_width beams by length
        new_beams.sort(key=lambda b: len(b[0]), reverse=True)
        beams = new_beams[:beam_width]
        
        # Track best
        for values, _ in beams:
            if len(values) > len(best_result):
                best_result = list(values)
    
    return best_result

def _local_search(initial, deadline, iterations=2000):
    """Local search: try to improve by swapping/removing/inserting elements."""
    if not initial:
        return initial
    
    best = list(initial)
    best_sums = set()
    n = len(best)
    for i in range(n):
        s = 0
        for j in range(i, n):
            s += best[j]
            best_sums.add(s)
    
    best_len = len(best)
    
    for it in range(iterations):
        if time.time() > deadline:
            break
        
        # Random perturbation: try to insert a new element
        success = False
        for _ in range(50):
            # Try inserting at a random position
            if len(best) == 0:
                break
            pos = random.randint(0, len(best))
            # Find a value that could fit at pos
            low = best[pos-1] + 1 if pos > 0 else 1
            high = best[pos] - 1 if pos < len(best) else 256
            if low > high:
                continue
            x = random.randint(low, high)
            
            # Check if inserting x at pos is valid
            test_seq = best[:pos] + [x] + best[pos:]
            test_sums = set()
            valid = True
            n_test = len(test_seq)
            for i in range(n_test):
                s = 0
                for j in range(i, n_test):
                    s += test_seq[j]
                    if s in test_sums:
                        valid = False
                        break
                    test_sums.add(s)
                if not valid:
                    break
            
            if valid:
                best = test_seq
                best_sums = test_sums
                best_len = len(best)
                success = True
                break
        
        if not success:
            # Try removing an element and re-adding a different one
            if len(best) > 1:
                rem_pos = random.randint(0, len(best) - 1)
                saved = best[rem_pos]
                best_without = best[:rem_pos] + best[rem_pos+1:]
                
                # Recompute sums for best_without
                bw_sums = set()
                n_bw = len(best_without)
                for i in range(n_bw):
                    s = 0
                    for j in range(i, n_bw):
                        s += best_without[j]
                        bw_sums.add(s)
                
                # Try to find a replacement
                low = best_without[rem_pos-1] + 1 if rem_pos > 0 else 1
                high = best_without[rem_pos] - 1 if rem_pos < len(best_without) else 256
                if low <= high:
                    for x in random.sample(range(low, high+1), min(10, high-low+1)):
                        if _is_valid_extension(best_without, bw_sums, x) if rem_pos == len(best_without) else True:
                            # Full check needed
                            test_seq = best_without[:rem_pos] + [x] + best_without[rem_pos:]
                            test_sums = set()
                            valid = True
                            n_test = len(test_seq)
                            for i in range(n_test):
                                s = 0
                                for j in range(i, n_test):
                                    s += test_seq[j]
                                    if s in test_sums:
                                        valid = False
                                        break
                                    test_sums.add(s)
                                if not valid:
                                    break
                            if valid and len(test_seq) >= best_len:
                                best = test_seq
                                best_sums = test_sums
                                best_len = len(best)
                                break
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin (90s limit)
    
    # Strategy 1: Greedy smallest
    def greedy_smallest():
        values, sums = [], set()
        for x in range(1, 257):
            if time.time() > deadline:
                break
            if _is_valid_extension(values, sums, x):
                tails = _compute_tails(values, x)
                values.append(x)
                sums.update(tails)
        return values
    
    # Strategy 2: Greedy largest
    def greedy_largest():
        values, sums = [], set()
        for x in range(256, 0, -1):
            if time.time() > deadline:
                break
            if _is_valid_extension(values, sums, x):
                tails = _compute_tails(values, x)
                values.append(x)
                sums.update(tails)
        return sorted(values)
    
    # Strategy 3: Beam search
    def beam_strategy():
        return _beam_search(deadline, beam_width=30, max_steps=50)
    
    # Strategy 4: Local search from greedy results
    def local_search_strategy(base_result):
        if not base_result:
            return base_result
        return _local_search(base_result, deadline, iterations=3000)
    
    # Run strategies in order, keeping the best
    best_result = []
    
    # First, get greedy results
    greedy_res = greedy_smallest()
    if len(greedy_res) > len(best_result):
        best_result = greedy_res
    
    greedy_res_l = greedy_largest()
    if len(greedy_res_l) > len(best_result):
        best_result = greedy_res_l
    
    # Try beam search
    beam_res = beam_strategy()
    if len(beam_res) > len(best_result):
        best_result = beam_res
    
    # Try local search from the best so far
    if time.time() < deadline - 10:
        local_res = local_search_strategy(best_result)
        if len(local_res) > len(best_result):
            best_result = local_res
    
    # If we have time, try more local search iterations
    if time.time() < deadline - 5 and best_result:
        local_res2 = _local_search(best_result, deadline, iterations=5000)
        if len(local_res2) > len(best_result):
            best_result = local_res2
    
    return best_result if best_result else greedy_smallest()
# EVOLVE-BLOCK-END
