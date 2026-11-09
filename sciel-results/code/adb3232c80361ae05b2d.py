"""Improved program for increasing sequences with unique consecutive-block sums."""
# EVOLVE-BLOCK-START
import time
import random
import heapq

def _compute_sums(seq):
    """Return set of all nonempty consecutive block sums."""
    sums = set()
    n = len(seq)
    for i in range(n):
        s = 0
        for j in range(i, n):
            s += seq[j]
            sums.add(s)
    return sums

def _try_append(seq, sums, x):
    """Try to append x to seq. Returns (True, new_sums) if valid, else (False, None)."""
    tails = [x]
    total = x
    for a in reversed(seq):
        total += a
        tails.append(total)
    for t in tails:
        if t in sums:
            return False, None
    new_sums = sums | set(tails)
    return True, new_sums

def greedy_smallest(deadline):
    values, sums = [], set()
    for x in range(1, 257):
        if time.time() > deadline:
            break
        ok, new_sums = _try_append(values, sums, x)
        if ok:
            values.append(x)
            sums = new_sums
    return values

def greedy_largest(deadline):
    values, sums = [], set()
    for x in range(256, 0, -1):
        if time.time() > deadline:
            break
        ok, new_sums = _try_append(values, sums, x)
        if ok:
            values.append(x)
            sums = new_sums
    return sorted(values)

def _extend_greedy(seq, sums, candidates, deadline):
    """Greedy extension of seq with candidates."""
    for x in candidates:
        if time.time() > deadline:
            break
        ok, new_sums = _try_append(seq, sums, x)
        if ok:
            seq.append(x)
            sums = new_sums
    return seq, sums

def _beam_search(deadline, beam_width=50):
    """Beam search: maintain top beam_width partial sequences."""
    # Each beam entry: (seq, sums, score)
    # Score: prefer shorter sequences with more room, or longer sequences
    beams = [([], set(), 0)]
    best = []
    
    while time.time() < deadline:
        new_beams = []
        for seq, sums, score in beams:
            if len(seq) > len(best):
                best = list(seq)
            if not seq:
                candidates = range(1, 257)
            else:
                candidates = range(seq[-1]+1, 257)
            
            for x in candidates:
                ok, new_sums = _try_append(seq, sums, x)
                if ok:
                    new_seq = seq + [x]
                    # Score: longer is better, but also prefer sequences that can extend further
                    # Use a heuristic: length * 100 - (last element) / 10
                    new_score = len(new_seq) * 1000 - (new_seq[-1] / 10.0)
                    new_beams.append((new_seq, new_sums, new_score))
        
        if not new_beams:
            break
        
        # Keep top beam_width by score
        new_beams.sort(key=lambda b: b[2], reverse=True)
        beams = new_beams[:beam_width]
        
        # If all beams have same length, we've likely reached a plateau
        lengths = set(len(b[0]) for b in beams)
        if len(lengths) == 1 and len(beams) > 10:
            # Check if we can extend any
            can_extend = False
            for seq, sums, _ in beams:
                for x in range(seq[-1]+1, 257):
                    ok, _ = _try_append(seq, sums, x)
                    if ok:
                        can_extend = True
                        break
                if can_extend:
                    break
            if not can_extend:
                break
    
    return best

def _local_search(start, deadline):
    """Hill-climbing with random restarts on individual elements."""
    if not start:
        return start
    
    current = list(start)
    current_sums = _compute_sums(current)
    best = list(current)
    best_len = len(current)
    
    max_iterations = 200
    for _ in range(max_iterations):
        if time.time() > deadline:
            break
        
        improved = False
        # Try replacing each element with a nearby value
        for i in range(len(current)):
            if time.time() > deadline:
                break
            orig = current[i]
            # Try a range of nearby values
            for delta in range(1, 8):
                for new_val in [orig - delta, orig + delta]:
                    if new_val < 1 or new_val > 256:
                        continue
                    if i > 0 and new_val <= current[i-1]:
                        continue
                    if i < len(current)-1 and new_val >= current[i+1]:
                        continue
                    
                    trial = current.copy()
                    trial[i] = new_val
                    
                    # Check validity efficiently
                    sums = set()
                    valid = True
                    for j in range(len(trial)):
                        s = 0
                        for k in range(j, len(trial)):
                            s += trial[k]
                            if s in sums:
                                valid = False
                                break
                            sums.add(s)
                        if not valid:
                            break
                    
                    if valid:
                        # Try to extend from here
                        ext, ext_sums = _extend_greedy(trial, sums, 
                                                       [c for c in range(trial[-1]+1, 257)], deadline)
                        if len(ext) > best_len:
                            current = ext
                            current_sums = ext_sums
                            best = list(current)
                            best_len = len(current)
                            improved = True
                            break
                if improved:
                    break
            if improved:
                break
        
        if not improved:
            # Try to extend the current best
            ext, ext_sums = _extend_greedy(current, current_sums,
                                           [c for c in range(current[-1]+1, 257)], deadline)
            if len(ext) > best_len:
                current = ext
                current_sums = ext_sums
                best = list(current)
                best_len = len(current)
            else:
                break
    
    return best

def _backtracking_search(initial, deadline, max_nodes=300000):
    """Depth-first backtracking with greedy pruning."""
    if not initial:
        return []
    
    sums = _compute_sums(initial)
    best = list(initial)
    nodes = 0
    
    def dfs(seq, sums, candidates):
        nonlocal best, nodes
        if time.time() > deadline or nodes > max_nodes:
            return
        nodes += 1
        if len(seq) > len(best):
            best = list(seq)
        if not candidates:
            return
        
        for i, x in enumerate(candidates):
            ok, new_sums = _try_append(seq, sums, x)
            if ok:
                remaining = candidates[i+1:]
                seq.append(x)
                dfs(seq, new_sums, remaining)
                seq.pop()
                if time.time() > deadline or nodes > max_nodes:
                    return
    
    candidates = [c for c in range(initial[-1]+1, 257)]
    dfs(list(initial), sums, candidates)
    return best

def _simulated_annealing(deadline):
    """SA approach: start with a good greedy solution, perturb and optimize."""
    start = greedy_smallest(deadline)
    if not start:
        return start
    
    current = list(start)
    current_sums = _compute_sums(current)
    best = list(current)
    best_len = len(current)
    
    temperature = 15.0
    cooling = 0.998
    
    while time.time() < deadline and temperature > 0.1:
        i = random.randint(0, len(current)-1)
        orig = current[i]
        attempts = 0
        while attempts < 8:
            new_val = orig + random.randint(-6, 6)
            if new_val < 1 or new_val > 256:
                attempts += 1
                continue
            if i > 0 and new_val <= current[i-1]:
                attempts += 1
                continue
            if i < len(current)-1 and new_val >= current[i+1]:
                attempts += 1
                continue
            
            trial = current.copy()
            trial[i] = new_val
            
            sums = set()
            valid = True
            for j in range(len(trial)):
                s = 0
                for k in range(j, len(trial)):
                    s += trial[k]
                    if s in sums:
                        valid = False
                        break
                    sums.add(s)
                if not valid:
                    break
            
            if valid:
                ext, ext_sums = _extend_greedy(trial, sums, 
                                               [c for c in range(trial[-1]+1, 257)], deadline)
                
                if len(ext) > best_len:
                    current = ext
                    current_sums = ext_sums
                    best = list(current)
                    best_len = len(current)
                elif len(ext) >= best_len and random.random() < 0.15:
                    current = ext
                    current_sums = ext_sums
                break
            attempts += 1
        
        temperature *= cooling
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    candidates = []
    
    # Strategy 1: Greedy smallest
    if time.time() < deadline:
        c1 = greedy_smallest(deadline)
        candidates.append(c1)
    
    # Strategy 2: Greedy largest
    if time.time() < deadline:
        c2 = greedy_largest(deadline)
        candidates.append(c2)
    
    # Strategy 3: Multiple greedy variants with different starting points
    for start in range(1, 60, 2):
        if time.time() > deadline - 10:
            break
        values, sums = [], set()
        for x in range(start, 257):
            if time.time() > deadline:
                break
            ok, new_sums = _try_append(values, sums, x)
            if ok:
                values.append(x)
                sums = new_sums
        if values:
            candidates.append(values)
    
    # Strategy 4: Beam search
    if time.time() < deadline - 15:
        beam_result = _beam_search(deadline, beam_width=80)
        if beam_result:
            candidates.append(beam_result)
    
    # Find best candidate so far
    best = max(candidates, key=len) if candidates else []
    
    # Refine best candidates using local search
    if time.time() < deadline - 15 and best:
        ls_result = _local_search(best, deadline)
        if len(ls_result) > len(best):
            best = ls_result
    
    # Use backtracking on the best candidate
    if time.time() < deadline - 10 and best:
        bt_result = _backtracking_search(best, deadline, max_nodes=250000)
        if len(bt_result) > len(best):
            best = bt_result
    
    # Try simulated annealing if time permits
    if time.time() < deadline - 8:
        sa_result = _simulated_annealing(deadline)
        if len(sa_result) > len(best):
            best = sa_result
    
    # Final greedy extension
    if best and time.time() < deadline:
        sums = _compute_sums(best)
        for x in range(best[-1]+1, 257):
            if time.time() > deadline:
                break
            ok, new_sums = _try_append(best, sums, x)
            if ok:
                best.append(x)
                sums = new_sums
    
    return best if best else greedy_smallest(deadline)
# EVOLVE-BLOCK-END
