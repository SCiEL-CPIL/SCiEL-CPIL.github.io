"""
Structural change: A* search with forward-checking constraint propagation,
beam search with density-based heuristic, and meet-in-the-middle extension.
Key insight: track sum-set density to prune branches where too many sums
are already occupied, and use forward-checking to eliminate candidates
before they enter the search tree.
"""
import time
import random
import sys
import heapq
from collections import defaultdict

def _check_valid(seq):
    """Verify strictly increasing in [1,256] and all consecutive block sums distinct."""
    n = len(seq)
    if n == 0:
        return True
    if seq[0] < 1 or seq[-1] > 256:
        return False
    for i in range(1, n):
        if seq[i] <= seq[i-1]:
            return False
    sums = set()
    for i in range(n):
        s = 0
        for j in range(i, n):
            s += seq[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _compute_sums(seq):
    """Compute all consecutive block sums for a sequence."""
    n = len(seq)
    sums = set()
    for i in range(n):
        s = 0
        for j in range(i, n):
            s += seq[j]
            sums.add(s)
    return sums

def _tails_for_value(seq, v):
    """Compute all new block sums that would be introduced by appending v to seq."""
    n = len(seq)
    total = v
    tails = [total]
    for i in range(n - 1, -1, -1):
        total += seq[i]
        tails.append(total)
    return tails

def _density_heuristic(sums, last_val, seq_len):
    """
    Estimate how many more elements can be added.
    Uses the fact that each new element of value v adds (seq_len+1) new sums.
    The total number of possible sums in range [1, 256*(256+1)//2] is bounded.
    We estimate remaining capacity based on sum-set density.
    """
    max_possible_sum = 256 * 256  # upper bound on any block sum
    # Number of distinct sums so far
    n_sums = len(sums)
    # Each new element adds (seq_len + 1) new sums
    # Total possible sums is at most max_possible_sum
    # Estimate how many more elements fit
    remaining_capacity = max(0, max_possible_sum - n_sums)
    est_elements = remaining_capacity // (seq_len + 2)
    # Also bounded by available values
    val_bound = 256 - last_val
    return min(est_elements, val_bound)

def _construct_powers():
    """Direct construction: powers-of-2-like gaps guarantee unique block sums."""
    seq = []
    s = 0
    x = 1
    while x <= 256:
        seq.append(x)
        s += x
        x = s + 1
    return seq

def _construct_with_forward_check(deadline, beam_width=50):
    """
    Beam search with forward-checking constraint propagation.
    At each step, maintain a beam of top states ranked by a heuristic
    that combines current length and estimated remaining capacity.
    Forward-checking eliminates values that would create sum conflicts
    before they are committed.
    """
    start_time = time.time()
    
    # State: (seq, sums, last_val)
    # Priority: negative of heuristic (for min-heap)
    initial_state = ([], set(), 0)
    
    beam = [(0, 0, initial_state)]  # (priority, counter, state)
    counter = 1
    best_seq = []
    
    max_steps = 40
    for step in range(max_steps):
        if time.time() > deadline - 5:
            break
        
        # Expand beam
        next_beam = []
        for priority, _, state in beam:
            seq, sums, last_val = state
            n = len(seq)
            
            # Find all valid extensions
            valid_candidates = []
            for v in range(last_val + 1, 257):
                tails = _tails_for_value(seq, v)
                if not any(t in sums for t in tails):
                    valid_candidates.append(v)
            
            if not valid_candidates:
                continue
            
            # Rank candidates: prefer those that leave more room
            # Use a scoring function: lower values tend to allow more future elements
            # but also check how many sums they add vs density
            scored = []
            for v in valid_candidates:
                new_sums_count = len(sums) + len(_tails_for_value(seq, v))
                # Heuristic: prefer values that keep sum-set sparse
                # and leave room for more elements
                remaining_vals = 256 - v
                density_penalty = new_sums_count / (n + 2)
                score = -n * 100 - density_penalty * 0.1 + remaining_vals * 0.5
                scored.append((score, v))
            
            # Take top beam_width candidates
            scored.sort(reverse=True)
            for score, v in scored[:beam_width]:
                new_seq = seq + [v]
                new_sums = sums.union(_tails_for_value(seq, v))
                new_last = v
                new_state = (new_seq, new_sums, new_last)
                
                # Compute heuristic for priority
                est_remaining = _density_heuristic(new_sums, new_last, n + 1)
                total_est = (n + 1) + est_remaining
                # Minimize negative of total estimate
                prio = -total_est
                next_beam.append((prio, counter, new_state))
                counter += 1
        
        if not next_beam:
            break
        
        # Keep only top beam_width states
        next_beam.sort(key=lambda x: x[0])
        beam = next_beam[:beam_width]
        
        # Track best
        for _, _, state in beam:
            seq = state[0]
            if len(seq) > len(best_seq):
                best_seq = list(seq)
        
        # Early termination if no improvement possible
        max_len_in_beam = max(len(s[0]) for s in beam)
        if max_len_in_beam == len(best_seq):
            # Check if we can still improve
            any_extendable = False
            for _, _, state in beam:
                seq, sums, last_val = state
                for v in range(last_val + 1, min(last_val + 10, 257)):
                    tails = _tails_for_value(seq, v)
                    if not any(t in sums for t in tails):
                        any_extendable = True
                        break
                if any_extendable:
                    break
            if not any_extendable:
                break
    
    return best_seq

def _construct_greedy_smallest():
    """Deterministic greedy smallest - fast baseline."""
    values, sums = [], set()
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

def _construct_greedy_with_lookahead(deadline, lookahead=3):
    """
    Greedy with lookahead: at each step, try each candidate value and
    estimate how many more elements can be added after choosing it.
    Pick the candidate that maximizes (immediate + estimated future) length.
    """
    start_time = time.time()
    values = []
    sums = set()
    
    while True:
        if time.time() > deadline - 5:
            break
        
        last_val = values[-1] if values else 0
        candidates = []
        
        for x in range(last_val + 1, 257):
            tails = _tails_for_value(values, x)
            if any(t in sums for t in tails):
                continue
            
            # Lookahead: try to extend by 'lookahead' more elements
            test_values = values + [x]
            test_sums = sums.union(tails)
            
            # Try to greedily extend further
            extra = 0
            tv = test_values
            ts = test_sums
            for _ in range(lookahead):
                lv = tv[-1] if tv else 0
                best_next = None
                for nx in range(lv + 1, 257):
                    nt = _tails_for_value(tv, nx)
                    if not any(t in ts for t in nt):
                        best_next = nx
                        break
                if best_next is None:
                    break
                tv = tv + [best_next]
                ts = ts.union(_tails_for_value(test_values, best_next))
                extra += 1
            
            total_est = len(values) + 1 + extra
            candidates.append((total_est, x))
        
        if not candidates:
            break
        
        # Pick best candidate
        candidates.sort(key=lambda c: (-c[0], c[1]))
        best_est, best_x = candidates[0]
        
        # Commit
        tails = _tails_for_value(values, best_x)
        values.append(best_x)
        sums.update(tails)
        
        # Check if we should stop (no improvement in estimate)
        if best_est <= len(values):
            break
    
    return values

def _construct_greedy_randomized(seed):
    """Greedy with randomized value ordering."""
    rng = random.Random(seed)
    values, sums = [], set()
    candidates = list(range(1, 257))
    rng.shuffle(candidates)
    for x in candidates:
        if values and x <= values[-1]:
            continue
        tails = [x]
        total = x
        for a in reversed(values):
            total += a
            tails.append(total)
        if not sums.intersection(tails):
            values.append(x)
            sums.update(tails)
    return values

def _astar_search(deadline, max_expansions=5000):
    """
    A* search with admissible heuristic.
    State: (seq, sums, last_val)
    Heuristic: upper bound on additional elements that can be added.
    Uses the constraint that each new element adds (len+1) distinct sums,
    and the total sum space is bounded.
    """
    start_time = time.time()
    
    # Priority queue: (f_score, counter, seq, sums, last_val)
    # f = g + h, where g = current length, h = estimated remaining
    initial = ([], set(), 0)
    open_set = [(0, 0, initial)]  # (f, counter, state)
    counter = 1
    best_seq = []
    best_len = 0
    expansions = 0
    
    # Track visited states by (last_val, frozenset of sums) - too expensive
    # Instead, use a simple limit on expansions
    
    while open_set and expansions < max_expansions:
        if time.time() > deadline - 5:
            break
        
        f, _, state = heapq.heappop(open_set)
        seq, sums, last_val = state
        g = len(seq)
        
        if g > best_len:
            best_len = g
            best_seq = list(seq)
        
        # Compute heuristic
        h = _density_heuristic(sums, last_val, g)
        
        if h == 0:
            continue
        
        # Generate successors
        for v in range(last_val + 1, 257):
            tails = _tails_for_value(seq, v)
            if any(t in sums for t in tails):
                continue
            
            new_seq = seq + [v]
            new_sums = sums.union(tails)
            new_g = g + 1
            new_h = _density_heuristic(new_sums, v, new_g)
            new_f = new_g + new_h
            
            # Only push if promising
            if new_f > best_len:
                heapq.heappush(open_set, (new_f, counter, (new_seq, new_sums, v)))
                counter += 1
        
        expansions += 1
    
    return best_seq

def _backtrack_fast(deadline, target_len=None, rng=None):
    """
    Fast backtracking using a set of block sums.
    Randomized candidate ordering for diversity across restarts.
    """
    if rng is None:
        rng = random.Random()
    
    best_seq = []
    best_len = 0
    
    def _try_extend(seq, sums, last_val):
        """Yield (v, new_sums_list) for each valid extension value v."""
        n = len(seq)
        for v in range(last_val + 1, 257):
            total = v
            new_sums_list = [total]
            valid = True
            for i in range(n - 1, -1, -1):
                total += seq[i]
                if total in sums:
                    valid = False
                    break
                new_sums_list.append(total)
            if valid:
                yield v, new_sums_list
    
    def dfs(seq, sums, last_val, depth):
        nonlocal best_seq, best_len
        if time.time() > deadline:
            return
        
        n = len(seq)
        if n > best_len:
            best_len = n
            best_seq = list(seq)
        
        if target_len is not None and n >= target_len:
            return
        
        # Pruning
        if n + (256 - last_val) <= best_len:
            return
        if target_len is not None and n + (256 - last_val) < target_len:
            return
        
        # Generate candidates
        candidates = list(_try_extend(seq, sums, last_val))
        if not candidates:
            return
        
        # Randomize order
        rng.shuffle(candidates)
        
        # Limit branching
        max_branch = 20
        if len(candidates) > max_branch:
            candidates.sort(key=lambda x: x[0])
            half = len(candidates) // 2
            chosen = candidates[:half]
            if half < max_branch:
                remaining = candidates[half:]
                if remaining:
                    chosen.extend(rng.sample(remaining, min(len(remaining), max_branch - half)))
            candidates = chosen[:max_branch]
        
        for v, new_sums_list in candidates:
            new_seq = seq + [v]
            new_sums = sums.union(new_sums_list)
            dfs(new_seq, new_sums, v, depth + 1)
            if time.time() > deadline:
                return
            if target_len is not None and best_len >= target_len:
                return
    
    dfs([], set(), 0, 0)
    return best_seq

def _meet_in_middle_extend(seq, sums, deadline, max_depth=5):
    """
    Try to extend a sequence using meet-in-the-middle:
    split the remaining search space and try to connect.
    This is useful when direct extension fails but a slightly different
    prefix could lead to a longer sequence.
    """
    if not seq:
        return seq
    
    n = len(seq)
    last_val = seq[-1]
    
    # Try each possible modification of the last few elements
    best_seq = list(seq)
    best_len = n
    
    for mod_depth in range(1, min(4, n + 1)):
        if time.time() > deadline - 3:
            break
        
        # Try changing the last mod_depth elements
        fixed_prefix = seq[:n - mod_depth]
        prefix_sums = _compute_sums(fixed_prefix) if fixed_prefix else set()
        
        # For the prefix, compute what sums are "locked in"
        # Then try to build the suffix optimally
        
        # Get all sums from the full original sequence for reference
        # Try greedy extension from the modified prefix
        for attempt in range(10):
            if time.time() > deadline - 3:
                break
            
            # Randomize the last few positions
            test_seq = list(fixed_prefix)
            test_sums = set(prefix_sums)
            
            # Add back some elements, possibly different
            start_val = fixed_prefix[-1] + 1 if fixed_prefix else 1
            rng = random.Random(hash((mod_depth, attempt)) % 100000)
            
            for _ in range(mod_depth):
                candidates = []
                for v in range(start_val, 257):
                    tails = _tails_for_value(test_seq, v)
                    if not any(t in test_sums for t in tails):
                        candidates.append(v)
                
                if not candidates:
                    break
                
                # Pick randomly from top candidates
                candidates.sort()
                if len(candidates) > 5:
                    v = rng.choice(candidates[:5])
                else:
                    v = rng.choice(candidates)
                
                test_seq.append(v)
                test_sums.update(_tails_for_value(test_seq[:-1], v))
                start_val = v + 1
            
            # Now greedily extend the rest
            lv = test_seq[-1] if test_seq else 0
            for v in range(lv + 1, 257):
                tails = _tails_for_value(test_seq, v)
                if not any(t in test_sums for t in tails):
                    test_seq.append(v)
                    test_sums.update(tails)
            
            if len(test_seq) > best_len:
                best_len = len(test_seq)
                best_seq = test_seq
    
    return best_seq

def _local_repair(seq, deadline, rng):
    """Try to extend or repair a sequence."""
    if not _check_valid(seq):
        return seq
    
    best_seq = list(seq)
    best_sums = _compute_sums(best_seq)
    
    # Try extending
    for _ in range(50):
        if time.time() > deadline:
            break
        last = best_seq[-1] if best_seq else 0
        candidates = []
        n = len(best_seq)
        for v in range(last + 1, 257):
            tails = _tails_for_value(best_seq, v)
            if not any(t in best_sums for t in tails):
                candidates.append(v)
        
        if not candidates:
            break
        
        v = rng.choice(candidates)
        new_seq = best_seq + [v]
        new_sums = best_sums.union(_tails_for_value(best_seq, v))
        if _check_valid(new_seq):
            best_seq = new_seq
            best_sums = new_sums
    
    # Try swapping
    for _ in range(100):
        if time.time() > deadline:
            break
        if len(best_seq) < 2:
            break
        i = rng.randint(0, len(best_seq) - 1)
        old_val = best_seq[i]
        min_val = best_seq[i-1] + 1 if i > 0 else 1
        max_val = best_seq[i+1] - 1 if i < len(best_seq) - 1 else 256
        if min_val >= max_val:
            continue
        new_val = rng.randint(min_val, max_val)
        if new_val == old_val:
            continue
        test_seq = best_seq[:i] + [new_val] + best_seq[i+1:]
        if _check_valid(test_seq):
            best_seq = test_seq
            best_sums = _compute_sums(best_seq)
    
    return best_seq

def _multi_restart(deadline):
    """Multiple randomized restarts with diverse search strategies."""
    best_seq = _construct_greedy_smallest()
    best_len = len(best_seq)
    
    # Phase 1: greedy with lookahead
    if time.time() < deadline - 20:
        seq = _construct_greedy_with_lookahead(deadline - 20, lookahead=3)
        if len(seq) > best_len:
            best_seq = seq
            best_len = len(seq)
    
    # Phase 2: beam search with forward checking
    if time.time() < deadline - 20:
        seq = _construct_with_forward_check(deadline - 20, beam_width=40)
        if len(seq) > best_len:
            best_seq = seq
            best_len = len(seq)
    
    # Phase 3: randomized greedy
    for seed in range(10):
        if time.time() > deadline - 15:
            break
        seq = _construct_greedy_randomized(seed)
        if len(seq) > best_len:
            best_seq = seq
            best_len = len(seq)
    
    # Phase 4: A* search
    if time.time() < deadline - 15:
        seq = _astar_search(deadline - 15, max_expansions=3000)
        if len(seq) > best_len:
            best_seq = seq
            best_len = len(seq)
    
    # Phase 5: backtracking with increasing targets
    target = best_len + 1
    max_target = best_len + 6
    while target <= max_target and time.time() < deadline - 10:
        found = False
        for attempt in range(3):
            if time.time() > deadline - 10:
                break
            rng = random.Random(target * 7919 + attempt)
            seq = _backtrack_fast(deadline - 10, target_len=target, rng=rng)
            if len(seq) >= target:
                best_seq = seq
                best_len = len(seq)
                found = True
                break
        if not found:
            break
        target = best_len + 1
    
    # Phase 6: meet-in-the-middle extension
    if time.time() < deadline - 8:
        seq = _meet_in_middle_extend(best_seq, _compute_sums(best_seq), deadline - 8)
        if len(seq) > best_len:
            best_seq = seq
            best_len = len(seq)
    
    # Phase 7: local repair
    if time.time() < deadline - 3:
        rng = random.Random(12345)
        repaired = _local_repair(best_seq, deadline - 3, rng)
        if len(repaired) > best_len:
            best_seq = repaired
            best_len = len(repaired)
    
    return best_seq

def construct():
    start_time = time.time()
    deadline = start_time + 80
    
    best = _construct_greedy_smallest()
    best_len = len(best)
    
    if time.time() > deadline - 10:
        if not _check_valid(best):
            best = _construct_powers()
        return best
    
    try:
        result = _multi_restart(deadline)
        if len(result) > best_len:
            best = result
            best_len = len(best)
    except Exception:
        pass
    
    if not _check_valid(best):
        best = _construct_powers()
    
    return best
