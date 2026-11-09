import time
import random
import sys

# EVOLVE-BLOCK-START

def _is_valid_full(values):
    """Check if a full sequence has all distinct consecutive block sums."""
    n = len(values)
    seen = set()
    for i in range(n):
        total = 0
        for j in range(i, n):
            total += values[j]
            if total in seen:
                return False
            seen.add(total)
    return True

def _greedy_forward(values, max_val=256):
    """Greedy: pick smallest valid next element."""
    current = list(values)
    cur_sums = set()
    for i in range(len(current)):
        total = 0
        for j in range(i, len(current)):
            total += current[j]
            cur_sums.add(total)
    
    for x in range((current[-1] if current else 0) + 1, max_val + 1):
        tails = [x]
        total = x
        for a in reversed(current):
            total += a
            tails.append(total)
        if not cur_sums.intersection(tails):
            current.append(x)
            cur_sums.update(tails)
    return current

def _compute_sums(values):
    """Compute all consecutive block sums for a given sequence."""
    sums = set()
    n = len(values)
    for i in range(n):
        total = 0
        for j in range(i, n):
            total += values[j]
            sums.add(total)
    return sums

def _beam_search(max_val=256, beam_width=50, deadline=None, initial_beam=None):
    """
    Beam Search to maximize sequence length.
    State: (seq, sums_set)
    Heuristic: Prefer sequences that can potentially grow longer.
    Score for a state: 
      1. Length (primary)
      2. Potential: max_val - last_val (more room = better)
      3. Density: Smaller gaps are generally better? Or maybe just room is enough.
    We keep the top 'beam_width' states based on (length, potential).
    """
    if deadline is None:
        deadline = time.time() + 50
        
    # Initialize beam
    if initial_beam is None:
        beam = [([], set())]
    else:
        beam = initial_beam
    
    best_seq = []
    best_len = 0
    
    # Update best from initial beam
    for seq, _ in beam:
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = list(seq)
            
    while time.time() < deadline:
        next_beam = []
        
        # Expand each state in the current beam
        for seq, sums in beam:
            if not seq:
                last_val = 0
            else:
                last_val = seq[-1]
                
            if last_val >= max_val:
                continue
                
            # Pruning: If we can't beat best_len even with max possible additions, skip?
            # Max additions is max_val - last_val.
            if len(seq) + (max_val - last_val) <= best_len:
                continue
                
            # Try extending
            # Optimization: Instead of trying all 1..256, we can iterate.
            # For each candidate x, check validity.
            for x in range(last_val + 1, max_val + 1):
                # Time check inside loop to be safe
                if time.time() > deadline:
                    return best_seq, best_len
                    
                # Check if x is valid
                tails = [x]
                total = x
                for a in reversed(seq):
                    total += a
                    tails.append(total)
                
                # Check intersection
                valid = True
                for t in tails:
                    if t in sums:
                        valid = False
                        break
                        
                if valid:
                    new_seq = seq + [x]
                    new_sums = set(sums)
                    new_sums.update(tails)
                    
                    # Update global best
                    if len(new_seq) > best_len:
                        best_len = len(new_seq)
                        best_seq = list(new_seq)
                        
                    # Add to next beam candidate
                    next_beam.append((new_seq, new_sums))
                    
                    # Early exit if we found a very long sequence? 
                    # No, keep exploring to ensure we don't miss a longer one.
                    # But we can break if we hit max possible length?
                    # Max length is limited by max_val.
        
        if not next_beam:
            break
            
        # Prune next_beam to keep only the best 'beam_width' states
        # Heuristic: 
        # 1. Length is most important.
        # 2. Among same length, prefer smaller last_val (more room).
        # 3. Among same length and last_val, maybe prefer smaller sum set size? (Harder to collide?)
        
        # Sort by (length, last_val)
        # We want to keep states with max length. If lengths are equal, keep those with smaller last_val.
        # Note: Python sort is stable. We want descending length, ascending last_val.
        
        # To avoid memory blowup, we can sort and slice.
        # next_beam can be large. Let's limit its size before sorting if it's huge?
        # Or just sort. For beam_width=50, next_beam size is ~ 50 * 256 = 12800. Sorting is fine.
        
        next_beam.sort(key=lambda item: (-len(item[0]), item[0][-1] if item[0] else 0))
        
        # Deduplicate states? 
        # Two states with same seq and same sums are identical.
        # We can use a set of tuples? Sums are sets, not hashable.
        # We can hash the seq and the frozenset of sums?
        # Or just keep the top N. Since we sorted, the top N are the best.
        # If there are duplicates in the top N, it wastes space.
        # Let's try to deduplicate the top beam_width * 2 candidates first.
        
        seen_states = set()
        unique_beam = []
        for seq, sums in next_beam:
            # Simple dedup key: seq tuple + min(sums) + max(sums)? Not unique enough.
            # Just use seq as key? No, different sums for same seq is impossible.
            # Wait, if seq is same, sums MUST be same.
            # So key = tuple(seq) is sufficient to dedup identical states.
            key = tuple(seq)
            if key not in seen_states:
                seen_states.add(key)
                unique_beam.append((seq, sums))
                if len(unique_beam) >= beam_width:
                    break
        
        beam = unique_beam
        
        # If the best length in the beam is not increasing, we might be stuck?
        # Beam search doesn't guarantee global optimum, but it's fast.
        
        # If the max length in the new beam is less than or equal to the previous max length 
        # and we've explored all extensions, we might be done? 
        # Actually, if next_beam is empty, we break.
        # If the max length in beam is L, and we can't extend any to L+1, we are done.
        # The sort ensures the first element is the best.
        if beam:
            current_best_in_beam = len(beam[0][0])
            # If this is not greater than the best_len we've seen so far? 
            # best_len is global. 
            # If the best state in the beam cannot be extended (handled by loop), 
            # and no new states were added that are longer, we might loop forever?
            # No, because last_val increases, so the search space shrinks.
            # Eventually last_val >= max_val for all, and next_beam becomes empty.
            
    return best_seq, best_len

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin for 60s limit
    
    best = []
    best_len = 0
    
    # 1. Get a good initial lower bound using Greedy
    greedy_seq = _greedy_forward([])
    if len(greedy_seq) > best_len:
        best = greedy_seq
        best_len = len(best)
        
    # 2. Run Beam Search
    # Use a reasonably wide beam. 
    # The search space is constrained by the distinct sums.
    # Beam width 100-200 should be fast enough in Python if optimized.
    
    # Run multiple beam searches with different random seeds for initialization?
    # Or just run one robust beam search.
    # Beam search from empty is good.
    
    found_seq, found_len = _beam_search(max_val=256, beam_width=100, deadline=deadline)
    if found_len > best_len:
        best = found_seq
        best_len = found_len
        
    # 3. If time remains, try to improve with a randomized local search or another beam pass?
    # Beam search is already stochastic-ish if we randomize the order of candidates?
    # Currently it tries candidates in increasing order. 
    # We can randomize the candidate order to find different local optima.
    
    if time.time() < deadline - 5:
        rng = random.Random(42)
        # Run a few more beam searches with randomized candidate ordering
        # This requires modifying _beam_search to accept a randomizer or we implement a variant.
        # For simplicity, let's just run the standard beam search again? 
        # It's deterministic, so it will find the same result.
        
        # Let's try a "Greedy + Beam" hybrid where we start the beam from a few random valid prefixes.
        
        for _ in range(3):
            if time.time() > deadline - 2:
                break
            
            # Generate a random valid prefix of length 2-4
            prefix = []
            cur_sums = set()
            last = 0
            for _ in range(3):
                found = False
                # Try a few random picks
                for _ in range(20):
                    x = rng.randint(last + 1, 256)
                    tails = [x]
                    total = x
                    for a in reversed(prefix):
                        total += a
                        tails.append(total)
                    if not cur_sums.intersection(tails):
                        prefix.append(x)
                        cur_sums.update(tails)
                        last = x
                        found = True
                        break
                if not found:
                    break
            
            if len(prefix) >= 2:
                # Run beam search starting from this prefix
                initial_beam = [(prefix, cur_sums)]
                found_seq, found_len = _beam_search(max_val=256, beam_width=50, deadline=deadline, initial_beam=initial_beam)
                if found_len > best_len:
                    best = found_seq
                    best_len = found_len

    # Final validation
    if not _is_valid_full(best):
        # Fallback
        best = _greedy_forward([])
        
    return best

# EVOLVE-BLOCK-END
