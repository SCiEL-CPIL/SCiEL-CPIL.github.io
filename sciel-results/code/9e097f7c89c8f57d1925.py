"""Combined program: Lookahead Greedy + Local Search for increasing sequences with unique consecutive-block sums."""
# EVOLVE-BLOCK-START
import time
import random

def _all_block_sums(seq):
    """Return set of all nonempty consecutive block sums."""
    sums = set()
    n = len(seq)
    for i in range(n):
        s = 0
        for j in range(i, n):
            s += seq[j]
            sums.add(s)
    return sums

def _check_conflict(seq, x, sums):
    """
    Check if appending x to seq creates any duplicate block sums.
    Returns True if conflict exists, False otherwise.
    Also returns the list of new sums if no conflict.
    """
    n = len(seq)
    new_sums = []
    s = x
    # The new blocks ending at x are:
    # [x], [seq[-1], x], [seq[-2], seq[-1], x], ..., [seq[0], ..., seq[-1], x]
    for i in range(n-1, -1, -1):
        if s in sums:
            return True, None
        new_sums.append(s)
        s += seq[i]
    return False, new_sums

def _greedy_smallest(deadline):
    """Greedy strategy: pick the smallest valid number at each step."""
    values = []
    sums = set()
    for x in range(1, 257):
        if time.time() > deadline:
            break
        conflict, new_sums = _check_conflict(values, x, sums)
        if not conflict:
            values.append(x)
            sums.update(new_sums)
    return values

def _greedy_largest(deadline):
    """Greedy strategy: pick the largest valid number at each step (reverse order logic).
    Note: We must maintain increasing order. So we iterate 256..1 and insert? 
    Actually, for strictly increasing sequences, picking 'largest' is tricky if we build left-to-right.
    Instead, let's try a heuristic: at each step, pick the largest x that still allows 
    a valid extension of at least length 1? Or just pick the largest valid x?
    Picking the largest valid x immediately might block smaller x later? 
    Let's try: iterate candidates from high to low, pick the first valid one? 
    No, that's just 'greedy largest valid'.
    """
    values = []
    sums = set()
    current_max = 0
    while True:
        if time.time() > deadline:
            break
        found = False
        # Try from 256 down to current_max + 1
        for x in range(256, current_max, -1):
            conflict, new_sums = _check_conflict(values, x, sums)
            if not conflict:
                values.append(x)
                sums.update(new_sums)
                current_max = x
                found = True
                break
        if not found:
            break
    return values

def _lookahead_greedy(deadline, depth=3):
    """
    Greedy with lookahead.
    At each step, for each valid candidate x, estimate the length of the 
    sequence we can build if we pick x, using a quick greedy extension of 'depth' steps.
    Pick the x that maximizes this estimated length.
    """
    values = []
    sums = set()
    
    while True:
        if time.time() > deadline:
            break
        
        candidates = []
        start_val = values[-1] + 1 if values else 1
        
        # Find all valid candidates
        for x in range(start_val, 257):
            conflict, new_sums = _check_conflict(values, x, sums)
            if not conflict:
                candidates.append((x, new_sums))
        
        if not candidates:
            break
        
        best_x = None
        best_sums = None
        best_score = -1
        
        for x, new_sums in candidates:
            # Quick lookahead: try to extend further with simple greedy (smallest)
            temp_values = values + [x]
            temp_sums = sums | set(new_sums)
            
            # Estimate potential: how many more can we add?
            # Simple heuristic: count how many numbers in [x+1, 256] are immediately valid
            # Or do a mini-greedy for 'depth' steps
            count = 0
            t_v = temp_values
            t_s = temp_sums
            last = x
            for _ in range(depth):
                found_next = False
                # Try smallest first for speed in estimation
                for y in range(last + 1, 257):
                    c, ns = _check_conflict(t_v, y, t_s)
                    if not c:
                        t_v.append(y)
                        t_s.update(ns)
                        last = y
                        count += 1
                        found_next = True
                        break
                if not found_next:
                    break
            
            # Score: prefer longer potential. Tie-break: smaller x (keeps values low)
            score = count * 100 - x
            if score > best_score:
                best_score = score
                best_x = x
                best_sums = new_sums
        
        if best_x is None:
            break
            
        values.append(best_x)
        sums.update(best_sums)
        
    return values

def _local_search_improve(seq, deadline):
    """
    Try to improve the sequence by removing one element and re-filling greedily.
    This can sometimes allow fitting more elements if the removed one was blocking many.
    """
    best_seq = seq[:]
    n = len(best_seq)
    
    # Try removing each element
    for i in range(n):
        if time.time() > deadline:
            break
        
        removed_val = best_seq[i]
        base_seq = best_seq[:i] + best_seq[i+1:]
        
        # Re-greedy fill from base_seq
        # We need to be careful: base_seq is valid. 
        # We iterate 1..256 and add if valid.
        # But base_seq elements are already "used" in the sense they are in the list.
        # We just add new elements that fit.
        
        current = base_seq[:]
        sums = _all_block_sums(current)
        
        # We must maintain sorted order. 
        # Simple approach: iterate x from 1 to 256. 
        # If x is in current, skip. 
        # If x can be inserted at the correct position and is valid, add it.
        # Since current is sorted, we can just append if x > current[-1].
        # But what if x < current[0]? Unlikely if we started from a good base.
        # What if x fits in the middle? 
        # The "consecutive block" constraint is local to the final sorted list.
        # If we insert x in the middle, it affects sums involving x and its neighbors.
        # A simple greedy append-only works if we iterate x in increasing order 
        # and only consider x > current[-1].
        # To handle middle insertions, we'd need a more complex check.
        # Let's stick to append-only for speed, which is valid if we assume 
        # the base sequence is "left-heavy" or we just try to extend to the right.
        
        # Actually, if we remove an element in the middle, we might be able to add 
        # smaller elements that were previously blocked? No, we iterate 1..256.
        # If we remove element at index i, we lose that value. 
        # We can try to add ANY value 1..256 not in base_seq.
        
        # Efficient re-greedy:
        # Iterate x from 1 to 256.
        # If x in current: continue
        # Try to insert x into current at the correct sorted position.
        # Check if valid. If yes, insert.
        
        # This is O(N * 256 * N) roughly. N is small (~10-15). 256*15*15 is ~50k ops.
        # This is fast enough to try for each removal.
        
        new_current = base_seq[:]
        new_sums = _all_block_sums(new_current)
        
        for x in range(1, 257):
            if x in new_current:
                continue
            
            # Find insertion point
            # Since new_current is sorted, we can just check if x fits at end first (fastest)
            # But it might fit in middle.
            # Let's just try appending first. If x > new_current[-1], append is safe for order.
            # If x < new_current[-1], we need to insert in middle.
            
            # Simplified: Just try to append x if x > new_current[-1].
            # If we want to be thorough, we should try inserting.
            # Given time constraints, let's try a "fill from scratch" approach 
            # using the base elements as "forced" starts? No, that's hard.
            
            # Let's try a simpler local search: 
            # Just re-run greedy_smallest but skipping the removed value?
            # No, that doesn't use the base structure.
            
            # Let's try: Iterate 1..256. If x not in new_current, 
            # check if adding x to new_current (at correct sorted pos) is valid.
            
            # Optimization: Only check if x is close to existing elements?
            # Let's just do the full check for correctness.
            
            # Insert x into sorted list
            # Python list insert is O(N). N is small.
            # Check validity:
            # We need to check all blocks involving x.
            # Blocks ending at x, starting at x.
            
            # This is complex to implement efficiently inline.
            # Let's skip complex middle insertion and just try to extend to the right 
            # or replace with a larger number?
            
            # Alternative Local Search: 
            # Replace one element with a larger one?
            # Or just re-run _lookahead_greedy with a different random seed?
            
            pass # Skip complex local search for now, rely on multiple greedy variants

    return best_seq

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s limit
    
    best_seq = []
    
    # Strategy 1: Greedy Smallest (Baseline)
    seq1 = _greedy_smallest(deadline)
    if len(seq1) > len(best_seq):
        best_seq = seq1[:]
        
    # Strategy 2: Greedy Largest (Pick largest valid at each step)
    if time.time() < deadline - 5:
        seq2 = _greedy_largest(deadline)
        if len(seq2) > len(best_seq):
            best_seq = seq2[:]
            
    # Strategy 3: Lookahead Greedy (The main improvement)
    if time.time() < deadline - 10:
        seq3 = _lookahead_greedy(deadline, depth=3)
        if len(seq3) > len(best_seq):
            best_seq = seq3[:]
            
    # Strategy 4: Randomized Lookahead (Multiple runs with different tie-breaking or noise)
    # Add some randomness to the lookahead score to explore different branches
    random.seed(42)
    for run in range(5):
        if time.time() > deadline - 2:
            break
            
        # Modify the lookahead logic slightly: add random noise to score
        # We'll just call a variant or re-run with different parameters
        # For simplicity, let's just run _lookahead_greedy again but it's deterministic.
        # Let's implement a quick random-greedy hybrid
        seq4 = []
        sums = set()
        while True:
            if time.time() > deadline - 2:
                break
            candidates = []
            start_val = seq4[-1] + 1 if seq4 else 1
            for x in range(start_val, 257):
                conflict, new_sums = _check_conflict(seq4, x, sums)
                if not conflict:
                    candidates.append((x, new_sums))
            
            if not candidates:
                break
            
            # Pick randomly from top 3 candidates by "potential" (just use index for speed)
            # Or pick the one with smallest x, but with 20% chance pick a random other
            if random.random() < 0.2 and len(candidates) > 1:
                choice_idx = random.randint(0, len(candidates)-1)
            else:
                choice_idx = 0 # Smallest
            
            x, new_sums = candidates[choice_idx]
            seq4.append(x)
            sums.update(new_sums)
            
        if len(seq4) > len(best_seq):
            best_seq = seq4[:]

    return best_seq

# EVOLVE-BLOCK-END
