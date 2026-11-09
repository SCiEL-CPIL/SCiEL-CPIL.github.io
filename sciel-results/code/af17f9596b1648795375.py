"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

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
    # Check for conflicts
    for t in tails:
        if t in sums:
            return False, None
    new_sums = sums | set(tails)
    return True, new_sums

def _greedy_extend(seq, sums, deadline):
    """Greedy extension of seq with smallest available values."""
    while time.time() < deadline:
        extended = False
        for x in range(seq[-1] + 1 if seq else 1, 257):
            ok, new_sums = _try_append(seq, sums, x)
            if ok:
                seq.append(x)
                sums = new_sums
                extended = True
                break
        if not extended:
            break
    return seq, sums

def beam_search(deadline, beam_width=50):
    """
    Beam search over sequence prefixes.
    State: (sequence, set_of_block_sums)
    At each level, keep top beam_width prefixes by length (then by sum flexibility).
    """
    # Initial beam: sequences of length 1
    beam = []
    for x in range(1, 257):
        beam.append(([x], {x}))
    
    best_seq = []
    best_len = 0
    
    while time.time() < deadline:
        if not beam:
            break
        
        # Update best
        for seq, sums in beam:
            if len(seq) > best_len:
                best_len = len(seq)
                best_seq = list(seq)
        
        # Generate next level
        next_beam = []
        seen_states = set()  # To avoid duplicate sum sets at same length
        
        for seq, sums in beam:
            last_val = seq[-1] if seq else 0
            # Try appending values greater than last_val
            for x in range(last_val + 1, 257):
                ok, new_sums = _try_append(seq, sums, x)
                if ok:
                    new_seq = seq + [x]
                    # Use frozenset of sums as state key (expensive but correct)
                    # For efficiency, we can use the sum of sums as a hash
                    state_key = (len(new_seq), sum(new_sums), len(new_sums))
                    if state_key not in seen_states:
                        seen_states.add(state_key)
                        next_beam.append((new_seq, new_sums))
        
        if not next_beam:
            break
        
        # Score and select top beam_width
        # Score: primarily length, secondarily "flexibility" (fewer sums = more room)
        def score(item):
            seq, sums = item
            # Prefer longer sequences
            # Among same length, prefer those with fewer distinct sums (more flexibility)
            # Also prefer those with smaller max sum (more room for larger values)
            return (len(seq), -len(sums), -max(sums) if sums else 0)
        
        next_beam.sort(key=score, reverse=True)
        beam = next_beam[:beam_width]
        
        # Early termination if we can't improve
        if not beam or len(beam[0][0]) <= best_len:
            # Check if any beam element can be extended
            can_extend = False
            for seq, sums in beam:
                last_val = seq[-1] if seq else 0
                for x in range(last_val + 1, 257):
                    ok, _ = _try_append(seq, sums, x)
                    if ok:
                        can_extend = True
                        break
                if can_extend:
                    break
            if not can_extend:
                break
    
    # Final greedy extension on best
    if best_seq:
        sums = _compute_sums(best_seq)
        best_seq, sums = _greedy_extend(best_seq, sums, deadline)
    
    return best_seq

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin
    
    best = []
    
    # Strategy 1: Beam search (main approach)
    if time.time() < deadline - 5:
        beam_result = beam_search(deadline, beam_width=30)
        if len(beam_result) > len(best):
            best = beam_result
    
    # Strategy 2: Greedy smallest for quick baseline
    if time.time() < deadline - 2:
        values, sums = [], set()
        for x in range(1, 257):
            if time.time() > deadline:
                break
            ok, new_sums = _try_append(values, sums, x)
            if ok:
                values.append(x)
                sums = new_sums
        if len(values) > len(best):
            best = values
    
    # Final greedy extension
    if best and time.time() < deadline:
        sums = _compute_sums(best)
        best, sums = _greedy_extend(best, sums, deadline)
    
    return best if best else []
# EVOLVE-BLOCK-END
