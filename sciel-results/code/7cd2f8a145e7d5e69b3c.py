import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 5s safety margin
    
    # We are looking for a Golomb ruler with m marks.
    # The sequence length n = m - 1.
    # Marks x_0=0 < x_1 < ... < x_{m-1}.
    # a_i = x_i - x_{i-1}.
    # Constraints:
    # 1. All |x_i - x_j| distinct.
    # 2. a_i <= 256.
    # 3. a_i >= 1 (implied by strict increase).
    
    best_seq = []
    best_len = 0
    
    # Beam search
    # State: (marks_list, diffs_set)
    # We start with mark 0.
    beam = [([], set())]
    
    # To allow for some diversity, we can run multiple beams or restarts.
    # But beam search with a decent width should be fine.
    
    K = 200  # Beam width
    max_marks = 35  # Try to find up to 35 marks (n=34)
    
    for step in range(max_marks):
        if time.time() > deadline:
            break
            
        new_beam = []
        
        for marks, diffs in beam:
            last = marks[-1] if marks else 0
            # Candidates for next mark
            # Must be > last
            # Gap <= 256
            # We prioritize smaller gaps to keep the ruler compact
            # Range: last+1 to last+256
            
            # Optimization: Instead of checking all 256 candidates for every state,
            # we can just check the first few valid ones?
            # No, we need to explore.
            # But 256 * 200 = 51200 checks per step.
            # Each check involves checking up to 'step' differences.
            # Total ops ~ 51200 * 30 = 1.5M. Fast enough.
            
            for c in range(last + 1, last + 257):
                # Check if any difference c - m is in diffs
                # Also need to ensure c - m > 0 (always true since c > last >= m)
                valid = True
                new_diffs = []
                for m in marks:
                    d = c - m
                    if d in diffs:
                        valid = False
                        break
                    new_diffs.append(d)
                
                if valid:
                    new_marks = marks + [c]
                    new_diffs_set = diffs.union(new_diffs)
                    new_beam.append((new_marks, new_diffs_set))
        
        if not new_beam:
            break
            
        # Update best
        # The length of the sequence is len(marks) - 1
        # We want to maximize this.
        # All states in new_beam have length step+1 marks.
        # So if we reach this step, we have a valid sequence of length step.
        best_len = step
        # Store one example
        if best_len > 0:
             # Convert first state in beam to sequence
             example_marks = new_beam[0][0]
             seq = [example_marks[i] - example_marks[i-1] for i in range(1, len(example_marks))]
             best_seq = seq
             
        # Prune beam
        # Sort by last mark (smallest is best, keeps options open)
        # Also maybe by "spread" or something?
        # Smallest last mark is the standard heuristic.
        new_beam.sort(key=lambda state: state[0][-1])
        
        if len(new_beam) > K:
            new_beam = new_beam[:K]
            
        beam = new_beam
        
    # Fallback: Powers of 2
    if not best_seq:
        seq = []
        s = 0
        x = 1
        while x <= 256:
            seq.append(x)
            s += x
            x = s + 1
        best_seq = seq
        best_len = len(best_seq)
        
    # Final check
    if not _check_valid(best_seq):
        # Generate powers of 2 again
        seq = []
        s = 0
        x = 1
        while x <= 256:
            seq.append(x)
            s += x
            x = s + 1
        best_seq = seq
        
    return best_seq

def _check_valid(seq):
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
